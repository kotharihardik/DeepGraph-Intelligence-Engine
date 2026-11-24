"""End-to-end GraphFusion QA: three graphs -> fused retrieval -> cited reader."""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

from .document import Document
from .reader import ExtractiveReader, LLMReader, ReaderOutput
from .entities import EntityIndex
from .retriever import DocIndex, GraphFusionRetriever, RetrieverConfig, RetrievalResult
from .span_graph import SpanGraph
from .text_utils import normalize_text


def light_index(doc: Document) -> DocIndex:
    """Index without neural encoders (spaCy + Span Graph only).

    Used by the reading phase of the evaluation, so the LLM never shares GPU
    memory with the retrieval encoders.
    """
    doc.annotate()
    ents = EntityIndex(doc)
    span = SpanGraph()
    span.build(doc, ents)
    return DocIndex(doc, ents, None, None, [], None, span=span)


def retrieval_to_json(ret: RetrievalResult) -> Dict:
    from dataclasses import asdict
    d = asdict(ret)
    d["sentence_scores"] = {str(k): v for k, v in ret.sentence_scores.items()}
    d["ce_prob"] = {str(k): v for k, v in ret.ce_prob.items()}
    return d


def retrieval_from_json(d: Dict) -> RetrievalResult:
    d = dict(d)
    d["sentence_scores"] = {int(k): v for k, v in d["sentence_scores"].items()}
    d["ce_prob"] = {int(k): v for k, v in d["ce_prob"].items()}
    return RetrievalResult(**d)


class GraphFusionQA:
    def __init__(self, retriever_cfg: RetrieverConfig = None, reader: str = "llm",
                 context_paras: int = 3, sp_mode: str = "union"):
        self.retriever_cfg = retriever_cfg or RetrieverConfig()
        self._retriever = None
        if reader == "llm":
            self.reader = LLMReader()
            if not self.reader.available():
                print("[GraphFusion] Ollama model not reachable - using extractive reader.")
                self.reader = ExtractiveReader()
        else:
            self.reader = ExtractiveReader()
        self.context_paras = context_paras
        self.sp_mode = sp_mode

    @property
    def retriever(self) -> GraphFusionRetriever:
        if self._retriever is None:
            self._retriever = GraphFusionRetriever(self.retriever_cfg)
        return self._retriever

    def release_encoders(self):
        """Free the embedder/reranker (and accelerator memory) before reading."""
        import gc
        from . import model_cache
        self._retriever = None
        model_cache._CACHE.clear()
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def index(self, doc: Document) -> DocIndex:
        return self.retriever.index(doc)

    def context_pids(self, ret: RetrievalResult) -> List[int]:
        pids = list(ret.chain)
        for p in ret.para_ranking:
            if len(pids) >= self.context_paras:
                break
            if p not in pids:
                pids.append(p)
        return pids

    def answer(self, idx: DocIndex, question: str) -> Dict:
        t0 = time.perf_counter()
        ret = self.retriever.retrieve(idx, question)
        return self.read(idx, ret, question, t_ret=time.perf_counter() - t0)

    def read(self, idx: DocIndex, ret: RetrievalResult, question: str, t_ret: float = 0.0) -> Dict:
        pids = self.context_pids(ret)
        cand_sids = [s for p in pids for s in idx.doc.paragraphs[p].sids]
        candidates = [sp.text for sp in idx.span.spans_in(cand_sids)] if idx.span else []
        out: ReaderOutput = self.reader.read(idx.doc, pids, question, candidates)

        if self.sp_mode == "retriever" or (self.sp_mode == "cited_or_retriever" and not out.cited_sids):
            sp = ret.sp_sids
        elif self.sp_mode == "union":
            sp = sorted(set(ret.sp_sids) | set(out.cited_sids))
        else:
            sp = out.cited_sids
        conf = self._confidence(idx, ret, out)
        return {
            "answer": out.answer,
            "raw_answer": out.raw_answer,
            "reasoning": out.reasoning,
            "cited_sids": out.cited_sids,
            "sp_sids": sp,
            "retriever_sp_sids": ret.sp_sids,
            "context_pids": pids,
            "context": out.context,
            "chain": ret.chain,
            "para_ranking": ret.para_ranking,
            "ranked_sids": ret.ranked_sids,
            "question_entities": ret.question_entities,
            "bridge_entities": ret.bridge_entities[:10],
            "graph_paths": ret.graph_paths,
            "confidence": conf,
            "postprocess": out.postprocess,
            "error": out.error,
            "timings": {**{f"index_{k}": v for k, v in idx.timings.items()},
                        **{f"ret_{k}": v for k, v in ret.timings.items()},
                        "retrieve": t_ret, "read": out.latency_s},
        }

    @staticmethod
    def _confidence(idx: DocIndex, ret: RetrievalResult, out: ReaderOutput) -> float:
        """Geometric mean of per-hop retrieval probability x answer grounding."""
        probs = []
        for p in ret.chain:
            ps = [ret.ce_prob.get(s, 0.0) for s in idx.doc.paragraphs[p].sids]
            probs.append(max(ps) if ps else 0.0)
        p_ret = math.exp(sum(math.log(max(p, 1e-6)) for p in probs) / max(1, len(probs)))
        ans = normalize_text(out.answer)
        if not ans or out.error:
            return 0.0
        grounded = ans in ("yes", "no") or ans in normalize_text(out.context)
        p_read = 1.0 if (grounded and out.cited_sids) else (0.6 if grounded else 0.2)
        return float(p_ret * p_read)
