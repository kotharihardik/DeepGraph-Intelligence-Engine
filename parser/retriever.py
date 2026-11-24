"""GraphFusion retriever: hybrid scoring + fused-graph PPR + multi-hop chains.

Per document (``index``)
  1. one spaCy pass, shared EntityIndex
  2. sentence embeddings ("title: sentence", BGE), BM25 index
  3. the three graphs (DRG, Span Graph, KG) and their fusion

Per question (``retrieve``)
  hop 1  dense + BM25 candidate pool -> cross-encoder rerank
         PPR on the fused graph seeded by question entities + top sentences
         sentence score = w_ce*z(CE) + w_ppr*z(log PPR) + w_dense*z(cos) + w_bm25*z(BM25)
  hop k  beam search over paragraph chains: the query is reformulated with
         the evidence already selected, PPR is re-seeded with the *bridge
         entities* found in that evidence, and paragraphs that are linked in
         the graph to the chain receive a link bonus
  output ranked sentences, the best paragraph chain, predicted supporting
         sentences and the graph paths that justify them.

All weights are z-scored per question, so they are scale-free; their values
were selected on a tuning split disjoint from the reported evaluation split
(see evaluation/tune.py).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from . import config
from .document import Document
from .drg_graph import DocumentReasoningGraph, sid_node
from .entities import EntityIndex, eid
from .graph_fusion import FusedGraph
from .knowledge_graph import KnowledgeGraph
from .model_cache import get_cross_encoder, get_embedder
from .span_graph import SpanGraph
from .text_utils import BM25, rrf


@dataclass
class RetrieverConfig:
    use_drg: bool = True
    use_span: bool = True
    use_kg: bool = True
    use_ppr: bool = True        # graph propagation signal
    use_links: bool = True      # graph link bonus between chained paragraphs
    use_chain: bool = True      # iterative (multi-hop) reformulated reranking
    use_rerank: bool = True
    max_candidates: int = 150
    sentence_seeds: int = 5
    entity_seed_mass: float = 0.3
    damping: float = 0.5
    w_ce: float = 1.0
    w_ppr: float = 0.5
    w_dense: float = 0.2
    w_bm25: float = 0.1
    beam: int = 3
    hops: int = 2
    link_bonus: float = 0.4
    hop_query_sents: int = 2
    sp_threshold: float = 0.8   # CE probability for extra supporting sentences

    @property
    def uses_graph(self) -> bool:
        return self.use_drg or self.use_span or self.use_kg

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DocIndex:
    doc: Document
    ents: EntityIndex
    emb: np.ndarray
    bm25: BM25
    texts: List[str]
    fused: Optional[FusedGraph]
    drg: Optional[DocumentReasoningGraph] = None
    span: Optional[SpanGraph] = None
    kg: Optional[KnowledgeGraph] = None
    para_mentions: Dict[int, Set[str]] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    ranked_sids: List[int]
    sentence_scores: Dict[int, float]
    ce_prob: Dict[int, float]
    para_ranking: List[int]
    chain: List[int]
    sp_sids: List[int]
    question_entities: List[str]
    bridge_entities: List[str]
    graph_paths: List[Dict]
    timings: Dict[str, float]


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def _release_accelerator_cache():
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


class GraphFusionRetriever:
    def __init__(self, cfg: RetrieverConfig = None):
        self.cfg = cfg or RetrieverConfig()
        self.embedder = get_embedder()
        self.reranker = get_cross_encoder() if self.cfg.use_rerank else None
        self._ce_cache: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------ index
    def index(self, doc: Document) -> DocIndex:
        cfg, t = self.cfg, {}
        t0 = time.perf_counter()
        doc.annotate()
        ents = EntityIndex(doc)
        t["nlp"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        texts = [f"{s.title}: {s.text}" for s in doc.sentences]
        emb = self.embedder.encode(texts, batch_size=64, normalize_embeddings=True,
                                   show_progress_bar=False)
        bm25 = BM25(texts)
        _release_accelerator_cache()
        t["encode"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        drg = span = kg = None
        if cfg.use_drg:
            drg = DocumentReasoningGraph()
            drg.build(doc, emb, ents)
        if cfg.use_span:
            span = SpanGraph()
            span.build(doc, ents)
        if cfg.use_kg:
            kg = KnowledgeGraph()
            kg.build(doc, ents)
        fused = None
        if cfg.uses_graph:
            fused = FusedGraph(drg.graph if drg else None, span.graph if span else None,
                               kg.fusion_edges() if kg else None,
                               [sid_node(s.sid) for s in doc.sentences])
        t["graphs"] = time.perf_counter() - t0

        idx = DocIndex(doc, ents, emb, bm25, texts, fused, drg, span, kg, timings=t)
        idx.para_mentions = self._paragraph_mentions(idx)
        return idx

    def refuse(self, idx: DocIndex) -> DocIndex:
        """Re-fuse an existing index using only the layers enabled in self.cfg.

        Lets ablations share one expensive index build per document.
        """
        cfg = self.cfg
        fused = None
        if cfg.uses_graph:
            fused = FusedGraph(idx.drg.graph if (cfg.use_drg and idx.drg) else None,
                               idx.span.graph if (cfg.use_span and idx.span) else None,
                               idx.kg.fusion_edges() if (cfg.use_kg and idx.kg) else None,
                               [sid_node(s.sid) for s in idx.doc.sentences])
        new = DocIndex(idx.doc, idx.ents, idx.emb, idx.bm25, idx.texts, fused,
                       idx.drg if cfg.use_drg else None, idx.span if cfg.use_span else None,
                       idx.kg if cfg.use_kg else None, timings=idx.timings)
        new.para_mentions = self._paragraph_mentions(new)
        return new

    def _paragraph_mentions(self, idx: DocIndex) -> Dict[int, Set[str]]:
        """Entities each paragraph points to, as visible in the fused graph."""
        out: Dict[int, Set[str]] = {p.pid: set() for p in idx.doc.paragraphs}
        if idx.fused is None:
            return out
        g = idx.fused.graph
        for s in idx.doc.sentences:
            node = sid_node(s.sid)
            if node not in g:
                continue
            for nb, d in g[node].items():
                if nb.startswith("E:") and d.get("type") != "about":
                    out[s.pid].add(nb)
                elif nb.startswith("P"):
                    out[s.pid].update(x for x in g[nb] if x.startswith("E:"))
                elif nb.startswith("S") and d.get("type") == "title_link":
                    out[s.pid].add(eid(d["entity"]))
        return out

    # ---------------------------------------------------------------- scoring
    def _ce(self, query: str, sids: List[int], idx: DocIndex) -> np.ndarray:
        if self.reranker is None:
            return np.zeros(len(sids))
        todo = [(query, idx.texts[s]) for s in sids if (query, idx.texts[s]) not in self._ce_cache]
        if todo:
            scores = self.reranker.predict(todo, batch_size=16, show_progress_bar=False)
            for pair, sc in zip(todo, np.atleast_1d(scores)):
                self._ce_cache[pair] = float(sc)
        if len(self._ce_cache) > 50000:
            self._ce_cache.clear()
        return np.array([self._ce_cache.get((query, idx.texts[s]), 0.0) for s in sids])

    def _ppr(self, idx: DocIndex, entity_seeds: Dict[str, float],
             sent_seeds: Dict[int, float]) -> Dict[int, float]:
        if not (self.cfg.use_ppr and idx.fused is not None):
            return {}
        mass_e = self.cfg.entity_seed_mass if entity_seeds else 0.0
        seeds: Dict[str, float] = {}
        tot_e = sum(entity_seeds.values()) or 1.0
        for n, w in entity_seeds.items():
            seeds[eid(n)] = seeds.get(eid(n), 0.0) + mass_e * w / tot_e
        tot_s = sum(sent_seeds.values()) or 1.0
        for sid, w in sent_seeds.items():
            seeds[sid_node(sid)] = seeds.get(sid_node(sid), 0.0) + (1 - mass_e) * w / tot_s
        pr = idx.fused.ppr(seeds, damping=self.cfg.damping)
        return {s.sid: pr.get(sid_node(s.sid), 0.0) for s in idx.doc.sentences}

    def _combine(self, sids: List[int], ce: np.ndarray, ppr: Dict[int, float],
                 dense: np.ndarray, bm: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        score = cfg.w_ce * _z(ce) if self.reranker is not None else np.zeros(len(sids))
        score = score + cfg.w_dense * _z(dense[sids]) + cfg.w_bm25 * _z(bm[sids])
        if ppr:
            p = np.array([math.log(ppr.get(s, 0.0) + 1e-9) for s in sids])
            score = score + cfg.w_ppr * _z(p)
        return score

    def _sentence_seeds(self, sids, score, k) -> Dict[int, float]:
        order = np.argsort(-score)[:k]
        w = np.exp(score[order] - score[order].max())
        return {sids[i]: float(wi) for i, wi in zip(order, w)}

    # -------------------------------------------------------------- retrieve
    def retrieve(self, idx: DocIndex, question: str) -> RetrievalResult:
        cfg, doc, t = self.cfg, idx.doc, {}
        t0 = time.perf_counter()
        q_emb = self.embedder.encode([config.EMBED_QUERY_PREFIX + question],
                                     normalize_embeddings=True, show_progress_bar=False)[0]
        dense = idx.emb @ q_emb
        bm = idx.bm25.scores(question)
        n = len(doc.sentences)
        if n > cfg.max_candidates:
            fused = rrf([list(np.argsort(-dense)), list(np.argsort(-bm))])
            cand = sorted(fused, key=fused.get, reverse=True)[:cfg.max_candidates]
        else:
            cand = list(range(n))
        cand = [int(c) for c in cand]
        t["first_stage"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ce = self._ce(question, cand, idx)
        t["rerank"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        q_ents = idx.ents.mentions_in(question)
        ent_seeds = {e: idx.ents.specificity(e) for e in q_ents}
        base = self._combine(cand, ce, {}, dense, bm)
        ppr = self._ppr(idx, ent_seeds, self._sentence_seeds(cand, base, cfg.sentence_seeds))
        score1 = self._combine(cand, ce, ppr, dense, bm)
        t["ppr"] = time.perf_counter() - t0

        s_score = {s: float(v) for s, v in zip(cand, score1)}
        ce_prob = {s: float(p) for s, p in zip(cand, _sigmoid(ce))}
        para1 = self._para_scores(doc, s_score)

        t0 = time.perf_counter()
        bridge: List[str] = []
        if cfg.use_chain and cfg.hops > 1 and len(para1) > 1:
            chain, bridge, ce_prob = self._chain_search(idx, question, q_ents, ent_seeds,
                                                        dense, bm, s_score, para1, ce_prob)
        else:
            chain = sorted(para1, key=para1.get, reverse=True)[:cfg.hops]
        t["chain"] = time.perf_counter() - t0

        _release_accelerator_cache()
        rest = sorted((p for p in para1 if p not in chain), key=para1.get, reverse=True)
        para_ranking = chain + rest
        ranked = sorted(s_score, key=s_score.get, reverse=True)
        sp = self._supporting(doc, chain, s_score, ce_prob)
        paths = self._explain(idx, q_ents, chain)
        return RetrievalResult(ranked, s_score, ce_prob, para_ranking, chain, sp,
                               q_ents, bridge, paths, t)

    @staticmethod
    def _para_scores(doc, s_score: Dict[int, float]) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for sid, v in s_score.items():
            pid = doc.sentences[sid].pid
            out[pid] = max(out.get(pid, -1e9), v)
        return out

    def _link(self, idx: DocIndex, a: int, b: int) -> float:
        if not (self.cfg.use_links and idx.fused is not None):
            return 0.0
        ta, tb = idx.ents.title_entity.get(a), idx.ents.title_entity.get(b)
        ma, mb = idx.para_mentions.get(a, set()), idx.para_mentions.get(b, set())
        if (tb and eid(tb) in ma) or (ta and eid(ta) in mb):
            return 1.0
        shared = (ma & mb) - {eid(ta) if ta else None, eid(tb) if tb else None}
        return 0.5 if shared else 0.0

    def _para_texts(self, idx: DocIndex) -> List[str]:
        if not hasattr(idx, "_ptexts"):
            idx._ptexts = [f"{p.title}: {idx.doc.paragraph_text(p.pid)}" for p in idx.doc.paragraphs]
        return idx._ptexts

    def _ce_texts(self, query: str, texts: List[str]) -> np.ndarray:
        if self.reranker is None:
            return np.zeros(len(texts))
        todo = [(query, t) for t in texts if (query, t) not in self._ce_cache]
        if todo:
            # Long (query + paragraph) pairs: small batches keep MPS memory bounded.
            scores = self.reranker.predict(todo, batch_size=4, show_progress_bar=False)
            for pair, sc in zip(todo, np.atleast_1d(scores)):
                self._ce_cache[pair] = float(sc)
        return np.array([self._ce_cache[(query, t)] for t in texts])

    def _chain_search(self, idx, question, q_ents, ent_seeds, dense, bm,
                      s_score, para1, ce_prob):
        """Beam search over paragraph chains of length cfg.hops.

        Hop k reformulates the query with the evidence chosen so far and
        scores *paragraphs* (cross-encoder on the reformulated query + PPR
        re-seeded with bridge entities + graph link bonus).
        """
        cfg, doc = self.cfg, idx.doc
        ptexts = self._para_texts(idx)
        beams = [([p], para1[p]) for p in sorted(para1, key=para1.get, reverse=True)[:cfg.beam]]
        bridge_all: List[str] = []
        best_q2: Dict[tuple, str] = {}
        for _ in range(cfg.hops - 1):
            new_beams = []
            for chain, sc in beams:
                used = set(chain)
                ev = []
                for p in chain:
                    top = sorted(doc.paragraphs[p].sids, key=lambda s: s_score.get(s, -1e9),
                                 reverse=True)[:cfg.hop_query_sents]
                    ev.extend(sorted(top))
                q2 = question + " " + " ".join(doc.sentences[s].text for s in ev)
                bridge = []
                for s in ev:
                    for e in idx.ents.sentence_entities(doc.sentences[s]):
                        if e not in q_ents and e not in bridge:
                            bridge.append(e)
                for p in chain:
                    te = idx.ents.title_entity.get(p)
                    if te in bridge:
                        bridge.remove(te)
                bridge_all.extend(b for b in bridge if b not in bridge_all)
                cands = [p.pid for p in doc.paragraphs if p.pid not in used and p.sids]
                if not cands:
                    continue
                ce2 = self._ce_texts(q2, [ptexts[p] for p in cands])
                sig = _z(ce2) * cfg.w_ce
                seeds_e = dict(ent_seeds)
                for e in bridge:
                    seeds_e[e] = seeds_e.get(e, 0.0) + idx.ents.specificity(e)
                ppr2 = self._ppr(idx, seeds_e, {s: 1.0 for s in ev})
                if ppr2:
                    pp = np.array([math.log(max(ppr2.get(s, 0.0) for s in doc.paragraphs[p].sids) + 1e-9)
                                   for p in cands])
                    sig = sig + cfg.w_ppr * _z(pp)
                h1 = np.array([para1.get(p, -5.0) for p in cands])
                sig = sig + cfg.w_dense * _z(h1)
                for p, v in zip(cands, sig):
                    bonus = cfg.link_bonus * max(self._link(idx, c, p) for c in chain)
                    new_beams.append((chain + [p], sc + float(v) + bonus))
                    best_q2[tuple(chain + [p])] = q2
            if not new_beams:
                break
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:cfg.beam]
        chain = beams[0][0]
        # Sentence-level relevance inside later hops w.r.t. the reformulated query.
        q2 = best_q2.get(tuple(chain))
        if q2:
            for p in chain[1:]:
                sids = doc.paragraphs[p].sids
                for s, v in zip(sids, _sigmoid(self._ce(q2, sids, idx))):
                    ce_prob[s] = max(ce_prob.get(s, 0.0), float(v))
        return chain, bridge_all, ce_prob

    def _supporting(self, doc, chain, s_score, ce_prob) -> List[int]:
        sp = []
        for p in chain:
            sids = [s for s in doc.paragraphs[p].sids if s in s_score]
            if not sids:
                continue
            best = max(sids, key=lambda s: s_score[s])
            sp.append(best)
            sp.extend(s for s in sids if s != best and ce_prob.get(s, 0) >= self.cfg.sp_threshold)
        return sorted(set(sp))

    def _explain(self, idx: DocIndex, q_ents: List[str], chain: List[int]) -> List[Dict]:
        """Human-readable graph justification for the selected chain."""
        paths = []
        for a, b in zip(chain, chain[1:]):
            ta, tb = idx.ents.title_entity.get(a), idx.ents.title_entity.get(b)
            if idx.drg is not None:
                for sa in idx.doc.paragraphs[a].sids:
                    for sb in idx.doc.paragraphs[b].sids:
                        d = idx.drg.graph.get_edge_data(sid_node(sa), sid_node(sb))
                        if d and d["type"] in ("title_link", "entity_overlap"):
                            paths.append({"graph": "DRG", "type": d["type"], "from": sa,
                                          "to": sb, "via": d.get("entity")})
            if idx.kg is not None and ta and tb:
                hops = idx.kg.explain_path(ta, tb)
                if hops:
                    paths.append({"graph": "KG", "type": "relation_path", "hops": hops})
        if idx.kg is not None:
            for e in q_ents[:3]:
                for p in chain:
                    te = idx.ents.title_entity.get(p)
                    if te and te != e:
                        hops = idx.kg.explain_path(e, te, cutoff=2)
                        if hops:
                            paths.append({"graph": "KG", "type": "question_entity_path",
                                          "hops": hops})
        return paths[:8]
