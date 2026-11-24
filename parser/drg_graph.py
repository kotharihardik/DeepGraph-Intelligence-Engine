"""Graph 1 - Document Reasoning Graph (DRG): sentence-level layer.

Nodes  : one per sentence, id "S<sid>".
Edges  (undirected, typed, weighted):
  next            consecutive sentences of a paragraph            w = 1.0
  same_paragraph  non-adjacent sentences of a paragraph           w = 0.5
  title_link      sentence mentions another paragraph's title     w = 1.0
                  (hyperlink analogue; the main bridge for multi-hop)
  entity_overlap  sentences in *different* paragraphs that share a
                  named entity, weighted by entity specificity
  semantic        mutual k-NN in embedding space across paragraphs (w = cos)

Changes vs. the original DRG
  * semantic edges are top-k mutual neighbours instead of a global 85th
    percentile threshold (which produced near-cliques in long documents);
  * entity edges are specificity-weighted, so a frequent entity ("United
    States") no longer creates an all-to-all clique;
  * structural edges are actually built in the evaluation path (the original
    HotpotQA runner never called add_structural_edges/add_kg_edges).
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, List

import networkx as nx
import numpy as np

from .entities import EntityIndex


def sid_node(sid: int) -> str:
    return f"S{sid}"


class DocumentReasoningGraph:
    def __init__(self, semantic_k: int = 3, semantic_min_sim: float = 0.6):
        self.semantic_k = semantic_k
        self.semantic_min_sim = semantic_min_sim
        self.graph = nx.Graph()

    def build(self, doc, embeddings: np.ndarray, ents: EntityIndex) -> nx.Graph:
        g = self.graph
        for s in doc.sentences:
            g.add_node(sid_node(s.sid), layer="sentence", sid=s.sid, pid=s.pid,
                       text=s.text, title=s.title)

        # Structural edges.
        for p in doc.paragraphs:
            for a, b in combinations(p.sids, 2):
                kind, w = ("next", 1.0) if abs(a - b) == 1 else ("same_paragraph", 0.5)
                g.add_edge(sid_node(a), sid_node(b), type=kind, weight=w)

        # Title links: sentence in paragraph i mentions the title entity of j.
        title_to_pid = {name: pid for pid, name in ents.title_entity.items()}
        for s in doc.sentences:
            for name in ents.mentions_in(s.text):
                pid = title_to_pid.get(name)
                if pid is None or pid == s.pid or not doc.paragraphs[pid].sids:
                    continue
                lead = doc.paragraphs[pid].sids[0]
                g.add_edge(sid_node(s.sid), sid_node(lead), type="title_link",
                           weight=1.0, entity=name)

        # Entity-overlap edges across paragraphs.
        by_entity: Dict[str, List[int]] = defaultdict(list)
        for s in doc.sentences:
            for e in ents.sentence_entities(s):
                by_entity[e].append(s.sid)
        max_spec = max([ents.specificity(e) for e in by_entity] or [1.0])
        for e, sids in by_entity.items():
            if len(sids) < 2 or len(sids) > 12:   # skip ubiquitous entities
                continue
            w = 0.8 * ents.specificity(e) / max_spec
            for a, b in combinations(sids, 2):
                if doc.sentences[a].pid == doc.sentences[b].pid:
                    continue
                u, v = sid_node(a), sid_node(b)
                if not g.has_edge(u, v):
                    g.add_edge(u, v, type="entity_overlap", weight=w, entity=e)

        # Semantic mutual k-NN across paragraphs.
        if len(doc.sentences) > 1 and embeddings is not None:
            sim = embeddings @ embeddings.T
            np.fill_diagonal(sim, -1)
            pids = np.array([s.pid for s in doc.sentences])
            sim[pids[:, None] == pids[None, :]] = -1
            k = min(self.semantic_k, len(doc.sentences) - 1)
            knn = np.argsort(-sim, axis=1)[:, :k]
            neigh = [set(row) for row in knn]
            for i, row in enumerate(knn):
                for j in row:
                    if i < j and i in neigh[j] and sim[i, j] >= self.semantic_min_sim:
                        u, v = sid_node(i), sid_node(int(j))
                        if not g.has_edge(u, v):
                            g.add_edge(u, v, type="semantic", weight=float(sim[i, j]))
        return g
