"""Fusion of the three graphs into one heterogeneous graph.

    sentence layer (DRG)  S<sid>   --in_sentence--  span layer  P<n>
          |                                             |
          +-----------about / kg_provenance-------- entity layer E:<name>
                                                       |  co_occurs (Span Graph)
                                                       |  kg:relation (KG)

Because the Span Graph and the KG share canonical entity ids, the union of
the three graphs is already aligned; no fuzzy cross-graph matching is needed.

Retrieval over the fused graph uses Personalized PageRank (Haveliwala, 2002)
seeded from (a) entities mentioned in the question, weighted by specificity,
and (b) the top cross-encoder sentences - the idea behind HippoRAG
(Gutierrez et al., NeurIPS 2024), applied here to a document-level graph.
Probability mass flows question entity -> KG relation -> bridge entity ->
sentences about the bridge entity, which is exactly a multi-hop step.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import networkx as nx


class FusedGraph:
    def __init__(self, drg: Optional[nx.Graph], span: Optional[nx.Graph],
                 kg_edges: Optional[Iterable], sentence_nodes: Iterable[str]):
        g = nx.Graph()
        g.add_nodes_from(sentence_nodes, layer="sentence")
        if drg is not None:
            g.add_edges_from(drg.edges(data=True))
        if span is not None:
            g.add_nodes_from(span.nodes(data=True))
            g.add_edges_from(span.edges(data=True))
        if kg_edges is not None:
            for u, v, d in kg_edges:
                if g.has_edge(u, v):
                    g[u][v]["weight"] = g[u][v].get("weight", 0.0) + d["weight"]
                else:
                    g.add_edge(u, v, **d)
        self.graph = g

    def ppr(self, seeds: Dict[str, float], damping: float = 0.5) -> Dict[str, float]:
        seeds = {n: w for n, w in seeds.items() if n in self.graph and w > 0}
        if not seeds or self.graph.number_of_edges() == 0:
            return {}
        try:
            return nx.pagerank(self.graph, alpha=damping, personalization=seeds,
                               weight="weight", max_iter=200, tol=1e-8)
        except nx.PowerIterationFailedConvergence:
            return {}

    def stats(self) -> Dict:
        layers = {}
        for _, d in self.graph.nodes(data=True):
            layers[d.get("layer", "?")] = layers.get(d.get("layer", "?"), 0) + 1
        return {"nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(), "layers": layers}
