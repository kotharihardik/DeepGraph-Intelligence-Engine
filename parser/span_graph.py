"""Graph 2 - Span Graph: fine-grained span + entity layer.

Nodes
  P<n>        span node: named-entity mention, value (date/number) or noun
              chunk, with char offsets into its sentence (answer candidates)
  E:<name>    canonical entity node (shared id space with the KG)
Edges
  in_sentence  span -> its sentence node "S<sid>"     (cross-layer to DRG)
  refers_to    span -> canonical entity                 (entity linking)
  about        title entity -> every sentence of its paragraph
               (Wikipedia-style implicit coreference: "He was born in ..."
               inside the 'Scott Derrickson' article is about him)
  co_occurs    entity <-> entity mentioned in the same sentence

Changes vs. the original Span Graph
  * span ids are unique (the original reused ids after de-duplication, so
    later spans silently overwrote earlier nodes);
  * the full sentence is no longer duplicated as a span (it is the DRG node);
  * substring "discourse markers" were dropped: matching "and", "so", "by",
    "no" as substrings tagged almost every span and added thousands of
    near-random edges (e.g. 1,389 edges for 120 spans in one HotpotQA item);
  * no per-span embeddings or O(n^2) structural loops - spans are reached
    through the graph, so the build is far cheaper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import networkx as nx

from .drg_graph import sid_node
from .entities import EntityIndex, VALUE_LABELS, canon, eid, is_pronoun


@dataclass
class Span:
    span_id: int
    sid: int
    text: str
    kind: str          # "ent:<LABEL>" | "value:<LABEL>" | "noun_chunk"
    start: int
    end: int
    entity: str = None  # canonical entity name when linked


def extract_spans(doc, ents: EntityIndex) -> List[Span]:
    spans: List[Span] = []
    for s in doc.sentences:
        if s.doc is None:
            continue
        covered = []
        for e in s.doc.ents:
            text = e.text.strip()
            if len(text) < 2:
                continue
            kind = f"value:{e.label_}" if e.label_ in VALUE_LABELS else f"ent:{e.label_}"
            linked = None if e.label_ in VALUE_LABELS else ents.resolve(text)
            spans.append(Span(len(spans), s.sid, text, kind, e.start_char, e.end_char, linked))
            covered.append((e.start_char, e.end_char))
        for nc in s.doc.noun_chunks:
            text = nc.text.strip()
            if len(text) < 3 or is_pronoun(text):
                continue
            if any(a <= nc.start_char and nc.end_char <= b for a, b in covered):
                continue   # already represented by an entity span
            linked = None
            for name in ents.mentions_in(text):
                linked = name
                break
            spans.append(Span(len(spans), s.sid, text, "noun_chunk",
                              nc.start_char, nc.end_char, linked))
    return spans


class SpanGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.spans: List[Span] = []

    def build(self, doc, ents: EntityIndex) -> nx.Graph:
        g = self.graph
        self.spans = extract_spans(doc, ents)

        for sp in self.spans:
            pn = f"P{sp.span_id}"
            g.add_node(pn, layer="span", text=sp.text, kind=sp.kind, sid=sp.sid,
                       start=sp.start, end=sp.end)
            g.add_edge(pn, sid_node(sp.sid), type="in_sentence", weight=1.0)
            if sp.entity:
                en = eid(sp.entity)
                g.add_node(en, layer="entity", text=sp.entity)
                g.add_edge(pn, en, type="refers_to",
                           weight=1.0 if sp.kind.startswith("ent:") else 0.5)

        # Title entity is the implicit subject of its paragraph.
        for pid, name in ents.title_entity.items():
            en = eid(name)
            g.add_node(en, layer="entity", text=name, is_title=True, pid=pid)
            for sid in doc.paragraphs[pid].sids:
                g.add_edge(en, sid_node(sid), type="about", weight=0.5)

        # Entity co-occurrence within a sentence.
        for s in doc.sentences:
            names = ents.sentence_entities(s)
            if len(names) < 2:
                continue
            w = 1.0 / (len(names) - 1)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = eid(names[i]), eid(names[j])
                    g.add_node(a, layer="entity", text=names[i])
                    g.add_node(b, layer="entity", text=names[j])
                    if not g.has_edge(a, b):
                        g.add_edge(a, b, type="co_occurs", weight=w, sid=s.sid)
        return g

    def spans_in(self, sids) -> List[Span]:
        sids = set(sids)
        return [sp for sp in self.spans if sp.sid in sids]

    def candidate_texts(self, sids) -> Dict[str, Span]:
        return {canon(sp.text): sp for sp in self.spans_in(sids)}
