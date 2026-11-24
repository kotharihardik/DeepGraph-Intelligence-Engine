"""Graph 3 - Knowledge Graph: (subject, relation, object) triples.

Triples are extracted from the cached spaCy dependency parse (no second model
pass). Changes vs. the original KG:

  * provenance: every triple keeps the sentence id it came from, so a KG path
    can always be turned back into citable evidence;
  * pronoun subjects ("He directed ...") are resolved to the paragraph's
    title entity instead of being dropped or creating a node called "he";
  * arguments are canonicalised with the shared EntityIndex, so KG nodes and
    Span-Graph entity nodes have the same ids ("E:<name>") and the graphs can
    be fused directly;
  * the "related_to" fallback between consecutive entities is marked as
    low-confidence (type="co_mention") instead of looking like a real relation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .drg_graph import sid_node
from .entities import EntityIndex, canon, eid, is_pronoun

_SUBJ = {"nsubj", "nsubjpass"}
_OBJ = {"dobj", "attr", "oprd", "dative"}
_NP_DEPS = {"compound", "amod", "nmod", "nummod", "poss", "flat"}


@dataclass
class Triple:
    subj: str
    rel: str
    obj: str
    sid: int
    kind: str  # "relation" | "is_a" | "co_mention"


def _phrase(tok) -> Tuple[str, int, int]:
    """Compact noun phrase around a head token (compounds/modifiers only)."""
    left = [t for t in tok.lefts if t.dep_ in _NP_DEPS]
    right = [t for t in tok.rights if t.dep_ in {"compound", "flat", "nummod"}]
    toks = sorted(left + [tok] + right, key=lambda t: t.i)
    return " ".join(t.text for t in toks), toks[0].idx, toks[-1].idx + len(toks[-1].text)


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.triples: List[Triple] = []

    # ---------------------------------------------------------------- extract
    def _arg(self, tok, sent, ents: EntityIndex, title_ent: Optional[str]) -> Optional[str]:
        if is_pronoun(tok.text):
            return title_ent
        # Prefer a full named entity covering the token.
        for e in sent.doc.ents:
            if e.start <= tok.i < e.end:
                r = ents.resolve(e.text)
                return r or canon(e.text)
        text, _, _ = _phrase(tok)
        r = ents.resolve(text)
        if r:
            return r
        found = ents.mentions_in(text)
        if found:
            return found[0]
        c = canon(text)
        return c if len(c) > 2 else None

    def _extract(self, sent, ents: EntityIndex, title_ent: Optional[str]) -> List[Triple]:
        doc = sent.doc
        out: List[Triple] = []
        for tok in doc:
            if tok.pos_ not in ("VERB", "AUX"):
                continue
            subj_toks = [c for c in tok.children if c.dep_ in _SUBJ]
            # Relative clauses: "X, who directed Y" -> subject is X.
            if not subj_toks and tok.dep_ == "relcl":
                subj_toks = [tok.head]
            if not subj_toks:
                continue
            passive = any(c.dep_ == "auxpass" for c in tok.children)
            neg = any(c.dep_ == "neg" for c in tok.children)
            prt = [c.text for c in tok.children if c.dep_ == "prt"]
            base = ("not " if neg else "") + " ".join([tok.lemma_] + prt)
            objs: List[Tuple[str, object]] = []
            for c in tok.children:
                if c.dep_ in _OBJ:
                    objs.append((base if tok.lemma_ != "be" else "is", c))
                    for conj in c.conjuncts:
                        objs.append((base if tok.lemma_ != "be" else "is", conj))
                elif c.dep_ in ("prep", "agent"):
                    for gc in c.children:
                        if gc.dep_ == "pobj":
                            rel = f"{base} {c.text}" if not passive else f"was {base} {c.text}"
                            objs.append((rel, gc))
                            for conj in gc.conjuncts:
                                objs.append((rel, conj))
                elif c.dep_ == "acomp":
                    objs.append(("is", c))
            for st in subj_toks:
                s = self._arg(st, sent, ents, title_ent)
                if not s:
                    continue
                for rel, ot in objs:
                    o = self._arg(ot, sent, ents, title_ent)
                    if o and o != s:
                        kind = "is_a" if rel == "is" else "relation"
                        out.append(Triple(s, rel.lower(), o, sent.sid, kind))
        if not out:
            names = ents.sentence_entities(sent)
            if title_ent and title_ent not in names and names:
                names = [title_ent] + names
            for a, b in zip(names, names[1:]):
                out.append(Triple(a, "co_mention", b, sent.sid, "co_mention"))
        return out

    def build(self, doc, ents: EntityIndex) -> nx.DiGraph:
        for s in doc.sentences:
            if s.doc is None:
                continue
            self.triples.extend(self._extract(s, ents, ents.title_entity.get(s.pid)))
        for t in self.triples:
            for n in (t.subj, t.obj):
                if not self.graph.has_node(eid(n)):
                    self.graph.add_node(eid(n), layer="entity", text=n)
            u, v = eid(t.subj), eid(t.obj)
            if self.graph.has_edge(u, v):
                d = self.graph[u][v]
                d["weight"] += 1.0
                d["sids"].append(t.sid)
                if t.kind != "co_mention" and d["kind"] == "co_mention":
                    d.update(relation=t.rel, kind=t.kind)
            else:
                self.graph.add_edge(u, v, relation=t.rel, kind=t.kind,
                                    weight=1.0, sids=[t.sid])
        return self.graph

    # ----------------------------------------------------------------- fusion
    def fusion_edges(self):
        """Undirected weighted edges for the fused graph.

        relation edges entity<->entity, plus provenance edges entity<->sentence
        (the KG's own alignment to the DRG; replaces the original regex scan).
        """
        for u, v, d in self.graph.edges(data=True):
            w = 0.3 if d["kind"] == "co_mention" else 1.0
            yield u, v, {"type": f"kg:{d['kind']}", "relation": d["relation"],
                         "weight": w * min(2.0, d["weight"])}
            for sid in set(d["sids"]):
                yield u, sid_node(sid), {"type": "kg_provenance", "weight": 0.5}
                yield v, sid_node(sid), {"type": "kg_provenance", "weight": 0.5}

    # ------------------------------------------------------------------ query
    def facts_about(self, name: str, limit: int = 10) -> List[str]:
        n = eid(canon(name))
        if n not in self.graph:
            return []
        facts = []
        for u, v, d in list(self.graph.out_edges(n, data=True)) + list(self.graph.in_edges(n, data=True)):
            if d["kind"] != "co_mention":
                facts.append(f"({u[2:]}) -[{d['relation']}]-> ({v[2:]})")
        return facts[:limit]

    def explain_path(self, a: str, b: str, cutoff: int = 3) -> Optional[List[Dict]]:
        """Shortest relation path between two entity names, with provenance."""
        ua, ub = eid(a), eid(b)
        und = self.graph.to_undirected(as_view=True)
        if ua not in und or ub not in und:
            return None
        try:
            path = nx.shortest_path(und, ua, ub)
        except nx.NetworkXNoPath:
            return None
        if len(path) - 1 > cutoff:
            return None
        hops = []
        for x, y in zip(path, path[1:]):
            d = self.graph.get_edge_data(x, y) or self.graph.get_edge_data(y, x)
            src, dst = (x, y) if self.graph.has_edge(x, y) else (y, x)
            hops.append({"subject": src[2:], "relation": d["relation"],
                         "object": dst[2:], "sids": sorted(set(d["sids"]))})
        return hops

    def stats(self) -> Dict:
        kinds = defaultdict(int)
        for _, _, d in self.graph.edges(data=True):
            kinds[d["kind"]] += 1
        return {"nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(), "edge_kinds": dict(kinds)}
