"""Entity canonicalisation and linking shared by the Span Graph and the KG.

Both graphs key their entity nodes by the same canonical id ("E:<name>"),
which is what lets the three graphs be fused into one without string-matching
every KG node against every sentence (the O(|KG| x |sentences|) regex loop in
the original code).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set

from .text_utils import STOP_WORDS, normalize_text

_LEADING = {"the", "a", "an", "this", "that", "these", "those", "his", "her",
            "their", "its", "our", "my", "your"}
_PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their",
             "this", "these", "who", "which"}
# NER labels that denote quantities rather than nameable entities. Kept as span
# nodes (answer candidates) but not merged into the entity layer.
VALUE_LABELS = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}


def canon(text: str) -> str:
    t = normalize_text(text)
    t = re.sub(r"[\"“”‘’`]", "", t)
    t = re.sub(r"'s\b", "", t)
    t = re.sub(r"[^\w\s&\-.]", " ", t)
    toks = t.split()
    while toks and toks[0] in _LEADING:
        toks = toks[1:]
    return " ".join(toks).strip(" .-")


def strip_disambiguation(title: str) -> str:
    """'Ed Wood (film)' -> 'Ed Wood'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def is_pronoun(text: str) -> bool:
    return normalize_text(text) in _PRONOUNS


def eid(name: str) -> str:
    return f"E:{name}"


class EntityIndex:
    """Vocabulary of canonical entities for one document.

    * every paragraph title is an entity (Wikipedia-style documents are about
      their title entity);
    * NER mentions are canonicalised; a single-token mention that is the
      last/first token of exactly one multi-token entity is treated as an alias
      ("Derrickson" -> "scott derrickson").
    """

    def __init__(self, doc):
        self.doc = doc
        self.title_entity: Dict[int, str] = {}
        self.names: Set[str] = set()
        self.label: Dict[str, str] = {}
        self.df: Dict[str, Set[int]] = defaultdict(set)  # entity -> paragraphs
        for p in doc.paragraphs:
            name = canon(strip_disambiguation(p.title))
            if name:
                self.title_entity[p.pid] = name
                self.names.add(name)
        for s in doc.sentences:
            for text, label in s.entities:
                if label in VALUE_LABELS:
                    continue
                c = canon(text)
                if self._valid(c):
                    self.names.add(c)
                    self.label.setdefault(c, label)
        self._alias = self._build_aliases()
        # Document frequency over paragraphs (for specificity weighting).
        for s in doc.sentences:
            for e in self.sentence_entities(s):
                self.df[e].add(s.pid)
            if s.pid in self.title_entity:
                self.df[self.title_entity[s.pid]].add(s.pid)
        self._by_len = sorted(self.names, key=len, reverse=True)

    @staticmethod
    def _valid(c: str) -> bool:
        return bool(c) and len(c) > 1 and c not in STOP_WORDS and not c.isdigit()

    def _build_aliases(self) -> Dict[str, str]:
        # Only people are routinely referred to by surname/first name alone;
        # aliasing places or organisations ("Denver" -> "Denver Broncos") is wrong.
        titles = set(self.title_entity.values())
        parts: Dict[str, List[str]] = defaultdict(list)
        for n in self.names:
            toks = n.split()
            if len(toks) >= 2 and self.label.get(n) == "PERSON":
                parts[toks[-1]].append(n)
                parts[toks[0]].append(n)
        return {tok: full[0] for tok, full in parts.items()
                if len(set(full)) == 1 and tok not in STOP_WORDS and len(tok) > 2
                and tok not in titles}

    def resolve(self, text: str) -> Optional[str]:
        c = canon(text)
        if not self._valid(c):
            return None
        if c in self._alias:
            return self._alias[c]
        return c if c in self.names else None

    def sentence_entities(self, sent) -> List[str]:
        out = []
        for text, label in sent.entities:
            if label in VALUE_LABELS:
                continue
            r = self.resolve(text)
            if r and r not in out:
                out.append(r)
        return out

    def mentions_in(self, text: str, exclude: Iterable[str] = ()) -> List[str]:
        """Known entity names occurring in free text (longest match first)."""
        t = " " + " ".join(re.findall(r"[\w&\-.]+", normalize_text(text))) + " "
        found: List[str] = []
        exclude = set(exclude)
        for name in self._by_len:
            if name in exclude or len(name) < 3:
                continue
            if f" {name} " in t and not any(name in f for f in found):
                found.append(name)
        return found

    def specificity(self, name: str) -> float:
        """HippoRAG-style node specificity: inverse paragraph frequency."""
        n = len(self.doc.paragraphs) or 1
        return math.log(1 + n / max(1, len(self.df.get(name, ())) or 1))
