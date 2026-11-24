"""Text normalisation, tokenisation and a vectorised BM25."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor
not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves also
""".split())

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOP_WORDS]


class BM25:
    """Okapi BM25 (Robertson & Zaragoza, 2009) with a pre-built term index.

    Documents are tokenised once in ``__init__``; scoring a query is
    O(|query| * |docs containing term|).
    """

    def __init__(self, documents: Sequence[str], k1: float = 1.2, b: float = 0.75,
                 drop_stopwords: bool = True):
        self.k1, self.b = k1, b
        self.drop_stopwords = drop_stopwords
        self.n_docs = len(documents)
        toks = [self._tok(d) for d in documents]
        self.doc_len = np.array([len(t) for t in toks], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.postings: Dict[str, Dict[int, int]] = {}
        for i, t in enumerate(toks):
            for term, tf in Counter(t).items():
                self.postings.setdefault(term, {})[i] = tf
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def _tok(self, text: str) -> List[str]:
        return content_tokens(text) if self.drop_stopwords else tokenize(text)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs or self.avgdl == 0:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in set(self._tok(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idx = np.fromiter(posting.keys(), dtype=np.int64)
            tf = np.fromiter(posting.values(), dtype=np.float32)
            out[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[idx])
        return out


def rrf(rankings: Sequence[Sequence[int]], k: int = 60,
        weights: Sequence[float] = None) -> Dict[int, float]:
    """Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).

    Rank-based, so it fuses scores on incomparable scales (BM25, cosine,
    cross-encoder logits) without ad-hoc min-max normalisation.
    """
    weights = weights or [1.0] * len(rankings)
    fused: Dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + w / (k + rank + 1)
    return fused


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
