"""Unified document model shared by the three graphs.

A Document is a list of paragraphs (Wikipedia articles in HotpotQA/MuSiQue,
detected sections or pages for PDFs), each holding sentences. Every sentence
gets a global integer id (``sid``) that is the node id in the DRG and the
provenance id used by the Span Graph and the Knowledge Graph.

spaCy is run exactly once per document (``annotate``) and its output is reused
by all graph builders.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .model_cache import get_spacy


@dataclass
class Sentence:
    sid: int
    pid: int
    idx: int            # position inside the paragraph
    text: str
    title: str
    page: Optional[int] = None
    # Filled by annotate():
    entities: List[Tuple[str, str]] = field(default_factory=list)   # (text, label)
    noun_chunks: List[str] = field(default_factory=list)
    doc: object = None  # spaCy Doc (kept in memory for KG extraction)


@dataclass
class Paragraph:
    pid: int
    title: str
    sids: List[int] = field(default_factory=list)


@dataclass
class Document:
    paragraphs: List[Paragraph]
    sentences: List[Sentence]
    annotated: bool = False

    def paragraph_text(self, pid: int) -> str:
        return " ".join(self.sentences[s].text for s in self.paragraphs[pid].sids)

    @property
    def titles(self) -> List[str]:
        return [p.title for p in self.paragraphs]

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_paragraphs(cls, items: Sequence[Tuple[str, Sequence[str]]],
                        pages: Sequence[Optional[int]] = None) -> "Document":
        """``items`` = [(title, [sentence, ...]), ...]."""
        paragraphs, sentences = [], []
        for pid, (title, sents) in enumerate(items):
            para = Paragraph(pid=pid, title=str(title).strip())
            idx = 0
            for s in sents:
                s = re.sub(r"\s+", " ", str(s)).strip()
                if not s:
                    continue
                sid = len(sentences)
                sentences.append(Sentence(sid=sid, pid=pid, idx=idx, text=s,
                                          title=para.title,
                                          page=pages[pid] if pages else None))
                para.sids.append(sid)
                idx += 1
            paragraphs.append(para)
        return cls(paragraphs=paragraphs, sentences=sentences)

    @classmethod
    def from_hotpot(cls, context: Sequence) -> "Document":
        # HotpotQA keeps original sentence indices (needed for supporting-fact
        # evaluation), so empty sentences are kept as-is rather than dropped.
        paragraphs, sentences = [], []
        for pid, (title, sents) in enumerate(context):
            para = Paragraph(pid=pid, title=str(title).strip())
            for idx, s in enumerate(sents):
                sid = len(sentences)
                sentences.append(Sentence(sid=sid, pid=pid, idx=idx,
                                          text=re.sub(r"\s+", " ", str(s)).strip() or ".",
                                          title=para.title))
                para.sids.append(sid)
            paragraphs.append(para)
        return cls(paragraphs=paragraphs, sentences=sentences)

    @classmethod
    def from_raw_paragraphs(cls, items: Sequence[Tuple[str, str]],
                            pages: Sequence[Optional[int]] = None) -> "Document":
        """Paragraphs given as raw text; sentences split with spaCy."""
        nlp = get_spacy()
        split = []
        for (title, text), doc in zip(items, nlp.pipe([t for _, t in items], batch_size=32)):
            split.append((title, [s.text for s in doc.sents if s.text.strip()]))
        return cls.from_paragraphs(split, pages=pages)

    @classmethod
    def from_pdf_pages(cls, pages: Sequence[Dict]) -> "Document":
        """Group PDF text into paragraphs (blank-line separated blocks)."""
        items, page_nums = [], []
        for p in pages:
            blocks = [b.strip() for b in re.split(r"\n\s*\n", p.get("text", "")) if b.strip()]
            if not blocks:
                continue
            heading = blocks[0].split("\n")[0][:80]
            for b in blocks:
                text = re.sub(r"\s*\n\s*", " ", b)
                if len(text) < 20:
                    continue
                items.append((f"{heading} (p.{p.get('page')})", text))
                page_nums.append(p.get("page"))
        return cls.from_raw_paragraphs(items, pages=page_nums)

    # ---------------------------------------------------------------- annotate
    def annotate(self) -> "Document":
        """Single spaCy pass: NER + noun chunks + dependency parse per sentence."""
        if self.annotated:
            return self
        nlp = get_spacy()
        texts = [s.text for s in self.sentences]
        for sent, doc in zip(self.sentences, nlp.pipe(texts, batch_size=64)):
            sent.doc = doc
            sent.entities = [(e.text.strip(), e.label_) for e in doc.ents
                             if len(e.text.strip()) > 1]
            sent.noun_chunks = [nc.text.strip() for nc in doc.noun_chunks
                                if len(nc.text.strip()) > 1]
        self.annotated = True
        return self
