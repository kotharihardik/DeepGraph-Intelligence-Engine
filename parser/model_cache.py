"""Process-wide model cache.

The original pipeline re-instantiated spaCy-trf, BART-MNLI, Flan-T5 and a
cross-encoder for *every question*, which is where most of its ~55 s/question
went. Every model here is loaded once per process and shared.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict

# Cap the Apple-MPS allocator: unbounded it caches GBs of activations and, on an
# 8 GB machine shared with a local LLM, pushes the system into swap.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from . import config

_CACHE: Dict[str, Any] = {}
_LOCK = threading.Lock()


def get_device() -> str:
    if config.DEVICE != "auto":
        return config.DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _cached(key: str, factory):
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = factory()
        return _CACHE[key]


def _half_on_accelerator(module):
    # fp16 on GPU/MPS halves memory and roughly doubles throughput; scores move
    # by ~1e-3, which does not change rankings in practice.
    if get_device() in ("mps", "cuda"):
        module.half()
    return module


def get_embedder(model_name: str = None):
    from sentence_transformers import SentenceTransformer
    name = model_name or config.EMBED_MODEL
    return _cached(f"emb::{name}",
                   lambda: _half_on_accelerator(SentenceTransformer(name, device=get_device())))


def get_cross_encoder(model_name: str = None, max_length: int = 512):
    from sentence_transformers import CrossEncoder
    name = model_name or config.RERANK_MODEL

    def _load():
        ce = CrossEncoder(name, max_length=max_length, device=get_device())
        _half_on_accelerator(ce.model)
        ce.model.to(get_device())
        return ce

    return _cached(f"ce::{name}", _load)


def get_spacy(model_name: str = None):
    import spacy
    name = model_name or config.SPACY_MODEL

    def _load():
        try:
            return spacy.load(name)
        except OSError:
            # Fall back to the small model rather than failing the whole run.
            return spacy.load("en_core_web_sm")

    return _cached(f"spacy::{name}", _load)


def get_extractive_qa(model_name: str = None):
    from transformers import pipeline
    name = model_name or config.EXTRACTIVE_QA_MODEL
    device = get_device()
    return _cached(
        f"qa::{name}",
        lambda: pipeline("question-answering", model=name,
                         device=0 if device == "cuda" else (device if device == "mps" else -1)),
    )
