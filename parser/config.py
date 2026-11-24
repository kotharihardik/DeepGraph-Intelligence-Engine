"""Runtime configuration for GraphFusion.

Every value can be overridden with an environment variable so experiments are
reproducible from the command line without code edits.

Model choices are deliberately small/base-sized: the whole stack (embedder,
reranker, spaCy, plus a local 8B LLM in Ollama) has to fit in 8 GB of RAM.
"""
import os


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


# --- Encoders ---------------------------------------------------------------
# Dense bi-encoder for first-stage retrieval and DRG semantic edges.
EMBED_MODEL = _env("GF_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# BGE models are trained with an instruction prefix on the *query* side only.
EMBED_QUERY_PREFIX = _env(
    "GF_EMBED_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)
# Cross-encoder reranker (query, passage) -> relevance.
RERANK_MODEL = _env("GF_RERANK_MODEL", "BAAI/bge-reranker-base")
# spaCy pipeline used once per document for NER, noun chunks and dependencies.
SPACY_MODEL = _env("GF_SPACY_MODEL", "en_core_web_sm")

# --- Reader (LLM) -----------------------------------------------------------
OLLAMA_URL = _env("GF_OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = _env("GF_LLM_MODEL", "llama3")
LLM_TIMEOUT_S = int(_env("GF_LLM_TIMEOUT_S", "120"))
LLM_NUM_CTX = int(_env("GF_LLM_NUM_CTX", "4096"))
# Extractive fallback reader used when no LLM is reachable.
EXTRACTIVE_QA_MODEL = _env("GF_EXTRACTIVE_QA_MODEL", "deepset/roberta-base-squad2")

# --- Evaluation-only models (never used to produce answers) ------------------
NLI_MODEL = _env("GF_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
SAS_MODEL = _env("GF_SAS_MODEL", "cross-encoder/stsb-distilroberta-base")

# --- Device -----------------------------------------------------------------
# "auto" picks cuda > mps > cpu.
DEVICE = _env("GF_DEVICE", "auto")

# --- PDF extraction (Streamlit app) ------------------------------------------
ENABLE_OCR = _env("PDF_ENABLE_OCR", "false").lower() in ("1", "true", "yes")
ENABLE_TABLES = _env("PDF_ENABLE_TABLES", "true").lower() in ("1", "true", "yes")
OUTPUT_DIR = _env("PDF_OUTPUT_DIR", ".")
