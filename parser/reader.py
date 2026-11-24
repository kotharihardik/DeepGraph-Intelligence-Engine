"""Readers: evidence-cited LLM reader (Ollama) and an extractive fallback.

The LLM sees only the retrieved paragraphs, each sentence tagged with an id,
and must return JSON {"reasoning", "answer", "evidence": [ids]}. The cited ids
are the system's supporting-fact prediction and are what the hallucination /
faithfulness metrics are computed against.

Post-processing (all deterministic):
  * yes/no questions are mapped to exactly "yes"/"no";
  * "X or Y" questions are snapped to the option the answer names;
  * span snapping: if the answer is not a verbatim substring of the evidence,
    it is replaced by the closest Span-Graph candidate (token F1 >= 0.6), so
    answers are copied from the document rather than paraphrased.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import requests

from . import config
from .text_utils import normalize_text, tokenize

_AUX = ("is", "are", "was", "were", "do", "does", "did", "can", "could", "has",
        "have", "had", "will", "would", "should")


def is_yes_no(question: str) -> bool:
    q = question.strip().lower()
    first = q.split(" ", 1)[0] if q else ""
    if first not in _AUX:
        return False
    # "Is X or Y older?" asks to choose, not yes/no.
    return not re.search(r"\bor\b", q)


def choice_options(question: str) -> List[str]:
    q = question.strip().rstrip("?")
    m = re.search(r",\s*([^,]+?)\s+or\s+([^,]+)$", q)
    if not m:
        m = re.search(r"\b(?:between|either)\s+(.+?)\s+(?:and|or)\s+(.+)$", q, re.I)
    if not m:
        return []
    return [re.sub(r"^(the|a|an)\s+", "", o.strip(), flags=re.I) for o in m.groups()]


SYSTEM = (
    "You are a precise question-answering system. Answer ONLY from the numbered "
    "context sentences. The question may need several steps: find the bridge fact "
    "first, then the final answer. Reply with JSON only."
)

FEWSHOT = [
    {
        "context": "Title: Harlow Dam\n[1] Harlow Dam impounds the Mere River in Ontario.\n"
                   "Title: Mere River\n[2] The Mere River is a river in Canada.\n"
                   "[3] It has a length of about 140 km (87 mi) and drains into Lake Huron.",
        "question": "How long is the river impounded by Harlow Dam?",
        "answer": {"reasoning": "[1] the dam impounds the Mere River; [3] it is about 140 km long.",
                   "answer": "about 140 km (87 mi)", "evidence": [1, 3]},
    },
    {
        "context": "Title: Lake Wobegon Days\n[1] Lake Wobegon Days is a 1985 book by Garrison Keillor.\n"
                   "Title: Garrison Keillor\n[2] Garrison Keillor is an American author born in Anoka, Minnesota.\n"
                   "[3] He created the radio show A Prairie Home Companion.",
        "question": "In which city was the author of Lake Wobegon Days born?",
        "answer": {"reasoning": "[1] the author is Garrison Keillor; [2] he was born in Anoka.",
                   "answer": "Anoka, Minnesota", "evidence": [1, 2]},
    },
    {
        "context": "Title: Mount Sorrel\n[1] Mount Sorrel is a peak in Canada with an elevation of 2,300 m.\n"
                   "Title: Pico Alto\n[2] Pico Alto is a mountain in Portugal.\n[3] It rises to 1,100 m.",
        "question": "Are Mount Sorrel and Pico Alto located in the same country?",
        "answer": {"reasoning": "[1] Mount Sorrel is in Canada; [2] Pico Alto is in Portugal; different.",
                   "answer": "no", "evidence": [1, 2]},
    },
]

INSTRUCTIONS = (
    'Return JSON: {"reasoning": "<one short sentence citing ids>", "answer": "<answer>", '
    '"evidence": [<ids of the sentences you used>]}.\n'
    "The answer is a short span copied exactly from the context (an entity, date, number or "
    "phrase), keeping its full name and any unit or qualifier attached to it, e.g. "
    "\"Lincoln Memorial\" not \"Lincoln\", \"4,512 at the 2010 census\" not \"4,512\". "
    "Use exactly \"yes\" or \"no\" only when the question is a yes/no question. "
    "Do not answer with a full sentence."
)
NO_ABSTAIN = (" The answer is always in the context: never reply \"unknown\", \"none\" or "
              "similar; give the most likely answer.")
_ABSTAIN_RE = re.compile(r"^(unknown|none|no one|nobody|not (mentioned|stated|given|specified|"
                         r"in the context)|n/?a|cannot be determined)\b", re.I)


@dataclass
class ReaderOutput:
    answer: str
    raw_answer: str
    cited_sids: List[int]
    reasoning: str = ""
    context: str = ""
    context_sids: List[int] = field(default_factory=list)
    latency_s: float = 0.0
    error: Optional[str] = None
    postprocess: List[str] = field(default_factory=list)


def build_context(doc, pids: Sequence[int]) -> (str, Dict[int, int]):
    """Numbered context; returns text and {local id -> global sid}."""
    lines, id_map, k = [], {}, 1
    for pid in pids:
        p = doc.paragraphs[pid]
        lines.append(f"Title: {p.title}")
        for sid in p.sids:
            lines.append(f"[{k}] {doc.sentences[sid].text}")
            id_map[k] = sid
            k += 1
    return "\n".join(lines), id_map


def _token_f1(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    common = sum((Counter(ta) & Counter(tb)).values())
    if not common:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def _matches_option(answer: str, options: Sequence[str]) -> bool:
    return any(_token_f1(answer, o) > 0 or normalize_text(o) in normalize_text(answer)
               for o in options)


def postprocess(question: str, answer: str, context: str,
                candidates: Sequence[str] = ()) -> (str, List[str]):
    steps = []
    a = (answer or "").strip()
    a = re.sub(r"^(the answer is|answer:)\s*", "", a, flags=re.I).rstrip(".").strip()
    a = a.strip("\"'“”‘’").strip()
    if is_yes_no(question):
        low = a.lower()
        if re.match(r"^(yes|no)\b", low):
            steps.append("yes_no")
            return low.split()[0].strip(",."), steps
    opts = choice_options(question)
    if opts and a:
        scored = [(max(_token_f1(a, o), float(normalize_text(o) in normalize_text(a))), o) for o in opts]
        best = max(scored)
        if best[0] > 0 and sum(1 for s, _ in scored if s == best[0]) == 1:
            if normalize_text(best[1]) != normalize_text(a):
                steps.append("option_snap")
            return best[1], steps
    if a and normalize_text(a) not in normalize_text(context) and candidates:
        best = max(candidates, key=lambda c: _token_f1(a, c))
        if _token_f1(a, best) >= 0.6:
            steps.append("span_snap")
            return best, steps
    return a, steps


class LLMReader:
    def __init__(self, model: str = None, url: str = None, reasoning: bool = True,
                 allow_abstain: bool = False):
        self.model = model or config.LLM_MODEL
        self.url = (url or config.OLLAMA_URL).rstrip("/")
        self.reasoning = reasoning
        # Benchmarks (HotpotQA, MuSiQue-Ans) are answerable by construction; the
        # app allows "not in the document".
        self.allow_abstain = allow_abstain

    def available(self, retries: int = 3) -> bool:
        for attempt in range(retries):
            try:
                r = requests.get(f"{self.url}/api/tags", timeout=15)
                return r.ok and any(self.model in m["name"] for m in r.json().get("models", []))
            except Exception:
                time.sleep(2 * (attempt + 1))
        return False

    def _messages(self, context: str, question: str) -> List[Dict]:
        msgs = [{"role": "system", "content": SYSTEM + "\n" + INSTRUCTIONS
                  + ("" if self.allow_abstain else NO_ABSTAIN)}]
        for ex in FEWSHOT:
            ans = dict(ex["answer"])
            if not self.reasoning:
                ans.pop("reasoning")
            msgs.append({"role": "user", "content": f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}"})
            msgs.append({"role": "assistant", "content": json.dumps(ans)})
        msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
        return msgs

    def unload(self):
        """Evict the model from (GPU) memory so retrieval encoders can use it."""
        try:
            requests.post(f"{self.url}/api/generate",
                          json={"model": self.model, "keep_alive": 0}, timeout=60)
        except Exception:
            pass

    @staticmethod
    def _parse(out: str) -> Dict:
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", out, re.S)
            data = json.loads(m.group(0)) if m else {"answer": out}
        return data if isinstance(data, dict) else {"answer": str(data)}

    def chat(self, messages: List[Dict], json_mode: bool = True, max_tokens: int = 160) -> str:
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"temperature": 0, "seed": 0, "num_ctx": config.LLM_NUM_CTX,
                            "num_predict": max_tokens}}
        if json_mode:
            body["format"] = "json"
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=config.LLM_TIMEOUT_S)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def read(self, doc, pids: Sequence[int], question: str,
             candidates: Sequence[str] = ()) -> ReaderOutput:
        context, id_map = build_context(doc, pids)
        t0 = time.perf_counter()
        raw, reasoning, cited, err = "", "", [], None
        try:
            msgs = self._messages(context, question)
            data = self._parse(self.chat(msgs))
            raw = str(data.get("answer", "")).strip()
            # Guard: one corrective re-ask if a non-yes/no question got yes/no, or
            # an answerable question got an abstention.
            bad = None
            if raw.lower() in ("yes", "no") and not is_yes_no(question) and not choice_options(question):
                bad = "This is not a yes/no question. Answer with the span from the context."
            elif choice_options(question) and not _matches_option(raw, choice_options(question)):
                opts = choice_options(question)
                bad = f"Answer with exactly one of: {opts[0]} or {opts[1]}."
            elif _ABSTAIN_RE.match(raw) and not self.allow_abstain:
                bad = "The answer is in the context. Give the most likely answer span."
            if bad:
                retry = self._parse(self.chat(msgs + [
                    {"role": "assistant", "content": json.dumps(data)},
                    {"role": "user", "content": bad}]))
                if str(retry.get("answer", "")).strip():
                    data = retry
                    raw = str(data.get("answer", "")).strip()
            reasoning = str(data.get("reasoning", ""))
            ev = data.get("evidence", []) or []
            if isinstance(ev, (int, str)):
                ev = [ev]
            for e in ev:
                try:
                    k = int(str(e).strip("[] S"))
                except ValueError:
                    continue
                if k in id_map:
                    cited.append(id_map[k])
        except Exception as exc:  # network / model errors are recorded, not hidden
            err = f"{type(exc).__name__}: {exc}"
        answer, steps = postprocess(question, raw, context, candidates)
        return ReaderOutput(answer=answer, raw_answer=raw, cited_sids=sorted(set(cited)),
                            reasoning=reasoning, context=context,
                            context_sids=list(id_map.values()),
                            latency_s=time.perf_counter() - t0, error=err, postprocess=steps)


class ExtractiveReader:
    """No-LLM fallback: extractive QA over the top paragraphs (SQuAD2 model)."""

    def __init__(self):
        from .model_cache import get_extractive_qa
        self.qa = get_extractive_qa()

    def read(self, doc, pids: Sequence[int], question: str,
             candidates: Sequence[str] = ()) -> ReaderOutput:
        context, id_map = build_context(doc, pids)
        plain = " ".join(doc.sentences[s].text for s in id_map.values())
        t0 = time.perf_counter()
        r = self.qa(question=question, context=plain, max_answer_len=30)
        ans = r.get("answer", "")
        cited = [s for s in id_map.values() if ans and ans in doc.sentences[s].text][:2]
        answer, steps = postprocess(question, ans, context, candidates)
        if is_yes_no(question) and answer not in ("yes", "no"):
            answer, steps = "yes", steps + ["yes_no_default"]
        return ReaderOutput(answer=answer, raw_answer=ans, cited_sids=cited, context=context,
                            context_sids=list(id_map.values()),
                            latency_s=time.perf_counter() - t0, postprocess=steps)
