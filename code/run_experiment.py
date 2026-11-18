"""End-to-end REAL experiment for VoI-Search on HotpotQA (distractor).

This runs actual agents against actual data with a real LLM backbone
(ollama) and real BM25 retrieval, and writes measured metrics to
../results/measured/. It is deliberately self-contained (no GPU, no external
index) so the core claims can be reproduced on a laptop.

Backbone: any ollama chat model (default qwen2.5:3b-instruct).
Data: HotpotQA distractor validation parquet (10-paragraph context/question).

Design:
  * search(q)   -> BM25 over either the question context or a corpus-wide
                   index, returns top-k sentences
  * Closed-book -> answer with no retrieval
  * RAG@k       -> one search, k sentences, answer
  * ReAct       -> up to S search steps (model proposes queries), then answer
  * VoI-Search  -> after each step, a trained critic predicts answer-now error
                   and future-best gain; a frozen value threshold controls
                   stopping.  The optional guarded variant also waits for one
                   consecutive normalized-answer repetition.
  * Oracle      -> hindsight-best truncation of the ReAct trajectory

Every ReAct trajectory is logged with the answer-now correctness at each
prefix, which yields (a) counterfactual value targets for the critic,
(b) the fixed-budget frontier, and (c) the oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import time
import urllib.request

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "measured"   # -> artifact/results/measured
OUT.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.environ.get("VOISEARCH_OLLAMA_URL",
                            "http://localhost:11434/api/chat")
_TOKENS = {"in": 0, "out": 0, "calls": 0, "latency_seconds": 0.0}
_LLM_SEED = 7
_LLM_TEMPERATURE = 0.0
_ANTHROPIC = {"client": None, "bedrock": None}


def model_metadata(model):
    """Return reproducible public model metadata without host identifiers."""
    meta = {"requested_model": model}
    if model.startswith("claude-") or model.startswith(_BEDROCK_PREFIXES):
        meta["backend"] = ("anthropic" if model.startswith("claude-")
                           else "amazon-bedrock")
        return meta
    meta["backend"] = "ollama"
    base = OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        with urllib.request.urlopen(base + "/api/version", timeout=5) as r:
            meta["ollama_version"] = json.loads(r.read()).get("version")
        with urllib.request.urlopen(base + "/api/tags", timeout=10) as r:
            entries = json.loads(r.read()).get("models", [])
        match = next((entry for entry in entries
                      if entry.get("name") == model or entry.get("model") == model), None)
        if match:
            meta["digest"] = match.get("digest")
            details = match.get("details", {})
            meta["family"] = details.get("family")
            meta["parameter_size"] = details.get("parameter_size")
            meta["quantization_level"] = details.get("quantization_level")
    except Exception as exc:  # metadata failure must not abort an experiment
        meta["metadata_status"] = f"unavailable:{type(exc).__name__}"
    return meta


# --------------------------------------------------------------------------- #
# LLM backend -- dispatches on model name: "claude-*" -> Anthropic API,
# anything else -> local ollama. The frozen backbone is identical across all
# control policies; only the model string changes between the two backbones.
# --------------------------------------------------------------------------- #
def _llm_ollama(messages, model, temperature, max_tokens):
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": temperature,
                           "num_predict": max_tokens, "num_ctx": 4096,
                           "seed": _LLM_SEED}}
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    _TOKENS["in"] += resp.get("prompt_eval_count", 0)
    _TOKENS["out"] += resp.get("eval_count", 0)
    return resp["message"]["content"].strip()


def _llm_anthropic(messages, model, temperature, max_tokens):
    if _ANTHROPIC["client"] is None:
        import anthropic
        _ANTHROPIC["client"] = anthropic.Anthropic()
    sys_txt = " ".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    resp = _ANTHROPIC["client"].messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=sys_txt or None, messages=turns)
    _TOKENS["in"] += resp.usage.input_tokens
    _TOKENS["out"] += resp.usage.output_tokens
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _llm_bedrock(messages, model, temperature, max_tokens):
    """Amazon Bedrock Converse backend (uses AWS_BEARER_TOKEN_BEDROCK).

    Verified inexpensive model: amazon.nova-micro-v1:0 (see WSDM/guide.md).
    """
    if _ANTHROPIC["bedrock"] is None:
        import boto3
        _ANTHROPIC["bedrock"] = boto3.client("bedrock-runtime",
                                             region_name="us-east-1")
    sys_txt = " ".join(m["content"] for m in messages if m["role"] == "system")
    turns = [{"role": m["role"], "content": [{"text": m["content"]}]}
             for m in messages if m["role"] != "system"]
    kwargs = {"modelId": model, "messages": turns,
              "inferenceConfig": {"maxTokens": max_tokens,
                                  "temperature": temperature}}
    if sys_txt:
        kwargs["system"] = [{"text": sys_txt}]
    r = _ANTHROPIC["bedrock"].converse(**kwargs)
    u = r.get("usage", {})
    _TOKENS["in"] += u.get("inputTokens", 0)
    _TOKENS["out"] += u.get("outputTokens", 0)
    # content may hold reasoning + text blocks (e.g. gpt-oss); keep the text.
    blocks = r["output"]["message"]["content"]
    texts = [b["text"] for b in blocks if isinstance(b, dict) and "text" in b]
    return " ".join(texts).strip()


_BEDROCK_PREFIXES = ("amazon.", "anthropic.", "openai.", "meta.", "deepseek.",
                     "qwen.", "mistral.", "moonshot", "minimax.", "cohere.",
                     "ai21.", "us.", "global.")


_REASONING = ("gpt-oss", "gpt-5", "deepseek", "r1-", ".r1", "magistral", "minimax", "thinking")


def llm(messages, model: str, temperature: float | None = None,
        max_tokens: int = 256,
        retries: int = 4) -> str:
    if temperature is None:
        temperature = _LLM_TEMPERATURE
    # reasoning models spend the token budget on hidden reasoning before any
    # text; give them headroom so the answer block is non-empty.
    if any(x in model for x in _REASONING):
        max_tokens = max(max_tokens, 800)
    if model.startswith("claude-"):        # direct Anthropic API (rate-limited)
        backend = _llm_anthropic
    elif model.startswith(_BEDROCK_PREFIXES):   # any Bedrock provider/region id
        backend = _llm_bedrock
    else:                                   # local ollama (e.g. qwen2.5:3b-instruct)
        backend = _llm_ollama
    last = None
    call_started = time.perf_counter()
    for _ in range(retries):
        try:
            out = backend(messages, model, temperature, max_tokens)
            _TOKENS["calls"] += 1
            _TOKENS["latency_seconds"] += time.perf_counter() - call_started
            return out
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0)
    raise RuntimeError(f"LLM call failed: {last}")


# --------------------------------------------------------------------------- #
# Data + retrieval
# --------------------------------------------------------------------------- #
def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class QuestionEnv:
    """Per-question retrieval environment over the distractor context.

    Supports two schemas:
      * HotpotQA / 2WikiMultihopQA parquet rows: context{title, sentences},
        supporting_facts{title}; retrieval units are title-prefixed sentences.
      * MuSiQue answerable JSONL rows: paragraphs[{title, paragraph_text,
        is_supporting}]; retrieval units are title-prefixed paragraphs, and
        `answer_aliases` are accepted as additional gold answers.
    """

    def __init__(self, row):
        self.qid = str(row.get("id", row.get("_id", "")))
        self.question = row["question"]
        self.answer = row["answer"]
        self.aliases = list(row["answer_aliases"]) if "answer_aliases" in row else []
        self.level = row["level"] if "level" in row else "n/a"
        self.sents, self.sent_src = [], []
        self.support_titles, self.support_sentences = [], []
        if "paragraphs" in row:                       # MuSiQue schema
            self.type = str(row.get("id", "hop"))\
                .split("_")[0].rstrip("123")           # 2hop/3hop/4hop
            for p in row["paragraphs"]:
                txt = p["paragraph_text"].strip()
                if txt:
                    self.sents.append(f"{p['title']}: {txt}")
                    self.sent_src.append(p["title"])
                    if p["is_supporting"]:
                        self.support_titles.append(p["title"])
                        self.support_sentences.append(txt)
            self.n_hops = sum(p["is_supporting"] for p in row["paragraphs"])
        else:                                          # HotpotQA / 2Wiki schema
            self.type = row["type"]
            titles = row["context"]["title"]
            sentences = row["context"]["sentences"]
            for t, sl in zip(titles, sentences):
                for s in sl:
                    s = s.strip()
                    if len(s) > 0:
                        self.sents.append(f"{t}: {s}")
                        self.sent_src.append(t)
            self.n_hops = len(set(row["supporting_facts"]["title"]))
            self.support_titles = list(dict.fromkeys(
                str(t) for t in row["supporting_facts"]["title"]))
            context_lookup = {str(t): list(sl)
                              for t, sl in zip(titles, sentences)}
            for title, sent_id in zip(row["supporting_facts"]["title"],
                                      row["supporting_facts"]["sent_id"]):
                candidates = context_lookup.get(str(title), [])
                if 0 <= int(sent_id) < len(candidates):
                    self.support_sentences.append(candidates[int(sent_id)].strip())
        self.bm25 = BM25Okapi([_tok(s) for s in self.sents]) if self.sents else None
        self.searches = 0

    def em(self, pred: str) -> int:
        return max(em(pred, g) for g in [self.answer] + self.aliases)

    def f1(self, pred: str) -> float:
        return max(f1(pred, g) for g in [self.answer] + self.aliases)

    def search(self, query: str, k: int = 3) -> list[str]:
        self.searches += 1
        if not self.bm25:
            return []
        if hasattr(self.bm25, "search"):
            return self.bm25.search(query, k)
        scores = self.bm25.get_scores(_tok(query))
        top = np.argsort(scores)[::-1][:k]
        return [self.sents[i] for i in top]


class CorpusBM25:
    """Fast in-memory SQLite FTS5/BM25 index for a natural corpus.

    ``rank_bm25`` is convenient for the small per-question environment but
    scores every document for every query.  FTS5 uses an inverted index, which
    makes the complete 7,405-question validation corpus practical on a laptop.
    """

    def __init__(self, documents):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE docs USING fts5(text)")
        self.db.executemany("INSERT INTO docs(text) VALUES (?)",
                            ((d,) for d in documents))
        self.db.commit()

    def search(self, query, k):
        tokens = _tok(query)
        if not tokens:
            return []
        # OR avoids returning nothing when a generated multi-term query has
        # no document containing every term. Tokens are alphanumeric by _tok.
        match = " OR ".join(f'"{t}"' for t in tokens)
        rows = self.db.execute(
            "SELECT text FROM docs WHERE docs MATCH ? "
            "ORDER BY bm25(docs) LIMIT ?", (match, int(k))).fetchall()
        return [row[0] for row in rows]


class SQLiteCorpusBM25:
    """Read-only BM25 retrieval over a prebuilt Wikipedia FTS5 index.

    Expected schema: ``sentences(page_id, line_number, text, ...)`` plus an
    external-content FTS5 table ``sentences_fts``.  This interface lets the
    exact controller run against a full corpus without copying a multi-GB
    index into the artifact.
    """

    def __init__(self, path, reranker=None, reranker_revision=None,
                 candidate_k=50):
        self.path = pathlib.Path(path)
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        tables = {r[0] for r in self.db.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        if not {"sentences", "sentences_fts"}.issubset(tables):
            raise ValueError("SQLite corpus must contain sentences and sentences_fts")
        self.sentence_count = int(self.db.execute(
            "SELECT COUNT(*) FROM sentences").fetchone()[0])
        self.provenance = ({row[0]: row[1] for row in self.db.execute(
            "SELECT key, value FROM provenance").fetchall()}
            if "provenance" in tables else {})
        self.reranker_name = reranker
        self.reranker_revision = reranker_revision
        self.candidate_k = int(candidate_k)
        self.reranker = self.reranker_tokenizer = self.reranker_device = None
        if reranker:
            import torch
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                reranker, revision=reranker_revision, local_files_only=True)
            self.reranker = AutoModelForSequenceClassification.from_pretrained(
                reranker, revision=reranker_revision, local_files_only=True)
            self.reranker_device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu")
            self.reranker.to(self.reranker_device).eval()

    def search(self, query, k):
        # Remove function words before a corpus-wide OR query: on 25M
        # sentences, terms such as "who" otherwise retrieve generic prose.
        stop = {"a", "an", "and", "are", "as", "at", "be", "by", "did",
                "do", "does", "for", "from", "had", "has", "have", "he",
                "her", "his", "how", "in", "is", "it", "its", "of", "on",
                "or", "she", "that", "the", "their", "them", "there", "they",
                "this", "to", "was", "were", "what", "when", "where", "which",
                "who", "why", "with", "would"}
        tokens = list(dict.fromkeys(t for t in _tok(query) if t not in stop))
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens)
        limit = max(int(k), self.candidate_k if self.reranker else int(k))
        rows = self.db.execute(
            "SELECT page_id, text FROM sentences_fts "
            "WHERE sentences_fts MATCH ? ORDER BY bm25(sentences_fts) LIMIT ?",
            (match, limit)).fetchall()
        documents = [f"{row['page_id'].replace('_', ' ')}: {row['text']}"
                     for row in rows]
        if not self.reranker or len(documents) <= k:
            return documents[:k]
        import torch
        batch = self.reranker_tokenizer(
            [query] * len(documents), documents, padding=True, truncation=True,
            max_length=256, return_tensors="pt")
        batch = {name: tensor.to(self.reranker_device)
                 for name, tensor in batch.items()}
        with torch.inference_mode():
            scores = self.reranker(**batch).logits.reshape(-1).detach().cpu().numpy()
        top = np.argsort(scores)[::-1][:int(k)]
        return [documents[i] for i in top]


def row_sentences(row):
    """Yield the exact title-prefixed retrieval units used by QuestionEnv."""
    if "paragraphs" in row:
        for paragraph in row["paragraphs"]:
            text = paragraph["paragraph_text"].strip()
            if text:
                yield f"{paragraph['title']}: {text}"
    else:
        for title, sentences in zip(row["context"]["title"],
                                    row["context"]["sentences"]):
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    yield f"{title}: {sentence}"


def file_sha256(path, chunk_bytes=8 * 1024 * 1024):
    """Stream a file hash so multi-GB corpus indexes are verified in place."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Answer grading
# --------------------------------------------------------------------------- #
_ART = re.compile(r"\b(a|an|the)\b")
_PUNC = re.compile(r"[^a-z0-9 ]")


def _norm(s: str) -> str:
    s = s.lower()
    s = _PUNC.sub(" ", s)
    s = _ART.sub(" ", s)
    return " ".join(s.split())


def em(pred: str, gold: str) -> int:
    return int(_norm(pred) == _norm(gold))


def f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for w in p:
        common[w] = min(p.count(w), g.count(w))
    ncommon = sum(common.values())
    if ncommon == 0:
        return 0.0
    prec, rec = ncommon / len(p), ncommon / len(g)
    return 2 * prec * rec / (prec + rec)


def extract_answer(text: str) -> str:
    m = re.search(r"(?i)answer\s*[:\-]\s*(.+)", text)
    ans = m.group(1) if m else text
    return ans.strip().split("\n")[0].strip(" .\"'")


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
SYS = ("You are a precise question-answering assistant. Give the shortest "
       "exact answer (a name, entity, number, or yes/no). Do not explain.")


def answer_from(env, evidence: list[str], model: str) -> str:
    ev = "\n".join(f"- {e}" for e in evidence) if evidence else "(no evidence)"
    msg = [
        {"role": "system", "content": SYS},
        {"role": "user", "content":
            f"Evidence:\n{ev}\n\nQuestion: {env.question}\n"
            f"Respond with exactly: Answer: <short answer>"},
    ]
    return extract_answer(llm(msg, model, max_tokens=40))


def gen_query(env, evidence: list[str], model: str) -> str:
    ev = "\n".join(f"- {e}" for e in evidence) if evidence else "(nothing yet)"
    msg = [
        {"role": "system", "content":
            "You issue one short search query to find missing facts for a "
            "multi-hop question. Output only the query."},
        {"role": "user", "content":
            f"Question: {env.question}\nKnown so far:\n{ev}\n"
            f"Next search query:"},
    ]
    raw = llm(msg, model, max_tokens=32).strip()
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    q = lines[0] if lines else ""
    return re.sub(r'^["\']|["\']$', "", q).strip() or env.question


def run_react(env, model: str, max_steps: int = 3, k: int = 3):
    """Run ReAct and return the per-prefix trajectory (evidence + answer-now)."""
    evidence, traj = [], []
    # prefix 0: closed-book (no search)
    before = dict(_TOKENS)
    a0 = answer_from(env, [], model)
    traj.append({"searches": 0, "evidence": [], "answer": a0,
                 "answer_usage": usage_delta(before),
                 "retrieval_latency_seconds": 0.0})
    for step in range(1, max_steps + 1):
        before = dict(_TOKENS)
        q = gen_query(env, evidence, model)
        query_usage = usage_delta(before)
        search_started = time.perf_counter()
        hits = env.search(q, k=k)
        retrieval_latency = time.perf_counter() - search_started
        for h in hits:
            if h not in evidence:
                evidence.append(h)
        before = dict(_TOKENS)
        a = answer_from(env, evidence, model)
        answer_usage = usage_delta(before)
        traj.append({"searches": step, "evidence": list(evidence), "answer": a,
                     "query": q, "query_usage": query_usage,
                     "answer_usage": answer_usage,
                     "retrieval_latency_seconds": retrieval_latency})
    return traj


def usage_delta(before):
    """Per-call usage delta from the process-global monotone counters."""
    return {k: round(_TOKENS[k] - before.get(k, 0), 6) for k in _TOKENS}


# --------------------------------------------------------------------------- #
# Adaptive-retrieval baselines (paper-grounded prompting variants; each is
# self-contained and counts its own searches).
#   Self-RAG-style : prompted retrieve/answer reflection each step (frozen
#                    approximation of Asai et al.'s reflection tokens).
#   FLARE-instruct : the FLARE paper's own prompting variant -- the model
#                    emits `Search: <q>` inline when it needs information,
#                    else `Answer: <a>` (Jiang et al., EMNLP 2023, Sec. 3.1).
#   SKR-prompt     : SKR's direct-prompting variant -- ask whether the
#                    question is within self-knowledge; route closed-book vs
#                    retrieval (Wang et al., Findings EMNLP 2023).
#   Adaptive-RAG   : complexity-routed budget {0,1,3} via a prompted router
#                    (frozen approximation of Jeong et al.'s classifier).
# --------------------------------------------------------------------------- #
def _first_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def run_selfrag(env, model, max_steps: int, k: int):
    evidence, searches = [], 0
    for _ in range(max_steps):
        ev = "\n".join(f"- {e}" for e in evidence) if evidence else "(none)"
        dec = llm([
            {"role": "system", "content":
             "You decide whether more retrieval is needed before answering. "
             "Reply with exactly one word: RETRIEVE or ANSWER."},
            {"role": "user", "content":
             f"Question: {env.question}\nEvidence so far:\n{ev}\n"
             f"Is the evidence sufficient to answer reliably?"},
        ], model, max_tokens=6).upper()
        if "RETRIEVE" not in dec:
            break
        q = gen_query(env, evidence, model)
        searches += 1
        for h in env.search(q, k=k):
            if h not in evidence:
                evidence.append(h)
    return answer_from(env, evidence, model), searches


def run_flare_instruct(env, model, max_steps: int, k: int):
    evidence, searches = [], 0
    for _ in range(max_steps + 1):
        ev = "\n".join(f"- {e}" for e in evidence) if evidence else "(none)"
        out = llm([
            {"role": "system", "content":
             "Answer the question if you are confident. If you need more "
             "information first, output exactly `Search: <short query>`. "
             "When ready, output exactly `Answer: <short answer>`."},
            {"role": "user", "content":
             f"Question: {env.question}\nEvidence:\n{ev}"},
        ], model, max_tokens=40)
        line = _first_line(out)
        m = re.match(r"(?i)search\s*[:\-]\s*(.+)", line)
        if m and searches < max_steps:
            searches += 1
            for h in env.search(m.group(1).strip(), k=k):
                if h not in evidence:
                    evidence.append(h)
            continue
        m = re.match(r"(?i)answer\s*[:\-]\s*(.+)", line)
        if m:
            return m.group(1).strip(" .\"'"), searches
        return extract_answer(out), searches
    return answer_from(env, evidence, model), searches


def run_skr(env, model, k: int):
    dec = llm([
        {"role": "system", "content":
         "Judge whether you can answer the question correctly from your own "
         "knowledge, without any retrieval. Reply exactly YES or NO."},
        {"role": "user", "content": f"Question: {env.question}"},
    ], model, max_tokens=4).upper()
    if dec.startswith("YES"):
        return answer_from(env, [], model), 0
    hits = env.search(env.question, k=5)
    return answer_from(env, hits, model), 1


def run_adaptive_rag(env, model, k: int):
    dec = llm([
        {"role": "system", "content":
         "Classify the question's retrieval complexity. Reply exactly one "
         "word: NONE (answerable from your knowledge), SINGLE (one lookup "
         "suffices), or MULTI (needs multi-step retrieval)."},
        {"role": "user", "content": f"Question: {env.question}"},
    ], model, max_tokens=4).upper()
    if dec.startswith("NONE"):
        return answer_from(env, [], model), 0
    if dec.startswith("SINGLE"):
        return answer_from(env, env.search(env.question, k=5), model), 1
    evidence, searches = [], 0
    for _ in range(3):
        q = gen_query(env, evidence, model)
        searches += 1
        for h in env.search(q, k=k):
            if h not in evidence:
                evidence.append(h)
    return answer_from(env, evidence, model), searches


# --------------------------------------------------------------------------- #
# Critic (features -> answer-now error probability + predicted FBG)
# --------------------------------------------------------------------------- #
def prefix_features(env, traj, t) -> list[float]:
    """Cheap, model-agnostic features over the trajectory prefix ending at t.

    All features are computable without any extra LLM call and without the
    backbone's logits, so the critic interface transfers across backbones.
    """
    p = traj[t]
    ev = p["evidence"]
    n = len(ev)
    ans = p["answer"]
    q_words = set(_tok(env.question))
    ev_tok = set(_tok(" ".join(ev))) if ev else set()
    cov = len(q_words & ev_tok) / len(q_words) if q_words else 0.0
    prev = traj[t - 1] if t > 0 else None
    new_sents = n - len(prev["evidence"]) if prev else n
    changed = float(_norm(ans) != _norm(prev["answer"])) if prev else 1.0
    # grounding: fraction of answer tokens present in the retrieved evidence
    ans_tok = set(_tok(ans))
    grounding = len(ans_tok & ev_tok) / max(len(ans_tok), 1)
    # answer stable across the last two steps (strong "done" signal)
    stable2 = float(t >= 2 and _norm(ans) == _norm(traj[t - 1]["answer"])
                    == _norm(traj[t - 2]["answer"]))
    # how many distinct answers seen so far (instability proxy)
    distinct = len({_norm(traj[j]["answer"]) for j in range(t + 1)})
    return [
        float(p["searches"]),                     # budget spent
        float(n),                                  # evidence size
        cov,                                       # question-term coverage
        float(new_sents),                          # novelty of last action
        changed,                                   # answer changed this step
        float(len(_norm(ans).split())),            # answer length
        grounding,                                 # answer grounded in evidence
        stable2,                                   # answer stable 2 steps
        float(distinct),                           # distinct answers so far
    ]


class VoICritic:
    """Two heads: P(answer-now wrong) and E[future-best gain | prefix]."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingRegressor
        self.err = LogisticRegression(max_iter=1000)
        self.val = GradientBoostingRegressor(max_depth=2, n_estimators=80)
        self._fitted = False

    def fit(self, X_err, y_err, X_val, y_val):
        y_err = np.asarray(y_err)
        if len(set(y_err.tolist())) < 2:      # guard: degenerate labels
            from sklearn.dummy import DummyClassifier
            self.err = DummyClassifier(strategy="prior").fit(X_err, y_err)
        else:
            self.err.fit(X_err, y_err)
        self.val.fit(X_val, y_val)
        self._fitted = True

    def err_prob(self, feats) -> float:
        p = self.err.predict_proba([feats])[0]
        classes = list(self.err.classes_)
        return float(p[classes.index(1)]) if 1 in classes else 0.0

    def value(self, feats) -> float:
        return float(self.val.predict([feats])[0])


def build_critic_data(train_envs, model, max_steps, k):
    """Counterfactual truncation.

    err head: at prefix t, is answering NOW wrong?  (backward state label)
    val head: at prefix t, what is the non-negative future-best F1 gain still
              reachable on this logged chain?  This is a hindsight target for
              supervised learning, not an upper bound on the fitted predictor.
    """
    X_err, y_err, X_val, y_val, traces = [], [], [], [], []
    for env in train_envs:
        traj = run_react(env, model, max_steps=max_steps, k=k)
        corr = [env.f1(p["answer"]) for p in traj]  # answer-now quality
        for t in range(len(traj)):
            feats = prefix_features(env, traj, t)
            X_err.append(feats)
            y_err.append(int(env.em(traj[t]["answer"]) == 0))  # wrong?
            # FBG: the best answer-quality improvement realised later
            # by continuing past t. This is 0 when the current answer already
            # matches the best the trajectory ever reaches, so the critic
            # supplies a less myopic target than a one-step difference.  This
            # is a logged hindsight target, not a Bellman continuation value.
            future = corr[t + 1:]
            vtg = max(max(future) - corr[t], 0.0) if future else 0.0
            X_val.append(feats)
            y_val.append(vtg)
        traces.append({"id": env.qid, "question": env.question,
                       "gold_answers": [env.answer] + env.aliases,
                       "support_titles": env.support_titles,
                       "support_sentences": env.support_sentences,
                       "corr": corr,
                       "searches": [p["searches"] for p in traj],
                       "trajectory": traj})
    return (X_err, y_err, X_val, y_val, traces)


# --------------------------------------------------------------------------- #
# Calibration + VoI policy
# --------------------------------------------------------------------------- #
def calibrate_threshold(err_hats, errors, alpha=0.05, delta=0.10):
    """Corrected SGR threshold with binary-search delta splitting.

    Returns ``None`` when no non-empty selection can be certified.  The input
    units must be one stopped state per calibration episode, rather than every
    correlated prefix from each episode.
    """
    import math
    from scipy.stats import beta as beta_dist
    pairs = sorted(zip(err_hats, errors))
    m = len(pairs)
    if not m:
        return None
    steps = max(1, math.ceil(math.log2(m)))
    level = 1 - delta / steps
    cum = np.cumsum([int(e) for _, e in pairs])
    lo, hi, best = 1, m, 0
    for _ in range(steps):
        if lo > hi:
            break
        mid = (lo + hi + 1) // 2
        k = int(cum[mid - 1])
        ub = float(beta_dist.ppf(level, k + 1, mid - k)) if k < mid else 1.0
        if ub <= alpha:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return float(pairs[best - 1][0]) if best else None


def chain_indices(X):
    """Recover episode boundaries: the searches feature resets to zero."""
    chains, current = [], []
    for i, row in enumerate(np.asarray(X)):
        if row[0] == 0 and current:
            chains.append(current)
            current = []
        current.append(i)
    if current:
        chains.append(current)
    return chains


def crossfit_critic_scores(X, y_val, y_err, chains, folds=5, seed=3):
    """Episode-level cross-fitting used for all training-only selection."""
    rng = np.random.default_rng(seed)
    order = np.arange(len(chains))
    rng.shuffle(order)
    value = np.zeros(len(X), float)
    errhat = np.zeros(len(X), float)
    for fold in range(folds):
        held = set(order[fold::folds].tolist())
        tr = [i for c, ch in enumerate(chains) if c not in held for i in ch]
        te = [i for c in held for i in chains[c]]
        critic = VoICritic()
        critic.fit(np.asarray(X)[tr], np.asarray(y_err)[tr],
                   np.asarray(X)[tr], np.asarray(y_val)[tr])
        value[te] = [critic.value(np.asarray(X)[i]) for i in te]
        errhat[te] = [critic.err_prob(np.asarray(X)[i]) for i in te]
    return value, errhat


def training_stop(ch, scores, X, threshold, rule):
    value_stop = ch[-1]
    for i in ch[:-1]:
        if scores[i] < threshold:
            value_stop = i
            break
    if rule == "value":
        return value_stop
    repeat_stop = ch[-1]
    for i in ch[1:-1]:
        if X[i][4] == 0:  # normalized answer unchanged from prior prefix
            repeat_stop = i
            break
    return max(value_stop, repeat_stop)


def select_frozen_lambda(scores, X, y_err, chains, grid, rule, band=1.0):
    """Choose the cheapest grid point within ``band`` training EM points."""
    reference = 100 * np.mean([1 - y_err[ch[-1]] for ch in chains])
    frontier = []
    for threshold in grid:
        stops = [training_stop(ch, scores, X, threshold, rule) for ch in chains]
        frontier.append({
            "lambda": threshold,
            "em": 100 * float(np.mean([1 - y_err[i] for i in stops])),
            "searches": float(np.mean([X[i][0] for i in stops])),
        })
    eligible = [x for x in frontier if x["em"] >= reference - band]
    if eligible:
        picked = min(eligible, key=lambda x: (x["searches"], -x["em"]))
        reason = "cheapest_within_band"
    else:
        best = max(x["em"] for x in frontier)
        picked = min((x for x in frontier if x["em"] == best),
                     key=lambda x: x["searches"])
        reason = "fallback_max_em"
    return float(picked["lambda"]), {
        "rule": rule, "band_em_points": band, "train_reference_em": reference,
        "reason": reason, "frontier": frontier,
    }


def voi_stop_on_trajectory(env, traj, critic: VoICritic, lam: float):
    """Evaluate the value-only FBG rule on an already-logged trajectory.

    Off-policy counterfactual evaluation (paper Sec. 5): reusing the logged
    Reusing the logged trajectory lets us sweep lambda without fresh calls and
    produce a measured quality-cost frontier.  The SGR error threshold only
    flags a subset of emitted answers; it never changes this stopping time.
    """
    for t in range(len(traj)):
        feats = prefix_features(env, traj, t)
        val = critic.value(feats)       # predicted future-best gain
        if t == len(traj) - 1:
            return t
        if val < lam:
            return t
    return len(traj) - 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    global _LLM_SEED, _LLM_TEMPERATURE
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--jsonl", default=None, help="MuSiQue-style JSONL input")
    ap.add_argument("--baselines", action="store_true",
                    help="also run Self-RAG/FLARE/SKR/Adaptive-RAG baselines")
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--llm_seed", type=int, default=7)
    ap.add_argument("--llm_temperature", type=float, default=0.0)
    ap.add_argument("--data_seed", type=int, default=13)
    ap.add_argument("--n_train", type=int, default=120)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--certificate_calibration_fraction", type=float,
                    default=0.20, help="episode fraction held out from both "
                    "critic fitting and threshold selection, then used only "
                    "to calibrate the SGR answer certificate")
    ap.add_argument("--tag", default="hotpotqa")
    ap.add_argument("--shared_index", action="store_true",
                    help="retrieve from ONE BM25 index over the union of all "
                         "sampled questions' paragraphs (cross-question "
                         "distractors) instead of each question's own "
                         "10-paragraph context")
    ap.add_argument("--corpus_index", action="store_true",
                    help="retrieve from one BM25 index over every context in "
                         "the complete input file, including questions not "
                         "sampled for train/test")
    ap.add_argument("--sqlite_corpus", default=None,
                    help="read-only full-corpus FTS5 index with sentences and "
                         "sentences_fts tables (e.g., the FEVER Wikipedia snapshot)")
    ap.add_argument("--sqlite_corpus_sha256", default=None,
                    help="precomputed full SHA-256 of --sqlite_corpus")
    ap.add_argument("--sqlite_corpus_source", default=None,
                    help="public source/snapshot label recorded with the "
                         "verified index hash")
    ap.add_argument("--sqlite_reranker", default=None,
                    help="optional frozen cross-encoder for BM25 candidate reranking")
    ap.add_argument("--sqlite_reranker_revision", default=None)
    ap.add_argument("--sqlite_candidate_k", type=int, default=50)
    ap.add_argument("--primary_rule", choices=("value", "guarded"),
                    default="value", help="predeclared stopping family; "
                    "guarded requires both the value crossing and one "
                    "consecutive normalized-answer repetition")
    args = ap.parse_args()
    if sum(bool(x) for x in [args.shared_index, args.corpus_index,
                             args.sqlite_corpus]) > 1:
        ap.error("choose only one of --shared_index, --corpus_index, --sqlite_corpus")
    if args.sqlite_reranker and not args.sqlite_corpus:
        ap.error("--sqlite_reranker requires --sqlite_corpus")
    _LLM_SEED = args.llm_seed
    _LLM_TEMPERATURE = args.llm_temperature

    if args.jsonl:
        rows = [json.loads(l) for l in open(args.jsonl)]
        rng = np.random.default_rng(args.data_seed)
        rng.shuffle(rows)
        train_envs = [QuestionEnv(r) for r in rows[: args.n_train]]
        test_envs = [QuestionEnv(r) for r in rows[args.n_train: args.n_train + args.n_test]]
    else:
        df_all = pd.read_parquet(args.parquet)
        df = df_all.sample(frac=1.0, random_state=args.data_seed).reset_index(drop=True)
        train_rows = df.iloc[: args.n_train]
        test_rows = df.iloc[args.n_train: args.n_train + args.n_test]
        train_envs = [QuestionEnv(r) for _, r in train_rows.iterrows()]
        test_envs = [QuestionEnv(r) for _, r in test_rows.iterrows()]

    retrieval_corpus_meta = None
    if args.shared_index:
        # one corpus, one index: sentences from every sampled question's
        # context, deduplicated, shared by all environments. Search now has
        # to reject topically-similar sentences from OTHER questions, which
        # is the failure mode the per-question distractor setting cannot
        # exhibit.
        pool, seen = [], set()
        for env in train_envs + test_envs:
            for s in env.sents:
                if s not in seen:
                    seen.add(s)
                    pool.append(s)
        shared_bm25 = BM25Okapi([_tok(s) for s in pool])
        for env in train_envs + test_envs:
            env.sents = pool
            env.bm25 = shared_bm25
        print(f"[shared-index] {len(pool)} sentences pooled from "
              f"{len(train_envs) + len(test_envs)} questions", flush=True)
        retrieval_corpus_meta = {
            "kind": "sampled-question shared BM25",
            "sentence_count": len(pool),
            "question_context_count": len(train_envs) + len(test_envs),
            "scope": "contexts of sampled training and test questions",
        }

    if args.corpus_index:
        pool, seen = [], set()
        corpus_rows = rows if args.jsonl else (
            row for _, row in df_all.iterrows())
        for corpus_row in corpus_rows:
            for sentence in row_sentences(corpus_row):
                if sentence not in seen:
                    seen.add(sentence)
                    pool.append(sentence)
        corpus_bm25 = CorpusBM25(pool)
        for env in train_envs + test_envs:
            env.sents = []
            env.bm25 = corpus_bm25
        print(f"[corpus-index] {len(pool)} unique sentences from "
              f"{len(rows) if args.jsonl else len(df_all)} input questions",
              flush=True)
        _src = pathlib.Path(args.jsonl or args.parquet).name.lower()
        if "musique" in _src:
            _kind = "MuSiQue-answerable dev paragraph FTS5/BM25"
        elif "2wiki" in _src:
            _kind = "2WikiMultihopQA dev-context FTS5/BM25"
        elif "multihoprag" in _src:
            _kind = "MultiHopRAG news-article FTS5/BM25 (non-Wikipedia)"
        else:
            _kind = "HotpotQA validation-context FTS5/BM25"
        retrieval_corpus_meta = {
            "kind": _kind,
            "sentence_count": len(pool),
            "question_context_count": len(rows) if args.jsonl else len(df_all),
            "scope": "all validation contexts, including questions outside train/test",
        }

    if args.sqlite_corpus:
        actual_corpus_sha256 = file_sha256(args.sqlite_corpus)
        if (args.sqlite_corpus_sha256
                and actual_corpus_sha256 != args.sqlite_corpus_sha256.lower()):
            raise ValueError(
                "SQLite corpus SHA-256 mismatch: expected "
                f"{args.sqlite_corpus_sha256}, observed {actual_corpus_sha256}")
        corpus_bm25 = SQLiteCorpusBM25(
            args.sqlite_corpus, reranker=args.sqlite_reranker,
            reranker_revision=args.sqlite_reranker_revision,
            candidate_k=args.sqlite_candidate_k)
        for env in train_envs + test_envs:
            env.sents = []
            env.bm25 = corpus_bm25
        retrieval_corpus_meta = {
            "filename": pathlib.Path(args.sqlite_corpus).name,
            "sentence_count": corpus_bm25.sentence_count,
            "size_bytes": pathlib.Path(args.sqlite_corpus).stat().st_size,
            "schema": "sentences + external-content FTS5 sentences_fts",
            "source_provenance": corpus_bm25.provenance,
            "source_label": args.sqlite_corpus_source,
            "sha256": actual_corpus_sha256,
            "reranker": args.sqlite_reranker,
            "reranker_revision": args.sqlite_reranker_revision,
            "reranker_device": (str(corpus_bm25.reranker_device)
                                  if args.sqlite_reranker else None),
            "reranker_candidate_k": (args.sqlite_candidate_k
                                      if args.sqlite_reranker else None),
        }
        print(f"[sqlite-corpus] {corpus_bm25.sentence_count} sentences from "
              f"{pathlib.Path(args.sqlite_corpus).name}", flush=True)

    t0 = time.time()
    # ---- resume support: Bedrock bearer tokens expire (~1h) mid-run, so every
    # expensive phase is cached under the run tag and skipped on relaunch.
    # The critic fit is deterministic (LogisticRegression lbfgs + GBR without
    # subsampling), so refitting cached arrays reproduces the same critic.
    import pickle
    _critic_pkl = OUT / f"critic_data_{args.tag}.pkl"
    _critic_resumed = _critic_pkl.exists()
    if _critic_resumed:
        _cd = pickle.loads(_critic_pkl.read_bytes())
        X_err, y_err = _cd["X_err"], _cd["y_err"]
        X_val, y_val = _cd["X_val"], _cd["y_val"]
        train_traces = _cd.get("traces", [])
        print(f"[critic] resume: {len(X_err)} cached prefixes from "
              f"{_critic_pkl.name}; skipping train collection", flush=True)
    else:
        print(f"[critic] building counterfactual data from {len(train_envs)} "
              f"train Qs...", flush=True)
        X_err, y_err, X_val, y_val, train_traces = build_critic_data(
            train_envs, args.model, args.max_steps, args.k)
        # cache counterfactual training data so resumes and critic-variant
        # ablations need no further LLM calls (see ablate.py).
        _critic_pkl.write_bytes(pickle.dumps(
            {"X_err": X_err, "y_err": y_err, "X_val": X_val,
             "y_val": y_val, "feature_names": [
                 "searches", "evidence_size", "question_coverage",
                 "new_evidence", "answer_changed", "answer_length",
                 "answer_grounding", "stable_two", "distinct_answers"],
             "traces": train_traces}))
    # lambda grid for the predicted-FBG frontier (F1-score units)
    lam_grid = [0.01, 0.02, 0.04, 0.07, 0.12, 0.20]
    all_chains = chain_indices(X_val)
    split_rng = np.random.default_rng(41)
    chain_order = np.arange(len(all_chains))
    split_rng.shuffle(chain_order)
    n_certificate = max(1, int(round(
        args.certificate_calibration_fraction * len(all_chains))))
    certificate_ids = set(chain_order[-n_certificate:].tolist())
    chains = [chain for i, chain in enumerate(all_chains)
              if i not in certificate_ids]
    certificate_chains = [chain for i, chain in enumerate(all_chains)
                          if i in certificate_ids]
    fit_rows = [i for chain in chains for i in chain]
    critic = VoICritic()
    critic.fit(np.asarray(X_err)[fit_rows], np.asarray(y_err)[fit_rows],
               np.asarray(X_val)[fit_rows], np.asarray(y_val)[fit_rows])
    print(f"[critic] fitted on {len(chains)} policy-training episodes / "
          f"{len(fit_rows)} prefixes; held out {len(certificate_chains)} "
          f"episodes for certificate calibration", flush=True)
    value_oof, err_oof = crossfit_critic_scores(
        X_val, y_val, y_err, chains)
    lam_main, selection = select_frozen_lambda(
        value_oof, X_val, y_err, chains, lam_grid, args.primary_rule)
    # Calibrate on exactly one stopped state per held-out episode.  The same
    # fitted critic scores these states and all test states; neither critic
    # fitting nor lambda selection touches the certificate episodes.
    deployed_value = np.asarray([critic.value(row) for row in X_val])
    deployed_errhat = np.asarray([critic.err_prob(row) for row in X_err])
    cal_stops = [training_stop(ch, deployed_value, X_val, lam_main,
                               args.primary_rule)
                 for ch in certificate_chains]
    q_alpha = calibrate_threshold(
        [deployed_errhat[i] for i in cal_stops],
        [y_err[i] for i in cal_stops],
        alpha=args.alpha)
    selection["n_policy_training_episodes"] = len(chains)
    selection["n_certificate_calibration_episodes"] = len(certificate_chains)
    selection["certificate_split_seed"] = 41
    selection["certificate_units"] = "one deployed-policy stop per held-out episode"
    print(f"[selection] rule={args.primary_rule} lambda={lam_main:.3f}; "
          f"certificate={q_alpha}", flush=True)

    # ---- evaluate all methods on the test set, logging per question ----
    # Resume from the last checkpoint if one exists (test env order is
    # deterministic: fixed-seed shuffle + slice, so skipping is sound).
    per_q = []
    _part = OUT / f"run_{args.tag}.partial.json"
    if _part.exists():
        per_q = json.loads(_part.read_text())["per_q"]
        print(f"[test] resume: {len(per_q)} questions from checkpoint",
              flush=True)
    _start = len(per_q)
    # flag resumed runs: their _TOKENS totals cover only the final process,
    # so token accounting for this tag is a known undercount.
    args.resumed = bool(_start) or _critic_resumed
    for i, env in enumerate(test_envs):
        if i < _start:
            continue
        traj = run_react(env, args.model, max_steps=args.max_steps, k=args.k)
        prefix_em = [env.em(p["answer"]) for p in traj]
        prefix_f1 = [env.f1(p["answer"]) for p in traj]
        # critic predictions at each prefix (for calibration + offline sweeps)
        prefix_errhat, prefix_value = [], []
        prefix_feature_values = []
        for t in range(len(traj)):
            feats = prefix_features(env, traj, t)
            prefix_feature_values.append(feats)
            prefix_errhat.append(critic.err_prob(feats))
            prefix_value.append(critic.value(feats))
        # RAG@5: a single search with more evidence
        rag_hits = env.search(env.question, k=5)
        rag = answer_from(env, rag_hits, args.model)
        # adaptive-retrieval baselines (paper-grounded prompting variants)
        base = {}
        if args.baselines:
            sr_a, sr_s = run_selfrag(env, args.model, args.max_steps, args.k)
            fl_a, fl_s = run_flare_instruct(env, args.model, args.max_steps, args.k)
            sk_a, sk_s = run_skr(env, args.model, args.k)
            ad_a, ad_s = run_adaptive_rag(env, args.model, args.k)
            base = {
                "selfrag_em": env.em(sr_a), "selfrag_searches": sr_s,
                "flare_em": env.em(fl_a), "flare_searches": fl_s,
                "skr_em": env.em(sk_a), "skr_searches": sk_s,
                "adaptive_em": env.em(ad_a), "adaptive_searches": ad_s,
            }
        # VoI (value mode) across the lambda grid -- the efficiency frontier,
        # computed counterfactually on the logged trajectory.
        voi = {}
        for lam in lam_grid:
            t = voi_stop_on_trajectory(env, traj, critic, lam)
            voi[str(lam)] = {"em": prefix_em[t], "f1": prefix_f1[t],
                             "searches": traj[t]["searches"],
                             "stopped_early": t < len(traj) - 1}
        # Answer-repetition baseline: stop at the first consecutive normalized
        # answer repeat. This is not entropy and is not full TASR (no logits).
        ent_t = len(traj) - 1
        for t in range(1, len(traj)):
            if _norm(traj[t]["answer"]) == _norm(traj[t - 1]["answer"]):
                ent_t = t
                break
        value_t = voi_stop_on_trajectory(env, traj, critic, lam_main)
        guarded_t = max(value_t, ent_t)
        primary_t = value_t if args.primary_rule == "value" else guarded_t
        # oracle: best prefix in hindsight (correct at fewest searches)
        best = max(range(len(traj)), key=lambda t: (prefix_em[t], -traj[t]["searches"]))
        rec = {
            "id": env.qid, "question": env.question,
            "gold_answers": [env.answer] + env.aliases,
            "support_titles": env.support_titles,
            "support_sentences": env.support_sentences,
            "type": env.type, "level": env.level, "n_hops": env.n_hops,
            "prefix_em": prefix_em, "prefix_f1": prefix_f1,
            "prefix_errhat": prefix_errhat, "prefix_value": prefix_value,
            "prefix_searches": [p["searches"] for p in traj],
            "prefix_features": prefix_feature_values,
            "trajectory": traj,
            "rag_em": env.em(rag), "rag_f1": env.f1(rag),
            "voi": voi,
            "answer_repeat_em": prefix_em[ent_t],
            "answer_repeat_searches": traj[ent_t]["searches"],
            "frozen_value": {"em": prefix_em[value_t],
                             "f1": prefix_f1[value_t],
                             "searches": traj[value_t]["searches"]},
            "frozen_guarded": {"em": prefix_em[guarded_t],
                               "f1": prefix_f1[guarded_t],
                               "searches": traj[guarded_t]["searches"]},
            "primary": {"rule": args.primary_rule, "em": prefix_em[primary_t],
                        "f1": prefix_f1[primary_t],
                        "searches": traj[primary_t]["searches"]},
            "oracle_em": prefix_em[best], "oracle_searches": traj[best]["searches"],
            **base,
        }
        per_q.append(rec)
        if (i + 1) % 20 == 0:
            vm = np.mean([r["voi"][str(lam_main)]["em"] for r in per_q])
            vs = np.mean([r["voi"][str(lam_main)]["searches"] for r in per_q])
            rm = np.mean([r["prefix_em"][-1] for r in per_q])
            print(f"  [{i+1}/{len(test_envs)}] voi_em={vm:.3f} voi_s={vs:.2f} "
                  f"react_em={rm:.3f} ({time.time()-t0:.0f}s)", flush=True)
            # checkpoint: a mid-run crash (e.g. API token expiry) must not
            # lose completed questions; analyze.py can read the partial file.
            (OUT / f"run_{args.tag}.partial.json").write_text(json.dumps({
                "config": vars(args), "q_alpha": q_alpha,
                "lam_grid": lam_grid, "lam_main": lam_main,
                "selection": selection,
                "tokens": dict(_TOKENS), "wall_seconds": time.time() - t0,
                "per_q": per_q, "partial": True}))

    public_config = vars(args).copy()
    input_path = args.jsonl or args.parquet
    if input_path:
        input_file = pathlib.Path(input_path)
        public_config["jsonl" if args.jsonl else "parquet"] = input_file.name
        digest = file_sha256(input_file)
    else:
        digest = None
    if public_config.get("sqlite_corpus"):
        public_config["sqlite_corpus"] = pathlib.Path(
            public_config["sqlite_corpus"]).name
    out = {
        "config": public_config, "input_sha256": digest,
        "q_alpha": q_alpha,
        "lam_grid": lam_grid, "lam_main": lam_main,
        "selection": selection,
        "retrieval_corpus": retrieval_corpus_meta,
        "model_metadata": model_metadata(args.model),
        "tokens": dict(_TOKENS),
        "wall_seconds": time.time() - t0,
        "per_q": per_q,
    }
    dest = OUT / f"run_{args.tag}.json"
    dest.write_text(json.dumps(out, indent=1))
    partial_path = OUT / f"run_{args.tag}.partial.json"
    if partial_path.exists():
        partial_path.unlink()
    print(f"[done] wrote {dest} in {out['wall_seconds']:.0f}s, "
          f"{_TOKENS['calls']} LLM calls", flush=True)


if __name__ == "__main__":
    main()
