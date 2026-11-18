"""Repack MultiHopRAG (news evidence) into the HotpotQA struct schema.

Input: the two Hugging Face parquet exports of ``yixuantt/MultiHopRAG``:
``MultiHopRAG/train`` (2,556 queries with an ``evidence_list``) and
``corpus/train`` (609 news articles).  Output: a parquet whose ``context``
column is {title: [...], sentences: [[...], ...]} and whose
``supporting_facts`` column is {title: [...], sent_id: [...]}, i.e. exactly
what ``run_experiment.QuestionEnv`` and ``row_sentences`` consume.

Only ``--question_type`` rows are kept (default ``inference_query``: short
entity answers, the answer form the paper's EM normalization was built for).
Distractor articles are assigned round-robin so that pooling every row's
context under ``--corpus_index`` covers all 609 articles.

    python3 prepare_multihoprag_parquet.py --qa qa.parquet \
        --corpus corpus.parquet --dst multihoprag_inference_hotpot_schema.parquet
"""
import argparse
import hashlib
import json
import re

import pandas as pd

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
MIN_TOKENS = 4


def split_sentences(body: str) -> list[str]:
    out = []
    for line in str(body).split("\n"):
        line = line.strip()
        if not line:
            continue
        for s in SENT_SPLIT.split(line):
            s = s.strip()
            if len(s.split()) >= MIN_TOKENS:
                out.append(s)
    return out


def _tok(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def locate(fact: str, sentences: list[str]) -> tuple[int, str]:
    fact = fact.strip()
    for i, s in enumerate(sentences):
        if s == fact:
            return i, "exact"
    for i, s in enumerate(sentences):
        if fact in s:
            return i, "substring"
    ft = _tok(fact)
    if not ft or not sentences:
        return -1, "none"
    best = max(range(len(sentences)),
               key=lambda i: len(ft & _tok(sentences[i])) / max(1, len(ft | _tok(sentences[i]))))
    return best, "overlap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--question_type", default="inference_query")
    ap.add_argument("--n_distractors", type=int, default=4)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    corpus = pd.read_parquet(args.corpus)
    qa = pd.read_parquet(args.qa)
    if args.question_type != "all":
        qa = qa[qa["question_type"] == args.question_type].reset_index(drop=True)

    titles = list(corpus["title"])
    assert len(set(titles)) == len(titles), "article titles are not unique"
    sents = {t: split_sentences(b) for t, b in zip(corpus["title"], corpus["body"])}
    source = dict(zip(corpus["title"], corpus["source"]))

    match_counts = {"exact": 0, "substring": 0, "overlap": 0, "none": 0}
    rows, cursor = [], 0
    for i, r in qa.iterrows():
        ev = list(r["evidence_list"])
        gold = list(dict.fromkeys(e["title"] for e in ev))
        # deterministic round-robin distractors, so the pooled index covers
        # every article in the corpus regardless of which rows are sampled.
        distractors = []
        while len(distractors) < args.n_distractors:
            cand = titles[cursor % len(titles)]
            cursor += 1
            if cand not in gold and cand not in distractors:
                distractors.append(cand)
        ctx_titles = gold + distractors
        ctx_sents = [sents[t] for t in ctx_titles]
        sf_title, sf_sid = [], []
        for e in ev:
            idx, how = locate(e["fact"], sents[e["title"]])
            match_counts[how] += 1
            if idx >= 0:
                sf_title.append(e["title"])
                sf_sid.append(int(idx))
        rows.append({
            "id": f"mhrag_{args.question_type}_{i:05d}",
            "question": str(r["query"]),
            "answer": str(r["answer"]),
            "type": str(r["question_type"]),
            "level": "n/a",
            "context": {"title": ctx_titles, "sentences": ctx_sents},
            "supporting_facts": {"title": sf_title, "sent_id": sf_sid},
        })

    out = pd.DataFrame(rows)
    out.to_parquet(args.dst, index=False)
    digest = hashlib.sha256(open(args.dst, "rb").read()).hexdigest()
    pool = {f"{t}: {s}" for row in rows
            for t, sl in zip(row["context"]["title"], row["context"]["sentences"])
            for s in sl}
    covered = {t for row in rows for t in row["context"]["title"]}
    report = {
        "dst": args.dst, "sha256": digest, "rows": len(out),
        "question_type": args.question_type,
        "articles_in_corpus": len(titles),
        "articles_reachable_from_pooled_contexts": len(covered),
        "pooled_unique_retrieval_units": len(pool),
        "distinct_news_outlets": int(corpus["source"].nunique()),
        "gold_fact_localisation": match_counts,
        "gold_articles_per_question": (
            out["supporting_facts"].apply(lambda d: len(set(d["title"]))).value_counts().to_dict()),
    }
    print(json.dumps(report, indent=1, default=str))
    if args.report:
        open(args.report, "w").write(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
