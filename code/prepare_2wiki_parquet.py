"""Repack the 2WikiMultihopQA dev split into the HotpotQA struct schema.

Input: ``dev.parquet`` from the Hugging Face mirror ``xanhho/2WikiMultihopQA``
(12,576 rows; ``context``/``supporting_facts`` stored as JSON strings in the
original 2Wiki list layout).  Output: a parquet whose ``context`` column is a
struct {title: [...], sentences: [[...], ...]} and whose ``supporting_facts``
column is a struct {title: [...], sent_id: [...]}, i.e. exactly what
``run_experiment.QuestionEnv`` and ``row_sentences`` consume for HotpotQA.

    python3 prepare_2wiki_parquet.py --src ../2wiki_dev.parquet \
        --dst ../2wiki_dev_hotpot_schema.parquet
"""
import argparse
import hashlib
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.src)
    rows = []
    for _, r in df.iterrows():
        ctx = json.loads(r["context"]) if isinstance(r["context"], str) else r["context"]
        sf = json.loads(r["supporting_facts"]) if isinstance(
            r["supporting_facts"], str) else r["supporting_facts"]
        rows.append({
            "id": r["_id"], "question": r["question"], "answer": r["answer"],
            "type": r["type"], "level": "n/a",
            "context": {"title": [c[0] for c in ctx],
                        "sentences": [list(c[1]) for c in ctx]},
            "supporting_facts": {"title": [s[0] for s in sf],
                                 "sent_id": [int(s[1]) for s in sf]},
        })
    out = pd.DataFrame(rows)
    out.to_parquet(args.dst, index=False)
    digest = hashlib.sha256(open(args.dst, "rb").read()).hexdigest()
    print(f"wrote {args.dst}: {len(out)} rows, sha256 {digest}")


if __name__ == "__main__":
    main()
