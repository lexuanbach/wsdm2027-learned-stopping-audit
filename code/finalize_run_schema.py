"""Losslessly migrate older completed runs to the public trajectory schema.

Only key names and missing provenance defaults are changed; policy outcomes,
scores, costs, and trajectories are untouched.  The script fails if a public
JSON still contains a local absolute-path marker.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import run_experiment as rx


MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--llm-seed", type=int, default=7)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--data-seed", type=int, default=13)
    parser.add_argument("--validation-corpus-sentences", type=int, default=None)
    parser.add_argument("--validation-corpus-questions", type=int, default=None)
    args = parser.parse_args()
    path = MEAS / f"run_{args.tag}.json"
    run = json.loads(path.read_text())
    config = run["config"]
    config.setdefault("llm_seed", args.llm_seed)
    config.setdefault("llm_temperature", args.llm_temperature)
    config.setdefault("data_seed", args.data_seed)
    for key in ["parquet", "jsonl", "sqlite_corpus"]:
        if config.get(key):
            config[key] = pathlib.PurePath(config[key]).name
    renamed = 0
    for record in run["per_q"]:
        if "ent_em" in record:
            record["answer_repeat_em"] = record.pop("ent_em"); renamed += 1
        if "ent_searches" in record:
            record["answer_repeat_searches"] = record.pop("ent_searches")
    if not run.get("model_metadata"):
        run["model_metadata"] = rx.model_metadata(config["model"])
    if args.validation_corpus_sentences and not run.get("retrieval_corpus"):
        run["retrieval_corpus"] = {
            "kind": "HotpotQA validation-context FTS5/BM25",
            "sentence_count": args.validation_corpus_sentences,
            "question_context_count": args.validation_corpus_questions,
            "scope": "all validation contexts, including questions outside train/test",
        }
    migrations = run.setdefault("schema_migrations", [])
    note = "renamed legacy ent_* answer-repetition keys; added missing seed/model provenance"
    if note not in migrations:
        migrations.append(note)
    text = json.dumps(run, indent=1) + "\n"
    forbidden = ["/" + "Users" + "/", "/private/" + "tmp/"]
    found = [marker for marker in forbidden if marker in text]
    if found:
        raise SystemExit(f"public path leak after migration: {found}")
    path.write_text(text)
    print(f"finalized {path.name}: {renamed} legacy records renamed")


if __name__ == "__main__":
    main()
