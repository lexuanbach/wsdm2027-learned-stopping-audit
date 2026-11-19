"""One machine-readable policy/configuration card per cell, plus a prompt-hash
and seed ledger, plus a flat per-question paired-outcome CSV.

Everything here is a JOIN of data that already exists on disk -- run_*.json
`config`/`model_metadata`/`retrieval_corpus`, the per-cell confirmation file,
and the per-question `per_q` arrays.  No number is created, no model is fit.

Outputs (in --out):
  policy_cards.json            one card per cell
  policy_cards.tex             a compact supplement table (one row per cell)
  prompt_and_seed_ledger.json  sha256 of every prompt constant + seed ledger
  per_question_paired.csv      one row per (cell, question, policy) outcome

    python3 policy_cards.py --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib

import numpy as np

import round9_analyses as r9
import run_experiment as rx

MEAS = r9.MEAS


def sha(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def prompt_ledger():
    """Hash the exact prompt constants the runs used."""
    answer_sys = rx.SYS
    query_sys = ("You issue one short search query to find missing facts for a "
                 "multi-hop question. Output only the query.")
    answer_user = ("Evidence:{NL}{evidence}{NL}{NL}Question: {question}{NL}"
                   "Respond with exactly: Answer: <short answer>")
    query_user = ("Question: {question}{NL}Known so far:{NL}{evidence}{NL}"
                  "Next search query:")
    parts = {
        "answer_system": answer_sys, "answer_user_template": answer_user,
        "query_system": query_sys, "query_user_template": query_user,
    }
    out = {k: {"sha256": sha(v), "text": v} for k, v in parts.items()}
    out["prompt_family_sha256"] = sha(" ".join(parts[k] for k in sorted(parts)))
    out["note"] = ("user templates are recorded with {NL} standing for a "
                   "newline so the hash is stable across encodings; the "
                   "authoritative text is in run_experiment.answer_from and "
                   "run_experiment.gen_query")
    out["source_file_sha256"] = {
        "run_experiment.py": hashlib.sha256(
            pathlib.Path(rx.__file__).read_bytes()).hexdigest()}
    return out


def card(tag, fname, label, short):
    cell = r9.Cell(tag, fname, label, short)
    run, cfg = cell.run, cell.run["config"]
    conf = cell.conf
    pts = cell.simple_points()
    pts.update(cell.learned_points())
    frontier = {k: {"em": round(v["em"], 3), "calls": round(v["calls"], 3)}
                for k, v in pts.items()}
    return {
        "tag": tag, "label": label,
        "data": {"input_file": cfg.get("parquet") or cfg.get("jsonl"),
                 "input_sha256": run.get("input_sha256"),
                 "data_seed": cfg["data_seed"], "n_train": cfg["n_train"],
                 "n_test": cfg["n_test"]},
        "backbone": {"model": cfg["model"], **run.get("model_metadata", {}),
                     "llm_seed": cfg["llm_seed"],
                     "temperature": cfg["llm_temperature"]},
        "retrieval": run.get("retrieval_corpus")
        or {"kind": "per-question distractor BM25"},
        "controller": {"max_steps": cfg["max_steps"], "k": cfg["k"],
                       "primary_rule": cfg["primary_rule"],
                       "alpha": cfg["alpha"],
                       "certificate_calibration_fraction":
                           cfg["certificate_calibration_fraction"],
                       "lambda_logged": cell.lam_logged,
                       "lambda_grid": run.get("lam_grid"),
                       "selection": run.get("selection")},
        "frozen_selection": {
            "rule": "cheapest grid point with train EM >= train reference EM "
                    "- 1.0 band; fallback argmax train EM; ties broken by "
                    "fewest searches; TRAINING CHAINS ONLY",
            "training_selected_fixed_budget": conf.get(
                "training_selected_fixed_budget"),
            "predeclared_primary": conf.get("predeclared_primary"),
        },
        "comparators_present": {
            "tasr": cell.tasr is not None,
            "semantic_guard": cell.sem_stop is not None,
            "verifier_inspired_substitute": cell.halt is not None,
            "stoprag": cell.stoprag is not None,
        },
        "frontier_em_at_calls": frontier,
    }


def per_question_csv(cells, path):
    fields = ["cell", "question_id", "n_hops", "type", "level", "policy",
              "stop_depth", "llm_calls", "input_tokens", "wall_seconds", "em"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for tag, fname, label, short in cells:
            cell = r9.Cell(tag, fname, label, short)
            named = {"fixed_%d" % k: np.full(cell.n, min(k, cell.last))
                     for k in range(cell.max_steps + 1)}
            named["answer_repeat"] = cell.rep_stop
            named["frozen_value"] = cell.v_stop
            named["guarded_fbg"] = cell.g_stop
            named["full_budget"] = np.full(cell.n, cell.last)
            for name, stops in named.items():
                stops = np.asarray(stops, int)
                for i, r in enumerate(cell.records):
                    c = cell.cost[i, stops[i]]
                    w.writerow({"cell": tag, "question_id": r["id"],
                                "n_hops": r.get("n_hops"), "type": r.get("type"),
                                "level": r.get("level"), "policy": name,
                                "stop_depth": int(stops[i]),
                                "llm_calls": round(float(c[0]), 3),
                                "input_tokens": round(float(c[1]), 1),
                                "wall_seconds": round(float(c[2]), 4),
                                "em": int(cell.em[i, stops[i]])})


def tex_table(cards, path):
    rows = []
    for c in cards:
        cells = [c["label"].replace("&", r"\&"),
                 str(c["backbone"]["model"]).replace("_", r"\_"),
                 str(c["data"]["data_seed"]),
                 str(c["data"]["n_test"]),
                 str(c["controller"]["max_steps"]),
                 "%.3f" % c["controller"]["lambda_logged"],
                 str(c["frozen_selection"]["training_selected_fixed_budget"]),
                 str(c["retrieval"].get("sentence_count", "--"))]
        rows.append(" & ".join(cells) + r" \\")
    pathlib.Path(path).write_text("\n".join(rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    # The stored cards (results/round10/policy_cards.json) list the Stop-RAG and
    # verifier-inspired comparators that were present in the round-9 analysis.
    r9.RESTRICT_BASELINES = True
    cells = r9.fresh_cells() + r9.fresh_cells(extra=True)
    cards = [card(*c) for c in cells]
    (outdir / "policy_cards.json").write_text(json.dumps(
        {"what": "one policy/configuration card per cell; a join of data "
                 "already on disk, no new numbers",
         "cards": cards}, indent=1, default=str))
    tex_table(cards, outdir / "policy_cards.tex")
    seeds = {
        "data_seeds": sorted({c["data"]["data_seed"] for c in cards}),
        "llm_seeds": sorted({c["backbone"]["llm_seed"] for c in cards}),
        "certificate_shuffle_seed": 41,
        "cross_fitting_seed": 3,
        "bootstrap_seeds": {"round9_analyses": 2029,
                            "analyze_corpus_confirmation": 101},
        "note": "documented in artifact/README.md 'Determinism and seeds'; "
                "this ledger is the consolidated view",
    }
    (outdir / "prompt_and_seed_ledger.json").write_text(json.dumps(
        {"prompts": prompt_ledger(), "seeds": seeds}, indent=1))
    per_question_csv(cells, outdir / "per_question_paired.csv")
    print("wrote %d cards, ledger, and per-question CSV to %s" % (len(cards), outdir))
    for c in cards:
        print("  %-40s lam=%.3f B*=%s corpus=%s" % (
            c["tag"], c["controller"]["lambda_logged"],
            c["frozen_selection"]["training_selected_fixed_budget"],
            c["retrieval"].get("sentence_count", "--")))


if __name__ == "__main__":
    main()
