"""Aggregate predeclared corpus-confirmation replications without seed picking.

The primary seed-7 run uses deterministic decoding on data seed 13.  Seeds
11/19 vary stochastic decoding on the *same* question sample; data seed 17
varies the question sample under deterministic decoding.  Every run is first
judged independently by analyze_corpus_confirmation.py.  This script reports
all outcomes and an equal-weight summary; it never chooses a favourable seed.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np


MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
RUNS = [
    ("seed7_data13_deterministic", "corpus_confirmation.json",
     "primary_predeclared"),
    ("seed11_data13_stochastic", "corpus_confirmation_corpus_guarded_qwen3b_seed11.json",
     "trajectory_replication"),
    ("seed19_data13_stochastic", "corpus_confirmation_corpus_guarded_qwen3b_seed19.json",
     "trajectory_replication"),
    ("seed7_data17_deterministic", "corpus_confirmation_corpus_guarded_qwen3b_data17.json",
     "question_sample_replication"),
    ("seed7_data23_deterministic", "corpus_confirmation_corpus_guarded_qwen3b_data23.json",
     "question_sample_replication"),
    ("seed7_data29_deterministic", "corpus_confirmation_corpus_guarded_qwen3b_data29.json",
     "question_sample_replication"),
    ("seed7_data31_deterministic", "corpus_confirmation_corpus_guarded_qwen3b_data31.json",
     "question_sample_replication"),
]
METHODS = ["full_budget", "fixed_3", "answer_repeat",
           "repeat_conditioned_gain", "semantic_guard",
           "value_only", "guarded_value"]


def main():
    found = []
    for label, filename, role in RUNS:
        path = MEAS / filename
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        found.append((label, role, filename, result))
    if not found:
        raise SystemExit("no completed corpus-confirmation analyses")

    per_run = []
    for label, role, filename, result in found:
        methods = result["methods"]
        per_run.append({
            "label": label,
            "role": role,
            "analysis_file": filename,
            "llm_seed": result.get("llm_seed"),
            "data_seed": result.get("data_seed"),
            "n_test": result["n_test"],
            "guarded_primary": result["predeclared_primary"],
            "repeat_conditioned": result["repeat_conditioned_confirmation"],
            "methods": {name: {
                key: methods[name][key]
                for key in ["em", "searches", "llm_calls"]
            } for name in METHODS if name in methods},
        })

    common = sorted(set.intersection(*[
        set(row["methods"]) for row in per_run
    ]))
    equal_weight = {}
    spread = {}
    for method in common:
        equal_weight[method] = {
            metric: round(float(np.mean([
                row["methods"][method][metric] for row in per_run
            ])), 3)
            for metric in ["em", "searches", "llm_calls"]
        }
        spread[method] = {
            metric: {
                "min": round(float(np.min(values)), 3),
                "max": round(float(np.max(values)), 3),
                "sd_across_runs": round(float(np.std(values, ddof=0)), 3),
            }
            for metric in ["em", "searches", "llm_calls"]
            for values in [[row["methods"][method][metric]
                            for row in per_run]]
        }

    output = {
        "aggregation_rule": (
            "Every completed run is reported; equal-weight means are descriptive. "
            "No run, seed, threshold, or policy is selected by test outcome."
        ),
        "replication_axes": {
            "trajectory_stochasticity": "data seed 13; LLM seeds 11 and 19 at temperature 0.2",
            "question_sampling": "LLM seed 7 at temperature 0; data seeds 13 and 17",
        },
        "n_completed": len(per_run),
        "per_run": per_run,
        "equal_weight_descriptive_mean": equal_weight,
        "descriptive_across_run_spread": spread,
        "guarded_pass_count": sum(
            bool(row["guarded_primary"]["confirmation_success"])
            for row in per_run
        ),
        "repeat_conditioned_fresh_pass_count": sum(
            bool(row["repeat_conditioned"]["confirmation_success"])
            for row in per_run if row["role"] != "primary_predeclared"
        ),
        "repeat_conditioned_fresh_run_count": sum(
            row["role"] != "primary_predeclared" for row in per_run
        ),
    }
    dest = MEAS / "corpus_robustness.json"
    dest.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
