"""Analyse ONE external cell against the frozen endpoint document.

Runs the paper's own confirmatory analysis and the round-9 per-cell replay on
a tag that is not part of the frozen 26-cell matrix, and writes everything to
a separate directory so no frozen file is touched and the 26-cell matrix is
unchanged.

    python3 external_cell_report.py --tag extnews_guarded_qwen3b --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import round9_analyses as r9


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # The frozen external analysis predates the verifier-inspired replay on
    # this cell, so it compares the same baselines as the round-9 tables.
    r9.RESTRICT_BASELINES = True
    tag = args.tag
    label = args.label or tag
    fname = f"corpus_confirmation_{tag}.json"
    cell = r9.Cell(tag, fname, label, "EXT")
    analysis = r9.analyse_cell(cell)

    conf = json.loads((r9.MEAS / fname).read_text())
    primary = conf.get("primary")
    simple = cell.simple_points()
    learned = cell.learned_points()
    idx = np.arange(cell.n)
    full = cell.point(np.full(cell.n, cell.last))
    g = cell.point(cell.g_stop)
    ci = r9.paired_ci(g["em_vec"] - full["em_vec"])

    def verdict(ci95):
        return ("win" if ci95[0] > 0 else
                ("loss" if ci95[1] < 0 else "unresolved"))

    report = {
        "tag": tag, "label": label,
        "frozen_endpoint_document":
            "results/round10/"
            "frozen-external-endpoint-2026-08-24.md",
        "primary_endpoint": {
            "quality_metric": "exact match (EM)",
            "cost_metric": "exact LLM calls (1 + 2T)",
            "contrast": "guarded FBG minus full budget",
            "predeclared_success_criterion":
                "paired-bootstrap 95% one-sided lower bound > -3.0 EM AND "
                "guarded stopping reduces mean LLM calls",
            "guarded_em": round(g["em"], 3),
            "guarded_calls": round(g["calls"], 3),
            "full_budget_em": round(full["em"], 3),
            "full_budget_calls": round(full["calls"], 3),
            "paired_delta_em": ci,
            "reduces_calls": bool(g["calls"] < full["calls"]),
            "criterion_met": bool(ci["lower95_one_sided"] > -3.0
                                  and g["calls"] < full["calls"]),
            "verdict_vs_full_budget": verdict(ci["ci95"]),
        },
        "confirmation_file_primary_block": primary,
        "simple_family": {k: {"em": round(v["em"], 3),
                              "calls": round(v["calls"], 3)}
                          for k, v in simple.items()},
        "learned_points": {k: {"em": round(v["em"], 3),
                               "calls": round(v["calls"], 3)}
                           for k, v in learned.items()},
        "round9_cell_analysis": to_jsonable(analysis),
        "single_draw_note":
            "one 400-question draw at data_seed 13; NOT a five-draw "
            "replication like the primary HotpotQA cell",
    }
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"external_cell_{tag}.json").write_text(
        json.dumps(report, indent=1, default=str))
    p = report["primary_endpoint"]
    print(json.dumps({k: p[k] for k in
                      ["guarded_em", "guarded_calls", "full_budget_em",
                       "full_budget_calls", "paired_delta_em", "reduces_calls",
                       "criterion_met", "verdict_vs_full_budget"]}, indent=1))
    print("simple family:", report["simple_family"])
    print("wrote", outdir / f"external_cell_{tag}.json")


if __name__ == "__main__":
    main()
