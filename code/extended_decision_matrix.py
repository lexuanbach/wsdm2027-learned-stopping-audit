"""One table separating IMPROVEMENT / FRONTIER / DOMINATED / UNRESOLVED for every stopping
policy, with the new baselines folded in.

Two vocabularies are kept apart on purpose:

  * dominance verdict against the simple
    family {fixed-0..fixed-B*, answer repetition} on the (EM, calls) plane):
        D = a simple policy Pareto-dominates the learned policy
        I = undominated AND strictly improves the training-selected B*
        F = undominated but does not strictly improve B*
  * interval verdict against the randomized fixed-budget mixture at MATCHED
    expected cost:
        win        = paired 95% CI entirely above 0
        loss       = paired 95% CI entirely below 0
        unresolved = the CI contains 0.  This is NOT equivalence; it is the
                     absence of a resolved difference.

Policies covered per cell: every fixed budget, answer repetition, FBG value
only, FBG + repeat guard, the same-feature stop classifier, the
sufficiency-style gate on the paper's own error head, TASR (native logits
only), Stop-RAG, the verifier-inspired substitute, and the NEW
gold-support-supervised sufficiency classifier.

    python3 extended_decision_matrix.py --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

import round9_analyses as r9
import sufficiency_classifier as sc


def interval_verdict(ci95):
    return ("win" if ci95[0] > 0 else
            ("loss" if ci95[1] < 0 else "unresolved"))


def analyse(tag, fname, label, short, suff_lookup):
    cell = r9.Cell(tag, fname, label, short)
    simple = cell.simple_points()
    learned = cell.learned_points()
    clf = cell.pipeline(target="stopclf")
    gate = cell.pipeline(target="suff")
    learned["stop_classifier"] = cell.point(clf["stops"]["classifier"])
    learned["sufficiency_gate"] = cell.point(gate["stops"]["gate"])

    entry = suff_lookup.get(tag)
    if entry is not None:
        for variant in ["same9", "lex12"]:
            v = entry["variants"][variant]
            learned["gold_support_sufficiency_clf_" + variant] = {
                "em": v["test_em"], "calls": v["test_calls"],
                "tokens": v["test_tokens"], "latency": v["test_latency_seconds"],
                "equal_cost_from_suffclf": v["equal_cost"],
            }

    bstar = cell.bstar
    simple_named = {("fixed-%d" % k): (simple["fixed_%d" % k]["em"],
                                       simple["fixed_%d" % k]["calls"])
                    for k in range(cell.max_steps + 1)}
    simple_named["repetition"] = (simple["answer_repeat"]["em"],
                                  simple["answer_repeat"]["calls"])
    bstar_em = simple["fixed_%d" % bstar]["em"]
    bstar_calls = simple["fixed_%d" % bstar]["calls"]

    pts = {k: {"cost": v["calls"], "em": v["em"], "em_vec": v["em_vec"]}
           for k, v in simple.items()}
    hull = r9.upper_hull(pts)

    rows = {}
    for name, p in learned.items():
        em, calls = p["em"], p["calls"]
        dominated_by = [k for k, (e, c) in simple_named.items()
                        if e >= em and c <= calls and (e > em or c < calls)]
        if dominated_by:
            dom = "D"
        elif em > bstar_em and calls <= bstar_calls:
            dom = "I"
        elif em >= bstar_em and calls < bstar_calls:
            dom = "I"
        else:
            dom = "F"
        row = {"em": round(em, 3), "calls": round(calls, 3),
               "dominance_verdict": dom, "dominated_by": dominated_by}
        if "em_vec" in p:
            mix, n1, n2, w = r9.hull_mixture(hull, calls, pts)
            pc = r9.paired_ci(p["em_vec"] - mix)
            row["vs_matched_mixture_on_calls"] = {
                "mixture_of": [n1, n2], "weight_first": round(w, 4),
                "delta_em": pc, "interval_verdict": interval_verdict(pc["ci95"])}
        elif "equal_cost_from_suffclf" in p:
            e = p["equal_cost_from_suffclf"]["calls"]
            row["vs_matched_mixture_on_calls"] = {
                "mixture_of": e["mixture"]["of"],
                "weight_first": e["mixture"]["weight_first"],
                "delta_em": e["delta_em_vs_mixture_at_equal_cost"],
                "interval_verdict": e["verdict"]}
        rows[name] = row
    return {"tag": tag, "label": label, "n_test": int(cell.n),
            "training_selected_fixed_budget": bstar,
            "simple_family": {k: {"em": round(v[0], 3), "calls": round(v[1], 3)}
                              for k, v in simple_named.items()},
            "policies": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffclf", default=None,
                    help="path to sufficiency_classifier.json")
    args = ap.parse_args()
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    suff_path = pathlib.Path(args.suffclf) if args.suffclf else \
        outdir / "sufficiency_classifier.json"
    suff_lookup = {}
    if suff_path.exists():
        suff_lookup = {c["tag"]: c for c in
                       json.loads(suff_path.read_text())["cells"]}

    cells = [analyse(*c, suff_lookup=suff_lookup)
             for c in r9.fresh_cells() + r9.fresh_cells(extra=True)]

    tally = collections.defaultdict(lambda: collections.Counter())
    coverage = collections.Counter()
    for c in cells:
        for name, row in c["policies"].items():
            tally[name][row["dominance_verdict"]] += 1
            iv = row.get("vs_matched_mixture_on_calls", {}).get("interval_verdict")
            if iv:
                tally[name]["mix_" + iv] += 1
            coverage[name] += 1
    summary = {name: {"cells_covered": coverage[name],
                      "dominance": {k: tally[name][k] for k in "IFD"},
                      "vs_matched_mixture": {
                          k: tally[name]["mix_" + k]
                          for k in ["win", "loss", "unresolved"]}}
               for name in sorted(coverage)}

    (outdir / "extended_decision_matrix.json").write_text(json.dumps(
        {"what": "per-policy dominance verdicts and matched-mixture interval "
                 "verdicts across every fresh and crossed cell, including the "
                 "new gold-support-supervised sufficiency classifier",
         "vocabulary": {
             "I": "undominated by the simple family AND strictly improves B*",
             "F": "undominated but does not strictly improve B*",
             "D": "Pareto-dominated by some simple policy",
             "win/loss/unresolved": "paired 95% CI against the randomized "
                                    "fixed-budget mixture at matched calls; "
                                    "'unresolved' is not equivalence"},
         "summary": summary, "cells": cells}, indent=1, default=str))

    print("%-42s %6s  %-14s %s" % ("policy", "cells", "dominance I/F/D",
                                   "vs matched mixture win/loss/unres"))
    for name, s in summary.items():
        d = s["dominance"]
        m = s["vs_matched_mixture"]
        print("%-42s %6d  %2d/%2d/%2d        %2d/%2d/%2d" % (
            name, s["cells_covered"], d["I"], d["F"], d["D"],
            m["win"], m["loss"], m["unresolved"]))
    print("wrote", outdir / "extended_decision_matrix.json")


if __name__ == "__main__":
    main()
