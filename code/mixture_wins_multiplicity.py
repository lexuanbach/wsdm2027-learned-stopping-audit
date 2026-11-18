"""Do the resolved equal-cost gains over the randomized mixture survive
multiplicity?

The manuscript states that no cell shows a resolvable gain over a randomized
fixed-budget mixture at matched cost.  That is exactly true for the FBG rules,
but results/measured/round9_analyses.json already contains three cells where a
DIFFERENT learned policy -- the same-feature stop classifier, and the
error-head sufficiency gate -- has a marginal 95% CI entirely above zero.  This
script quantifies those cases and asks whether they survive a family-wise
correction over the cells each policy was evaluated on.

For each policy family it computes, on the calls axis:
  * the per-cell paired difference vs. the matched mixture,
  * the marginal 95% CI,
  * a max-t simultaneous band over that policy's own cells,
  * a Bonferroni interval over that policy's own cells.

    python3 mixture_wins_multiplicity.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import round9_analyses as r9

N_BOOT = 20000
ALPHA = 0.05
POLICIES = ["guarded_value", "value_only", "stop_classifier",
            "sufficiency_gate", "tasr", "stoprag", "semantic_guard"]


def cell_vectors(tag, fname, label, short):
    cell = r9.Cell(tag, fname, label, short)
    simple = cell.simple_points()
    learned = cell.learned_points()
    clf = cell.pipeline(target="stopclf")
    gate = cell.pipeline(target="suff")
    learned["stop_classifier"] = cell.point(clf["stops"]["classifier"])
    learned["sufficiency_gate"] = cell.point(gate["stops"]["gate"])
    pts = {k: {"cost": v["calls"], "em": v["em"], "em_vec": v["em_vec"]}
           for k, v in simple.items()}
    hull = r9.upper_hull(pts)
    out = {}
    for pol in POLICIES:
        if pol not in learned:
            continue
        p = learned[pol]
        mix, n1, n2, w = r9.hull_mixture(hull, p["calls"], pts)
        out[pol] = np.asarray(p["em_vec"], float) - np.asarray(mix, float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cells = r9.fresh_cells() + r9.fresh_cells(extra=True)
    data = {}
    for c in cells:
        for pol, vec in cell_vectors(*c).items():
            data.setdefault(pol, {})[c[0]] = vec

    rng = np.random.default_rng(2029)
    report = {}
    for pol, per_cell in data.items():
        tags = sorted(per_cell)
        boots = {}
        for t in tags:
            d = per_cell[t]
            n = len(d)
            rs = d[rng.integers(0, n, (N_BOOT, n))]
            means = 100 * rs.mean(axis=1)
            ses = 100 * rs.std(axis=1, ddof=1) / np.sqrt(n)
            boots[t] = {"means": means, "ses": np.where(ses > 0, ses, np.inf),
                        "obs": 100 * float(d.mean()),
                        "se": 100 * float(d.std(ddof=1) / np.sqrt(n))}
        T = np.max(np.vstack([np.abs(boots[t]["means"] - boots[t]["obs"])
                              / boots[t]["ses"] for t in tags]), axis=0)
        crit = float(np.percentile(T, 100 * (1 - ALPHA)))
        K = len(tags)
        rows, counts = [], {"marginal_win": 0, "bonferroni_win": 0,
                            "maxt_win": 0, "marginal_loss": 0}
        for t in tags:
            b = boots[t]
            m = b["means"]
            marg = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
            bq = 100 * (ALPHA / K) / 2
            bonf = [float(np.percentile(m, bq)), float(np.percentile(m, 100 - bq))]
            maxt = [b["obs"] - crit * b["se"], b["obs"] + crit * b["se"]]
            counts["marginal_win"] += int(marg[0] > 0)
            counts["marginal_loss"] += int(marg[1] < 0)
            counts["bonferroni_win"] += int(bonf[0] > 0)
            counts["maxt_win"] += int(maxt[0] > 0)
            rows.append({"cell": t, "mean_em_points": round(b["obs"], 3),
                         "marginal_ci95": [round(x, 3) for x in marg],
                         "bonferroni_ci": [round(x, 3) for x in bonf],
                         "maxt_ci": [round(x, 3) for x in maxt]})
        report[pol] = {"n_cells": K, "maxt_critical_value": round(crit, 4),
                       "counts": counts, "cells": rows}
        print("%-18s cells=%2d  marginal wins=%d  Bonferroni wins=%d  "
              "max-t wins=%d  (marginal losses=%d)" % (
                  pol, K, counts["marginal_win"], counts["bonferroni_win"],
                  counts["maxt_win"], counts["marginal_loss"]))

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "mixture_wins_multiplicity.json").write_text(json.dumps(
        {"what": "equal-cost (calls axis) gains over the randomized "
                 "fixed-budget mixture, per policy family, with family-wise "
                 "corrections over that policy's own cells",
         "axis": "exact LLM calls",
         "policies": report}, indent=1))
    print("wrote", outdir / "mixture_wins_multiplicity.json")


if __name__ == "__main__":
    main()
