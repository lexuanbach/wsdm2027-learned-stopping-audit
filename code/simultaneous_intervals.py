"""Multiplicity-aware SIMULTANEOUS intervals over the cell family.

The manuscript already applies Holm to the exploratory sign tests, but the
per-cell paired intervals in the executive matrix are marginal 95% intervals.
This script adds the family-wise view the panel asked for, computed offline
from the per-question paired outcomes already on disk.

Two adjustments, both reported:
  * Bonferroni  -- per-cell percentile intervals at level alpha / K.
  * max-t band  -- a studentized (max-t) simultaneous band: bootstrap the
    questions inside each cell independently, take the maximum absolute
    studentized deviation across the K cells per replicate, and use its
    95th percentile as one common critical value.  This is the honest
    "how many cells stay resolved once you look at all of them" answer, and
    it is less conservative than Bonferroni.

Nothing is selected here.  The cells, the policy and the contrast were all
fixed long before this script existed; only the interval width changes.

    python3 simultaneous_intervals.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import meta_review_analyses as mra

MEAS = mra.MEAS
N_BOOT = 20000
ALPHA = 0.05


def paired_vectors(tag):
    """Per-question (guarded - full budget) EM differences for one cell."""
    run = json.loads((MEAS / f"run_{tag}.json").read_text())
    d = []
    for r in run["per_q"]:
        last = len(r["prefix_em"]) - 1
        stop = int(r["frozen_guarded"]["searches"]) if "frozen_guarded" in r \
            else int(r["primary"]["searches"])
        d.append(float(r["prefix_em"][min(stop, last)]) - float(r["prefix_em"][last]))
    return np.asarray(d, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dm = json.loads((MEAS / "meta_review_analyses.json").read_text())["decision_matrix"]
    label = {c["cell"]: c["label"] for c in dm["cells"]}
    # Only cells whose per-question guarded stop is logged in the run file can
    # have their paired vector reconstructed exactly.  The 12 historical cells
    # predate that field, so they are NOT included; their marginal intervals
    # in the executive matrix are unchanged and this restriction is reported.
    import round9_analyses as r9
    keep, vecs = [], {}
    for tag, fname, lab, short in r9.fresh_cells() + r9.fresh_cells(extra=True):
        vecs[tag] = paired_vectors(tag)
        label.setdefault(tag, lab)
        keep.append(tag)
    K = len(keep)

    rng = np.random.default_rng(2029)
    boots, stats_t = {}, []
    for t in keep:
        d = vecs[t]
        n = len(d)
        idx = rng.integers(0, n, (N_BOOT, n))
        rs = d[idx]
        means = 100 * rs.mean(axis=1)
        ses = 100 * rs.std(axis=1, ddof=1) / np.sqrt(n)
        boots[t] = {"means": means, "ses": np.where(ses > 0, ses, np.inf),
                    "obs_mean": 100 * float(d.mean()),
                    "obs_se": 100 * float(d.std(ddof=1) / np.sqrt(n)), "n": n}
    T = np.max(np.vstack([np.abs(boots[t]["means"] - boots[t]["obs_mean"])
                          / boots[t]["ses"] for t in keep]), axis=0)
    crit = float(np.percentile(T, 100 * (1 - ALPHA)))

    rows, resolved = [], {"marginal": 0, "bonferroni": 0, "maxt": 0}
    for t in keep:
        b = boots[t]
        m = b["means"]
        marg = [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
        bq = 100 * (ALPHA / K) / 2
        bonf = [float(np.percentile(m, bq)), float(np.percentile(m, 100 - bq))]
        maxt = [b["obs_mean"] - crit * b["obs_se"], b["obs_mean"] + crit * b["obs_se"]]

        def res(ci):
            return "loss" if ci[1] < 0 else ("win" if ci[0] > 0 else "unresolved")
        v = {"marginal": res(marg), "bonferroni": res(bonf), "maxt": res(maxt)}
        for k in resolved:
            resolved[k] += int(v[k] != "unresolved")
        rows.append({"cell": t, "label": label[t], "n": b["n"],
                     "mean_em_points": round(b["obs_mean"], 3),
                     "marginal_ci95": [round(x, 3) for x in marg],
                     "bonferroni_ci": [round(x, 3) for x in bonf],
                     "maxt_simultaneous_ci": [round(x, 3) for x in maxt],
                     "verdicts": v})
        print("%-28s %+7.2f  marg[%+6.2f,%+6.2f]=%-10s bonf[%+6.2f,%+6.2f]=%-10s "
              "maxt[%+6.2f,%+6.2f]=%s" % (
                  t[:28], b["obs_mean"], marg[0], marg[1], v["marginal"],
                  bonf[0], bonf[1], v["bonferroni"], maxt[0], maxt[1], v["maxt"]),
              flush=True)

    out = {
        "what": "simultaneous / multiplicity-adjusted intervals for the "
                "guarded-minus-full-budget EM contrast across the cell family",
        "contrast": "guarded FBG minus full budget, per-question paired EM",
        "n_cells": K,
        "coverage_note": ("restricted to the cells whose per-question guarded "
                          "stop is logged (the fresh corpus-index block plus "
                          "the crossed backbone block). The 12 historical "
                          "cells predate that logged field, so they are "
                          "excluded rather than approximated."),
        "family_wise_alpha": ALPHA,
        "maxt_critical_value": round(crit, 4),
        "n_boot": N_BOOT,
        "resolved_cells": resolved,
        "reading": ("'unresolved' means the interval contains zero. It is NOT "
                    "evidence of equivalence; the paper's non-inferiority "
                    "claims continue to rest on the pre-registered 3-EM "
                    "margin, not on interval overlap."),
        "cells": rows,
    }
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "simultaneous_intervals.json").write_text(json.dumps(out, indent=1))
    print()
    print("resolved cells (interval excludes 0):", resolved, "of", K)
    print("max-t critical value:", round(crit, 3))
    print("wrote", outdir / "simultaneous_intervals.json")


if __name__ == "__main__":
    main()
