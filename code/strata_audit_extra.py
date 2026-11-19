"""Additional subgroup strata, computed offline.

Extends (does not replace) strata_audit.py, which already reports
bridge/comparison, level, and gold-support-recall strata.  New strata, all
read from fields the runs already carry:

  * hop depth      -- MuSiQue `n_hops` in {2,3,4} (mixed depth WITHIN
                      multi-hop; HotpotQA cells are uniformly 2-hop, so the
                      contrast is only available on the MuSiQue cells and
                      that limitation is recorded per cell);
  * evidence length -- terciles of the total token length of the retrieved
                      evidence at the full budget (short / medium / long);
  * retrieval ambiguity -- terciles of a proxy computed from the logged
                      retrieval only: the mean pairwise Jaccard similarity of
                      the retrieved evidence units at the full budget.  High
                      similarity == the corpus offered many near-duplicate
                      candidates, i.e. an ambiguous retrieval neighbourhood.

Per stratum: n, full-budget EM, guarded EM, and the paired guarded-minus-full
contrast with a percentile-bootstrap 95% interval and a four-way verdict
(win / loss / unresolved; "tie" is reserved for the pre-registered
non-inferiority margin used in the main matrix and is not claimed here).

    python3 strata_audit_extra.py --out <dir>
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import re

import numpy as np

import round9_analyses as r9


def _tok(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def paired(a, b, seed=2029, n_boot=10000):
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = 100 * d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {"n": int(len(d)), "mean_em_points": round(100 * float(d.mean()), 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "verdict": "win" if lo > 0 else ("loss" if hi < 0 else "unresolved")}


def terciles(values):
    v = np.asarray(values, float)
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    return np.where(v <= q1, "low", np.where(v <= q2, "mid", "high"))


def cell_strata(tag, fname, label, short):
    cell = r9.Cell(tag, fname, label, short)
    idx = np.arange(cell.n)
    full_em = cell.em[:, cell.last].astype(float)
    guard_em = cell.em[idx, cell.g_stop].astype(float)

    ev_len, ambiguity, hops = [], [], []
    for r in cell.records:
        ev = r["trajectory"][cell.last].get("evidence", []) if r["trajectory"] else []
        ev_len.append(sum(len(str(e).split()) for e in ev))
        toks = [_tok(e) for e in ev]
        sims = [len(a & b) / max(1, len(a | b)) for a, b in itertools.combinations(toks, 2)]
        ambiguity.append(float(np.mean(sims)) if sims else 0.0)
        hops.append(r.get("n_hops"))

    out = {"tag": tag, "label": label, "n_test": int(cell.n),
           "full_budget_em": round(100 * float(full_em.mean()), 3),
           "guarded_em": round(100 * float(guard_em.mean()), 3),
           "strata": {}}

    hop_vals = sorted({h for h in hops if h is not None})
    if len(hop_vals) > 1:
        out["strata"]["hop_depth"] = {
            str(h): dict(paired(guard_em[np.asarray(hops) == h],
                                full_em[np.asarray(hops) == h]),
                         full_budget_em=round(100 * float(full_em[np.asarray(hops) == h].mean()), 3))
            for h in hop_vals}
    else:
        out["strata"]["hop_depth"] = None
        out["hop_depth_note"] = (f"single hop depth {hop_vals} in this cell; "
                                 "a mixed-depth contrast is not available here")

    for name, values in [("evidence_length_tokens", ev_len),
                         ("retrieval_ambiguity_jaccard", ambiguity)]:
        lab = terciles(values)
        block = {}
        for level in ["low", "mid", "high"]:
            m = lab == level
            if m.sum() == 0:
                continue
            block[level] = dict(
                paired(guard_em[m], full_em[m]),
                full_budget_em=round(100 * float(full_em[m].mean()), 3),
                stratum_mean_value=round(float(np.asarray(values, float)[m].mean()), 4))
        out["strata"][name] = block
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cells = r9.fresh_cells() + r9.fresh_cells(extra=True)
    rows = [cell_strata(*c) for c in cells]
    for r in rows:
        h = r["strata"]["hop_depth"]
        hs = ("  ".join(f"{k}h:{v['mean_em_points']:+.1f}{v['verdict'][0]}"
                        for k, v in h.items()) if h else "single-depth")
        a = r["strata"]["retrieval_ambiguity_jaccard"]
        asum = "  ".join(f"{k}:{v['mean_em_points']:+.1f}{v['verdict'][0]}" for k, v in a.items())
        print(f"{r['tag']:40s} hops[{hs}] ambig[{asum}]", flush=True)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "strata_audit_extra.json").write_text(json.dumps(
        {"what": "hop-depth, evidence-length and retrieval-ambiguity strata; "
                 "paired guarded-minus-full contrasts inside each stratum",
         "verdict_vocabulary": {"win": "95% CI entirely above 0",
                                "loss": "95% CI entirely below 0",
                                "unresolved": "95% CI contains 0; NOT evidence "
                                              "of equivalence"},
         "note": "no easy/hard contrast is available: the HotpotQA cells are "
                 "100% level='hard' and the other datasets carry no difficulty "
                 "label. That subgroup is not reported rather than invented.",
         "cells": rows}, indent=1))
    print("wrote", outdir / "strata_audit_extra.json")


if __name__ == "__main__":
    main()
