#!/usr/bin/env python3
"""Training-set-size learning curve for the guarded FBG rule (CPU only).

The question is whether the negative result is an artifact of too few
policy-training episodes.  The selection-protocol sensitivity already varies
training size as a side effect of the certificate split (80/90/96 episodes),
but that axis is narrow and confounded with calibration size.  This script
varies it directly: for each fresh cell and each size n it draws n of the
cell's policy-training chains uniformly at random, runs the *identical* frozen
pipeline (cross-fitted scores, frozen threshold selection, replay on the
untouched test questions), and records the guarded rule's EM, calls, and
verdict against the simple family.

Direction is pre-committed before looking at the output: a flat curve is
evidence the null is not data starvation; a curve still improving at n=90 is
evidence that it might be, and is to be reported as such.  Subsampling can only
go DOWN from 90 -- going above 90 needs new backbone generation, which is out
of scope for this submission.

Output: results/measured/training_size_curve.json
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import round9_analyses as r9                                   # noqa: E402

MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"


def run_cell(cell, sizes, seeds):
    full_chains = list(cell.fit_chains)
    n_full = len(full_chains)
    simple = cell.simple_points()
    simple_named = {k: (v["em"], v["calls"]) for k, v in simple.items()}
    bstar_key = f"fixed_{cell.bstar}"
    out = {}
    for n in sizes:
        if n > n_full:
            continue
        draws = []
        for s in seeds:
            rng = np.random.default_rng(10_000 * s + n)
            idx = rng.choice(n_full, size=n, replace=False)
            cell.fit_chains = [full_chains[i] for i in idx]
            try:
                r = cell.pipeline()
            except Exception as exc:                    # keep the curve honest
                draws.append({"seed": s, "error": repr(exc)})
                continue
            p = cell.point(r["stops"]["guarded"])
            v, doms = r9.verdict_eps(p["em"], p["calls"], simple_named, bstar_key, 0.0)
            draws.append({"seed": s, "threshold": r["threshold"],
                          "em": round(p["em"], 3), "calls": round(p["calls"], 3),
                          "verdict": v, "dominated_by": doms})
        cell.fit_chains = full_chains
        ok = [d for d in draws if "error" not in d]
        out[str(n)] = {
            "n_draws": len(ok),
            "n_failed": len(draws) - len(ok),
            "mean_em": round(float(np.mean([d["em"] for d in ok])), 3) if ok else None,
            "em_range": [round(min(d["em"] for d in ok), 3),
                         round(max(d["em"] for d in ok), 3)] if ok else None,
            "mean_calls": round(float(np.mean([d["calls"] for d in ok])), 3) if ok else None,
            "dominated_rate": round(sum(d["verdict"] == "L" for d in ok) / len(ok), 4) if ok else None,
            "verdicts": {v: sum(d["verdict"] == v for d in ok) for v in ("W", "T", "L")},
            "draws": draws,
        }
    return out


def emit_rows(result):
    """LaTeX rows for the supplement's training-size table."""
    tex = (pathlib.Path(__file__).resolve().parents[1] / "outputs" /
           "tables" / "training_size_rows.tex")
    tex.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for n in result["sizes"]:
        e = result["pooled"][str(n)]
        lines.append(f"{n} & {e['n_draws_total']} & "
                     f"{100 * e['dominated_rate']:.1f}\\% & "
                     f"{e['mean_em_over_cells']:.2f} & "
                     f"{e['mean_calls_over_cells']:.2f} \\\\")
    tex.write_text("\n".join(lines) + "\n")
    print("wrote", tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="30,45,60,75,90")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="first N cells (0 = all)")
    ap.add_argument("--out", default=str(MEAS / "training_size_curve.json"))
    ap.add_argument("--emit-only", action="store_true",
                    help="regenerate the LaTeX rows from an existing JSON")
    a = ap.parse_args()
    if a.emit_only:
        emit_rows(json.loads(pathlib.Path(a.out).read_text()))
        return 0
    sizes = [int(x) for x in a.sizes.split(",")]
    seeds = list(range(1, a.seeds + 1))

    tags = r9.fresh_cells()
    if a.limit:
        tags = tags[:a.limit]
    cells = {}
    t0 = time.time()
    for tag, fname, label, short in tags:
        c = r9.Cell(tag, fname, label, short)
        cells[tag] = {"label": label, "n_policy_training_episodes": len(c.fit_chains),
                      "sizes": run_cell(c, sizes, seeds)}
        print(f"[{time.time()-t0:7.1f}s] {tag}: "
              + " ".join(f"n={n}:L={cells[tag]['sizes'][n]['verdicts']['L']}"
                         f"/{cells[tag]['sizes'][n]['n_draws']}"
                         for n in cells[tag]["sizes"]), flush=True)

    # pooled curve across cells: how often is the guard dominated, by size
    pooled = {}
    for n in map(str, sizes):
        ems, doms, tot, calls = [], 0, 0, []
        for tag, c in cells.items():
            e = c["sizes"].get(n)
            if not e or not e["n_draws"]:
                continue
            ems.append(e["mean_em"]); calls.append(e["mean_calls"])
            doms += e["verdicts"]["L"]; tot += e["n_draws"]
        if tot:
            pooled[n] = {
                "n_cells": len(ems),
                "n_draws_total": tot,
                "dominated_rate": round(doms / tot, 4),
                "mean_em_over_cells": round(float(np.mean(ems)), 3),
                "mean_calls_over_cells": round(float(np.mean(calls)), 3),
            }
    result = {
        "description": ("Guarded FBG under the frozen pipeline with the policy-"
                        "training chains subsampled to n episodes, averaged over "
                        f"{len(seeds)} random draws per size per cell. Sizes go "
                        "down from the logged 90; going above 90 requires new "
                        "backbone generation."),
        "sizes": sizes,
        "n_seeds_per_size": len(seeds),
        "pooled": pooled,
        "cells": cells,
        "wall_seconds": round(time.time() - t0, 1),
    }
    pathlib.Path(a.out).write_text(json.dumps(result, indent=1))
    print("wrote", a.out)
    emit_rows(result)
    print(json.dumps(pooled, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
