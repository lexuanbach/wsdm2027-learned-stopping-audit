"""Calibration and selective-risk audit of the learned stopping scores.

Offline replay only; no LLM calls.  Two heads are audited per fresh cell:

  * the ERROR head (P(answer is wrong at this prefix)) -- the head the
    sufficiency-style gate thresholds;
  * the FBG VALUE head (predicted future gain) -- the head the guarded
    controller thresholds; it is a regression score, so it gets a
    selective-risk / risk--coverage curve rather than a probability
    calibration curve.

Reported per cell:
  ECE (15 equal-width bins), adaptive ECE (15 equal-count bins), Brier score
  and reliability bins, computed twice -- once on the cross-fitted TRAINING
  chains (the honest in-protocol view) and once on the untouched TEST
  prefixes (an evaluation, never a selection).  Plus a risk--coverage curve
  for the deployed guarded policy: questions ordered by the controller's own
  confidence at its stop, risk = 1 - EM.

    python3 calibration_audit.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import round9_analyses as r9

BINS = 15


def ece(p, y, bins=BINS, adaptive=False):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -1e-9, 1 + 1e-9
    else:
        edges = np.linspace(0, 1, bins + 1)
        edges[0], edges[-1] = -1e-9, 1 + 1e-9
    total, out = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        total += m.mean() * abs(conf - acc)
        out.append({"lo": round(float(lo), 4), "hi": round(float(hi), 4),
                    "n": int(m.sum()), "mean_prediction": round(conf, 4),
                    "observed_rate": round(acc, 4)})
    return float(total), out


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def risk_coverage(conf, correct):
    """Selective risk as a function of coverage, ordered by confidence."""
    order = np.argsort(-np.asarray(conf, float))
    c = np.asarray(correct, float)[order]
    n = len(c)
    curve = []
    for frac in np.arange(0.05, 1.0001, 0.05):
        k = max(1, int(round(frac * n)))
        curve.append({"coverage": round(float(k / n), 4),
                      "risk": round(float(1 - c[:k].mean()), 4)})
    # area under the risk-coverage curve (lower is better)
    risks = np.array([x["risk"] for x in curve])
    return curve, float(risks.mean())


def run_cell(tag, fname, label, short):
    cell = r9.Cell(tag, fname, label, short)
    fit_rows = [i for ch in cell.fit_chains for i in ch]
    pos = {i: k for k, i in enumerate(fit_rows)}
    chains_f = [[pos[i] for i in ch] for ch in cell.fit_chains]
    Xf = cell.X[fit_rows]

    # ---- error head -------------------------------------------------- #
    y_err_f = cell.y_err[fit_rows]
    oof_err = r9.crossfit(Xf, y_err_f.astype(float), chains_f, 5, 3, "logit")
    train_ece, train_bins = ece(oof_err, y_err_f)
    train_aece, _ = ece(oof_err, y_err_f, adaptive=True)
    flat_feats = cell.feats.reshape(-1, cell.feats.shape[-1])
    test_err_hat = r9.fit_predict(Xf, y_err_f.astype(float), flat_feats,
                                  3, "logit").reshape(cell.n, cell.width)
    y_test_err = (1 - cell.em).astype(float)
    t_ece, test_bins = ece(test_err_hat.ravel(), y_test_err.ravel())
    t_aece, _ = ece(test_err_hat.ravel(), y_test_err.ravel(), adaptive=True)

    # ---- FBG value head: risk--coverage at the deployed stop ---------- #
    y_val_f = cell.y_val[fit_rows]
    test_val_hat = r9.fit_predict(Xf, y_val_f, flat_feats, 3,
                                  "gbr").reshape(cell.n, cell.width)
    idx = np.arange(cell.n)
    g_stop = cell.g_stop
    correct = cell.em[idx, g_stop].astype(float)
    # confidence at the stop: low predicted future gain == "nothing more to
    # gain"; and separately 1 - predicted error.
    conf_fbg = -test_val_hat[idx, g_stop]
    conf_err = 1.0 - test_err_hat[idx, g_stop]
    rc_fbg, aurc_fbg = risk_coverage(conf_fbg, correct)
    rc_err, aurc_err = risk_coverage(conf_err, correct)
    rng = np.random.default_rng(7)
    rand_aurc = float(np.mean([risk_coverage(rng.random(cell.n), correct)[1]
                               for _ in range(20)]))
    return {
        "tag": tag, "label": label, "n_test": int(cell.n),
        "error_head": {
            "train_crossfit": {"ece_15_equal_width": round(train_ece, 4),
                               "adaptive_ece_15_equal_count": round(train_aece, 4),
                               "brier": round(brier(oof_err, y_err_f), 4),
                               "base_error_rate": round(float(y_err_f.mean()), 4),
                               "reliability_bins": train_bins},
            "test_prefixes": {"ece_15_equal_width": round(t_ece, 4),
                              "adaptive_ece_15_equal_count": round(t_aece, 4),
                              "brier": round(brier(test_err_hat.ravel(),
                                                   y_test_err.ravel()), 4),
                              "base_error_rate": round(float(y_test_err.mean()), 4),
                              "n_prefixes": int(y_test_err.size),
                              "reliability_bins": test_bins},
        },
        "selective_risk_at_guarded_stop": {
            "policy_em": round(100 * float(correct.mean()), 3),
            "aurc_fbg_value_head": round(aurc_fbg, 4),
            "aurc_error_head": round(aurc_err, 4),
            "aurc_random_ordering_mean_of_20": round(rand_aurc, 4),
            "curve_fbg_value_head": rc_fbg,
            "curve_error_head": rc_err,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", default=None)
    args = ap.parse_args()
    cells = r9.fresh_cells() + r9.fresh_cells(extra=True)
    if args.tags:
        keep = set(args.tags.split(","))
        cells = [c for c in cells if c[0] in keep]
    results = []
    for tag, fname, label, short in cells:
        r = run_cell(tag, fname, label, short)
        results.append(r)
        e = r["error_head"]
        print(f"{tag:40s} ECE_train={e['train_crossfit']['ece_15_equal_width']:.3f} "
              f"Brier_train={e['train_crossfit']['brier']:.3f} "
              f"ECE_test={e['test_prefixes']['ece_15_equal_width']:.3f} "
              f"Brier_test={e['test_prefixes']['brier']:.3f} "
              f"AURC_fbg={r['selective_risk_at_guarded_stop']['aurc_fbg_value_head']:.3f} "
              f"AURC_rand={r['selective_risk_at_guarded_stop']['aurc_random_ordering_mean_of_20']:.3f}",
              flush=True)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "calibration_audit.json").write_text(json.dumps(
        {"what": "ECE / Brier / reliability bins for the error head and "
                 "risk--coverage for the FBG value head, offline replay",
         "note": "test-prefix calibration is an EVALUATION; no threshold, "
                 "policy or cell was selected on it",
         "cells": results}, indent=1))
    print("wrote", outdir / "calibration_audit.json")


if __name__ == "__main__":
    main()
