"""Diagnostics and recovery experiments on logged trajectories.

This module only replays already logged trajectories.  It deliberately keeps
exploratory policy-family comparisons separate from confirmatory claims.  The
outputs are machine-readable and are regenerated from the measured run files:

* fixed-budget, answer-repeat, value-gate, and guarded-value Pareto results;
* paired uncertainty and Holm-corrected exact sign tests;
* out-of-fold score calibration and policy regret on training chains;
* feature-family ablations on identical out-of-fold folds;
* chain-bootstrap threshold-selection stability;
* SGR certificates calibrated on training episodes and evaluated on test
  episodes (never calibrated on a held-out test half);
* exact LLM-call counts implied by each selected prefix.

The guarded-value rule stops only after BOTH signals have fired: the learned
value score has crossed its frozen threshold and the normalized answer has
repeated on consecutive prefixes.  Because old public logs record the first
answer-repeat prefix but not the answers themselves, replay uses the equivalent
index max(first_value_crossing, first_answer_repeat).  This is an exploratory
recovery analysis for existing runs; a fresh corpus-index run must predeclare
the rule before collecting its test outcomes.
"""
from __future__ import annotations

import json
import pathlib
import pickle

import numpy as np
from scipy.stats import binomtest, spearmanr
from sklearn.ensemble import GradientBoostingRegressor

import frozen_protocol as fp


MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
OUT = MEAS / "revision_experiments.json"
FEATURE_NAMES = [
    "searches", "evidence_size", "question_coverage", "new_evidence",
    "answer_changed", "answer_length", "answer_grounding", "stable_two",
    "distinct_answers",
]
FEATURE_FAMILIES = {
    "full": list(range(9)),
    "step_only": [0],
    "stability_only": [4, 7, 8],
    "retrieval_only_no_answer_proxies": [0, 1, 2, 3],
    "answer_only_no_step_or_evidence_size": [4, 5, 6, 7, 8],
    "full_without_step": list(range(1, 9)),
    "full_without_stability": [0, 1, 2, 3, 5, 6],
}
N_BOOT = 10_000


def _value_stop(r, lam):
    return next((i for i, v in enumerate(r["prefix_value"][:-1]) if v < lam),
                len(r["prefix_value"]) - 1)


def _indices(qs, policy, lam=None, budget=None):
    out = []
    for r in qs:
        last = len(r["prefix_em"]) - 1
        if policy == "react":
            i = last
        elif policy == "value":
            i = _value_stop(r, lam)
        elif policy == "answer_repeat":
            i = int(r.get("answer_repeat_searches", r.get("ent_searches")))
        elif policy == "guarded_value":
            repeat = r.get("answer_repeat_searches", r.get("ent_searches"))
            i = max(_value_stop(r, lam), int(repeat))
        elif policy == "fixed":
            i = int(budget)
        else:
            raise ValueError(policy)
        out.append(min(i, last))
    return np.asarray(out, int)


def _paired(a, b, seed=19):
    """Paired EM difference a-b with bootstrap CI and exact sign test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(d), (N_BOOT, len(d)))
    bd = 100 * d[ix].mean(axis=1)
    wins, losses = int(np.sum(d > 0)), int(np.sum(d < 0))
    p = (binomtest(wins, wins + losses, 0.5).pvalue
         if wins + losses else 1.0)
    return {
        "mean_diff_em_points": round(100 * float(d.mean()), 3),
        "ci95": [round(float(np.percentile(bd, 2.5)), 3),
                 round(float(np.percentile(bd, 97.5)), 3)],
        "wins": wins, "losses": losses, "ties": int(np.sum(d == 0)),
        "exact_two_sided_p": float(p),
    }


def _holm(items):
    """Holm correction over [(id, p)], returning adjusted p by id."""
    ordered = sorted(items, key=lambda x: x[1])
    m, running, out = len(ordered), 0.0, {}
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        out[key] = running
    return out


def _summary(qs, idx):
    em = np.asarray([r["prefix_em"][i] for r, i in zip(qs, idx)], float)
    f1 = np.asarray([r["prefix_f1"][i] for r, i in zip(qs, idx)], float)
    searches = np.asarray([r["prefix_searches"][i]
                           for r, i in zip(qs, idx)], float)
    # One initial answer, then one query-generation and one answer call/search.
    calls = 1 + 2 * searches
    return {
        "em": round(100 * float(em.mean()), 2),
        "f1": round(100 * float(f1.mean()), 2),
        "searches": round(float(searches.mean()), 3),
        "llm_calls": round(float(calls.mean()), 3),
        "em_vector": em,
    }


def _pareto(rows):
    """Label rows dominated in (higher EM, fewer searches) space."""
    for name, row in rows.items():
        row["pareto_dominated_by"] = sorted([
            other for other, q in rows.items()
            if other != name
            and q["em"] >= row["em"] and q["searches"] <= row["searches"]
            and (q["em"] > row["em"] or q["searches"] < row["searches"])
        ])


def _chain_folds(chains, folds=5, seed=3):
    rng = np.random.default_rng(seed)
    order = np.arange(len(chains))
    rng.shuffle(order)
    return [set(order[f::folds].tolist()) for f in range(folds)]


def _oof_regression(X, y, chains, cols, seed=3):
    pred = np.zeros(len(X), float)
    for held in _chain_folds(chains, seed=seed):
        tr = [i for c, ch in enumerate(chains) if c not in held for i in ch]
        te = [i for c in held for i in chains[c]]
        model = GradientBoostingRegressor(
            max_depth=2, n_estimators=80, random_state=seed)
        model.fit(X[np.ix_(tr, cols)], y[tr])
        pred[te] = model.predict(X[np.ix_(te, cols)])
    return pred


def _calibration(scores, target, bins=5):
    scores, target = np.asarray(scores), np.asarray(target)
    mae = float(np.mean(np.abs(scores - target)))
    rmse = float(np.sqrt(np.mean((scores - target) ** 2)))
    rho = float(spearmanr(scores, target).statistic)
    order = np.argsort(scores, kind="stable")
    groups = np.array_split(order, bins)
    curve = [{
        "n": len(g),
        "mean_prediction": round(float(scores[g].mean()), 4),
        "mean_target": round(float(target[g].mean()), 4),
        "positive_target_rate": round(float(np.mean(target[g] > 0)), 4),
    } for g in groups if len(g)]
    return {"mae": round(mae, 4), "rmse": round(rmse, 4),
            "spearman": round(rho, 4),
            "underprediction_rate": round(float(np.mean(scores < target)), 4),
            "mean_signed_error_prediction_minus_target": round(
                float(np.mean(scores - target)), 4),
            "equal_count_bins": curve}


def _train_policy(scores, X, y_err, chains, lam):
    em, searches = fp.replay_chains(scores, X, y_err, chains, lam, "lt")
    oracle = np.asarray([max(1 - y_err[i] for i in ch) for ch in chains], float)
    full = np.asarray([1 - y_err[ch[-1]] for ch in chains], float)
    return {
        "em": round(100 * float(em.mean()), 2),
        "searches": round(float(searches.mean()), 3),
        "oracle_regret_em_points": round(100 * float((oracle - em).mean()), 2),
        "full_budget_diff_em_points": round(100 * float((em - full).mean()), 2),
    }


def _threshold_bootstrap(X, y_vtg, y_err, chains, grid, reps=100, seed=29):
    """Bootstrap the full chain-resample/refit/OOB-score/select pipeline."""
    rng = np.random.default_rng(seed)
    selected, oob_perf = [], []
    n = len(chains)
    for rep in range(reps):
        sampled = rng.integers(0, n, n)
        present = set(sampled.tolist())
        oob = [c for c in range(n) if c not in present]
        if len(oob) < max(5, n // 10):
            continue
        tr_rows = [i for c in sampled for i in chains[c]]
        te_rows = [i for c in oob for i in chains[c]]
        model = GradientBoostingRegressor(
            max_depth=2, n_estimators=80, random_state=seed + rep)
        model.fit(X[tr_rows], y_vtg[tr_rows])
        sc = np.zeros(len(X), float)
        sc[te_rows] = model.predict(X[te_rows])
        oob_chains = [chains[c] for c in oob]
        th, meta = fp.select_threshold(sc, X, y_err, oob_chains, grid)
        selected.append(float(th))
        picked = next(x for x in meta["train_frontier"] if x["th"] == th)
        oob_perf.append(picked)
    vals, counts = np.unique(selected, return_counts=True)
    return {
        "requested_repetitions": reps,
        "valid_oob_repetitions": len(selected),
        "threshold_frequency": {str(float(v)): int(c)
                                for v, c in zip(vals, counts)},
        "median_threshold": float(np.median(selected)) if selected else None,
        "selection_probability_modal": (round(float(max(counts) / len(selected)), 3)
                                         if selected else None),
        "oob_selected_em_mean": (round(float(np.mean([x["em"] for x in oob_perf])), 2)
                                 if oob_perf else None),
        "oob_selected_searches_mean": (
            round(float(np.mean([x["s"] for x in oob_perf])), 3)
            if oob_perf else None),
    }


def _sgr_train_to_test(qs, train_scores, X, y_err, chains, lam,
                       alphas=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
    train_stops = []
    for ch in chains:
        stop = ch[-1]
        for i in ch[:-1]:
            if train_scores[i] < lam:
                stop = i
                break
        train_stops.append(stop)
    cal_hat = [train_scores[i] for i in train_stops]
    cal_err = [int(y_err[i]) for i in train_stops]
    test_stops = _indices(qs, "value", lam=lam)
    out = []
    for alpha in alphas:
        th, ncal = fp.sgr_threshold(cal_hat, cal_err, alpha, delta=0.10)
        if th is None:
            out.append({"alpha": alpha, "fired": False, "n_cal": len(cal_hat)})
            continue
        selected = [(r["prefix_errhat"][i], 1 - r["prefix_em"][i])
                    for r, i in zip(qs, test_stops)
                    if r["prefix_errhat"][i] <= th]
        out.append({
            "alpha": alpha, "fired": True, "q_alpha": round(float(th), 5),
            "n_cal": len(cal_hat), "n_cal_selected": int(ncal),
            "test_coverage": round(len(selected) / len(qs), 4),
            "test_risk": (round(float(np.mean([e for _, e in selected])), 4)
                          if selected else None),
            "n_test_selected": len(selected),
        })
    return out


def run_cell(cell):
    tag, lam = cell["tag"], float(cell["lambda"])
    run = fp.load_run(tag)
    qs = run["per_q"]
    grid = [float(v) for v in run["lam_grid"]]
    X, y_vtg, y_err, chains = fp.load_chains(tag)
    value_oof, _ = fp.oof_scores(X, y_vtg, y_err, chains)

    policies = {}
    for name, kw in [
        ("react", {}), ("value", {"lam": lam}),
        ("answer_repeat", {}), ("guarded_value", {"lam": lam}),
    ]:
        policies[name] = _summary(qs, _indices(qs, name, **kw))
    max_budget = max(len(r["prefix_em"]) for r in qs) - 1
    for b in range(max_budget + 1):
        policies[f"fixed_{b}"] = _summary(
            qs, _indices(qs, "fixed", budget=b))
    _pareto(policies)

    comparisons = {}
    react_vec = policies["react"]["em_vector"]
    for name in ["value", "answer_repeat", "guarded_value"]:
        comparisons[f"{name}_vs_react"] = _paired(
            policies[name]["em_vector"], react_vec)

    ablations = {}
    for name, cols in FEATURE_FAMILIES.items():
        scores = _oof_regression(X, y_vtg, chains, cols)
        selected, meta = fp.select_threshold(scores, X, y_err, chains, grid)
        ablations[name] = {
            "features": [FEATURE_NAMES[i] for i in cols],
            "selected_lambda": selected,
            "selection_rule": meta["rule"],
            "oof_training_policy": _train_policy(
                scores, X, y_err, chains, selected),
            "score_calibration": _calibration(scores, y_vtg),
        }

    # Remove private vectors from JSON after paired comparisons are computed.
    for p in policies.values():
        p.pop("em_vector")
    return {
        "tag": tag, "n_test": len(qs), "n_train_chains": len(chains),
        "frozen_lambda": lam, "policies": policies,
        "paired_comparisons": comparisons,
        "full_oof_score_calibration": _calibration(value_oof, y_vtg),
        "feature_family_ablations": ablations,
        "threshold_pipeline_bootstrap": _threshold_bootstrap(
            X, y_vtg, y_err, chains, grid),
        "sgr_train_calibrated_test_evaluated": _sgr_train_to_test(
            qs, value_oof, X, y_err, chains, lam),
        "cost_accounting_note": (
            "llm_calls=1+2*searches for the evaluated loop: one initial "
            "answer plus query-generation and answer-generation per search; "
            "retrieval calls are reported separately as searches. Historical "
            "logs do not support honest per-prefix token or latency totals."),
    }


def main():
    frozen = json.loads((MEAS / "frozen_protocol.json").read_text())
    cells = [run_cell(c) for c in
             frozen["hotpot_cells"] + frozen["dataset_cells"]]

    tests = []
    for c in cells:
        for comp, row in c["paired_comparisons"].items():
            tests.append((f"{c['tag']}::{comp}", row["exact_two_sided_p"]))
    adjusted = _holm(tests)
    for c in cells:
        for comp, row in c["paired_comparisons"].items():
            key = f"{c['tag']}::{comp}"
            row["holm_p_across_all_reported_policy_tests"] = adjusted[key]

    result = {
        "status": "exploratory_replay_of_existing_test_logs",
        "confirmatory_boundary": (
            "No policy family was chosen from these test results. The guarded "
            "rule must be frozen before evaluation on a fresh corpus-index "
            "test set to support a confirmatory claim."),
        "multiple_comparisons_family": (
            "All value/repeat/guarded versus ReAct exact paired sign tests "
            "across all replayed cells; Holm family-wise correction."),
        "cells": cells,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}")
    for c in cells:
        p = c["policies"]
        print(c["tag"], "|", " | ".join(
            f"{n} {p[n]['em']:.1f}@{p[n]['searches']:.2f}"
            for n in ("react", "value", "answer_repeat", "guarded_value")))


if __name__ == "__main__":
    main()
