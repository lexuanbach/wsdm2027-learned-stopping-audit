"""Can we predict, before looking at any test outcome, whether a learned
stopping rule will be worth deploying in a cell?

LearnStop reports that learned exits help only where no scalar signal is
reliable and early correctness is common.  That is a testable hypothesis about
*our* failure map, and it is the difference between "the learned rule usually
loses" and "here is when it loses".  This script therefore:

1. computes, for every evaluation cell, a small set of properties that a
   practitioner can measure **from the training chains alone** -- no test
   question, no test outcome, nothing selected after the fact:
     early_correct_rate     share of training episodes already correct by the
                            first search (LearnStop's early-correctness axis);
     repeat_precision       training EM at the first answer-repeat prefix
                            divided by training EM at the full budget --- how
                            reliable the free scalar signal is;
     repeat_signal_auc      AUC of "the answer just repeated" as a classifier
                            of "answering now is already correct";
     saturation_gap_em      training EM at the full budget minus at fixed-2;
     fbg_positive_rate      share of training prefixes whose FBG target > 0;
     oof_spearman           cross-fitted predicted-FBG vs target rank
                            correlation on the training chains;
     threshold_stability    share of chain-bootstrap refits that reselect the
                            modal threshold;
     train_em_full          training full-budget EM (accuracy level).

2. reads the outcome for that cell from the frozen registry
   (`meta_review_analyses.json` decision matrix, and for the fresh cells the
   pooled-frontier / 1-EM-band winner in `round9_analyses.json`);

3. evaluates the predeclared LearnStop rule --- predict a learned win iff
   early correctness is above the median **and** the scalar signal is below
   the median --- under **leave-one-cell-out** cross-validation, where the two
   medians are recomputed from the other cells only, plus a leave-one-out
   logistic regression over all predictors as an exploratory ceiling, and a
   label-permutation test for both.

If the rule holds out of sample the paper gains a positive, actionable finding;
if it does not, the audit gains a sharper negative --- that no cheaply
measurable property tells you in advance where learning pays.

    python3 when_learned_wins.py [--n-boot 50]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

import run_experiment as rx
import meta_review_analyses as mra

MEAS = mra.MEAS
TABLES = mra.TABLES
GRID = [0.01, 0.02, 0.04, 0.07, 0.12, 0.20]
HIST = list(mra.HIST_PRETTY)
PREDICTORS = ["early_correct_rate", "repeat_precision", "repeat_signal_auc",
              "saturation_gap_em", "fbg_positive_rate", "oof_spearman",
              "threshold_stability", "train_em_full"]
# Predeclared LearnStop rule: high early correctness AND unreliable scalar
# signal.  Direction is fixed before any outcome is read.
RULE = {"early_correct_rate": +1, "repeat_precision": -1}


def cell_tags():
    fresh = [(tag, label) for tag, _f, label, _s in mra.FRESH
             if (MEAS / f"run_{tag}.json").exists()]
    hist = [(tag, mra.HIST_PRETTY[tag]) for tag in HIST
            if (MEAS / f"run_{tag}.json").exists()]
    return hist + fresh


def training_chains(tag):
    """Policy-training chains only (the certificate split is excluded)."""
    run = json.loads((MEAS / f"run_{tag}.json").read_text())
    cache = pickle.loads((MEAS / f"critic_data_{tag}.pkl").read_bytes())
    X = np.asarray(cache["X_val"], float)
    y_val = np.asarray(cache["y_val"], float)
    y_err = np.asarray(cache["y_err"], int)
    chains = rx.chain_indices(X)
    frac = run["config"].get("certificate_calibration_fraction") or 0.0
    if frac:
        rng = np.random.default_rng(41)
        order = np.arange(len(chains))
        rng.shuffle(order)
        ncal = max(1, int(round(frac * len(chains))))
        chains = [chains[i] for i in order[:-ncal]]
    return X, y_val, y_err, chains, run


def auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def threshold_stability(X, y_val, y_err, chains, n_boot, seed=31):
    """Share of chain-bootstrap refits that reselect the modal threshold."""
    rng = np.random.default_rng(seed)
    picks = []
    for b in range(n_boot):
        np.random.seed(seed * 100 + b)
        idx = rng.integers(0, len(chains), len(chains))
        Xb, yvb, yeb, chains_b = [], [], [], []
        for i in idx:
            ch = chains[i]
            start = len(Xb)
            Xb.extend(X[ch].tolist())
            yvb.extend(y_val[ch].tolist())
            yeb.extend(y_err[ch].tolist())
            chains_b.append(list(range(start, start + len(ch))))
        Xb = np.asarray(Xb, float)
        oof, _ = rx.crossfit_critic_scores(Xb, np.asarray(yvb, float),
                                           np.asarray(yeb, int), chains_b)
        lam, _meta = rx.select_frozen_lambda(oof, Xb, np.asarray(yeb, int),
                                             chains_b, GRID, "guarded")
        picks.append(lam)
    values, counts = np.unique(picks, return_counts=True)
    return float(counts.max() / len(picks)), {str(float(v)): int(c)
                                              for v, c in zip(values, counts)}


def cell_predictors(tag, n_boot):
    X, y_val, y_err, chains, run = training_chains(tag)
    correct = 1 - y_err
    early = [float(max(correct[ch[t]] for t in range(min(2, len(ch)))))
             for ch in chains]
    full = np.asarray([correct[ch[-1]] for ch in chains], float)
    fixed2 = np.asarray([correct[ch[min(2, len(ch) - 1)]] for ch in chains], float)
    repeat_stop, repeat_correct = [], []
    for ch in chains:
        stop = ch[-1]
        for i in ch[1:-1]:
            if X[i][4] == 0:                      # normalized answer unchanged
                stop = i
                break
        repeat_stop.append(stop)
        repeat_correct.append(correct[stop])
    repeat_em = float(np.mean(repeat_correct))
    full_em = float(full.mean())
    rows = [i for ch in chains for i in ch[1:]]   # prefixes with a predecessor
    signal = [1.0 - X[i][4] for i in rows]        # 1 == the answer just repeated
    labels = [int(correct[i]) for i in rows]
    oof, _ = rx.crossfit_critic_scores(X, y_val, y_err, chains)
    fit_rows = [i for ch in chains for i in ch]
    rho = spearmanr(oof[fit_rows], y_val[fit_rows]).statistic
    stability, freq = threshold_stability(X, y_val, y_err, chains, n_boot)
    return {
        "tag": tag,
        "n_training_episodes": len(chains),
        "early_correct_rate": round(float(np.mean(early)), 4),
        "repeat_precision": round(repeat_em / full_em, 4) if full_em else None,
        "repeat_signal_auc": round(auc(signal, labels), 4),
        "saturation_gap_em": round(100 * float(full.mean() - fixed2.mean()), 3),
        "fbg_positive_rate": round(float(np.mean(y_val[fit_rows] > 1e-9)), 4),
        "oof_spearman": round(float(rho), 4),
        "threshold_stability": round(stability, 4),
        "threshold_frequency": freq,
        "train_em_full": round(100 * full_em, 3),
    }


def cell_outcomes():
    meta = json.loads((MEAS / "meta_review_analyses.json").read_text())
    out = {}
    for c in meta["decision_matrix"]["cells"]:
        out[c["cell"]] = {
            "label": c["label"], "setting": c["setting"],
            "guarded_verdict": c["guarded_verdict"],
            "value_verdict": c["value_verdict"],
            "guarded_not_dominated": int(c["guarded_verdict"] != "L"),
            "value_not_dominated": int(c["value_verdict"] != "L"),
            "any_fbg_not_dominated": int(c["guarded_verdict"] != "L"
                                         or c["value_verdict"] != "L"),
        }
    path = MEAS / "round9_analyses.json"
    if path.exists():
        r9 = json.loads(path.read_text())
        family = {"value_only", "guarded_value", "semantic_guard"}
        for c in r9["fresh_cells"]:
            entry = out.setdefault(c["tag"], {})
            nd = set(c["pooled_pareto"]["nondominated"])
            entry["fbg_on_pooled_frontier"] = int(bool(nd & family))
            entry["fbg_wins_1em_band"] = int(
                c["pooled_pareto"]["band_winners"]["1.0"]["winner"] in family)
            entry["fbg_wins_2em_band"] = int(
                c["pooled_pareto"]["band_winners"]["2.0"]["winner"] in family)
    return out


def loo_rule(rows, outcome_key, rule=None):
    """Leave-one-cell-out evaluation of a median-split rule."""
    rule = RULE if rule is None else rule
    keys = list(rule)
    values = {k: np.asarray([r[k] for r in rows], float) for k in keys}
    y = np.asarray([r[outcome_key] for r in rows], int)
    preds = []
    for i in range(len(rows)):
        others = [j for j in range(len(rows)) if j != i]
        vote = all(
            (values[k][i] > np.median(values[k][others])) if rule[k] > 0
            else (values[k][i] < np.median(values[k][others]))
            for k in keys)
        preds.append(int(vote))
    preds = np.asarray(preds, int)
    return preds, y


def scores(preds, y):
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    acc = (tp + tn) / len(y)
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    bal = None if sens is None or spec is None else 0.5 * (sens + spec)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(acc, 4),
            "base_rate_accuracy": round(max(y.mean(), 1 - y.mean()), 4),
            "sensitivity": None if sens is None else round(sens, 4),
            "specificity": None if spec is None else round(spec, 4),
            "balanced_accuracy": None if bal is None else round(bal, 4),
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else None}


def permutation_p(preds, y, seed=7, n_perm=20_000):
    rng = np.random.default_rng(seed)
    observed = float((preds == y).mean())
    count = 0
    for _ in range(n_perm):
        if float((preds == rng.permutation(y)).mean()) >= observed:
            count += 1
    return round((count + 1) / (n_perm + 1), 5)


def loo_logistic(rows, outcome_key, predictors=PREDICTORS):
    X = np.asarray([[r[k] for k in predictors] for r in rows], float)
    y = np.asarray([r[outcome_key] for r in rows], int)
    prob = np.zeros(len(y), float)
    for i in range(len(y)):
        tr = [j for j in range(len(y)) if j != i]
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
        sd[sd == 0] = 1.0
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit((X[tr] - mu) / sd, y[tr])
        prob[i] = model.predict_proba(((X[i] - mu) / sd).reshape(1, -1))[0][
            list(model.classes_).index(1)] if 1 in model.classes_ else 0.0
    return prob, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse cached per-cell predictors")
    args = ap.parse_args()
    dest = MEAS / "when_learned_wins.json"
    cached = json.loads(dest.read_text()) if (args.reuse and dest.exists()) else None

    outcomes = cell_outcomes()
    rows = []
    for tag, label in cell_tags():
        if cached and tag in {r["tag"] for r in cached["cells"]}:
            row = next(r for r in cached["cells"] if r["tag"] == tag)
            row = {k: v for k, v in row.items() if k not in outcomes.get(tag, {})}
        else:
            print(f"[predictors] {tag}", flush=True)
            row = cell_predictors(tag, args.n_boot)
        row.update(outcomes.get(tag, {}))
        row["label"] = outcomes.get(tag, {}).get("label", label)
        rows.append(row)

    result = {
        "status": "predictors_computed_from_training_chains_only",
        "n_cells": len(rows), "n_bootstrap_refits_per_cell": args.n_boot,
        "predictors": PREDICTORS,
        "predeclared_rule": {
            "hypothesis": ("LearnStop: a learned exit helps where early "
                           "correctness is high and the free scalar signal is "
                           "unreliable"),
            "rule": "predict a learned win iff early_correct_rate > median and "
                    "repeat_precision < median (medians from the other cells)",
            "directions": RULE},
        "cells": rows,
    }

    # marginal association of every predictor with every outcome
    assoc = {}
    for outcome in ["guarded_not_dominated", "value_not_dominated",
                    "any_fbg_not_dominated"]:
        y = np.asarray([r[outcome] for r in rows], int)
        assoc[outcome] = {"n_positive": int(y.sum()), "n_cells": len(y)}
        for k in PREDICTORS:
            x = np.asarray([r[k] for r in rows], float)
            rho = spearmanr(x, y)
            assoc[outcome][k] = {
                "spearman": round(float(rho.statistic), 4),
                "p": round(float(rho.pvalue), 4),
                "mean_when_win": round(float(x[y == 1].mean()), 4) if y.sum() else None,
                "mean_when_loss": round(float(x[y == 0].mean()), 4) if (1 - y).sum() else None,
            }
    result["marginal_association"] = assoc

    # predeclared rule + logistic ceiling, leave-one-cell-out
    evals = {}
    for outcome in ["guarded_not_dominated", "value_not_dominated",
                    "any_fbg_not_dominated"]:
        preds, y = loo_rule(rows, outcome)
        entry = {"predeclared_rule": scores(preds, y)}
        entry["predeclared_rule"]["permutation_p"] = permutation_p(preds, y)
        prob, _ = loo_logistic(rows, outcome)
        hard = (prob >= 0.5).astype(int)
        entry["loo_logistic"] = scores(hard, y)
        entry["loo_logistic"]["auc"] = (round(auc(prob, y), 4)
                                        if auc(prob, y) is not None else None)
        entry["loo_logistic"]["permutation_p"] = permutation_p(hard, y)
        evals[outcome] = entry
    result["leave_one_cell_out"] = evals

    # Sensitivity: the same leave-one-out protocol under other readings of
    # "the scalar signal is unreliable", and under each predictor alone.  The
    # conjunction in RULE is LearnStop's hypothesis; these variants show how
    # much of the result depends on that particular operationalisation.
    variants = {"predeclared (early + repeat precision)": dict(RULE),
                "early + repeat-signal AUC":
                    {"early_correct_rate": +1, "repeat_signal_auc": -1},
                "early + threshold stability":
                    {"early_correct_rate": +1, "threshold_stability": -1},
                "repeat precision alone": {"repeat_precision": -1},
                "early correctness alone": {"early_correct_rate": +1},
                "FBG-positive rate alone": {"fbg_positive_rate": +1}}
    sens = {}
    for outcome in ["guarded_not_dominated", "value_not_dominated",
                    "any_fbg_not_dominated"]:
        sens[outcome] = {}
        for name, rule in variants.items():
            preds, y = loo_rule(rows, outcome, rule)
            entry = scores(preds, y)
            entry["permutation_p"] = permutation_p(preds, y)
            sens[outcome][name] = entry
    result["rule_sensitivity"] = sens

    fresh = [r for r in rows if "fbg_on_pooled_frontier" in r]
    if fresh:
        sub = {}
        for outcome in ["fbg_on_pooled_frontier", "fbg_wins_1em_band"]:
            y = np.asarray([r[outcome] for r in fresh], int)
            if 0 < y.sum() < len(y):
                preds, _ = loo_rule(fresh, outcome)
                sub[outcome] = {"n_cells": len(fresh), "n_positive": int(y.sum()),
                                "predeclared_rule": scores(preds, y)}
                sub[outcome]["predeclared_rule"]["permutation_p"] = permutation_p(preds, y)
            else:
                sub[outcome] = {"n_cells": len(fresh), "n_positive": int(y.sum()),
                                "note": "degenerate outcome; no evaluation"}
        result["fresh_cell_outcomes"] = sub

    dest.write_text(json.dumps(result, indent=1) + "\n")
    write_table(result)
    print(json.dumps({"marginal": {k: {p: v[p]["spearman"] for p in PREDICTORS}
                                   for k, v in assoc.items()},
                      "loo": evals}, indent=1))
    print(f"wrote {dest}")


def write_table(result):
    TABLES.mkdir(exist_ok=True)
    pretty = {"early_correct_rate": "early correctness",
              "repeat_precision": "scalar-signal precision",
              "repeat_signal_auc": "scalar-signal AUC",
              "saturation_gap_em": "saturation gap (EM)",
              "fbg_positive_rate": "FBG-positive rate",
              "oof_spearman": "cross-fitted $\\rho$",
              "threshold_stability": "threshold stability",
              "train_em_full": "training EM"}
    rows = []
    a = result["marginal_association"]["any_fbg_not_dominated"]
    g = result["marginal_association"]["guarded_not_dominated"]
    for k in PREDICTORS:
        rows.append(
            f"{pretty[k]} & {a[k]['mean_when_win']:.3f} & {a[k]['mean_when_loss']:.3f} & "
            f"{a[k]['spearman']:+.2f} & {a[k]['p']:.2f} & "
            f"{g[k]['spearman']:+.2f} & {g[k]['p']:.2f} \\\\")
    (TABLES / "when_wins_rows.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
