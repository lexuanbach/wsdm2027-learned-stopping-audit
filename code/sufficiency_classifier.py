"""Gold-support-supervised evidence-sufficiency classifier (Sufficient-Context
family) replayed as a stopping rule.  Offline: no LLM calls, no GPU.

This is NOT a HALT reproduction and is not a reproduction of any released
sufficiency verifier.  It is the strongest sufficiency baseline that the
measured logs can support: a classifier trained on *gold-support coverage*
(does the retrieved evidence at this prefix contain the annotated supporting
sentences?) rather than on the paper's own answer-error signal, then used to
stop.  It asks whether the guarded controller loses only
because its supervision is weak, without inventing a verifier.

Signal declaration (fairness):
  * variant ``same9``  -- exactly the nine logit-free controller features.
    Identical test-time signals to the FBG controller and the stop classifier.
  * variant ``lex12``  -- those nine plus three offline lexical
    evidence-quality features (question/evidence token recall, evidence unit
    count, best single-unit overlap).  A declared SUPERSET, computed from the
    logged retrieval text only; still no logits and no extra LLM call.
  Gold support sentences are used for TRAINING LABELS ONLY.  They are never
  read at test time by either variant.

Protocol (identical to every other policy in the paper):
  * labels and fits use only that cell's 90 policy-training episodes
    (the seed-41 split; certificate episodes untouched);
  * scores on training chains are 5-fold cross-fitted by chain (seed 3);
  * tau is chosen on TRAINING chains with the frozen band rule
    ("cheapest tau whose training EM is within 1.0 EM of the training
    full-budget EM; else argmax train EM, ties broken by fewest searches");
  * the test replay happens once, on the untouched 400 test trajectories.

    python3 sufficiency_classifier.py --out <dir> [--tags a,b,c]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np
from sklearn.linear_model import LogisticRegression

import round9_analyses as r9
import frontier_analyses as mra

MEAS = r9.MEAS
TAU_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
COVER_THRESHOLD = 0.80      # fixed a priori, never tuned
FOLDS, XSEED = 5, 3


def _tok(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def covered(gold_sentence, evidence):
    """Is one gold supporting sentence present in the retrieved evidence?"""
    g = _tok(gold_sentence)
    if not g:
        return True
    gs = set(g)
    best = 0.0
    for e in evidence:
        es = set(_tok(e))
        if not es:
            continue
        best = max(best, len(gs & es) / len(gs))
    return best >= COVER_THRESHOLD


def sufficiency_labels(trace):
    """1 at prefix t iff every gold supporting sentence is retrieved by t."""
    gold = [s for s in trace.get("support_sentences", []) if str(s).strip()]
    out = []
    for prefix in trace["trajectory"]:
        ev = prefix.get("evidence", [])
        out.append(int(bool(gold) and all(covered(s, ev) for s in gold)))
    return out


def lex_features(question, evidence):
    q = set(_tok(question))
    if not evidence:
        return [0.0, 0.0, 0.0]
    units = [set(_tok(e)) for e in evidence]
    union = set().union(*units)
    recall = len(q & union) / max(1, len(q))
    best = max(len(q & u) / max(1, len(q)) for u in units)
    return [recall, float(len(evidence)), best]


def build(cell, variant):
    """Training design matrix / labels and the test-time score matrix."""
    X_tr, y_tr, chains = [], [], []
    row = 0
    trace_by_pos = {i: cell.traces[i] for i in range(len(cell.traces))}
    for i in cell.fit_ids:
        trace = trace_by_pos[i]
        chain = cell.all_chains[i]
        labels = sufficiency_labels(trace)
        assert len(labels) == len(chain)
        idx = []
        for t, gidx in enumerate(chain):
            feats = list(cell.X[gidx])
            if variant == "lex12":
                feats += lex_features(trace["question"],
                                      trace["trajectory"][t].get("evidence", []))
            X_tr.append(feats)
            y_tr.append(labels[t])
            idx.append(row)
            row += 1
        chains.append(idx)
    # test features, prefix-aligned
    X_te = np.zeros((cell.n, cell.width, len(X_tr[0])), float)
    for i, r in enumerate(cell.records):
        for t in range(cell.width):
            feats = list(cell.feats[i, t])
            if variant == "lex12":
                ev = (r["trajectory"][t].get("evidence", [])
                      if t < len(r["trajectory"]) else [])
                feats += lex_features(r["question"], ev)
            X_te[i, t] = feats
    return np.asarray(X_tr, float), np.asarray(y_tr, int), chains, X_te


def crossfit_logit(X, y, chains, folds, seed):
    oof = np.zeros(len(y), float)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(chains))
    for f in range(folds):
        te_chains = [chains[j] for k, j in enumerate(order) if k % folds == f]
        te = [i for ch in te_chains for i in ch]
        tr = [i for i in range(len(y)) if i not in set(te)]
        if not te:
            continue
        oof[te] = _fit_predict(X[tr], y[tr], X[te], seed)
    return oof


def _fit_predict(Xtr, ytr, Xte, seed):
    if len(set(ytr.tolist())) < 2:
        return np.full(len(Xte), float(ytr.mean()))
    m = LogisticRegression(max_iter=2000, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def select_ge(scores, cell, chains, grid, band=1.0):
    """Frozen band rule for a stop-when-score>=tau policy, train chains only."""
    y_err = cell.y_err
    fit_rows = [cell.all_chains[i] for i in cell.fit_ids]
    reference = 100 * np.mean([1 - y_err[ch[-1]] for ch in fit_rows])
    frontier = []
    for tau in grid:
        ems, searches = [], []
        for local, gchain in zip(chains, fit_rows):
            stop_t = len(gchain) - 1
            for t in range(len(gchain) - 1):
                if scores[local[t]] >= tau:
                    stop_t = t
                    break
            ems.append(1 - y_err[gchain[stop_t]])
            searches.append(cell.X[gchain[stop_t]][0])
        frontier.append({"tau": tau, "em": 100 * float(np.mean(ems)),
                         "searches": float(np.mean(searches))})
    eligible = [x for x in frontier if x["em"] >= reference - band]
    if eligible:
        picked = min(eligible, key=lambda x: (x["searches"], -x["em"]))
        reason = "cheapest_within_band"
    else:
        best = max(x["em"] for x in frontier)
        picked = min((x for x in frontier if x["em"] == best),
                     key=lambda x: x["searches"])
        reason = "fallback_max_em"
    return float(picked["tau"]), {"reason": reason,
                                  "train_reference_em": round(reference, 3),
                                  "frontier": frontier}


def verdict(delta_ci):
    """Four distinct verdicts, defined once."""
    lo, hi = delta_ci
    if lo > 0:
        return "win"
    if hi < 0:
        return "loss"
    return "unresolved"


def run_cell(tag, fname, label, short):
    cell = r9.Cell(tag, fname, label, short)
    simple = cell.simple_points()
    full = cell.point(np.full(cell.n, cell.last))
    out = {"tag": tag, "label": label, "n_test": int(cell.n),
           "cover_threshold": COVER_THRESHOLD, "tau_grid": TAU_GRID,
           "variants": {}}
    for variant in ["same9", "lex12"]:
        X_tr, y_tr, chains, X_te = build(cell, variant)
        base_rate = float(y_tr.mean())
        oof = crossfit_logit(X_tr, y_tr, chains, FOLDS, XSEED)
        auc = _auc(y_tr, oof)
        tau, meta = select_ge(oof, cell, chains, TAU_GRID)
        flat = X_te.reshape(-1, X_te.shape[-1])
        scores = _fit_predict(X_tr, y_tr, flat, XSEED).reshape(cell.n, cell.width)
        stops = np.full(cell.n, cell.last, int)
        for i in range(cell.n):
            for t in range(cell.width - 1):
                if scores[i, t] >= tau:
                    stops[i] = t
                    break
        pt = cell.point(stops)
        d_full = pt["em_vec"] - full["em_vec"]
        ci_full = r9.paired_ci(d_full)
        best_simple = min(
            (k for k in simple if simple[k]["em"] >= pt["em"] - 1e-9),
            key=lambda k: simple[k]["calls"], default=None)
        dominated_by = [k for k, p in simple.items()
                        if p["em"] >= pt["em"] and p["calls"] <= pt["calls"]
                        and (p["em"] > pt["em"] or p["calls"] < pt["calls"])]
        out["variants"][variant] = {
            "n_train_prefixes": int(len(y_tr)),
            "n_policy_training_episodes": int(len(chains)),
            "label_base_rate": round(base_rate, 4),
            "crossfit_auc": None if auc is None else round(auc, 4),
            "tau": tau, "selection_reason": meta["reason"],
            "train_reference_em": meta["train_reference_em"],
            "train_frontier": meta["frontier"],
            "test_em": round(pt["em"], 3), "test_calls": round(pt["calls"], 3),
            "test_tokens": round(pt["tokens"], 1),
            "test_latency_seconds": round(pt["latency"], 3),
            "mean_stop_depth": round(float(stops.mean()), 3),
            "delta_em_vs_full_budget": ci_full,
            "verdict_vs_full_budget": verdict(ci_full["ci95"]),
            "dominated_by_simple_policies": dominated_by,
            "cheapest_simple_policy_at_or_above_its_em": best_simple,
            "full_budget_em": round(full["em"], 3),
            "full_budget_calls": round(full["calls"], 3),
            "guarded_fbg_em": round(cell.point(cell.g_stop)["em"], 3),
            "guarded_fbg_calls": round(cell.point(cell.g_stop)["calls"], 3),
        }
        # paired comparison against the paper's own guarded controller
        g = cell.point(cell.g_stop)
        ci_g = r9.paired_ci(pt["em_vec"] - g["em_vec"])
        out["variants"][variant]["delta_em_vs_guarded_fbg"] = ci_g
        out["variants"][variant]["verdict_vs_guarded_fbg"] = verdict(ci_g["ci95"])
        # the paper's strongest control: a randomized fixed-budget mixture at
        # MATCHED expected cost, on each of the three measured cost axes.
        equal = {}
        for coord, key in [("calls", "calls"), ("input_tokens", "tokens"),
                           ("wall_seconds", "latency")]:
            pts = {k: {"cost": v[key], "em": v["em"], "em_vec": v["em_vec"]}
                   for k, v in simple.items()}
            hull = r9.upper_hull(pts)
            mix, n1, n2, w = r9.hull_mixture(hull, pt[key], pts)
            pc = r9.paired_ci(pt["em_vec"] - mix)
            equal[coord] = {
                "learned_cost": round(pt[key], 3),
                "mixture": {"of": [n1, n2], "weight_first": round(w, 4),
                            "em": round(100 * float(mix.mean()), 3)},
                "delta_em_vs_mixture_at_equal_cost": pc,
                "verdict": verdict(pc["ci95"]),
                "pareto_improvement_at_equal_cost": bool(pc["ci95"][0] > 0),
            }
        out["variants"][variant]["equal_cost"] = equal
    out["simple_family"] = {k: {"em": round(v["em"], 3),
                                "calls": round(v["calls"], 3)}
                            for k, v in simple.items()}
    return out


def _auc(y, s):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


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
        print(f"[suffclf] {tag}", flush=True)
        results.append(run_cell(tag, fname, label, short))
        v = results[-1]["variants"]["same9"]
        print(f"   same9 tau={v['tau']} EM={v['test_em']} calls={v['test_calls']} "
              f"(full {v['full_budget_em']}@{v['full_budget_calls']}, "
              f"guarded {v['guarded_fbg_em']}@{v['guarded_fbg_calls']}) "
              f"AUC={v['crossfit_auc']} base={v['label_base_rate']}", flush=True)
        v = results[-1]["variants"]["lex12"]
        print(f"   lex12 tau={v['tau']} EM={v['test_em']} calls={v['test_calls']} "
              f"AUC={v['crossfit_auc']}", flush=True)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "what": "gold-support-supervised evidence-sufficiency classifier "
                "(Sufficient-Context family) replayed as a stopping rule",
        "not_a_reproduction": "This is NOT HALT and NOT a reproduction of any "
                              "released sufficiency verifier; no released "
                              "verifier weights were run.",
        "supervision": "gold support-sentence coverage at each retrieval "
                       "prefix, threshold 0.80 token recall, fixed a priori",
        "test_time_signals": {"same9": "the nine logit-free controller features",
                              "lex12": "those nine plus three offline lexical "
                                       "evidence features; gold support is "
                                       "never read at test time"},
        "selection": "tau on that cell's own training chains only, frozen band "
                     "rule, 5-fold cross-fitting by chain (seed 3)",
        "cells": results,
    }
    (outdir / "sufficiency_classifier.json").write_text(
        json.dumps(payload, indent=1))
    print("wrote", outdir / "sufficiency_classifier.json")


if __name__ == "__main__":
    main()
