"""Analyze the predeclared corpus-wide guarded-policy confirmation.

The confirmatory success criterion was frozen before test evaluation:
paired-bootstrap 95% lower bound for (guarded - full-budget) EM must exceed
-3.0 points, and guarded stopping must reduce mean searches/LLM calls.
All other policy comparisons are descriptive and share one Pareto table.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import re

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

import run_experiment as rx


MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
ART = pathlib.Path(__file__).resolve().parents[1]
MANU = ART / "manuscript"
if not MANU.exists():
    MANU = pathlib.Path(__file__).resolve().parents[2] / "manuscript"
TABLES = MANU / "tables"
TAG = "corpus_guarded_qwen3b"
MARGIN_EM = 3.0
TARGETED_GRID = [0.01, 0.02, 0.04, 0.07, 0.12, 0.20]
TARGETED_TRAIN_BAND_EM = 1.0
N_BOOT = 20_000


def norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def bootstrap_difference(a, b, seed=101):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(d), (N_BOOT, len(d)))
    means = 100 * d[ix].mean(axis=1)
    return {
        "mean_em_points": round(100 * float(d.mean()), 3),
        "ci95": [round(float(np.percentile(means, 2.5)), 3),
                 round(float(np.percentile(means, 97.5)), 3)],
        "lower95_one_sided": round(float(np.percentile(means, 5)), 3),
    }


def cumulative_cost(record, stop):
    trajectory = record["trajectory"][: stop + 1]
    usage = {"in": 0.0, "out": 0.0, "calls": 0.0,
             "latency_seconds": 0.0, "retrieval_latency_seconds": 0.0}
    for prefix in trajectory:
        for part in ["query_usage", "answer_usage"]:
            for key, value in prefix.get(part, {}).items():
                usage[key] = usage.get(key, 0.0) + float(value)
        usage["retrieval_latency_seconds"] += float(
            prefix.get("retrieval_latency_seconds", 0.0))
    return usage


def gold_coverage_stop(record, mode):
    supports = (record["support_titles"] if mode == "titles"
                else record["support_sentences"])
    supports = [norm(x) for x in supports if norm(x)]
    if not supports:
        return len(record["trajectory"]) - 1
    for t, prefix in enumerate(record["trajectory"]):
        evidence = norm(" ".join(prefix["evidence"]))
        if all(item in evidence for item in supports):
            return t
    return len(record["trajectory"]) - 1


def policy_stop(record, name):
    last = len(record["prefix_em"]) - 1
    if name == "full_budget":
        return last
    if name == "guarded_value":
        return int(record["primary"]["searches"])
    if name == "value_only":
        return int(record["frozen_value"]["searches"])
    if name == "answer_repeat":
        return int(record.get("answer_repeat_searches",
                              record.get("ent_searches")))
    if name == "repeat_conditioned_gain":
        return int(record["repeat_conditioned_gain_stop"])
    if name == "tasr":
        return int(record["tasr_stop"])
    if name == "semantic_guard":
        return int(record["semantic_guard_stop"])
    if name.startswith("fixed_"):
        return min(int(name.split("_")[1]), last)
    if name == "gold_title_coverage_diagnostic":
        return gold_coverage_stop(record, "titles")
    if name == "gold_sentence_coverage_diagnostic":
        return gold_coverage_stop(record, "sentences")
    raise ValueError(name)


def summarize(records, name):
    stops = [policy_stop(r, name) for r in records]
    if name == "tasr":
        evaluations = [r["tasr_evaluation"] for r in records]
        em = np.asarray([row["em"] for row in evaluations], float)
        f1 = np.asarray([row["f1"] for row in evaluations], float)
        costs = [{
            "calls": row["deployment_cost"]["calls"],
            "in": row["deployment_cost"]["input_tokens"],
            "out": row["deployment_cost"]["output_tokens"],
            "latency_seconds": row["deployment_cost"]["llm_latency_seconds"],
            "retrieval_latency_seconds": row["deployment_cost"][
                "retrieval_latency_seconds"],
        } for row in evaluations]
    else:
        em = np.asarray([r["prefix_em"][t]
                         for r, t in zip(records, stops)], float)
        f1 = np.asarray([r["prefix_f1"][t]
                         for r, t in zip(records, stops)], float)
        costs = [cumulative_cost(r, t) for r, t in zip(records, stops)]
    searches = np.asarray(stops, float)
    return {
        "em": round(100 * float(em.mean()), 2),
        "f1": round(100 * float(f1.mean()), 2),
        "searches": round(float(searches.mean()), 3),
        "llm_calls": round(float(np.mean([c["calls"] for c in costs])), 3),
        "input_tokens": round(float(np.mean([c["in"] for c in costs])), 1),
        "output_tokens": round(float(np.mean([c["out"] for c in costs])), 1),
        "llm_latency_seconds": round(float(np.mean(
            [c["latency_seconds"] for c in costs])), 3),
        "retrieval_latency_seconds": round(float(np.mean(
            [c["retrieval_latency_seconds"] for c in costs])), 4),
        "stopped_early_rate": round(float(np.mean([
            t < len(r["prefix_em"]) - 1 for r, t in zip(records, stops)])), 3),
        "_em": em, "_stops": stops,
    }


def pareto(methods):
    for name, row in methods.items():
        row["pareto_dominated_by"] = sorted([
            other for other, q in methods.items()
            if other != name and q["em"] >= row["em"]
            and q["llm_calls"] <= row["llm_calls"]
            and (q["em"] > row["em"] or q["llm_calls"] < row["llm_calls"])
        ])


def score_reliability(records):
    prediction, target = [], []
    for record in records:
        f1s = record["prefix_f1"]
        for t, score in enumerate(record["prefix_value"]):
            future = f1s[t + 1:]
            y = max(max(future) - f1s[t], 0.0) if future else 0.0
            prediction.append(score)
            target.append(y)
    prediction, target = np.asarray(prediction), np.asarray(target)
    groups = np.array_split(np.argsort(prediction, kind="stable"), 10)
    return {
        "n_prefixes": len(target),
        "mae": round(float(np.mean(np.abs(prediction - target))), 4),
        "rmse": round(float(np.sqrt(np.mean((prediction - target) ** 2))), 4),
        "spearman": round(float(spearmanr(prediction, target).statistic), 4),
        "underprediction_rate": round(float(np.mean(prediction < target)), 4),
        "mean_signed_error_prediction_minus_target": round(
            float(np.mean(prediction - target)), 4),
        "deciles": [{
            "n": len(g), "predicted": round(float(prediction[g].mean()), 4),
            "realized": round(float(target[g].mean()), 4),
            "positive_target_rate": round(float(np.mean(target[g] > 0)), 4),
        } for g in groups],
    }


def certificate_curve(run, cache, records):
    X = np.asarray(cache["X_val"], float)
    y_val = np.asarray(cache["y_val"], float)
    y_err = np.asarray(cache["y_err"], int)
    all_chains = rx.chain_indices(X)
    rng = np.random.default_rng(41)
    order = np.arange(len(all_chains))
    rng.shuffle(order)
    ncal = max(1, int(round(run["config"]["certificate_calibration_fraction"]
                            * len(all_chains))))
    cal_ids = set(order[-ncal:].tolist())
    fit_chains = [ch for i, ch in enumerate(all_chains) if i not in cal_ids]
    cal_chains = [ch for i, ch in enumerate(all_chains) if i in cal_ids]
    fit_rows = [i for ch in fit_chains for i in ch]
    critic = rx.VoICritic()
    critic.fit(X[fit_rows], y_err[fit_rows], X[fit_rows], y_val[fit_rows])
    train_value = np.asarray([critic.value(x) for x in X])
    train_error = np.asarray([critic.err_prob(x) for x in X])
    cal_stops = [rx.training_stop(ch, train_value, X, run["lam_main"],
                                  run["config"]["primary_rule"])
                 for ch in cal_chains]
    cal_hat = [train_error[i] for i in cal_stops]
    cal_err = [y_err[i] for i in cal_stops]
    test_stops = [policy_stop(r, "guarded_value") for r in records]
    test_hat = [critic.err_prob(r["prefix_features"][t])
                for r, t in zip(records, test_stops)]
    test_err = [1 - r["prefix_em"][t]
                for r, t in zip(records, test_stops)]
    curve = []
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        threshold = rx.calibrate_threshold(cal_hat, cal_err, alpha, 0.1)
        if threshold is None:
            curve.append({"alpha": alpha, "fired": False, "n_cal": len(cal_hat)})
            continue
        selected = [e for h, e in zip(test_hat, test_err) if h <= threshold]
        curve.append({
            "alpha": alpha, "fired": True,
            "threshold": round(float(threshold), 5), "n_cal": len(cal_hat),
            "test_coverage": round(len(selected) / len(test_err), 4),
            "test_risk": round(float(np.mean(selected)), 4) if selected else None,
            "n_test_selected": len(selected),
        })
    return curve


def fit_repeat_conditioned_policy(cache, run, records, tag):
    """Train-only selection for the exploratory repeat-conditioned repair.

    The regressor is fitted only on nonterminal states where the normalized
    answer just repeated. At deployment it stops at the first such state whose
    predicted future-best gain is below theta; otherwise it runs to budget.
    """
    X = np.asarray(cache["X_val"], float)
    y_val = np.asarray(cache["y_val"], float)
    y_err = np.asarray(cache["y_err"], int)
    all_chains = rx.chain_indices(X)
    rng = np.random.default_rng(41)
    order = np.arange(len(all_chains))
    rng.shuffle(order)
    ncal = max(1, int(round(run["config"]["certificate_calibration_fraction"]
                            * len(all_chains))))
    fit_chains = [all_chains[i] for i in order[:-ncal]]
    fold_rng = np.random.default_rng(73)
    fold_order = np.arange(len(fit_chains))
    fold_rng.shuffle(fold_order)
    oof = np.full(len(X), np.nan)
    for fold in range(5):
        held = set(fold_order[fold::5].tolist())
        train_rows = [i for c, ch in enumerate(fit_chains) if c not in held
                      for i in ch[:-1] if i > ch[0] and X[i, 4] == 0]
        test_rows = [i for c in held for i in fit_chains[c][1:-1]
                     if X[i, 4] == 0]
        if not train_rows or not test_rows:
            continue
        model = GradientBoostingRegressor(max_depth=2, n_estimators=80,
                                          random_state=73 + fold)
        model.fit(X[train_rows], y_val[train_rows])
        oof[test_rows] = model.predict(X[test_rows])

    def replay_train(theta):
        stops = []
        for ch in fit_chains:
            stop = ch[-1]
            for i in ch[1:-1]:
                if X[i, 4] == 0 and np.isfinite(oof[i]) and oof[i] < theta:
                    stop = i
                    break
            stops.append(stop)
        return (100 * float(np.mean([1 - y_err[i] for i in stops])),
                float(np.mean([X[i, 0] for i in stops])))

    ref = 100 * float(np.mean([1 - y_err[ch[-1]] for ch in fit_chains]))
    frontier = []
    for theta in TARGETED_GRID:
        em, searches = replay_train(theta)
        frontier.append({"theta": theta, "em": em, "searches": searches})
    eligible = [x for x in frontier if x["em"] >= ref - TARGETED_TRAIN_BAND_EM]
    if eligible:
        pick = min(eligible, key=lambda x: (x["searches"], -x["em"]))
        reason = "cheapest_within_1em_training_band"
    else:
        best = max(x["em"] for x in frontier)
        pick = min((x for x in frontier if x["em"] == best),
                   key=lambda x: x["searches"])
        reason = "fallback_max_training_em"

    fit_rows = [i for ch in fit_chains for i in ch[:-1]
                if i > ch[0] and X[i, 4] == 0]
    model = GradientBoostingRegressor(max_depth=2, n_estimators=80,
                                      random_state=73)
    model.fit(X[fit_rows], y_val[fit_rows])
    for record in records:
        stop = len(record["prefix_em"]) - 1
        for t in range(1, stop):
            feats = np.asarray(record["prefix_features"][t], float)
            if feats[4] == 0 and model.predict(feats.reshape(1, -1))[0] < pick["theta"]:
                stop = t
                break
        record["repeat_conditioned_gain_stop"] = int(stop)
    return {
        "evaluation_status": ("exploratory_family_defined_after_partial_seed7_look"
                              if tag == TAG else
                              "predeclared_before_this_fresh_seed_run"),
        "confirmatory_use": "policy frozen for independent seed11/seed19 runs",
        "fit_seed": 73, "features": "all_nine_prefix_features",
        "training_reference_em": ref, "training_band_em": TARGETED_TRAIN_BAND_EM,
        "selected_theta": pick["theta"], "selection_reason": reason,
        "training_frontier": frontier, "n_repeat_training_prefixes": len(fit_rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=TAG)
    args = parser.parse_args()
    tag = args.tag
    run_path = MEAS / f"run_{tag}.json"
    cache_path = MEAS / f"critic_data_{tag}.pkl"
    run = json.loads(run_path.read_text())
    cache = pickle.loads(cache_path.read_bytes())
    records = run["per_q"]
    tasr_path = MEAS / f"tasr_{tag}.json"
    if tasr_path.exists():
        tasr = json.loads(tasr_path.read_text())
        tasr_stops = {row["id"]: row["stop"] for row in tasr["per_q"]}
        tasr_rows = {row["id"]: row for row in tasr["per_q"]}
        expected_ids = {str(row["id"]) for row in records}
        if tasr.get("tag") != tag or set(tasr_stops) != expected_ids:
            raise ValueError("TASR result tag/question IDs do not match run")
        for record in records:
            record["tasr_stop"] = int(tasr_stops[str(record["id"])])
            record["tasr_evaluation"] = tasr_rows[str(record["id"])]
    semantic_path = MEAS / f"semantic_guard_{tag}.json"
    if semantic_path.exists():
        semantic = json.loads(semantic_path.read_text())
        semantic_stops = {row["id"]: row["stop"] for row in semantic["per_q"]}
        expected_ids = {str(row["id"]) for row in records}
        if semantic.get("tag") != tag or set(semantic_stops) != expected_ids:
            raise ValueError("semantic-guard tag/question IDs do not match run")
        for record in records:
            record["semantic_guard_stop"] = int(semantic_stops[str(record["id"])])
    targeted_meta = fit_repeat_conditioned_policy(cache, run, records, tag)
    X_train = np.asarray(cache["X_val"], float)
    y_train_err = np.asarray(cache["y_err"], int)
    all_chains = rx.chain_indices(X_train)
    rng = np.random.default_rng(41)
    order = np.arange(len(all_chains))
    rng.shuffle(order)
    ncal = max(1, int(round(run["config"]["certificate_calibration_fraction"]
                            * len(all_chains))))
    fit_ids = order[:-ncal]
    fit_chains = [all_chains[i] for i in fit_ids]
    budget_em = []
    for budget in range(run["config"]["max_steps"] + 1):
        budget_em.append(float(np.mean([
            1 - y_train_err[ch[min(budget, len(ch) - 1)]]
            for ch in fit_chains])))
    bstar = int(np.argmax(budget_em))
    names = ["full_budget", "guarded_value", "value_only", "answer_repeat",
             "repeat_conditioned_gain",
             "fixed_0", "fixed_1", "fixed_2", "fixed_3", "fixed_4",
             "gold_title_coverage_diagnostic",
             "gold_sentence_coverage_diagnostic"]
    if tasr_path.exists():
        names.insert(4, "tasr")
    if semantic_path.exists():
        names.insert(4, "semantic_guard")
    methods = {name: summarize(records, name) for name in names}
    pareto(methods)
    primary = bootstrap_difference(methods["guarded_value"]["_em"],
                                   methods["full_budget"]["_em"])
    primary["predeclared_margin_em_points"] = MARGIN_EM
    primary["passes_noninferiority"] = (
        primary["lower95_one_sided"] > -MARGIN_EM)
    primary["reduces_calls"] = (methods["guarded_value"]["llm_calls"]
                                < methods["full_budget"]["llm_calls"])
    primary["confirmation_success"] = bool(
        primary["passes_noninferiority"] and primary["reduces_calls"])
    targeted = bootstrap_difference(methods["repeat_conditioned_gain"]["_em"],
                                    methods["full_budget"]["_em"], seed=103)
    targeted["predeclared_margin_em_points"] = MARGIN_EM
    targeted["passes_noninferiority"] = targeted["lower95_one_sided"] > -MARGIN_EM
    targeted["reduces_calls"] = (methods["repeat_conditioned_gain"]["llm_calls"]
                                 < methods["full_budget"]["llm_calls"])
    targeted["confirmation_success"] = bool(
        targeted["passes_noninferiority"] and targeted["reduces_calls"])
    targeted["status"] = ("exploratory_on_seed7"
                          if tag == TAG else "predeclared_fresh_seed_confirmation")
    comparisons = {}
    comparison_pairs = [
        ("guarded_value", "value_only"),
        ("guarded_value", "answer_repeat"),
        ("value_only", f"fixed_{bstar}"),
        ("value_only", "fixed_2"),
        ("value_only", "answer_repeat"),
        ("repeat_conditioned_gain", "answer_repeat"),
    ]
    if "tasr" in methods:
        comparison_pairs += [("value_only", "tasr"), ("tasr", "answer_repeat")]
    if "semantic_guard" in methods:
        comparison_pairs += [("semantic_guard", "value_only"),
                             ("semantic_guard", "answer_repeat")]
    for left, right in comparison_pairs:
        comparisons[f"{left}_minus_{right}"] = bootstrap_difference(
            methods[left]["_em"], methods[right]["_em"], seed=107)
    full_em = methods["full_budget"]["_em"].copy()
    for name, row in methods.items():
        if name != "full_budget":
            row["paired_vs_full"] = bootstrap_difference(
                row["_em"], full_em)
        row.pop("_em")
        row.pop("_stops")

    result = {
        "status": "confirmatory_test_evaluated_once_after_training_selection",
        "predeclared_primary": primary,
        "repeat_conditioned_confirmation": targeted,
        "selection": run["selection"],
        "repeat_conditioned_gain": targeted_meta,
        "paired_policy_comparisons": comparisons,
        "training_selected_fixed_budget": bstar,
        "training_fixed_budget_em": budget_em,
        "input_sha256": run["input_sha256"],
        "retrieval_corpus": run.get("retrieval_corpus"),
        "model": run["config"]["model"],
        "model_metadata": run.get("model_metadata"),
        "llm_seed": run["config"].get("llm_seed", 7),
        "llm_temperature": run["config"].get("llm_temperature", 0.0),
        "data_seed": run["config"].get("data_seed", 13),
        "n_test": len(records), "methods": methods,
        "heldout_test_fbg_reliability": score_reliability(records),
        "dedicated_calibration_sgr": certificate_curve(run, cache, records),
        "cost_scope": (
            "Per-prefix live inference only. Training/logging amortization and "
            "the extra diagnostic RAG answer are excluded. Local-model monetary "
            "cost is not assigned; calls, tokens, and wall latency are measured."),
        "halt_scope": (
            "Gold coverage rows are diagnostic ceilings, not HALT. Matched "
            "generated-claim HALT requires its supervised verifier and is "
            "reported separately only if run on these exact trajectories."),
    }
    if tasr_path.exists():
        result["tasr_metadata"] = {
            key: tasr[key] for key in ["method", "rule", "threshold",
                                       "selection", "replay_answer_match_rate",
                                       "test_replay_answer_match_rate",
                                       "isotonic_calibration_episodes",
                                       "excluded_certificate_episodes",
                                       "split_seed",
                                       "deployment_cost_note"]
        }
    if semantic_path.exists():
        result["semantic_guard_metadata"] = {
            key: semantic[key] for key in ["method", "status", "nli_model",
                                           "nli_revision", "threshold"]
        }
    output = (MEAS / "corpus_confirmation.json" if tag == TAG
              else MEAS / f"corpus_confirmation_{tag}.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    TABLES.mkdir(exist_ok=True)
    rows = []
    labels = {
        "full_budget": "Full budget", f"fixed_{bstar}": f"$B^*{{=}}{bstar}$",
        "fixed_2": "Fixed-2",
        "answer_repeat": "Answer repetition",
        "repeat_conditioned_gain": "Repeat-conditioned gain",
        "tasr": "TASR (full logit rule)",
        "semantic_guard": "FBG + NLI + repeat guard",
        "value_only": "FBG value only", "guarded_value": "FBG + repeat guard",
        "gold_title_coverage_diagnostic": "Gold-title coverage (diag.)",
    }
    display = ["full_budget"]
    if bstar != 2:
        display.append("fixed_2")
    display += [f"fixed_{bstar}", "answer_repeat"]
    if "tasr" in methods:
        display.append("tasr")
    if "semantic_guard" in methods:
        display.append("semantic_guard")
    display += [
                 "repeat_conditioned_gain", "value_only", "guarded_value"]
    for name in display:
        row = methods[name]
        mark = "yes" if not row["pareto_dominated_by"] else "no"
        wall = row["llm_latency_seconds"] + row["retrieval_latency_seconds"]
        rows.append(f"{labels[name]} & {row['em']:.1f} & {row['searches']:.2f} & "
                    f"{row['llm_calls']:.2f} & {row['input_tokens']:.0f} & "
                    f"{wall:.1f} & {mark} \\\\")
    if tag == TAG:
        (TABLES / "corpus_confirmation_rows.tex").write_text("\n".join(rows) + "\n")
    print(json.dumps({"primary": primary, "methods": {
        k: {q: v for q, v in row.items() if q in {"em", "searches", "llm_calls", "pareto_dominated_by"}}
        for k, row in methods.items()}}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
