"""Frontier analyses computed from the frozen measured runs (no new LLM calls).

Three analyses:

1. ``nested_refit_select``: for every fresh prefix-instrumented run, bootstrap
   the 90 policy-training episodes, re-run the *entire* frozen pipeline on the
   resample (5-fold cross-fitted scores -> guarded-rule threshold selection ->
   fixed-budget selection -> critic refit), then evaluate the resulting frozen
   policies on the untouched test questions.  This propagates critic-fitting
   and operating-point-selection uncertainty into the reported test contrasts.
   A second variant additionally resamples the paired test questions so that
   training, selection and test-sampling uncertainty are combined.

2. ``hierarchical_variance``: the four corpus-index runs form two question
   samples (data seeds 13 and 17) and, within data seed 13, three stochastic
   trajectory runs (LLM seeds 7/11/19).  We decompose the per-question paired
   contrasts into question and within-question (run) variance components, pool
   all runs with a question-clustered bootstrap, and report paired contrasts
   against the strongest simple policies (fixed-2 / fixed-3 / answer repetition)
   instead of only against the full budget.

3. ``decision_matrix``: a compact win / tie / loss / refusal matrix over every
   reported cell (nine HotpotQA backbones, 2Wiki, two MuSiQue cells, four
   corpus-index runs, and the Wikipedia hybrid).  The learned policy is judged
   against the *simple family* {fixed-0..fixed-B, answer repetition}: L if some
   simple policy Pareto-dominates it (EM >= and calls <=, one strict); W if it
   is undominated and strictly dominates the training-selected fixed budget
   B*; T otherwise.  The certificate column records whether corrected SGR
   fired at any target.

Outputs: results/measured/frontier_analyses.json and the LaTeX fragments
outputs/tables/{decision_matrix_rows,nested_uncertainty_rows,
simple_policy_paired_rows}.tex.  Nothing here selects a seed, threshold, or
policy by test outcome; every replicate and every cell is reported.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import numpy as np
from display_numbers import display, percent

import run_experiment as rx
from voi_controller import guarded_stop_index, stop_index

MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
ART = pathlib.Path(__file__).resolve().parents[1]
MANU = ART / "outputs"  # generated tables/ and figures/ (not tracked)
(MANU / "tables").mkdir(parents=True, exist_ok=True)
(MANU / "figures").mkdir(parents=True, exist_ok=True)
TABLES = MANU / "tables"

GRID = [0.01, 0.02, 0.04, 0.07, 0.12, 0.20]
MARGIN_EM = 3.0
FRESH = [
    ("corpus_guarded_qwen3b", "corpus_confirmation.json",
     "Corpus (seed 7, q-sample A)", "corpus_s7"),
    ("corpus_guarded_qwen3b_seed11",
     "corpus_confirmation_corpus_guarded_qwen3b_seed11.json",
     "Corpus (seed 11, q-sample A)", "corpus_s11"),
    ("corpus_guarded_qwen3b_seed19",
     "corpus_confirmation_corpus_guarded_qwen3b_seed19.json",
     "Corpus (seed 19, q-sample A)", "corpus_s19"),
    ("corpus_guarded_qwen3b_data17",
     "corpus_confirmation_corpus_guarded_qwen3b_data17.json",
     "Corpus (seed 7, q-sample B)", "corpus_qB"),
    ("corpus_guarded_qwen3b_data23",
     "corpus_confirmation_corpus_guarded_qwen3b_data23.json",
     "Corpus (seed 7, q-sample C)", "corpus_qC"),
    ("corpus_guarded_qwen3b_data29",
     "corpus_confirmation_corpus_guarded_qwen3b_data29.json",
     "Corpus (seed 7, q-sample D)", "corpus_qD"),
    ("corpus_guarded_qwen3b_data31",
     "corpus_confirmation_corpus_guarded_qwen3b_data31.json",
     "Corpus (seed 7, q-sample E)", "corpus_qE"),
    ("corpus2wiki_guarded_qwen3b_data17",
     "corpus_confirmation_corpus2wiki_guarded_qwen3b_data17.json",
     "2Wiki corpus (seed 7, draw 1)", "w2_d1"),
    ("corpus2wiki_guarded_qwen3b_data23",
     "corpus_confirmation_corpus2wiki_guarded_qwen3b_data23.json",
     "2Wiki corpus (seed 7, draw 2)", "w2_d2"),
    ("corpus2wiki_guarded_qwen3b_data29",
     "corpus_confirmation_corpus2wiki_guarded_qwen3b_data29.json",
     "2Wiki corpus (seed 7, draw 3)", "w2_d3"),
    ("corpusmusique_guarded_qwen3b_data17",
     "corpus_confirmation_corpusmusique_guarded_qwen3b_data17.json",
     "MuSiQue corpus (seed 7, draw 1)", "mq_d1"),
    ("corpusmusique_guarded_qwen3b_data23",
     "corpus_confirmation_corpusmusique_guarded_qwen3b_data23.json",
     "MuSiQue corpus (seed 7, draw 2)", "mq_d2"),
    ("corpusmusique_guarded_qwen3b_data29",
     "corpus_confirmation_corpusmusique_guarded_qwen3b_data29.json",
     "MuSiQue corpus (seed 7, draw 3)", "mq_d3"),
    ("wiki25m_guarded_qwen3b",
     "corpus_confirmation_wiki25m_guarded_qwen3b.json",
     "Wikipedia 25M hybrid", "wiki25m"),
]
HIST_PRETTY = {
    "final_haiku": "HotpotQA / Haiku 4.5",
    "final_sonnet45": "HotpotQA / Sonnet 4.5",
    "final_gptoss120b": "HotpotQA / GPT-oss-120B",
    "final_novalite": "HotpotQA / Nova Lite",
    "final_qwen3b": "HotpotQA / Qwen2.5-3B",
    "final_llama3b": "HotpotQA / Llama-3.2-3B",
    "final_commandr35b": "HotpotQA / Command-R-35B",
    "final_kimik25": "HotpotQA / Kimi K2.5",
    "final_minimaxm25": "HotpotQA / MiniMax M2.5",
    "final2wiki_qwen3b": "2Wiki / Qwen2.5-3B",
    "rev2_musique_qwen3b": "MuSiQue / Qwen2.5-3B",
    "rev2_musique_llama3b": "MuSiQue / Llama-3.2-3B",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_fresh(tag):
    run = json.loads((MEAS / f"run_{tag}.json").read_text())
    cache = pickle.loads((MEAS / f"critic_data_{tag}.pkl").read_bytes())
    return run, cache


def fit_chain_split(run, cache):
    """Reproduce the 90/30 policy-training / certificate split (seed 41)."""
    X = np.asarray(cache["X_val"], float)
    all_chains = rx.chain_indices(X)
    rng = np.random.default_rng(41)
    order = np.arange(len(all_chains))
    rng.shuffle(order)
    ncal = max(1, int(round(run["config"]["certificate_calibration_fraction"]
                            * len(all_chains))))
    fit_chains = [all_chains[i] for i in order[:-ncal]]
    return X, fit_chains


def bstar_from_chains(chains, y_err, max_steps):
    budget_em = [float(np.mean([1 - y_err[ch[min(b, len(ch) - 1)]]
                                for ch in chains]))
                 for b in range(max_steps + 1)]
    return int(np.argmax(budget_em))


def test_arrays(records):
    """Per-question arrays of prefix EM, answers, and features."""
    em = np.asarray([r["prefix_em"] for r in records], float)      # n x (B+1)
    feats = np.asarray([r["prefix_features"] for r in records], float)
    answers = [[p["answer"] for p in r["trajectory"]] for r in records]
    rep_stop = np.asarray([int(r.get("answer_repeat_searches"))
                           for r in records], int)
    return em, feats, answers, rep_stop


def pareto_dominated(em, calls, others):
    """True if any (em_o, calls_o) in others dominates (em, calls)."""
    for em_o, calls_o in others:
        if em_o >= em - 1e-12 and calls_o <= calls + 1e-12 and (
                em_o > em + 1e-12 or calls_o < calls - 1e-12):
            return True
    return False


def strictly_dominates(em, calls, em_o, calls_o):
    return em >= em_o - 1e-12 and calls <= calls_o + 1e-12 and (
        em > em_o + 1e-12 or calls < calls_o - 1e-12)


def pct(x, q):
    return round(float(np.percentile(x, q)), 3)


# --------------------------------------------------------------------------- #
# 1. nested refit / reselect bootstrap
# --------------------------------------------------------------------------- #
def nested_refit_select(tag, n_boot, seed):
    run, cache = load_fresh(tag)
    records = run["per_q"]
    X, fit_chains = fit_chain_split(run, cache)
    y_val = np.asarray(cache["y_val"], float)
    y_err = np.asarray(cache["y_err"], int)
    max_steps = run["config"]["max_steps"]
    em, feats, answers, rep_stop = test_arrays(records)
    n, width = em.shape
    last = width - 1
    full_em = em[:, last]
    fixed_em = {k: em[:, min(k, last)] for k in range(max_steps + 1)}
    fixed_calls = {k: 1 + 2 * min(k, last) for k in range(max_steps + 1)}
    rep_em = em[np.arange(n), rep_stop]
    rep_calls = float(np.mean(1 + 2 * rep_stop))

    # provenance: the logged primary stop is the guarded rule on logged scores
    logged_match = np.mean([
        guarded_stop_index(r["prefix_value"], a, run["lam_main"])
        == int(r["primary"]["searches"])
        for r, a in zip(records, answers)])
    # provenance: refitting on the original 90 chains reproduces logged scores.
    # VoICritic's gradient-boosted head draws its split-tie permutation from
    # numpy's global RNG, so seed it before every fit for bit-reproducibility.
    np.random.seed(seed)
    fit_rows = [i for ch in fit_chains for i in ch]
    critic0 = rx.VoICritic()
    critic0.fit(X[fit_rows], y_err[fit_rows], X[fit_rows], y_val[fit_rows])
    refit_scores = critic0.val.predict(feats.reshape(-1, feats.shape[-1]))
    logged_scores = np.asarray([r["prefix_value"] for r in records],
                               float).reshape(-1)
    refit_max_abs_dev = float(np.max(np.abs(refit_scores - logged_scores)))
    refit_stop_agreement = float(np.mean([
        guarded_stop_index(s_, a, run["lam_main"]) == int(r["primary"]["searches"])
        for s_, a, r in zip(refit_scores.reshape(n, width), answers, records)]))

    rng = np.random.default_rng(seed)
    rows = []
    flat_feats = feats.reshape(-1, feats.shape[-1])
    for b in range(n_boot):
        np.random.seed(seed * 10_000 + b)
        idx = rng.integers(0, len(fit_chains), len(fit_chains))
        # build a re-indexed resample so duplicated chains are distinct rows
        Xb, yvb, yeb, chains_b = [], [], [], []
        for i in idx:
            ch = fit_chains[i]
            start = len(Xb)
            Xb.extend(X[ch].tolist())
            yvb.extend(y_val[ch].tolist())
            yeb.extend(y_err[ch].tolist())
            chains_b.append(list(range(start, start + len(ch))))
        Xb = np.asarray(Xb, float)
        yvb = np.asarray(yvb, float)
        yeb = np.asarray(yeb, int)
        oof_value, _ = rx.crossfit_critic_scores(Xb, yvb, yeb, chains_b)
        lam_b, _meta = rx.select_frozen_lambda(oof_value, Xb, yeb, chains_b,
                                               GRID, "guarded")
        bstar_b = bstar_from_chains(chains_b, yeb, max_steps)
        critic = rx.VoICritic()
        critic.fit(Xb, yeb, Xb, yvb)
        scores = critic.val.predict(flat_feats).reshape(n, width)
        g_stop = np.asarray([guarded_stop_index(s, a, lam_b)
                             for s, a in zip(scores, answers)], int)
        v_stop = np.asarray([stop_index(s, lam_b) for s in scores], int)
        g_em = em[np.arange(n), g_stop]
        v_em = em[np.arange(n), v_stop]
        g_calls = float(np.mean(1 + 2 * g_stop))
        v_calls = float(np.mean(1 + 2 * v_stop))
        simple = [(100 * fixed_em[k].mean(), fixed_calls[k])
                  for k in fixed_em] + [(100 * rep_em.mean(), rep_calls)]
        simple_named = {f"fixed-{k}": (100 * fixed_em[k].mean(), fixed_calls[k])
                        for k in fixed_em}
        simple_named["repetition"] = (100 * rep_em.mean(), rep_calls)
        bstar_em = 100 * fixed_em[bstar_b].mean()
        bstar_calls = fixed_calls[bstar_b]
        # total-uncertainty variant: also resample paired test questions
        tix = rng.integers(0, n, n)
        rows.append({
            "lambda": lam_b, "bstar": bstar_b,
            "guarded_em": 100 * g_em.mean(), "guarded_calls": g_calls,
            "value_em": 100 * v_em.mean(), "value_calls": v_calls,
            "guarded_minus_full": 100 * (g_em - full_em).mean(),
            "value_minus_full": 100 * (v_em - full_em).mean(),
            "guarded_minus_bstar": 100 * (g_em - fixed_em[bstar_b]).mean(),
            "guarded_minus_fixed2": 100 * (g_em - fixed_em[2]).mean(),
            "value_minus_fixed2": 100 * (v_em - fixed_em[2]).mean(),
            "guarded_dominated": pareto_dominated(
                100 * g_em.mean(), g_calls, simple),
            "value_dominated": pareto_dominated(
                100 * v_em.mean(), v_calls, simple),
            "guarded_dominators": dominator_name(
                100 * g_em.mean(), g_calls, simple_named),
            "value_dominators": dominator_name(
                100 * v_em.mean(), v_calls, simple_named),
            "guarded_beats_bstar": strictly_dominates(
                100 * g_em.mean(), g_calls, bstar_em, bstar_calls),
            "value_beats_bstar": strictly_dominates(
                100 * v_em.mean(), v_calls, bstar_em, bstar_calls),
            "total_guarded_minus_full": 100 * (g_em[tix] - full_em[tix]).mean(),
            "total_guarded_minus_fixed2": 100 * (
                g_em[tix] - fixed_em[2][tix]).mean(),
            "total_value_minus_full": 100 * (v_em[tix] - full_em[tix]).mean(),
        })
    lam_freq = {str(g): int(sum(r["lambda"] == g for r in rows)) for g in GRID}
    comp_names = [f"fixed-{k}" for k in range(max_steps + 1)] + ["repetition"]
    guarded_dom_freq = {c: round(float(np.mean([c in r["guarded_dominators"]
                                                for r in rows])), 3)
                        for c in comp_names}
    value_dom_freq = {c: round(float(np.mean([c in r["value_dominators"]
                                              for r in rows])), 3)
                      for c in comp_names}
    bstar_freq = {str(k): int(sum(r["bstar"] == k for r in rows))
                  for k in range(max_steps + 1)}
    logged_lambda = run["lam_main"]

    def dist(key):
        x = np.asarray([r[key] for r in rows], float)
        return {"mean": round(float(x.mean()), 3), "p2.5": pct(x, 2.5),
                "p50": pct(x, 50), "p97.5": pct(x, 97.5)}

    def prob(key):
        return round(float(np.mean([bool(r[key]) for r in rows])), 3)

    return {
        "tag": tag, "n_boot": n_boot, "n_test": n,
        "n_policy_training_episodes": len(fit_chains),
        "resampling_unit": "policy-training episode (chain), with replacement",
        "pipeline_per_replicate": [
            "5-fold cross-fitted critic scores on the resample",
            "guarded-rule threshold selection (cheapest within 1 EM band)",
            "fixed-budget selection B* (max training EM)",
            "critic refit on the full resample",
            "frozen policies replayed on the untouched test questions",
        ],
        "logged_guarded_replay_match_rate": round(float(logged_match), 4),
        "refit_on_original_chains_max_abs_score_deviation":
            round(refit_max_abs_dev, 6),
        "refit_on_original_chains_guarded_stop_agreement":
            round(refit_stop_agreement, 4),
        "logged_lambda": logged_lambda,
        "lambda_frequency": lam_freq,
        "lambda_logged_frequency": round(lam_freq[str(logged_lambda)] / n_boot, 3),
        "bstar_frequency": bstar_freq,
        "guarded_em": dist("guarded_em"),
        "guarded_calls": dist("guarded_calls"),
        "value_em": dist("value_em"),
        "value_calls": dist("value_calls"),
        "guarded_minus_full_em": dist("guarded_minus_full"),
        "value_minus_full_em": dist("value_minus_full"),
        "guarded_minus_bstar_em": dist("guarded_minus_bstar"),
        "guarded_minus_fixed2_em": dist("guarded_minus_fixed2"),
        "value_minus_fixed2_em": dist("value_minus_fixed2"),
        "p_guarded_point_below_margin": round(float(np.mean(
            [r["guarded_minus_full"] < -MARGIN_EM for r in rows])), 3),
        "p_guarded_dominated_by_simple": prob("guarded_dominated"),
        "p_value_dominated_by_simple": prob("value_dominated"),
        "guarded_dominator_frequency": guarded_dom_freq,
        "value_dominator_frequency": value_dom_freq,
        "dominator_note": ("share of replicates in which each simple competitor "
                           "(fixed-k or answer repetition) Pareto-dominates the "
                           "refit rule; a replicate can have several dominators"),
        "p_guarded_strictly_beats_bstar": prob("guarded_beats_bstar"),
        "p_value_strictly_beats_bstar": prob("value_beats_bstar"),
        "total_uncertainty_guarded_minus_full_em":
            dist("total_guarded_minus_full"),
        "total_uncertainty_guarded_minus_fixed2_em":
            dist("total_guarded_minus_fixed2"),
        "total_uncertainty_value_minus_full_em":
            dist("total_value_minus_full"),
        "interpretation": (
            "Intervals labelled 'guarded_minus_*' vary only training "
            "episodes (critic fit + threshold + B* selection) with the test "
            "sample fixed; 'total_uncertainty_*' additionally resample the "
            "paired test questions."),
    }


# --------------------------------------------------------------------------- #
# 2. hierarchical variance across the four corpus-index runs
# --------------------------------------------------------------------------- #
def contrasts_for_run(tag):
    run = json.loads((MEAS / f"run_{tag}.json").read_text())
    records = run["per_q"]
    em, feats, answers, rep_stop = test_arrays(records)
    n, width = em.shape
    last = width - 1
    g_stop = np.asarray([int(r["primary"]["searches"]) for r in records])
    v_stop = np.asarray([int(r["frozen_value"]["searches"]) for r in records])
    full = em[:, last]
    out = {
        "ids": [str(r["id"]) for r in records],
        "full": full,
        "guarded": em[np.arange(n), g_stop],
        "value": em[np.arange(n), v_stop],
        "repeat": em[np.arange(n), rep_stop],
        "fixed2": em[:, min(2, last)],
        "fixed3": em[:, min(3, last)],
        "guarded_calls": 1 + 2 * g_stop,
        "value_calls": 1 + 2 * v_stop,
        "repeat_calls": 1 + 2 * rep_stop,
    }
    return out


def paired_ci(d, seed=2027, n_boot=20_000):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = 100 * d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return {"mean_em_points": round(100 * float(d.mean()), 3),
            "ci95": [pct(means, 2.5), pct(means, 97.5)],
            "lower95_one_sided": pct(means, 5)}


def one_way_components(matrix):
    """Random-effects decomposition for a questions x runs matrix."""
    q, k = matrix.shape
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    msb = k * np.sum((row_means - grand) ** 2) / (q - 1)
    msw = np.sum((matrix - row_means[:, None]) ** 2) / (q * (k - 1))
    var_q = max((msb - msw) / k, 0.0)
    var_run = msw
    total = var_q + var_run
    return {
        "n_questions": int(q), "n_runs": int(k),
        "var_question": round(float(var_q), 5),
        "var_run_within_question": round(float(var_run), 5),
        "icc_question": round(float(var_q / total), 4) if total > 0 else None,
        "se_mean_question_sampling_only_em_points":
            round(100 * float(np.sqrt(total / q)), 3),
        "se_mean_run_only_em_points":
            round(100 * float(np.sqrt(var_run / (q * k))), 3),
        "note": ("Per-question paired contrast d = policy EM - comparator EM; "
                 "var_run is the stochastic-trajectory component at fixed "
                 "questions, var_question the between-question component. "
                 "SEs are for a 400-question mean.")
    }


INDEPENDENT_SAMPLE_TAGS = [
    ("corpus_guarded_qwen3b_data17", "B", 17),
    ("corpus_guarded_qwen3b_data23", "C", 23),
    ("corpus_guarded_qwen3b_data29", "D", 29),
    ("corpus_guarded_qwen3b_data31", "E", 31),
]


def hierarchical_variance(seed=2027, n_boot=20_000):
    """Question-sample vs decoding variance across every corpus-index run.

    Sample A (data seed 13) has three stochastic runs (LLM seeds 7/11/19);
    each further data seed is an independent 400-question draw evaluated once
    with deterministic decoding.  All runs share the logging agent, corpus,
    critic protocol, and frozen rules.
    """
    tags_a = ["corpus_guarded_qwen3b", "corpus_guarded_qwen3b_seed11",
              "corpus_guarded_qwen3b_seed19"]
    indep = [(t, lab, ds) for t, lab, ds in INDEPENDENT_SAMPLE_TAGS
             if (MEAS / f"run_{t}.json").exists()]
    runs_a = [contrasts_for_run(t) for t in tags_a]
    runs_i = [(lab, t, contrasts_for_run(t)) for t, lab, _ in indep]
    assert runs_a[0]["ids"] == runs_a[1]["ids"] == runs_a[2]["ids"]
    ids_a = runs_a[0]["ids"]
    sample_ids = {"A": ids_a}
    for lab, _t, r in runs_i:
        sample_ids[lab] = r["ids"]
    labels = list(sample_ids)
    overlap = {f"{x}-{y}": len(set(sample_ids[x]) & set(sample_ids[y]))
               for i, x in enumerate(labels) for y in labels[i + 1:]}
    contrasts = {
        "guarded_minus_full": ("guarded", "full"),
        "guarded_minus_fixed3": ("guarded", "fixed3"),
        "guarded_minus_fixed2": ("guarded", "fixed2"),
        "guarded_minus_repeat": ("guarded", "repeat"),
        "value_minus_full": ("value", "full"),
        "value_minus_fixed3": ("value", "fixed3"),
        "value_minus_fixed2": ("value", "fixed2"),
        "fixed3_minus_full": ("fixed3", "full"),
        "fixed2_minus_full": ("fixed2", "full"),
    }
    result = {
        "question_sample_A_runs": tags_a,
        "independent_question_samples": {lab: t for lab, t, _ in runs_i},
        "n_independent_question_samples": 1 + len(runs_i),
        "n_runs": 3 + len(runs_i),
        "n_questions_per_sample": len(ids_a),
        "question_id_overlap_between_samples": overlap,
        "overlap_note": ("All samples are independent draws from the same "
                         "7,405-question validation pool, so a few ids recur "
                         "by chance; the pooled bootstrap clusters on question "
                         "id across all runs."),
        "contrasts": {},
    }
    all_ids = list(ids_a) + [q for _lab, _t, r in runs_i for q in r["ids"]]
    uniq = sorted(set(all_ids))
    pos = {q: i for i, q in enumerate(uniq)}
    rng = np.random.default_rng(seed)
    cluster_draws = rng.integers(0, len(uniq), (n_boot, len(uniq)))
    for name, (a, b) in contrasts.items():
        mat_a = np.stack([r[a] - r[b] for r in runs_a], axis=1)   # 400 x 3
        comp = one_way_components(mat_a)
        per_run = {t: paired_ci(r[a] - r[b], seed=seed)
                   for t, r in zip(tags_a, runs_a)}
        for _lab, t, r in runs_i:
            per_run[t] = paired_ci(r[a] - r[b], seed=seed)
        cluster_a = mat_a.mean(axis=1)
        sample_means = {"A": 100 * float(cluster_a.mean())}
        sample_se = {"A": 100 * float(np.sqrt(cluster_a.var(ddof=1) / len(cluster_a)))}
        for lab, _t, r in runs_i:
            d = r[a] - r[b]
            sample_means[lab] = 100 * float(d.mean())
            sample_se[lab] = 100 * float(np.sqrt(d.var(ddof=1) / len(d)))
        means = np.asarray(list(sample_means.values()))
        ses = np.asarray(list(sample_se.values()))
        k = len(means)
        between = {}
        if k >= 2:
            sd_obs = float(means.std(ddof=1))
            exp_var = float(np.mean(ses ** 2))          # pure question sampling
            excess = max(sd_obs ** 2 - exp_var, 0.0)
            between = {
                "observed_sd_of_sample_means_em": round(sd_obs, 3),
                "expected_sd_under_question_sampling_em": round(np.sqrt(exp_var), 3),
                "excess_between_sample_sd_em": round(float(np.sqrt(excess)), 3),
                "range_of_sample_means_em": [round(float(means.min()), 3),
                                             round(float(means.max()), 3)],
                "n_samples": k,
            }
        # observations: every run x question, clustered by question id
        obs = [mat_a.reshape(-1)]
        cl = [pos[q] for q in ids_a for _ in range(3)]
        for _lab, _t, r in runs_i:
            obs.append(r[a] - r[b])
            cl.extend(pos[q] for q in r["ids"])
        obs = np.concatenate(obs)
        obs_cluster = np.asarray(cl)
        csum = np.bincount(obs_cluster, weights=obs, minlength=len(uniq))
        ccnt = np.bincount(obs_cluster, minlength=len(uniq)).astype(float)
        draws = 100 * csum[cluster_draws].sum(axis=1) / ccnt[cluster_draws].sum(axis=1)
        entry = {
            "per_run": per_run,
            "sample_means_em_points": {lab: round(v, 3) for lab, v in sample_means.items()},
            "sample_question_sampling_se_em_points": {lab: round(v, 3) for lab, v in sample_se.items()},
            "sample_A_three_run_mean_em_points": round(sample_means["A"], 3),
            "sample_A_variance_components": comp,
            "between_sample": between,
            "pooled_all_run_mean_em_points": round(float(100 * obs.mean()), 3),
            "pooled_question_clustered_ci95": [pct(draws, 2.5), pct(draws, 97.5)],
            "pooled_lower95_one_sided": pct(draws, 5),
        }
        # backward-compatible fields for the first independent sample (B)
        if runs_i:
            lab_b = runs_i[0][0]
            entry["sample_B_em_points"] = round(sample_means[lab_b], 3)
            gap = sample_means[lab_b] - sample_means["A"]
            se_gap = float(np.sqrt(sample_se["A"] ** 2 + sample_se[lab_b] ** 2))
            entry["between_sample_gap_em_points"] = round(gap, 3)
            entry["gap_se_question_sampling_em_points"] = round(se_gap, 3)
            entry["gap_z"] = round(gap / se_gap, 3) if se_gap else None
        entry["pooled_four_run_mean_em_points"] = entry["pooled_all_run_mean_em_points"]
        result["contrasts"][name] = entry
    # margin pass count over independent question samples (guarded vs full)
    g = result["contrasts"]["guarded_minus_full"]["per_run"]
    passes = {t: bool(g[t]["lower95_one_sided"] > -MARGIN_EM) for t in g}
    result["guarded_margin_pass_by_run"] = passes
    result["independent_sample_margin_passes"] = {
        "passed": int(sum(passes[t] for _l, t, _r in runs_i) + passes[tags_a[0]]),
        "of": 1 + len(runs_i),
        "note": "primary sample A counted once (seed-7 run) plus each independent draw",
    }
    result["pooled_calls"] = {
        "guarded": round(float(np.mean([r["guarded_calls"].mean()
                                        for r in runs_a + [x[2] for x in runs_i]])), 3),
        "value": round(float(np.mean([r["value_calls"].mean()
                                      for r in runs_a + [x[2] for x in runs_i]])), 3),
        "repeat": round(float(np.mean([r["repeat_calls"].mean()
                                       for r in runs_a + [x[2] for x in runs_i]])), 3),
        "fixed2": 5.0, "fixed3": 7.0, "full": 9.0,
    }
    result["design_note"] = (
        "Sample A carries three stochastic runs on fixed questions; each "
        "further data seed is an independent question draw evaluated once. "
        "'between_sample' compares the spread of sample means with the spread "
        "expected from question sampling alone; with fewer than three "
        "independent samples it is descriptive only.")
    return result


# --------------------------------------------------------------------------- #
# 3. decision matrix
# --------------------------------------------------------------------------- #
def verdict(em, calls, simple, bstar_em, bstar_calls):
    if pareto_dominated(em, calls, simple):
        return "L"
    if strictly_dominates(em, calls, bstar_em, bstar_calls):
        return "W"
    return "T"


def dominator_name(em, calls, simple_named):
    names = [name for name, (e, c) in simple_named.items()
             if e >= em - 1e-12 and c <= calls + 1e-12
             and (e > em + 1e-12 or c < calls - 1e-12)]
    return names


def decision_matrix():
    rev = json.loads((MEAS / "revision_experiments.json").read_text())
    fz = json.loads((MEAS / "frozen_protocol.json").read_text())
    bstar = {c["tag"]: c["b_star"] for c in fz["hotpot_cells"] + fz["dataset_cells"]}
    cells = []
    for c in rev["cells"]:
        p = c["policies"]
        simple_named = {f"fixed-{k}": (p[f"fixed_{k}"]["em"], p[f"fixed_{k}"]["llm_calls"])
                        for k in range(5) if f"fixed_{k}" in p}
        simple_named["repetition"] = (p["answer_repeat"]["em"],
                                      p["answer_repeat"]["llm_calls"])
        simple = list(simple_named.values())
        bs = bstar[c["tag"]]
        bstar_em, bstar_calls = p[f"fixed_{bs}"]["em"], p[f"fixed_{bs}"]["llm_calls"]
        entry = {
            "cell": c["tag"], "label": HIST_PRETTY[c["tag"]], "n": c["n_test"],
            "setting": "historical", "full_em": p["react"]["em"],
            "bstar": bs, "bstar_em": bstar_em, "bstar_calls": bstar_calls,
            "repeat_em": p["answer_repeat"]["em"],
            "repeat_calls": p["answer_repeat"]["llm_calls"],
            # historical cells lack raw prefix features for a deployment-
            # matched dedicated-split certificate; their score-mismatched
            # curves are quarantined (frozen_protocol.json), so no verdict.
            "certificate": "n/a",
        }
        for pol, key in [("value", "value"), ("guarded", "guarded_value")]:
            em, calls = p[key]["em"], p[key]["llm_calls"]
            entry[f"{pol}_em"] = em
            entry[f"{pol}_calls"] = calls
            entry[f"{pol}_verdict"] = verdict(em, calls, simple, bstar_em, bstar_calls)
            entry[f"{pol}_dominated_by"] = dominator_name(em, calls, simple_named)
            paired = c["paired_comparisons"][f"{key}_vs_react"]
            entry[f"{pol}_delta_vs_full"] = paired["mean_diff_em_points"]
            entry[f"{pol}_ci95_vs_full"] = paired["ci95"]
        cells.append(entry)
    for tag, fname, label, short in FRESH:
        path = MEAS / fname
        if not path.exists():
            continue
        cc = json.loads(path.read_text())
        m = cc["methods"]
        simple_named = {f"fixed-{k}": (m[f"fixed_{k}"]["em"], m[f"fixed_{k}"]["llm_calls"])
                        for k in range(5) if f"fixed_{k}" in m}
        simple_named["repetition"] = (m["answer_repeat"]["em"],
                                      m["answer_repeat"]["llm_calls"])
        simple = list(simple_named.values())
        bs = cc["training_selected_fixed_budget"]
        bstar_em, bstar_calls = m[f"fixed_{bs}"]["em"], m[f"fixed_{bs}"]["llm_calls"]
        sgr_fired = any(x["fired"] for x in cc["dedicated_calibration_sgr"])
        entry = {
            "cell": tag, "label": label, "short": short, "n": cc["n_test"],
            "setting": "fresh", "full_em": m["full_budget"]["em"],
            "bstar": bs, "bstar_em": bstar_em, "bstar_calls": bstar_calls,
            "repeat_em": m["answer_repeat"]["em"],
            "repeat_calls": m["answer_repeat"]["llm_calls"],
            "certificate": "fired" if sgr_fired else "refuse",
        }
        for pol, key in [("value", "value_only"), ("guarded", "guarded_value")]:
            em, calls = m[key]["em"], m[key]["llm_calls"]
            entry[f"{pol}_em"] = em
            entry[f"{pol}_calls"] = calls
            entry[f"{pol}_verdict"] = verdict(em, calls, simple, bstar_em, bstar_calls)
            entry[f"{pol}_dominated_by"] = dominator_name(em, calls, simple_named)
            paired = m[key]["paired_vs_full"]
            entry[f"{pol}_delta_vs_full"] = paired["mean_em_points"]
            entry[f"{pol}_ci95_vs_full"] = paired["ci95"]
        cells.append(entry)
    tally = {}
    for pol in ["value", "guarded"]:
        tally[pol] = {v: int(sum(c[f"{pol}_verdict"] == v for c in cells))
                      for v in ["W", "T", "L"]}
    tally["certificate_refusals"] = int(sum(c["certificate"] == "refuse" for c in cells))
    tally["certificate_evaluable_cells"] = int(sum(c["certificate"] != "n/a" for c in cells))
    tally["n_cells"] = len(cells)
    return {
        "verdict_rule": (
            "Against the simple family {fixed-0..fixed-B, answer repetition}: "
            "L = some simple policy Pareto-dominates the learned policy on "
            "(EM, LLM calls); W = undominated and strictly dominates the "
            "training-selected fixed budget B*; T = undominated but does not "
            "strictly dominate B*. Certificate: corrected SGR at any target on "
            "the dedicated split (fresh runs only; historical cells are n/a)."),
        "cells": cells, "tally": tally,
    }


# --------------------------------------------------------------------------- #
# LaTeX fragments
# --------------------------------------------------------------------------- #
def write_tables(out):
    TABLES.mkdir(exist_ok=True)
    display_verdict = {"W": "I", "T": "F", "L": "D"}
    rows = []
    def ci(c, pol):
        lo, hi = c[f"{pol}_ci95_vs_full"]
        return f"{display(c[f'{pol}_delta_vs_full'], signed=True)}\\,[{display(lo)},{display(hi)}]"
    for c in out["decision_matrix"]["cells"]:
        label = c['label'].replace('q-sample', 'sample').replace('Wikipedia 25M hybrid', 'Wikipedia 25M')
        rows.append(
            f"{label} & {c['n']} & {display(c['full_em'])} & "
            f"{display(c['bstar_em'])}/{display(c['bstar_calls'])}\\,({c['bstar']}) & "
            f"{display(c['repeat_em'])}/{display(c['repeat_calls'])} & "
            f"{display(c['value_em'])}/{display(c['value_calls'])} & {ci(c, 'value')} & "
            f"{display_verdict[c['value_verdict']]} & "
            f"{display(c['guarded_em'])}/{display(c['guarded_calls'])} & {ci(c, 'guarded')} & "
            f"{display_verdict[c['guarded_verdict']]} & "
            f"{'NA' if c['certificate'] == 'n/a' else c['certificate']} " + r"\\")
    (TABLES / "decision_matrix_rows.tex").write_text("\n".join(rows) + "\n")

    rows = []
    short = {t: s for t, _, _, s in FRESH}
    pretty = {"corpus_s7": "Corpus s7, A", "corpus_s11": "Corpus s11, A",
              "corpus_s19": "Corpus s19, A", "corpus_qB": "Corpus s7, B",
              "corpus_qC": "Corpus s7, C", "corpus_qD": "Corpus s7, D",
              "corpus_qE": "Corpus s7, E",
              "w2_d1": "2Wiki d1", "w2_d2": "2Wiki d2", "w2_d3": "2Wiki d3",
              "mq_d1": "MuSiQue d1", "mq_d2": "MuSiQue d2", "mq_d3": "MuSiQue d3",
              "wiki25m": "Wikipedia 25M"}
    for tag in sorted([t for t, *_ in FRESH if t in out["nested_refit_select"]],
                      key=lambda t: (0 if t.startswith("corpus_guarded") else
                                     1 if t.startswith("wiki25m") else
                                     2 if t.startswith("corpus2wiki") else 3)):
        res = out["nested_refit_select"][tag]
        g = res["guarded_minus_full_em"]
        rows.append(
            f"{pretty[short[tag]]} & {percent(res['lambda_logged_frequency'])}\\% & "
            f"{g['mean']:+.2f}\\,[{g['p2.5']:.2f},{g['p97.5']:.2f}] & "
            f"{res['guarded_calls']['mean']:.2f} & "
            f"{percent(res['p_guarded_dominated_by_simple'])}\\% & "
            f"{percent(res['p_value_dominated_by_simple'])}\\% & "
            f"{percent(res['p_guarded_strictly_beats_bstar'])}\\%"
            + "".join(f" & {percent(res.get('guarded_dominator_frequency', {}).get(k, 0))}\\%"
                      for k in ["fixed-2", "fixed-3", "repetition"])
            + " \\\\")
    (TABLES / "nested_uncertainty_rows.tex").write_text("\n".join(rows) + "\n")

    hv_all = out["hierarchical_variance"]
    hv = hv_all["contrasts"]
    sample_labels = ["A"] + list(hv_all["independent_question_samples"].keys())
    rows = []
    names = [("guarded_minus_full", "G $-$ full"),
             ("guarded_minus_fixed3", "G $-$ fixed-3"),
             ("guarded_minus_fixed2", "G $-$ fixed-2"),
             ("guarded_minus_repeat", "G $-$ repetition"),
             ("value_minus_full", "V $-$ full"),
             ("value_minus_fixed3", "V $-$ fixed-3"),
             ("value_minus_fixed2", "V $-$ fixed-2")]
    for key, label in names:
        c = hv[key]
        cells = [display(c['sample_means_em_points'][lab], signed=True) for lab in sample_labels]
        between = c.get("between_sample", {})
        spread = (f"{between['observed_sd_of_sample_means_em']:.2f}/"
                  f"{between['expected_sd_under_question_sampling_em']:.2f}"
                  if between else "--")
        rows.append(
            f"{label} & " + " & ".join(cells) + f" & {spread} & "
            f"{c['pooled_all_run_mean_em_points']:+.2f}\\,"
            f"[{c['pooled_question_clustered_ci95'][0]:.2f},"
            f"{c['pooled_question_clustered_ci95'][1]:.2f}] \\\\")
    (TABLES / "simple_policy_paired_rows.tex").write_text("\n".join(rows) + "\n")
    header = ("Contrast & " + " & ".join(sample_labels)
              + " & SD obs/exp & Pooled [CI] \\\\")
    spec = "l" + "r" * len(sample_labels) + "rr"
    table = ("\\begin{tabular}{" + spec + "}\n\\toprule\n" + header
             + "\n\\midrule\n" + "\n".join(rows)
             + "\n\\bottomrule\n\\end{tabular}\n")
    (TABLES / "simple_policy_paired_table.tex").write_text(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--reuse-nested", action="store_true",
                    help="reuse the nested bootstrap block of an existing "
                         "frontier_analyses.json instead of recomputing")
    args = ap.parse_args()
    out = {"status": "derived_from_frozen_measured_runs_no_new_llm_calls",
           "n_boot": args.n_boot, "seed": args.seed}
    out["nested_refit_select"] = {}
    previous = None
    dest = MEAS / "frontier_analyses.json"
    if args.reuse_nested and dest.exists():
        previous = json.loads(dest.read_text())
        out["n_boot"] = previous.get("n_boot", args.n_boot)
        out["seed"] = previous.get("seed", args.seed)
    for tag, fname, _label, _short in FRESH:
        if not (MEAS / f"run_{tag}.json").exists():
            continue
        if previous and tag in previous.get("nested_refit_select", {}):
            out["nested_refit_select"][tag] = previous["nested_refit_select"][tag]
            print(f"[nested] {tag} (reused)", flush=True)
            continue
        print(f"[nested] {tag}", flush=True)
        out["nested_refit_select"][tag] = nested_refit_select(
            tag, args.n_boot, args.seed)
        r = out["nested_refit_select"][tag]
        print(f"  lambda freq={r['lambda_frequency']} logged={r['logged_lambda']} "
              f"dG-full={r['guarded_minus_full_em']} "
              f"P(dom)={r['p_guarded_dominated_by_simple']} "
              f"P(val dom)={r['p_value_dominated_by_simple']} "
              f"P(beats B*)={r['p_guarded_strictly_beats_bstar']} "
              f"match={r['logged_guarded_replay_match_rate']} "
              f"refit dev={r['refit_on_original_chains_max_abs_score_deviation']}",
              flush=True)
    print("[hierarchical]", flush=True)
    out["hierarchical_variance"] = hierarchical_variance(seed=args.seed)
    for k, v in out["hierarchical_variance"]["contrasts"].items():
        print(f"  {k}: A={v['sample_A_three_run_mean_em_points']} "
              f"B={v['sample_B_em_points']} gap={v['between_sample_gap_em_points']} "
              f"z={v['gap_z']} icc={v['sample_A_variance_components']['icc_question']} "
              f"pooled={v['pooled_four_run_mean_em_points']} "
              f"{v['pooled_question_clustered_ci95']}", flush=True)
    print("[matrix]", flush=True)
    out["decision_matrix"] = decision_matrix()
    for c in out["decision_matrix"]["cells"]:
        print(f"  {c['label']:28s} full={c['full_em']:5.1f} B*={c['bstar_em']:5.1f}/"
              f"{c['bstar_calls']:.1f} rep={c['repeat_em']:5.1f}/{c['repeat_calls']:.2f} "
              f"val={c['value_em']:5.1f}/{c['value_calls']:.2f} {c['value_verdict']} "
              f"{c['value_dominated_by']} guard={c['guarded_em']:5.1f}/"
              f"{c['guarded_calls']:.2f} {c['guarded_verdict']} "
              f"{c['guarded_dominated_by']} cert={c['certificate']}", flush=True)
    print("  tally:", out["decision_matrix"]["tally"], flush=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    write_tables(out)
    print(f"wrote {dest} and tex fragments in {TABLES}")


if __name__ == "__main__":
    main()
