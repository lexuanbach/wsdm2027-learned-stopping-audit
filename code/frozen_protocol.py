"""Frozen-selection evaluation protocol (revision of the test-selected one).

Every knob is selected on TRAINING data only and frozen before any test
outcome is read:

  * lambda*  : per-cell VoI threshold, chosen by replaying the deployed
               critic on the logged TRAINING chains (critic_data_<tag>.pkl)
               with the predeclared rule "cheapest grid point whose training
               EM is within 1.0 EM of the training full-budget EM; else the
               highest-EM point, ties broken by fewest searches".
  * lambda_g : one GLOBAL threshold selected the same way on the pooled
               training chains of all nine HotpotQA cells.
  * B*       : best fixed budget by training-chain EM (smallest budget
               achieving the max), the symmetric competitor to lambda*.
  * tau*     : sufficiency-stop threshold (SIM-RAG-style: stop when the
               error head predicts sufficiently low error), same rule,
               same training chains -- isolates value-head vs error-head
               stopping under an identical protocol.
  * transfer : source-trained critic AND source-selected lambda* applied
               unchanged to the target's logged chains (no target sweep).

Test-set evaluation then happens ONCE per frozen knob, with paired
statistics:

  * paired per-question EM differences (VoI - ReAct) with percentile
    bootstrap CIs and one-sided lower bounds (non-inferiority reading);
  * question-clustered flip inference across backbones (cluster bootstrap
    + question-level sign-flip permutation test) -- the same questions recur
    across backbones, so flips are clustered, not independent;
  * SGR risk calibration with the Geifman--El-Yaniv delta-splitting
    correction (delta / ceil(log2 m) per binary-search step), calibrated on
    cross-fitted TRAINING-episode stops of the deployed first-crossing
    policy at lambda*, then evaluated once on all test episodes.

Everything is an offline replay of measured files; no LLM calls.
Writes results/measured/frozen_protocol.json and tex fragments into
outputs/tables/.
"""
from __future__ import annotations

import json
import math
import pathlib
import pickle

import numpy as np
from scipy.stats import beta as beta_dist

MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
ART = pathlib.Path(__file__).resolve().parents[1]
MANU = ART / "outputs"  # generated tables/ and figures/ (not tracked)
(MANU / "tables").mkdir(parents=True, exist_ok=True)
(MANU / "figures").mkdir(parents=True, exist_ok=True)
TABDIR = MANU / "tables"

HOTPOT_TAGS = [
    "final_haiku", "final_sonnet45", "final_gptoss120b", "final_novalite",
    "final_qwen3b", "final_llama3b", "final_commandr35b", "final_kimik25",
    "final_minimaxm25",
]
DATASET_TAGS = ["final2wiki_qwen3b", "rev2_musique_qwen3b",
                "rev2_musique_llama3b"]
PRETTY = {
    "final_haiku": "Claude Haiku 4.5", "final_sonnet45": "Claude Sonnet 4.5",
    "final_gptoss120b": "GPT-oss-120B", "final_novalite": "Amazon Nova Lite",
    "final_qwen3b": "Qwen2.5-3B", "final_llama3b": "Llama-3.2-3B",
    "final_commandr35b": "Command-R-35B", "final_kimik25": "Kimi K2.5",
    "final_minimaxm25": "MiniMax M2.5",
    "final2wiki_qwen3b": "2Wiki / Qwen2.5-3B",
    "rev2_musique_qwen3b": "MuSiQue / Qwen2.5-3B",
    "rev2_musique_llama3b": "MuSiQue / Llama-3.2-3B",
}
TAU_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
BAND = 1.0          # predeclared matched-accuracy band, EM points
N_BOOT = 10_000


# --------------------------------------------------------------------------- #
# Loading + deterministic critic refit
# --------------------------------------------------------------------------- #
def load_run(tag):
    return json.loads((MEAS / f"run_{tag}.json").read_text())


def load_chains(tag):
    """Training chains from the cached counterfactual data.

    Rows are appended prefix-by-prefix in trajectory order and feature 0 is
    the searches count, so a 0 in column 0 starts a new chain.
    """
    d = pickle.loads((MEAS / f"critic_data_{tag}.pkl").read_bytes())
    X = np.asarray(d["X_val"], float)
    y_vtg = np.asarray(d["y_val"], float)
    y_err = np.asarray(d["y_err"], int)
    chains, cur = [], []
    for i, row in enumerate(X):
        if row[0] == 0 and cur:
            chains.append(cur)
            cur = []
        cur.append(i)
    if cur:
        chains.append(cur)
    return X, y_vtg, y_err, chains


def refit_critic(X, y_vtg, y_err):
    """Deterministic refit of the run-time critic (same estimators/data)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    val = GradientBoostingRegressor(max_depth=2, n_estimators=80).fit(X, y_vtg)
    if len(set(y_err.tolist())) < 2:
        from sklearn.dummy import DummyClassifier
        err = DummyClassifier(strategy="prior").fit(X, y_err)
    else:
        err = LogisticRegression(max_iter=1000).fit(X, y_err)
    return val, err


def err_probs(err_clf, X):
    p = err_clf.predict_proba(X)
    classes = list(err_clf.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return p[:, classes.index(1)]


def oof_scores(X, y_vtg, y_err, chains, folds=5, seed=3):
    """Out-of-fold critic scores over the training chains.

    The deployed critic is fit on ALL training chains, so its scores on
    those same chains are overfit-accurate; selecting lambda on them chooses
    an over-aggressive threshold (the first run of this protocol measured
    4--9 EM test losses that the training replay never predicted). Cross-
    fitting restores deployment-realistic score noise while still never
    touching test data: each chain is scored by a critic that did not train
    on it, and the pooled scores drive the threshold selection.
    """
    rng = np.random.default_rng(seed)
    order = np.arange(len(chains))
    rng.shuffle(order)
    v_out = np.zeros(len(X))
    e_out = np.zeros(len(X))
    for f in range(folds):
        te = set(order[f::folds].tolist())
        tr_rows = [i for c in range(len(chains)) if c not in te
                   for i in chains[c]]
        te_rows = [i for c in te for i in chains[c]]
        if not tr_rows or not te_rows:
            continue
        val, errclf = refit_critic(X[tr_rows], y_vtg[tr_rows], y_err[tr_rows])
        v_out[te_rows] = val.predict(X[te_rows])
        e_out[te_rows] = err_probs(errclf, X[te_rows])
    return v_out, e_out


# --------------------------------------------------------------------------- #
# Replay primitives
# --------------------------------------------------------------------------- #
def replay_chains(scores, X, y_err, chains, th, stop_if="lt"):
    """Stop at the first non-terminal prefix whose score crosses th."""
    ems, ss = [], []
    for ch in chains:
        stop = ch[-1]
        for i in ch[:-1]:
            fire = scores[i] < th if stop_if == "lt" else scores[i] <= th
            if fire:
                stop = i
                break
        ems.append(1 - y_err[stop])
        ss.append(X[stop][0])
    return np.array(ems, float), np.array(ss, float)


def chain_ref(X, y_err, chains):
    em = np.array([1 - y_err[c[-1]] for c in chains], float)
    s = np.array([X[c[-1]][0] for c in chains], float)
    return em, s


def select_threshold(scores, X, y_err, chains, grid, stop_if="lt"):
    """Predeclared rule on TRAINING chains: cheapest point within BAND EM of
    the full-budget reference; fallback highest EM, ties fewest searches."""
    ref_em, _ = chain_ref(X, y_err, chains)
    ref = 100 * ref_em.mean()
    front = []
    for th in grid:
        em, s = replay_chains(scores, X, y_err, chains, th, stop_if)
        front.append({"th": th, "em": 100 * em.mean(), "s": s.mean()})
    cand = [f for f in front if f["em"] >= ref - BAND]
    if cand:
        pick = min(cand, key=lambda f: (f["s"], -f["em"]))
        rule = "band"
    else:
        best = max(f["em"] for f in front)
        pick = min([f for f in front if f["em"] == best], key=lambda f: f["s"])
        rule = "fallback_max_em"
    return pick["th"], {"train_ref_em": round(ref, 1), "rule": rule,
                        "train_frontier": [{k: round(v, 2) for k, v in f.items()}
                                           for f in front]}


def test_stop_index(r, lam):
    """Deployed first-crossing value rule on a logged test record."""
    for t in range(len(r["prefix_value"]) - 1):
        if r["prefix_value"][t] < lam:
            return t
    return len(r["prefix_value"]) - 1


def test_stop_index_err(r, tau):
    for t in range(len(r["prefix_errhat"]) - 1):
        if r["prefix_errhat"][t] <= tau:
            return t
    return len(r["prefix_errhat"]) - 1


def paired_stats(em_a, em_b, seed=0):
    """Paired bootstrap for mean(a-b): CI, one-sided lower bound."""
    a, b = np.asarray(em_a, float), np.asarray(em_b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    means = 100 * d[idx].mean(axis=1)
    return {
        "mean_diff": round(100 * d.mean(), 2),
        "ci95": [round(float(np.percentile(means, 2.5)), 2),
                 round(float(np.percentile(means, 97.5)), 2)],
        "lower90_onesided": round(float(np.percentile(means, 10)), 2),
        "lower95_onesided": round(float(np.percentile(means, 5)), 2),
    }


def boot_ci_mean(vals, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (N_BOOT, len(v)))
    means = 100 * v[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 1),
            round(float(np.percentile(means, 97.5)), 1)]


# --------------------------------------------------------------------------- #
# SGR with the Geifman--El-Yaniv delta-splitting correction
# --------------------------------------------------------------------------- #
def cp_upper(k, n, level):
    """Clopper-Pearson upper bound at confidence `level` (k errors of n)."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(level, k + 1, n - k))


def sgr_threshold(errhat, err, alpha, delta=0.10):
    """Threshold found by binary search over
    nested selection sizes; each of the <= ceil(log2 m) tests runs at level
    delta / ceil(log2 m) (union bound => simultaneous 1-delta validity).

    Returns (threshold, n_selected) or (None, 0) if even the smallest
    non-empty selection cannot be certified.
    """
    order = np.argsort(errhat, kind="stable")
    eh = np.asarray(errhat, float)[order]
    ee = np.asarray(err, int)[order]
    m = len(eh)
    if m == 0:
        return None, 0
    steps = max(1, math.ceil(math.log2(m)))
    level = 1 - delta / steps
    cum = np.cumsum(ee)
    # Threshold candidates end after complete tie groups: deployment selects
    # every score <= threshold, never an arbitrary prefix inside a tie.
    ends = [i for i in range(1, m + 1)
            if i == m or eh[i] != eh[i - 1]]
    lo, hi = 0, len(ends) - 1
    best = None
    for _ in range(steps):
        if lo > hi:
            break
        mid = (lo + hi + 1) // 2
        n_selected = ends[mid]
        if cp_upper(int(cum[n_selected - 1]), n_selected, level) <= alpha:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        return None, 0
    n_selected = ends[best]
    return float(eh[n_selected - 1]), n_selected


# --------------------------------------------------------------------------- #
# Per-cell protocol
# --------------------------------------------------------------------------- #
def run_cell(tag, lam_override=None):
    d = load_run(tag)
    Q = d["per_q"]
    grid = [float(x) for x in d["lam_grid"]]
    X, y_vtg, y_err, chains = load_chains(tag)
    vscores, escores = oof_scores(X, y_vtg, y_err, chains)

    lam, lam_meta = select_threshold(vscores, X, y_err, chains, grid, "lt")
    if lam_override is not None:
        lam = lam_override
    tau, tau_meta = select_threshold(escores, X, y_err, chains, TAU_GRID, "le")

    # B* on training chains: smallest budget with max training EM.
    max_pref = max(len(c) for c in chains)
    b_em = []
    for b in range(max_pref):
        b_em.append(np.mean([1 - y_err[c[min(b, len(c) - 1)]] for c in chains]))
    b_star = int(np.argmax(b_em))

    react_em = np.array([r["prefix_em"][-1] for r in Q], float)
    react_s = np.array([r["prefix_searches"][-1] for r in Q], float)

    stops = [test_stop_index(r, lam) for r in Q]
    voi_em = np.array([r["prefix_em"][t] for r, t in zip(Q, stops)], float)
    voi_s = np.array([r["prefix_searches"][t] for r, t in zip(Q, stops)], float)

    stops_tau = [test_stop_index_err(r, tau) for r in Q]
    suf_em = np.array([r["prefix_em"][t] for r, t in zip(Q, stops_tau)], float)
    suf_s = np.array([r["prefix_searches"][t] for r, t in zip(Q, stops_tau)],
                     float)

    bs_em = np.array([r["prefix_em"][min(b_star, len(r["prefix_em"]) - 1)]
                      for r in Q], float)
    bs_s = np.array([r["prefix_searches"][min(b_star, len(r["prefix_em"]) - 1)]
                     for r in Q], float)

    out = {
        "tag": tag, "n": len(Q), "lambda": lam, "lambda_meta": lam_meta,
        "tau": tau, "tau_meta": tau_meta, "b_star": b_star,
        "react": {"em": round(100 * react_em.mean(), 1),
                  "em_ci": boot_ci_mean(react_em),
                  "s": round(react_s.mean(), 2)},
        "voi_frozen": {"em": round(100 * voi_em.mean(), 1),
                       "em_ci": boot_ci_mean(voi_em),
                       "s": round(voi_s.mean(), 2),
                       "save_pct": round(100 * (1 - voi_s.mean()
                                                / react_s.mean()))},
        "paired_voi_vs_react": paired_stats(voi_em, react_em),
        "suff_frozen": {"em": round(100 * suf_em.mean(), 1),
                        "s": round(suf_s.mean(), 2)},
        "bstar_frozen": {"em": round(100 * bs_em.mean(), 1),
                         "s": round(bs_s.mean(), 2)},
        "paired_voi_vs_bstar": paired_stats(voi_em, bs_em, seed=1),
        "closed_book_em": round(100 * np.mean([r["prefix_em"][0] for r in Q]), 1),
        "_voi_em": voi_em.tolist(), "_react_em": react_em.tolist(),
        "_stops": stops,
    }
    return out, d


# --------------------------------------------------------------------------- #
# Cluster-aware flip analysis across HotpotQA backbones
# --------------------------------------------------------------------------- #
def flips_clustered(cells, runs):
    """Oracle-reachability and controller-realised flips with question-level
    clustering. Question identity = absolute index in the seed-13 shuffle
    (n_train + position), constant across runs of the same parquet."""
    recs = []   # (qkey, bad_reach, good_reach, voi_gain, voi_loss, tag)
    for c in cells:
        d = runs[c["tag"]]
        n_train = d["config"]["n_train"]
        for j, r in enumerate(d["per_q"]):
            qkey = n_train + j
            best_early = max(r["prefix_em"][:-1] or [0])
            last = r["prefix_em"][-1]
            v = c["_voi_em"][j]
            recs.append((qkey, int(best_early > last), int(last > best_early),
                         int(v > last), int(last > v), c["tag"]))
    bad = sum(r[1] for r in recs)
    good = sum(r[2] for r in recs)
    vgain = sum(r[3] for r in recs)
    vloss = sum(r[4] for r in recs)

    keys = sorted({r[0] for r in recs})
    by_key = {k: [r for r in recs if r[0] == k] for k in keys}
    rng = np.random.default_rng(2)
    diffs, ratios = [], []
    for _ in range(N_BOOT):
        smp = rng.integers(0, len(keys), len(keys))
        b = g = 0
        for i in smp:
            for r in by_key[keys[i]]:
                b += r[1]
                g += r[2]
        diffs.append(b - g)
        ratios.append((b + 0.5) / (g + 0.5))
    # question-level sign-flip permutation on net counts
    net = np.array([sum(r[1] - r[2] for r in by_key[k]) for k in keys], float)
    obs = net.sum()
    flips = rng.integers(0, 2, (N_BOOT, len(net))) * 2 - 1
    perm = (flips * net).sum(axis=1)
    p_perm = float((perm >= obs).mean())
    per_backbone = {}
    for c in cells:
        t = c["tag"]
        rs = [r for r in recs if r[5] == t]
        per_backbone[t] = {"bad": sum(r[1] for r in rs),
                           "good": sum(r[2] for r in rs),
                           "voi_gain": sum(r[3] for r in rs),
                           "voi_loss": sum(r[4] for r in rs),
                           "n": len(rs)}
    return {
        "n_records": len(recs), "n_questions": len(keys),
        "reachability": {"bad": bad, "good": good,
                         "diff_ci95": [int(np.percentile(diffs, 2.5)),
                                       int(np.percentile(diffs, 97.5))],
                         "ratio_ci95": [round(float(np.percentile(ratios, 2.5)), 2),
                                        round(float(np.percentile(ratios, 97.5)), 2)],
                         "p_signflip_clustered": p_perm},
        "controller_realised": {"voi_gain": vgain, "voi_loss": vloss},
        "per_backbone": per_backbone,
    }


# --------------------------------------------------------------------------- #
# Historical certificate diagnostic (superseded)
# --------------------------------------------------------------------------- #
def rq4_cell(cell, d, alphas=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), delta=0.10):
    """Legacy cross-fit/full-fit score-mismatched diagnostic, not a guarantee.

    Historical runs did not retain raw prefix features, so their test states
    cannot be rescored by a critic fitted before dedicated calibration.  This
    function is preserved only to audit the old numbers and its output is
    explicitly quarantined below.  Formal certificate evidence comes only
    from fresh raw-trajectory runs with a dedicated calibration split.
    """
    Q = d["per_q"]
    test_stops = cell["_stops"]
    X, y_vtg, y_err, chains = load_chains(cell["tag"])
    train_value, train_errhat = oof_scores(X, y_vtg, y_err, chains)
    train_stops = []
    for chain in chains:
        stop = chain[-1]
        for i in chain[:-1]:
            if train_value[i] < cell["lambda"]:
                stop = i
                break
        train_stops.append(stop)
    cal = [(train_errhat[i], int(y_err[i])) for i in train_stops]
    tst = [(Q[j]["prefix_errhat"][test_stops[j]],
            1 - Q[j]["prefix_em"][test_stops[j]]) for j in range(len(Q))]
    curve = []
    for alpha in alphas:
        th, nsel = sgr_threshold([u[0] for u in cal], [u[1] for u in cal],
                                 alpha, delta)
        if th is None:
            curve.append({"alpha": alpha, "fired": False,
                          "n_cal": len(cal)})
            continue
        sel = [e for (eh, e) in tst if eh <= th]
        curve.append({"alpha": alpha, "fired": True,
                      "q_alpha": round(th, 3),
                      "coverage": round(len(sel) / len(tst), 2),
                      "risk": round(float(np.mean(sel)), 3) if sel else 0.0,
                      "n_sel": len(sel), "n_cal": len(cal),
                      "n_cal_selected": nsel})
    return curve


# --------------------------------------------------------------------------- #
# Frozen zero-shot transfer
# --------------------------------------------------------------------------- #
def transfer_frozen(a, b, lam_a, boots=2000):
    Xa, ya, ea, cha = load_chains(a)
    Xb, yb, eb, chb = load_chains(b)
    val_a, _ = refit_critic(Xa, ya, ea)
    scores = val_a.predict(Xb)
    em, s = replay_chains(scores, Xb, eb, chb, lam_a, "lt")
    ref_em, ref_s = chain_ref(Xb, eb, chb)
    rng = np.random.default_rng(5)
    ems, diffs = [], []
    per_chain_ref = ref_em
    for _ in range(boots):
        smp = rng.integers(0, len(chb), len(chb))
        ems.append(100 * em[smp].mean())
        diffs.append(100 * (em[smp] - per_chain_ref[smp]).mean())
    return {"from": a, "to": b, "lambda_source": lam_a,
            "em": round(100 * em.mean(), 1),
            "ci": [round(float(np.percentile(ems, 2.5)), 1),
                   round(float(np.percentile(ems, 97.5)), 1)],
            "s": round(s.mean(), 2),
            "save": round(100 * (1 - s.mean() / ref_s.mean())),
            "ref_em": round(100 * ref_em.mean(), 1),
            "ref_s": round(ref_s.mean(), 2),
            "paired_diff_ci": [round(float(np.percentile(diffs, 2.5)), 1),
                               round(float(np.percentile(diffs, 97.5)), 1)]}


# --------------------------------------------------------------------------- #
def main():
    out = {"protocol": {
        "band_em": BAND,
        "rule": "cheapest grid point with train EM >= train ReAct EM - band; "
                "fallback argmax train EM (ties: fewest searches); all "
                "selection on training chains only, frozen before test",
        "n_boot": N_BOOT}}

    runs = {}
    cells = []
    for tag in HOTPOT_TAGS:
        cell, d = run_cell(tag)
        runs[tag] = d
        cells.append(cell)
        print(f"[{tag}] lam*={cell['lambda']} ({cell['lambda_meta']['rule']}) "
              f"tau*={cell['tau']} B*={cell['b_star']} | "
              f"VoI {cell['voi_frozen']['em']}@{cell['voi_frozen']['s']} "
              f"({cell['voi_frozen']['save_pct']}%) vs ReAct "
              f"{cell['react']['em']}@{cell['react']['s']} | "
              f"dEM {cell['paired_voi_vs_react']['mean_diff']} "
              f"CI {cell['paired_voi_vs_react']['ci95']}")

    # global lambda over pooled hotpot training chains
    pooled = {"scores": [], "X": [], "y": [], "chains": []}
    off = 0
    for tag in HOTPOT_TAGS:
        X, y_vtg, y_err, chains = load_chains(tag)
        sc, _ = oof_scores(X, y_vtg, y_err, chains)
        pooled["scores"].append(sc)
        pooled["X"].append(X)
        pooled["y"].append(y_err)
        pooled["chains"] += [[i + off for i in c] for c in chains]
        off += len(X)
    Xp = np.vstack(pooled["X"])
    sp = np.concatenate(pooled["scores"])
    yp = np.concatenate(pooled["y"])
    grid = [float(x) for x in runs[HOTPOT_TAGS[0]]["lam_grid"]]
    lam_g, lam_g_meta = select_threshold(sp, Xp, yp, pooled["chains"], grid)
    out["global_lambda"] = {"lambda": lam_g, "meta": lam_g_meta}
    print(f"[global] lam_g={lam_g} ({lam_g_meta['rule']})")
    glob_cells = {}
    for tag in HOTPOT_TAGS:
        cell_g, _ = run_cell(tag, lam_override=lam_g)
        glob_cells[tag] = {
            "em": cell_g["voi_frozen"]["em"], "s": cell_g["voi_frozen"]["s"],
            "save_pct": cell_g["voi_frozen"]["save_pct"],
            "paired": cell_g["paired_voi_vs_react"]}
    out["global_lambda"]["cells"] = glob_cells

    # dataset cells (per-cell frozen lambda)
    ds_cells = []
    for tag in DATASET_TAGS:
        cell, d = run_cell(tag)
        runs[tag] = d
        ds_cells.append(cell)
        print(f"[{tag}] lam*={cell['lambda']} VoI "
              f"{cell['voi_frozen']['em']}@{cell['voi_frozen']['s']} "
              f"({cell['voi_frozen']['save_pct']}%) vs "
              f"{cell['react']['em']}@{cell['react']['s']}")

    out["flips"] = flips_clustered(cells, runs)
    f = out["flips"]
    print(f"[flips] reach bad/good {f['reachability']['bad']}/"
          f"{f['reachability']['good']} p={f['reachability']['p_signflip_clustered']:.2e} "
          f"| realised gain/loss {f['controller_realised']['voi_gain']}/"
          f"{f['controller_realised']['voi_loss']}")

    out["certificate_status"] = (
        "Historical runs lack raw prefix features and cannot support a "
        "deployment-matched dedicated-split certificate. The following "
        "cross-fit/full-fit score-mismatched values are quarantined and are "
        "not used in the paper; formal evidence is in corpus_confirmation.json."
    )
    out["rq4_historical_score_mismatch"] = {
        c["tag"]: rq4_cell(c, runs[c["tag"]]) for c in cells}

    pairs = [("final_haiku", "final_qwen3b"), ("final_qwen3b", "final_haiku"),
             ("final_llama3b", "final_commandr35b"),
             ("final_commandr35b", "final_llama3b")]
    lam_by = {c["tag"]: c["lambda"] for c in cells}
    out["transfer"] = [transfer_frozen(a, b, lam_by[a]) for a, b in pairs]
    for t in out["transfer"]:
        print(f"[transfer] {t['from']}->{t['to']} lam={t['lambda_source']} "
              f"em={t['em']} {t['ci']} s={t['s']} save={t['save']}% "
              f"ref {t['ref_em']}@{t['ref_s']} dCI {t['paired_diff_ci']}")

    # strip bulky per-question arrays before saving
    for c in cells + ds_cells:
        c.pop("_voi_em"), c.pop("_react_em"), c.pop("_stops")
    out["hotpot_cells"] = cells
    out["dataset_cells"] = ds_cells

    (MEAS / "frozen_protocol.json").write_text(json.dumps(out, indent=1))
    print(f"[wrote] {MEAS / 'frozen_protocol.json'}")

    # --- tex fragments -----------------------------------------------------
    TABDIR.mkdir(exist_ok=True)
    rows = []
    for c in cells:
        p = c["paired_voi_vs_react"]
        rows.append(
            f"{PRETTY[c['tag']]} & {c['n']} & {c['closed_book_em']:.0f} & "
            f"{c['react']['em']:.1f}\\,[{c['react']['em_ci'][0]:.1f},{c['react']['em_ci'][1]:.1f}] & "
            f"{c['voi_frozen']['em']:.1f}\\,[{c['voi_frozen']['em_ci'][0]:.1f},{c['voi_frozen']['em_ci'][1]:.1f}] & "
            f"{p['mean_diff']:+.1f}\\,[{p['ci95'][0]:.1f},{p['ci95'][1]:.1f}] & "
            f"{c['voi_frozen']['s']:.2f} & {c['voi_frozen']['save_pct']}\\% \\\\")
    (TABDIR / "backbones_frozen_rows.tex").write_text("\n".join(rows))

    SHORT = {"final_haiku": "Haiku", "final_sonnet45": "Sonnet",
             "final_qwen3b": "Qwen-3B", "final_llama3b": "Llama-3B",
             "final_commandr35b": "Cmd-R"}
    rows = []
    for t in out["transfer"]:
        A = SHORT[t["from"]]
        B = SHORT[t["to"]]
        rows.append(f"{A} $\\to$ {B} & {t['ref_em']:.1f} & "
                    f"{t['em']:.1f}\\,[{t['ci'][0]},{t['ci'][1]}] & "
                    f"{t['s']:.2f} & {t['save']}\\% \\\\")
    (TABDIR / "transfer_frozen_rows.tex").write_text("\n".join(rows))

    rows = []
    name = {"final2wiki_qwen3b": ("2Wiki", "Qwen2.5-3B"),
            "rev2_musique_qwen3b": ("MuSiQue", "Qwen2.5-3B"),
            "rev2_musique_llama3b": ("MuSiQue", "Llama-3.2-3B")}
    for c in ds_cells:
        ds, bb = name[c["tag"]]
        rows.append(f"{ds} & {bb} & {c['n']} & {c['react']['em']:.1f} & "
                    f"{c['react']['s']:.1f} & {c['voi_frozen']['em']:.1f} & "
                    f"{c['voi_frozen']['s']:.2f} & "
                    f"{c['voi_frozen']['save_pct']}\\% \\\\")
    (TABDIR / "datasets_frozen_rows.tex").write_text("\n".join(rows))
    print(f"[wrote] tex fragments in {TABDIR}")


if __name__ == "__main__":
    main()
