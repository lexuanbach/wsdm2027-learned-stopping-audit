"""Round-9 analyses: offline replays and re-fits on the frozen measured runs.

No new LLM calls.  Every block reads results/measured/ and writes
results/measured/round9_analyses.json plus LaTeX fragments in
outputs/tables/round9_*.tex.  Blocks:

1. pooled_pareto  -- per fresh corpus-index cell, the nondominated set after
   pooling every fixed budget, answer repetition, full TASR (where logits
   exist), the NLI-augmented guard, a same-feature stop classifier, the
   sufficiency-style gate, Stop-RAG (where trained on that cell's logs),
   FBG value-only and FBG + repeat guard, all on the (EM, exact LLM calls)
   plane; dominators of each learned point; margin passes at 1/2/3 EM.
2. epsilon_dominance -- the W/T/L verdict of the executive matrix recomputed
   under a frozen epsilon-dominance rule (EM differences within epsilon are
   ties) for epsilon in {0, 0.5, 1, 2} over every cell.
3. equal_cost -- the learned points compared with the randomized
   simple-family frontier (upper concave hull of fixed budgets and answer
   repetition) at EQUAL mean exact LLM calls, input tokens, and wall
   latency, with per-question paired bootstrap CIs against the matching
   mixture policy.
4. target_variants -- the value head retrained on cost-adjusted targets
   (net future gain minus c per extra search; discounted future gain;
   one-step gain), plus a same-feature stop classifier and the
   sufficiency-style gate, each taken through the frozen selection protocol
   and replayed on the untouched test questions.
5. selection_sensitivity -- threshold grid, selection band, cross-fitting
   folds and seed, and certificate-calibration split fraction varied one at
   a time; selected lambda and test outcome per configuration.
6. joint_ablation -- jointly removing step, evidence-count, and stability
   proxies from the critic features (frozen selection, test replay).

Nothing selects a cell, seed, threshold, or policy by test outcome; every
configuration is reported.  Seeds are fixed; GBR fits use random_state so
the file is bit-reproducible.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import numpy as np
from display_numbers import display
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

import run_experiment as rx
from voi_controller import guarded_stop_index, stop_index
from analyze_corpus_confirmation import cumulative_cost
import frontier_analyses as mra

MEAS = mra.MEAS
TABLES = mra.TABLES
GRID6 = [0.01, 0.02, 0.04, 0.07, 0.12, 0.20]
GRID12 = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.16, 0.20, 0.30]
TAU_GRID = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
CLF_GRID = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
MARGINS = [1.0, 2.0, 3.0]
EPSILONS = [0.0, 0.5, 1.0, 2.0]
FEATURES = ["searches", "evidence_size", "question_coverage", "new_evidence",
            "answer_changed", "answer_length", "answer_grounding",
            "stable_two", "distinct_answers"]
N_BOOT = 10_000

# Additional fresh cells (present only if the runs were executed).
FRESH_EXTRA = [
    ("corpus_guarded_llama3b_data17",
     "corpus_confirmation_corpus_guarded_llama3b_data17.json",
     "Corpus / Llama-3.2-3B (q-sample B)", "llama_qB"),
    ("corpus_guarded_llama3b_data23",
     "corpus_confirmation_corpus_guarded_llama3b_data23.json",
     "Corpus / Llama-3.2-3B (q-sample C)", "llama_qC"),
    ("corpus_guarded_llama3b_data29",
     "corpus_confirmation_corpus_guarded_llama3b_data29.json",
     "Corpus / Llama-3.2-3B (q-sample D)", "llama_qD"),
    ("corpus_guarded_haiku45_data17",
     "corpus_confirmation_corpus_guarded_haiku45_data17.json",
     "Corpus / Haiku 4.5 (q-sample B)", "haiku_qB"),
    ("corpus_guarded_haiku45_data23",
     "corpus_confirmation_corpus_guarded_haiku45_data23.json",
     "Corpus / Haiku 4.5 (q-sample C)", "haiku_qC"),
    ("corpus_guarded_haiku45_data29",
     "corpus_confirmation_corpus_guarded_haiku45_data29.json",
     "Corpus / Haiku 4.5 (q-sample D)", "haiku_qD"),
]
SHORT = {"corpus_s7": "A s7", "corpus_s11": "A s11", "corpus_s19": "A s19",
         "corpus_qB": "B", "corpus_qC": "C", "corpus_qD": "D", "corpus_qE": "E",
         "w2_d1": "2Wiki d1", "w2_d2": "2Wiki d2", "w2_d3": "2Wiki d3",
         "mq_d1": "MuSiQue d1", "mq_d2": "MuSiQue d2", "mq_d3": "MuSiQue d3",
         "wiki25m": "Wikipedia 25M",
         "llama_qB": "Llama B", "llama_qC": "Llama C", "llama_qD": "Llama D",
         "haiku_qB": "Haiku B", "haiku_qC": "Haiku C", "haiku_qD": "Haiku D"}
CODES = {"fixed_0": "F0", "fixed_1": "F1", "fixed_2": "F2", "fixed_3": "F3",
         "fixed_4": "F4", "answer_repeat": "Rep", "tasr": "TASR",
         "semantic_guard": "NLI", "stop_classifier": "Clf",
         "sufficiency_gate": "Err", "stoprag": "StopRAG",
         "halt_nli": "HALT$_\\mathrm{nli}$", "halt_relevance": "HALT$_\\mathrm{rel}$",
         "value_only": "V", "guarded_value": "G"}


# Cells whose Stop-RAG and verifier-inspired (HALT) replays entered the round-9
# analysis behind supplement Tables 6 to 11, 14, 18 and 19. Both baselines were
# later extended to every fresh cell; that extension is analysed separately in
# results/round10/extended_decision_matrix.json (paper Table 5, supplement
# Table 5). main() restricts the round-9 tables to these cells so that they
# match the paper. Other scripts that import Cell see every available replay.
ROUND9_STOPRAG_CELLS = frozenset({
    "corpus_guarded_qwen3b", "corpus_guarded_qwen3b_data23",
    "corpus2wiki_guarded_qwen3b_data17", "corpusmusique_guarded_qwen3b_data17",
    "wiki25m_guarded_qwen3b"})
ROUND9_HALT_CELLS = ROUND9_STOPRAG_CELLS | {"corpus_guarded_qwen3b_data29"}
RESTRICT_BASELINES = False   # set by main()


def fresh_cells(extra=False):
    """Primary 14 corpus-index cells, or the crossed backbone x population set.

    The crossed cells are kept in a separate block so that adding a backbone
    never changes the 26-cell executive matrix or any number quoted from the
    primary block.
    """
    cells = []
    for tag, fname, label, short in (FRESH_EXTRA if extra else list(mra.FRESH)):
        if (MEAS / f"run_{tag}.json").exists() and (MEAS / fname).exists():
            cells.append((tag, fname, label, short))
    return cells


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def paired_ci(d, seed=2029, n_boot=N_BOOT):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = 100 * d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return {"mean_em_points": round(100 * float(d.mean()), 3),
            "ci95": [round(float(np.percentile(means, 2.5)), 3),
                     round(float(np.percentile(means, 97.5)), 3)],
            "lower95_one_sided": round(float(np.percentile(means, 5)), 3)}


def chain_split(run, cache, fraction=None):
    """Policy-training / certificate chains (seed 41), optional fraction."""
    X = np.asarray(cache["X_val"], float)
    all_chains = rx.chain_indices(X)
    rng = np.random.default_rng(41)
    order = np.arange(len(all_chains))
    rng.shuffle(order)
    if fraction is None:
        fraction = run["config"]["certificate_calibration_fraction"]
    ncal = max(1, int(round(fraction * len(all_chains))))
    fit_ids = order[:-ncal]
    return X, all_chains, [all_chains[i] for i in fit_ids], list(fit_ids)


def crossfit(Xc, y, chains, folds, seed, kind):
    """Episode-level cross-fitted scores; kind in {'gbr','logit'}."""
    rng = np.random.default_rng(seed)
    order = np.arange(len(chains))
    rng.shuffle(order)
    out = np.zeros(len(Xc), float)
    for fold in range(folds):
        held = set(order[fold::folds].tolist())
        tr = [i for c, ch in enumerate(chains) if c not in held for i in ch]
        te = [i for c in held for i in chains[c]]
        if not tr or not te:
            continue
        out[te] = fit_predict(Xc[tr], y[tr], Xc[te], seed, kind)
    return out


def fit_predict(Xtr, ytr, Xte, seed, kind):
    if kind == "gbr":
        m = GradientBoostingRegressor(max_depth=2, n_estimators=80,
                                      random_state=seed).fit(Xtr, ytr)
        return m.predict(Xte)
    y = np.asarray(ytr).astype(int)
    if len(set(y.tolist())) < 2:
        return np.full(len(Xte), float(y.mean()))
    m = LogisticRegression(max_iter=1000).fit(Xtr, y)
    return m.predict_proba(Xte)[:, list(m.classes_).index(1)]


def replay_scores(scores, answers, lam, rule):
    """Per-question stop indices for a score matrix (n x width)."""
    if rule == "guarded":
        return np.asarray([guarded_stop_index(s, a, lam)
                           for s, a in zip(scores, answers)], int)
    return np.asarray([stop_index(s, lam) for s in scores], int)


def err_stop_index(errhat, tau):
    for t in range(len(errhat) - 1):
        if errhat[t] <= tau:
            return t
    return len(errhat) - 1


class Cell:
    """Everything needed to replay policies on one fresh corpus-index run."""

    def __init__(self, tag, fname, label, short):
        self.tag, self.label, self.short = tag, label, short
        self.run = json.loads((MEAS / f"run_{tag}.json").read_text())
        self.cache = pickle.loads((MEAS / f"critic_data_{tag}.pkl").read_bytes())
        self.conf = json.loads((MEAS / fname).read_text())
        self.records = self.run["per_q"]
        self.max_steps = self.run["config"]["max_steps"]
        self.em, self.feats, self.answers, self.rep_stop = mra.test_arrays(self.records)
        self.n, self.width = self.em.shape
        self.last = self.width - 1
        self.full_em = self.em[:, self.last]
        self.X, self.all_chains, self.fit_chains, self.fit_ids = chain_split(
            self.run, self.cache)
        self.y_val = np.asarray(self.cache["y_val"], float)
        self.y_err = np.asarray(self.cache["y_err"], int)
        self.traces = self.cache["traces"]
        self.lam_logged = float(self.run["lam_main"])
        self.g_stop = np.asarray([int(r["primary"]["searches"]) for r in self.records])
        self.v_stop = np.asarray([int(r["frozen_value"]["searches"]) for r in self.records])
        # per-prefix cumulative costs (calls, input tokens, wall latency)
        self.cost = np.zeros((self.n, self.width, 3), float)
        for i, r in enumerate(self.records):
            for t in range(self.width):
                c = cumulative_cost(r, t)
                self.cost[i, t] = [c["calls"], c["in"],
                                   c["latency_seconds"] + c["retrieval_latency_seconds"]]
        self.bstar = int(self.conf["training_selected_fixed_budget"])
        # optional replays
        self.tasr = None
        p = MEAS / f"tasr_{tag}.json"
        if p.exists():
            t = json.loads(p.read_text())
            rows = {str(x["id"]): x for x in t["per_q"]}
            self.tasr = [rows[str(r["id"])] for r in self.records]
        self.sem_stop = None
        p = MEAS / f"semantic_guard_{tag}.json"
        if p.exists():
            s = json.loads(p.read_text())
            rows = {str(x["id"]): int(x["stop"]) for x in s["per_q"]}
            self.sem_stop = np.asarray([rows[str(r["id"])] for r in self.records])
        self.halt = None
        p = MEAS / f"halt_{tag}.json"
        if p.exists() and (not RESTRICT_BASELINES or tag in ROUND9_HALT_CELLS):
            h = json.loads(p.read_text())
            if h["n_test"] == self.n:
                self.halt = h
        self.stoprag = None
        p = MEAS / f"stoprag_result_{tag}.json"
        if p.exists() and (not RESTRICT_BASELINES or tag in ROUND9_STOPRAG_CELLS):
            s = json.loads(p.read_text())
            if "test_per_q" in s and len(s["test_per_q"]) == self.n:
                self.stoprag = s

    # ---- policy points ----------------------------------------------- #
    def point(self, stops, em_override=None):
        stops = np.asarray(stops, int)
        em = (self.em[np.arange(self.n), stops] if em_override is None
              else np.asarray(em_override, float))
        cost = self.cost[np.arange(self.n), stops]
        return {"em_vec": em, "em": 100 * float(em.mean()),
                "calls": float(cost[:, 0].mean()),
                "tokens": float(cost[:, 1].mean()),
                "latency": float(cost[:, 2].mean()),
                "cost_vec": cost, "stops": stops}

    def simple_points(self):
        pts = {f"fixed_{k}": self.point(np.full(self.n, min(k, self.last)))
               for k in range(self.max_steps + 1)}
        pts["answer_repeat"] = self.point(self.rep_stop)
        return pts

    def learned_points(self):
        pts = {"value_only": self.point(self.v_stop),
               "guarded_value": self.point(self.g_stop)}
        if self.tasr is not None:
            stops = [int(x["stop"]) for x in self.tasr]
            pts["tasr"] = self.point(stops, em_override=[x["em"] for x in self.tasr])
        if self.sem_stop is not None:
            pts["semantic_guard"] = self.point(self.sem_stop)
        if self.stoprag is not None:
            pts["stoprag"] = self.point([int(x["stop"]) for x in self.stoprag["test_per_q"]])
        if self.halt is not None:
            for name, variant in self.halt["variants"].items():
                rows = {r["id"]: r for r in variant["per_q"]}
                stops = np.asarray([int(rows[str(r["id"])]["stop"])
                                    for r in self.records], int)
                pt = self.point(stops)
                # one extra claim-generation call per question, charged in
                # calls; its tokens/latency are not logged per question, so the
                # token and wall columns understate HALT (in its favour).
                pt["calls"] = float(pt["calls"] + 1.0)
                pt["cost_vec"] = pt["cost_vec"].copy()
                pt["cost_vec"][:, 0] += 1.0
                pts[f"halt_{name}"] = pt
        return pts

    # ---- frozen pipeline on a target / feature subset / config --------- #
    def pipeline(self, *, cols=None, target="fbg", grid=GRID6, band=1.0,
                 folds=5, xseed=3, fraction=None, rule="guarded",
                 head="gbr", select_grid=None):
        """Cross-fit -> frozen selection -> refit -> test replay.

        target: 'fbg' (logged y_val), 'net_<c>', 'disc_<g>', 'onestep',
        'stopclf' (label FBG>0, logistic head), 'suff' (error head,
        stop when errhat <= tau).  Returns the selected threshold and per-
        question stop indices for the guarded and value-only rules (or the
        single rule for classifier/gate families).
        """
        if fraction is None:
            X, chains, fit_ids = self.X, self.fit_chains, self.fit_ids
        else:
            X, _all, chains, fit_ids = chain_split(self.run, self.cache, fraction)
        cols = list(range(X.shape[1])) if cols is None else cols
        y = self.targets()[target] if target not in ("stopclf", "suff") else None
        if target == "stopclf":
            y = (self.y_val > 1e-9).astype(int)
            head = "logit"
        if target == "suff":
            y = self.y_err
            head = "logit"
        fit_rows = [i for ch in chains for i in ch]
        # re-index the chains onto the fit-row subset
        pos = {i: k for k, i in enumerate(fit_rows)}
        chains_f = [[pos[i] for i in ch] for ch in chains]
        Xf = X[fit_rows][:, cols]
        yf = y[fit_rows]
        Xfull_f = X[fit_rows]
        oof = crossfit(Xf, yf, chains_f, folds, xseed, head)
        if target == "suff":
            g = TAU_GRID if select_grid is None else select_grid
            th, meta = select_le(oof, Xfull_f, self.y_err[fit_rows], chains_f, g, band)
        else:
            g = (CLF_GRID if target == "stopclf" else grid) if select_grid is None else select_grid
            th, meta = rx.select_frozen_lambda(oof, Xfull_f, self.y_err[fit_rows],
                                               chains_f, g, rule if target not in ("stopclf",) else "value",
                                               band=band)
        test_scores = fit_predict(Xf, yf, self.feats.reshape(-1, self.feats.shape[-1])[:, cols],
                                  xseed, head).reshape(self.n, self.width)
        out = {"threshold": float(th), "selection_reason": meta["reason"],
               "train_reference_em": round(float(meta["train_reference_em"]), 3),
               "n_policy_training_episodes": len(chains)}
        if target == "suff":
            out["stops"] = {"gate": np.asarray([err_stop_index(s, th) for s in test_scores], int)}
        elif target == "stopclf":
            out["stops"] = {"classifier": replay_scores(test_scores, self.answers, th, "value")}
        else:
            out["stops"] = {"guarded": replay_scores(test_scores, self.answers, th, "guarded"),
                            "value": replay_scores(test_scores, self.answers, th, "value")}
        return out

    def targets(self):
        if hasattr(self, "_targets"):
            return self._targets
        T = {"fbg": self.y_val.copy()}
        for name in ["net_0.02", "net_0.04", "net_0.08", "disc_0.5", "disc_0.8",
                     "onestep", "fbg_recomputed"]:
            T[name] = np.zeros(len(self.y_val), float)
        row = 0
        for ch, tr in zip(self.all_chains, self.traces):
            u = np.asarray(tr["corr"], float)
            assert len(u) == len(ch), (self.tag, len(u), len(ch))
            for t in range(len(ch)):
                fut = u[t + 1:]
                steps = np.arange(1, len(fut) + 1)
                i = ch[t]
                T["fbg_recomputed"][i] = max(float(np.max(fut - u[t])), 0.0) if len(fut) else 0.0
                for c in [0.02, 0.04, 0.08]:
                    T[f"net_{c}"][i] = (max(float(np.max(fut - u[t] - c * steps)), 0.0)
                                        if len(fut) else 0.0)
                for gmm in [0.5, 0.8]:
                    T[f"disc_{gmm}"][i] = (max(float(np.max((gmm ** steps) * (fut - u[t]))), 0.0)
                                           if len(fut) else 0.0)
                T["onestep"][i] = max(float(fut[0] - u[t]), 0.0) if len(fut) else 0.0
            row += len(ch)
        self.fbg_target_max_abs_dev = float(np.max(np.abs(T["fbg_recomputed"] - T["fbg"])))
        self._targets = T
        return T


def select_le(scores, X, y_err, chains, grid, band):
    """Band rule for the sufficiency gate (stop when errhat <= tau)."""
    reference = 100 * np.mean([1 - y_err[ch[-1]] for ch in chains])
    frontier = []
    for th in grid:
        stops = []
        for ch in chains:
            stop = ch[-1]
            for i in ch[:-1]:
                if scores[i] <= th:
                    stop = i
                    break
            stops.append(stop)
        frontier.append({"tau": th,
                         "em": 100 * float(np.mean([1 - y_err[i] for i in stops])),
                         "searches": float(np.mean([X[i][0] for i in stops]))})
    eligible = [x for x in frontier if x["em"] >= reference - band]
    if eligible:
        picked = min(eligible, key=lambda x: (x["searches"], -x["em"]))
        reason = "cheapest_within_band"
    else:
        best = max(x["em"] for x in frontier)
        picked = min((x for x in frontier if x["em"] == best), key=lambda x: x["searches"])
        reason = "fallback_max_em"
    return float(picked["tau"]), {"reason": reason, "train_reference_em": reference,
                                  "frontier": frontier}


# --------------------------------------------------------------------------- #
# dominance helpers
# --------------------------------------------------------------------------- #
def eps_dominates(em_s, c_s, em_p, c_p, eps):
    """S epsilon-dominates P on (EM, calls): EM within eps counts as a tie."""
    em_ge = em_s >= em_p - eps - 1e-12
    c_le = c_s <= c_p + 1e-12
    strict = (em_s > em_p + eps + 1e-12) or (c_s < c_p - 1e-12)
    return em_ge and c_le and strict


def verdict_eps(em, calls, simple_named, bstar_key, eps):
    dominators = [k for k, (e, c) in simple_named.items()
                  if eps_dominates(e, c, em, calls, eps)]
    if dominators:
        return "L", dominators
    e_b, c_b = simple_named[bstar_key]
    if eps_dominates(em, calls, e_b, c_b, eps):
        return "W", []
    return "T", []


def nondominated(points, key=("em", "calls")):
    names = list(points)
    nd = []
    for a in names:
        ea, ca = points[a][key[0]], points[a][key[1]]
        dominated = any(
            (points[b][key[0]] >= ea - 1e-12 and points[b][key[1]] <= ca + 1e-12
             and (points[b][key[0]] > ea + 1e-12 or points[b][key[1]] < ca - 1e-12))
            for b in names if b != a)
        if not dominated:
            nd.append(a)
    return nd


def upper_hull(points):
    """Upper concave hull of (cost, em) mean points; returns ordered vertices."""
    pts = sorted({(round(p["cost"], 9), round(p["em"], 9), name)
                  for name, p in points.items()})
    hull = []
    for x, y, name in pts:
        while len(hull) >= 2:
            (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
            # remove middle point if it lies below the segment (not concave)
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= -1e-12:
                hull.pop()
            else:
                break
        hull.append((x, y, name))
    # drop trailing vertices with lower EM than an earlier cheaper vertex
    out = []
    best = -1
    for x, y, name in hull:
        if y > best + 1e-12:
            out.append((x, y, name))
            best = y
    return out


def hull_mixture(hull, cost, points):
    """Per-question EM vector of the randomized hull policy at mean cost."""
    if cost <= hull[0][0] + 1e-12:
        return points[hull[0][2]]["em_vec"], hull[0][2], hull[0][2], 1.0
    for (x1, y1, n1), (x2, y2, n2) in zip(hull, hull[1:]):
        if x1 - 1e-12 <= cost <= x2 + 1e-12:
            w = (x2 - cost) / (x2 - x1) if x2 > x1 else 1.0
            return (w * points[n1]["em_vec"] + (1 - w) * points[n2]["em_vec"],
                    n1, n2, float(w))
    return points[hull[-1][2]]["em_vec"], hull[-1][2], hull[-1][2], 1.0


# --------------------------------------------------------------------------- #
# per-cell analyses
# --------------------------------------------------------------------------- #
def analyse_cell(cell: Cell, seed=2029):
    simple = cell.simple_points()
    learned = cell.learned_points()
    # additional learned alternatives from the frozen pipeline
    clf = cell.pipeline(target="stopclf")
    suff = cell.pipeline(target="suff")
    learned["stop_classifier"] = cell.point(clf["stops"]["classifier"])
    learned["sufficiency_gate"] = cell.point(suff["stops"]["gate"])
    allpts = {**simple, **learned}
    bstar_key = f"fixed_{cell.bstar}"
    simple_named = {k: (v["em"], v["calls"]) for k, v in simple.items()}

    # 1. pooled Pareto ------------------------------------------------------
    nd = nondominated(allpts)
    pooled = {
        "nondominated": nd,
        "n_candidates": len(allpts),
        "candidates": sorted(allpts),
        "points": {k: {"em": round(v["em"], 3), "calls": round(v["calls"], 3),
                       "input_tokens": round(v["tokens"], 1),
                       "wall_seconds": round(v["latency"], 3)}
                   for k, v in allpts.items()},
        "fbg_value_nondominated": "value_only" in nd,
        "fbg_guarded_nondominated": "guarded_value" in nd,
        "any_fbg_nondominated": ("value_only" in nd) or ("guarded_value" in nd),
        "learned_nondominated": [k for k in nd if k in learned],
        "simple_nondominated": [k for k in nd if k in simple],
        "dominators": {k: [o for o in allpts if o != k and
                           (allpts[o]["em"] >= allpts[k]["em"] - 1e-12
                            and allpts[o]["calls"] <= allpts[k]["calls"] + 1e-12
                            and (allpts[o]["em"] > allpts[k]["em"] + 1e-12
                                 or allpts[o]["calls"] < allpts[k]["calls"] - 1e-12))]
                       for k in learned},
        "stop_classifier_threshold": clf["threshold"],
        "sufficiency_gate_tau": suff["threshold"],
    }
    # what won where: cheapest policy whose EM is within b points of the full
    # budget (ties -> higher EM, then name); learned policies compete with
    # every simple and learned alternative under one rule.
    full_em_pct = 100 * float(cell.full_em.mean())
    winners = {}
    for b in [1.0, 2.0, 3.0]:
        elig = [(v["calls"], -v["em"], k) for k, v in allpts.items()
                if v["em"] >= full_em_pct - b - 1e-12]
        elig.sort()
        winners[str(b)] = {"winner": elig[0][2], "calls": round(elig[0][0], 3),
                           "em": round(-elig[0][1], 3),
                           "cheapest_learned": next(((round(c, 3), k) for c, _e, k in elig
                                                     if k in learned), None)}
    pooled["band_winners"] = winners
    # margins
    margins = {}
    for pol in ["guarded_value", "value_only"]:
        pc = paired_ci(allpts[pol]["em_vec"] - cell.full_em, seed=seed)
        margins[pol] = {**pc,
                        "pass_at_margin": {str(m): bool(pc["lower95_one_sided"] > -m)
                                           for m in MARGINS},
                        "reduces_calls": bool(allpts[pol]["calls"] < allpts["fixed_%d" % cell.last]["calls"]),
                        "smallest_passing_margin_em": round(-pc["lower95_one_sided"], 3)}
    pooled["margin_sensitivity"] = margins
    # power of the paired non-inferiority test at the observed contrast SD
    # (true difference 0, one-sided alpha 0.05): P(lower bound > -m)
    from scipy.stats import norm as _norm
    d = allpts["guarded_value"]["em_vec"] - cell.full_em
    sd = float(np.std(d, ddof=1))
    se = 100 * sd / np.sqrt(len(d))
    pooled["margin_power"] = {
        "contrast_sd_fraction": round(sd, 4), "se_em_points": round(se, 3),
        "power_at_margin": {str(m): round(float(_norm.cdf(m / se - _norm.ppf(0.95))), 3)
                            for m in MARGINS},
        "margin_with_80pct_power_em": round(float(se * (_norm.ppf(0.95) + _norm.ppf(0.80))), 3),
        "note": "normal approximation; true difference 0; n = test questions"}

    # 2. epsilon-dominance verdicts -------------------------------------
    eps_v = {}
    for eps in EPSILONS:
        eps_v[str(eps)] = {}
        for pol in ["value_only", "guarded_value"]:
            v, doms = verdict_eps(allpts[pol]["em"], allpts[pol]["calls"],
                                  simple_named, bstar_key, eps)
            eps_v[str(eps)][pol] = {"verdict": v, "dominated_by": doms}

    # 3. equal-cost comparison -----------------------------------------
    equal = {}
    for coord, key in [("calls", "calls"), ("input_tokens", "tokens"),
                       ("wall_seconds", "latency")]:
        pts = {k: {"cost": v[key], "em": v["em"], "em_vec": v["em_vec"]}
               for k, v in simple.items()}
        hull = upper_hull(pts)
        equal[coord] = {"hull_vertices": [{"policy": n, "cost": round(x, 3),
                                           "em": round(y, 3)} for x, y, n in hull]}
        for pol in ["guarded_value", "value_only", "tasr", "stop_classifier",
                    "sufficiency_gate", "stoprag"]:
            if pol not in allpts:
                continue
            c = allpts[pol][key]
            mix, n1, n2, w = hull_mixture(hull, c, pts)
            pc = paired_ci(allpts[pol]["em_vec"] - mix, seed=seed)
            # horizontal gap: cheapest hull cost reaching the learned EM
            em_l = allpts[pol]["em"]
            cost_at_em = None
            for (x1, y1, _), (x2, y2, _) in zip(hull, hull[1:]):
                if y1 - 1e-12 <= em_l <= y2 + 1e-12:
                    cost_at_em = x1 + (x2 - x1) * (em_l - y1) / (y2 - y1) if y2 > y1 else x1
                    break
            if cost_at_em is None:
                cost_at_em = hull[0][0] if em_l <= hull[0][1] else None
            equal[coord][pol] = {
                "learned_cost": round(c, 3), "learned_em": round(em_l, 3),
                "mixture": {"of": [n1, n2], "weight_first": round(w, 4),
                            "em": round(100 * float(mix.mean()), 3)},
                "delta_em_vs_mixture_at_equal_cost": pc,
                "pareto_improvement_at_equal_cost": bool(pc["ci95"][0] > 0),
                "hull_cost_at_learned_em": (round(cost_at_em, 3)
                                            if cost_at_em is not None else None),
                "cost_saving_vs_hull_at_equal_em": (
                    round(cost_at_em - c, 3) if cost_at_em is not None else None),
            }

    # 4. target variants ------------------------------------------------
    variants = {}
    logged = {"threshold": cell.lam_logged, "selection_reason": "logged_run",
              "n_policy_training_episodes": len(cell.fit_chains),
              "stops": {"guarded": cell.g_stop, "value": cell.v_stop}}
    for tname in ["fbg", "fbg_refit", "net_0.02", "net_0.04", "net_0.08", "disc_0.5",
                  "disc_0.8", "onestep"]:
        r = logged if tname == "fbg" else cell.pipeline(target="fbg" if tname == "fbg_refit" else tname)
        entry = {"threshold": r["threshold"], "selection_reason": r["selection_reason"]}
        for rule in ["guarded", "value"]:
            p = cell.point(r["stops"][rule])
            v, doms = verdict_eps(p["em"], p["calls"], simple_named, bstar_key, 0.0)
            entry[rule] = {"em": round(p["em"], 3), "calls": round(p["calls"], 3),
                           "verdict": v, "dominated_by": doms,
                           "paired_vs_full": paired_ci(p["em_vec"] - cell.full_em, seed=seed)}
        if tname == "fbg_refit":
            entry["reproduces_logged_lambda"] = bool(abs(r["threshold"] - cell.lam_logged) < 1e-12)
            entry["guarded_stop_agreement_with_logged"] = round(float(np.mean(
                r["stops"]["guarded"] == cell.g_stop)), 4)
        variants[tname] = entry
    for tname, rulekey in [("stopclf", "classifier"), ("suff", "gate")]:
        r = clf if tname == "stopclf" else suff
        p = cell.point(r["stops"][rulekey])
        v, doms = verdict_eps(p["em"], p["calls"], simple_named, bstar_key, 0.0)
        variants[tname] = {"threshold": r["threshold"], "selection_reason": r["selection_reason"],
                           rulekey: {"em": round(p["em"], 3), "calls": round(p["calls"], 3),
                                     "verdict": v, "dominated_by": doms,
                                     "paired_vs_full": paired_ci(p["em_vec"] - cell.full_em, seed=seed)}}
    variants["fbg_target_recomputation_max_abs_dev"] = round(cell.fbg_target_max_abs_dev, 9)

    # 5. selection sensitivity -------------------------------------------
    configs = [("base", None), ("refit_s3", {}), ("grid12", {"grid": GRID12}),
               ("band0.5", {"band": 0.5}), ("band2.0", {"band": 2.0}),
               ("folds3", {"folds": 3}), ("folds10", {"folds": 10}),
               ("xseed11", {"xseed": 11}), ("xseed29", {"xseed": 29}),
               ("cal0.20", {"fraction": 0.20}), ("cal0.33", {"fraction": 0.33})]
    sens = {}
    for name, kw in configs:
        r = logged if kw is None else cell.pipeline(**kw)
        p = cell.point(r["stops"]["guarded"])
        pv = cell.point(r["stops"]["value"])
        v, doms = verdict_eps(p["em"], p["calls"], simple_named, bstar_key, 0.0)
        vv, _ = verdict_eps(pv["em"], pv["calls"], simple_named, bstar_key, 0.0)
        pc = paired_ci(p["em_vec"] - cell.full_em, seed=seed)
        sens[name] = {"lambda": r["threshold"], "n_policy_training_episodes": r["n_policy_training_episodes"],
                      "guarded_em": round(p["em"], 3), "guarded_calls": round(p["calls"], 3),
                      "guarded_verdict": v, "guarded_dominated_by": doms,
                      "value_em": round(pv["em"], 3), "value_calls": round(pv["calls"], 3),
                      "value_verdict": vv,
                      "guarded_minus_full": pc,
                      "passes_3em_margin": bool(pc["lower95_one_sided"] > -3.0)}
    g_ems = [s["guarded_em"] for s in sens.values()]
    g_calls = [s["guarded_calls"] for s in sens.values()]
    sens_summary = {"n_configs": len(sens),
                    "lambda_values": sorted({s["lambda"] for s in sens.values()}),
                    "guarded_em_range": [round(min(g_ems), 3), round(max(g_ems), 3)],
                    "guarded_calls_range": [round(min(g_calls), 3), round(max(g_calls), 3)],
                    "n_guarded_dominated": int(sum(s["guarded_verdict"] == "L" for s in sens.values())),
                    "n_guarded_win": int(sum(s["guarded_verdict"] == "W" for s in sens.values())),
                    "n_pass_3em": int(sum(s["passes_3em_margin"] for s in sens.values()))}

    # 6. joint ablation ---------------------------------------------------
    ablation_sets = {
        "full": list(range(9)),
        "no_step_evidence_stability": [2, 3, 5, 6],
        "answer_content_only": [2, 5, 6],
        "no_stability": [0, 1, 2, 3, 5, 6],
        "no_step_no_evidence_size": [2, 3, 4, 5, 6, 7, 8],
    }
    abl = {}
    for name, cols in ablation_sets.items():
        r = cell.pipeline(cols=cols)
        entry = {"features": [FEATURES[i] for i in cols], "threshold": r["threshold"]}
        for rule in ["guarded", "value"]:
            p = cell.point(r["stops"][rule])
            v, doms = verdict_eps(p["em"], p["calls"], simple_named, bstar_key, 0.0)
            entry[rule] = {"em": round(p["em"], 3), "calls": round(p["calls"], 3),
                           "verdict": v, "dominated_by": doms,
                           "paired_vs_full": paired_ci(p["em_vec"] - cell.full_em, seed=seed)}
        abl[name] = entry

    return {"tag": cell.tag, "label": cell.label, "short": cell.short,
            "n_test": cell.n, "bstar": cell.bstar, "logged_lambda": cell.lam_logged,
            "model": cell.run["config"]["model"],
            "pooled_pareto": pooled, "epsilon_dominance": eps_v,
            "equal_cost": equal, "target_variants": variants,
            "selection_sensitivity": sens, "selection_sensitivity_summary": sens_summary,
            "joint_ablation": abl}


# --------------------------------------------------------------------------- #
# historical cells: epsilon-dominance only (replayed policies in revision JSON)
# --------------------------------------------------------------------------- #
def historical_epsilon():
    rev = json.loads((MEAS / "revision_experiments.json").read_text())
    fz = json.loads((MEAS / "frozen_protocol.json").read_text())
    bstar = {c["tag"]: c["b_star"] for c in fz["hotpot_cells"] + fz["dataset_cells"]}
    out = {}
    for c in rev["cells"]:
        p = c["policies"]
        simple_named = {f"fixed_{k}": (p[f"fixed_{k}"]["em"], p[f"fixed_{k}"]["llm_calls"])
                        for k in range(5) if f"fixed_{k}" in p}
        simple_named["answer_repeat"] = (p["answer_repeat"]["em"], p["answer_repeat"]["llm_calls"])
        bkey = f"fixed_{bstar[c['tag']]}"
        entry = {}
        for eps in EPSILONS:
            entry[str(eps)] = {}
            for pol, key in [("value_only", "value"), ("guarded_value", "guarded_value")]:
                v, doms = verdict_eps(p[key]["em"], p[key]["llm_calls"], simple_named, bkey, eps)
                entry[str(eps)][pol] = {"verdict": v, "dominated_by": doms}
        out[c["tag"]] = {"label": mra.HIST_PRETTY[c["tag"]], "epsilon_dominance": entry}
    return out


# --------------------------------------------------------------------------- #
# LaTeX fragments
# --------------------------------------------------------------------------- #
def fmt_ci(pc):
    return f"{pc['mean_em_points']:+.2f}\\,[{pc['ci95'][0]:.2f},{pc['ci95'][1]:.2f}]"


DISPLAY_VERDICT = {"W": "I", "T": "F", "L": "D"}


def write_tables(out):
    TABLES.mkdir(exist_ok=True)
    cells = out["fresh_cells"]
    # pooled what-won-where table (body): compact single-column form
    rows = []
    for c in cells:
        pp = c["pooled_pareto"]
        ndl = ", ".join(CODES.get(k, k) for k in pp["learned_nondominated"]) or "NA"
        dg = ", ".join(CODES.get(k, k) for k in pp["dominators"]["guarded_value"]) or "none"
        mg = pp["margin_sensitivity"]["guarded_value"]
        marks = "".join("\\checkmark" if mg["pass_at_margin"][str(m)] else "$\\times$"
                        for m in [1.0, 2.0, 3.0])
        e0 = c["epsilon_dominance"]["0.0"]["guarded_value"]["verdict"]
        e1 = c["epsilon_dominance"]["1.0"]["guarded_value"]["verdict"]
        bw = pp["band_winners"]
        win = "/".join(CODES.get(bw[b]["winner"], bw[b]["winner"]) for b in ["1.0", "2.0", "3.0"])
        rows.append(f"{SHORT.get(c['short'], c['short'])} & {ndl} & {dg} & {win} & "
                    f"{mg['lower95_one_sided']:+.2f}\\,{marks} & "
                    f"{DISPLAY_VERDICT[e0]}/{DISPLAY_VERDICT[e1]} \\\\")
    (TABLES / "round9_pooled_rows.tex").write_text("\n".join(rows) + "\n")

    # target variants (supplement): per cell, EM/calls/verdict for each target under the guarded rule and value rule
    names = [("fbg", "FBG (logged)"), ("fbg_refit", "FBG (refit)"), ("net_0.02", "net, $c{=}0.02$"),
             ("net_0.04", "net, $c{=}0.04$"), ("net_0.08", "net, $c{=}0.08$"),
             ("disc_0.5", "disc., $\\gamma{=}0.5$"), ("disc_0.8", "disc., $\\gamma{=}0.8$"),
             ("onestep", "one-step gain")]
    rows_g, rows_v = [], []
    for c in cells:
        tv = c["target_variants"]
        pg, pv = [], []
        for key, _lab in names:
            g = tv[key]["guarded"]
            v = tv[key]["value"]
            pg.append(f"{display(g['em'], places=1)}/{display(g['calls'])}\\,{DISPLAY_VERDICT[g['verdict']]}")
            pv.append(f"{display(v['em'], places=1)}/{display(v['calls'])}\\,{DISPLAY_VERDICT[v['verdict']]}")
        clf = tv["stopclf"]["classifier"]
        suf = tv["suff"]["gate"]
        rows_g.append(f"{SHORT.get(c['short'], c['short'])} & " + " & ".join(pg)
                      + f" & {display(clf['em'], places=1)}/{display(clf['calls'])}\\,{DISPLAY_VERDICT[clf['verdict']]}"
                      + f" & {display(suf['em'], places=1)}/{display(suf['calls'])}\\,{DISPLAY_VERDICT[suf['verdict']]} \\\\")
        rows_v.append(f"{SHORT.get(c['short'], c['short'])} & " + " & ".join(pv) + " \\\\")
    (TABLES / "round9_target_rows.tex").write_text("\n".join(rows_g) + "\n")
    (TABLES / "round9_target_value_rows.tex").write_text("\n".join(rows_v) + "\n")

    # sensitivity (supplement): per cell, lambda set, EM range, calls range, #dominated/10, #pass/10
    rows = []
    for c in cells:
        s = c["selection_sensitivity_summary"]
        lam = ", ".join(f"{x:g}" for x in s["lambda_values"])
        rows.append(f"{SHORT.get(c['short'], c['short'])} & {c['logged_lambda']:g} & {lam} & "
                    f"{s['guarded_em_range'][0]:.1f} to {s['guarded_em_range'][1]:.1f} & "
                    f"{s['guarded_calls_range'][0]:.2f} to {s['guarded_calls_range'][1]:.2f} & "
                    f"{s['n_guarded_dominated']}/{s['n_configs']} & {s['n_guarded_win']}/{s['n_configs']} & "
                    f"{s['n_pass_3em']}/{s['n_configs']} \\\\")
    (TABLES / "round9_sensitivity_rows.tex").write_text("\n".join(rows) + "\n")

    # joint ablation (supplement)
    rows = []
    for c in cells:
        a = c["joint_ablation"]
        parts = []
        for key in ["full", "no_step_evidence_stability", "answer_content_only", "no_stability"]:
            g = a[key]["guarded"]
            v = a[key]["value"]
            parts.append(f"{display(g['em'], places=1)}/{display(g['calls'])}\\,{DISPLAY_VERDICT[g['verdict']]} & "
                         f"{display(v['em'], places=1)}/{display(v['calls'])}\\,{DISPLAY_VERDICT[v['verdict']]}")
        rows.append(f"{SHORT.get(c['short'], c['short'])} & " + " & ".join(parts) + " \\\\")
    (TABLES / "round9_ablation_rows.tex").write_text("\n".join(rows) + "\n")

    # equal-cost (supplement): per cell, G and V deltas at equal calls/tokens/latency
    rows = []
    for c in cells:
        e = c["equal_cost"]
        parts = []
        for pol in ["guarded_value", "value_only"]:
            for coord in ["calls", "input_tokens", "wall_seconds"]:
                parts.append(fmt_ci(e[coord][pol]["delta_em_vs_mixture_at_equal_cost"]))
        rows.append(f"{SHORT.get(c['short'], c['short'])} & " + " & ".join(parts) + " \\\\")
    (TABLES / "round9_equalcost_rows.tex").write_text("\n".join(rows) + "\n")

    # matched generated-claim HALT (supplement)
    rows = []
    for c in cells:
        pp = c["pooled_pareto"]
        if "halt_nli" not in pp["points"]:
            continue
        pts = pp["points"]
        dom = pp["dominators"]
        simple = {f"fixed_{k}" for k in range(5)} | {"answer_repeat"}
        def cell_str(key):
            names = dom.get(key, [])
            simple_doms = ", ".join(CODES.get(k, k) for k in names if k in simple)
            extra = len([k for k in names if k not in simple])
            label = simple_doms or ("NA" if not extra else "")
            if extra:
                label += f"{', ' if simple_doms else ''}+{extra} learned"
            return f"{display(pts[key]['em'], places=1)}/{display(pts[key]['calls'])} & {label}"
        rows.append(f"{SHORT.get(c['short'], c['short'])} & "
                    f"{display(pts['fixed_2']['em'], places=1)}/{display(pts['fixed_2']['calls'])} & "
                    f"{display(pts['guarded_value']['em'], places=1)}/{display(pts['guarded_value']['calls'])} & "
                    + cell_str("halt_nli") + " & " + cell_str("halt_relevance") + " \\\\")
    if rows:
        (TABLES / "round9_halt_rows.tex").write_text("\n".join(rows) + "\n")

    # epsilon tallies (body, small)
    rows = []
    t = out["epsilon_tallies"]
    for eps in EPSILONS:
        r = t[str(eps)]
        rows.append(f"{eps:g} & {r['value_only']['L']}/{r['value_only']['T']}/{r['value_only']['W']} & "
                    f"{r['guarded_value']['L']}/{r['guarded_value']['T']}/{r['guarded_value']['W']} \\\\")
    (TABLES / "round9_epsilon_rows.tex").write_text("\n".join(rows) + "\n")


def write_crossed_table(cells):
    rows = []
    for c in cells:
        pp = c["pooled_pareto"]
        dg = ", ".join(CODES.get(k, k) for k in pp["dominators"]["guarded_value"]) or "NA"
        mg = pp["margin_sensitivity"]["guarded_value"]
        marks = "".join("\\checkmark" if mg["pass_at_margin"][str(m)] else "$\\times$"
                        for m in [1.0, 2.0, 3.0])
        eq = c["equal_cost"]["calls"]["guarded_value"]["delta_em_vs_mixture_at_equal_cost"]
        g = pp["points"]["guarded_value"]
        v = pp["points"]["value_only"]
        e0 = c["epsilon_dominance"]["0.0"]["guarded_value"]["verdict"]
        rows.append(f"{SHORT.get(c['short'], c['short'])} & {pp['points']['fixed_%d' % c['bstar']]['em']:.1f}"
                    f"/{pp['points']['fixed_%d' % c['bstar']]['calls']:.1f}\\,({c['bstar']}) & "
                    f"{display(v['em'], places=1)}/{display(v['calls'])} & {display(g['em'], places=1)}/{display(g['calls'])} & "
                    f"{DISPLAY_VERDICT[e0]} & {dg} & "
                    f"{mg['lower95_one_sided']:+.2f} {marks} & "
                    f"{eq['mean_em_points']:+.2f}\\,[{eq['ci95'][0]:.2f},{eq['ci95'][1]:.2f}] \\\\")
    (TABLES / "round9_crossed_rows.tex").write_text("\n".join(rows) + "\n")


def summarise(out):
    cells = out["fresh_cells"]
    n = len(cells)
    summ = {"n_fresh_cells": n}
    summ["cells_any_fbg_nondominated"] = int(sum(c["pooled_pareto"]["any_fbg_nondominated"] for c in cells))
    summ["cells_guarded_nondominated"] = int(sum(c["pooled_pareto"]["fbg_guarded_nondominated"] for c in cells))
    summ["cells_value_nondominated"] = int(sum(c["pooled_pareto"]["fbg_value_nondominated"] for c in cells))
    summ["cells_tasr_nondominated"] = int(sum("tasr" in c["pooled_pareto"]["nondominated"] for c in cells))
    summ["cells_with_tasr"] = int(sum("tasr" in c["pooled_pareto"]["candidates"] for c in cells))
    summ["cells_stop_classifier_nondominated"] = int(sum("stop_classifier" in c["pooled_pareto"]["nondominated"] for c in cells))
    summ["cells_sufficiency_nondominated"] = int(sum("sufficiency_gate" in c["pooled_pareto"]["nondominated"] for c in cells))
    summ["cells_halt_present"] = int(sum("halt_nli" in c["pooled_pareto"]["candidates"] for c in cells))
    summ["cells_halt_nondominated"] = int(sum(
        ("halt_nli" in c["pooled_pareto"]["nondominated"])
        or ("halt_relevance" in c["pooled_pareto"]["nondominated"]) for c in cells))
    summ["cells_halt_dominated_by_fixed2"] = int(sum(
        "fixed_2" in c["pooled_pareto"]["dominators"].get("halt_nli", [])
        for c in cells))
    summ["cells_stoprag_present"] = int(sum("stoprag" in c["pooled_pareto"]["candidates"] for c in cells))
    summ["cells_stoprag_nondominated"] = int(sum("stoprag" in c["pooled_pareto"]["nondominated"] for c in cells))
    summ["cells_fixed2_nondominated"] = int(sum("fixed_2" in c["pooled_pareto"]["nondominated"] for c in cells))
    summ["cells_repeat_nondominated"] = int(sum("answer_repeat" in c["pooled_pareto"]["nondominated"] for c in cells))
    summ["cells_only_simple_nondominated"] = int(sum(
        not c["pooled_pareto"]["learned_nondominated"] for c in cells))
    summ["band_winner_counts"] = {
        b: {} for b in ["1.0", "2.0", "3.0"]}
    for b in ["1.0", "2.0", "3.0"]:
        for c in cells:
            w = c["pooled_pareto"]["band_winners"][b]["winner"]
            summ["band_winner_counts"][b][w] = summ["band_winner_counts"][b].get(w, 0) + 1
    summ["margin_power_primary"] = next((c["pooled_pareto"]["margin_power"] for c in cells
                                         if c["tag"] == "corpus_guarded_qwen3b"), None)
    summ["margin_with_80pct_power_range_em"] = [
        round(min(c["pooled_pareto"]["margin_power"]["margin_with_80pct_power_em"] for c in cells), 3),
        round(max(c["pooled_pareto"]["margin_power"]["margin_with_80pct_power_em"] for c in cells), 3)]
    summ["cells_guarded_nondominated_nontrivial"] = int(sum(
        c["pooled_pareto"]["fbg_guarded_nondominated"] for c in cells))
    # margins
    for pol in ["guarded_value", "value_only"]:
        summ[f"{pol}_margin_passes"] = {
            str(m): int(sum(c["pooled_pareto"]["margin_sensitivity"][pol]["pass_at_margin"][str(m)]
                            for c in cells)) for m in MARGINS}
    # equal cost
    # The per-cell records carry every learned policy, not only the two FBG
    # rules; summarising just those two left four rows of the who-wins table
    # dashed even though the data existed. Policies absent from a cell (TASR,
    # Stop-RAG, HALT are not everywhere) are skipped, and n_cells records how
    # many cells actually backed each mean.
    equal_pols = sorted({p for c in cells for p in c["equal_cost"]["calls"]
                         if p != "hull_vertices"})
    for coord in ["calls", "input_tokens", "wall_seconds"]:
        for pol in equal_pols:
            d = [c["equal_cost"][coord][pol]["delta_em_vs_mixture_at_equal_cost"]
                 for c in cells if pol in c["equal_cost"][coord]]
            if not d:
                continue
            summ[f"equal_{coord}_{pol}"] = {
                "n_cells": len(d),
                "n_ci_above_zero": int(sum(x["ci95"][0] > 0 for x in d)),
                "n_ci_below_zero": int(sum(x["ci95"][1] < 0 for x in d)),
                "n_point_above_zero": int(sum(x["mean_em_points"] > 0 for x in d)),
                "mean_delta_em": round(float(np.mean([x["mean_em_points"] for x in d])), 3),
                "min_delta_em": round(float(min(x["mean_em_points"] for x in d)), 3),
                "max_delta_em": round(float(max(x["mean_em_points"] for x in d)), 3)}
    # target variants: best variant per cell by guarded verdict/EM
    tv_summary = {}
    for tname in ["fbg", "fbg_refit", "net_0.02", "net_0.04", "net_0.08", "disc_0.5", "disc_0.8", "onestep"]:
        for rule in ["guarded", "value"]:
            vs = [c["target_variants"][tname][rule]["verdict"] for c in cells]
            ems = [c["target_variants"][tname][rule]["em"] for c in cells]
            calls = [c["target_variants"][tname][rule]["calls"] for c in cells]
            tv_summary[f"{tname}_{rule}"] = {
                "W": vs.count("W"), "T": vs.count("T"), "L": vs.count("L"),
                "mean_em": round(float(np.mean(ems)), 3),
                "mean_calls": round(float(np.mean(calls)), 3)}
    for tname, key in [("stopclf", "classifier"), ("suff", "gate")]:
        vs = [c["target_variants"][tname][key]["verdict"] for c in cells]
        ems = [c["target_variants"][tname][key]["em"] for c in cells]
        calls = [c["target_variants"][tname][key]["calls"] for c in cells]
        tv_summary[tname] = {"W": vs.count("W"), "T": vs.count("T"), "L": vs.count("L"),
                             "mean_em": round(float(np.mean(ems)), 3),
                             "mean_calls": round(float(np.mean(calls)), 3)}
    summ["target_variants"] = tv_summary
    summ["fbg_refit_reproduces_logged_lambda_cells"] = int(sum(
        c["target_variants"]["fbg_refit"]["reproduces_logged_lambda"] for c in cells))
    summ["fbg_refit_guarded_stop_agreement_min"] = round(min(
        c["target_variants"]["fbg_refit"]["guarded_stop_agreement_with_logged"] for c in cells), 4)
    # sensitivity
    summ["sensitivity"] = {
        "total_configs": int(sum(c["selection_sensitivity_summary"]["n_configs"] for c in cells)),
        "total_guarded_dominated": int(sum(c["selection_sensitivity_summary"]["n_guarded_dominated"] for c in cells)),
        "total_guarded_win": int(sum(c["selection_sensitivity_summary"]["n_guarded_win"] for c in cells)),
        "total_pass_3em": int(sum(c["selection_sensitivity_summary"]["n_pass_3em"] for c in cells)),
        "max_em_range_width": round(max(c["selection_sensitivity_summary"]["guarded_em_range"][1]
                                        - c["selection_sensitivity_summary"]["guarded_em_range"][0]
                                        for c in cells), 3),
        "max_calls_range_width": round(max(c["selection_sensitivity_summary"]["guarded_calls_range"][1]
                                           - c["selection_sensitivity_summary"]["guarded_calls_range"][0]
                                           for c in cells), 3),
        "cells_lambda_changes": int(sum(len(c["selection_sensitivity_summary"]["lambda_values"]) > 1 for c in cells)),
    }
    # ablation
    abl_summary = {}
    for key in ["full", "no_step_evidence_stability", "answer_content_only", "no_stability",
                "no_step_no_evidence_size"]:
        for rule in ["guarded", "value"]:
            vs = [c["joint_ablation"][key][rule]["verdict"] for c in cells]
            ems = [c["joint_ablation"][key][rule]["em"] for c in cells]
            calls = [c["joint_ablation"][key][rule]["calls"] for c in cells]
            abl_summary[f"{key}_{rule}"] = {"W": vs.count("W"), "T": vs.count("T"), "L": vs.count("L"),
                                            "mean_em": round(float(np.mean(ems)), 3),
                                            "mean_calls": round(float(np.mean(calls)), 3)}
    summ["joint_ablation"] = abl_summary
    return summ


def main():
    global RESTRICT_BASELINES
    RESTRICT_BASELINES = True
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2029)
    ap.add_argument("--only", default=None, help="comma-separated tags")
    args = ap.parse_args()
    out = {"status": "derived_from_frozen_measured_runs_no_new_llm_calls",
           "seed": args.seed,
           "verdict_rule": ("Against the simple family {fixed-0..fixed-B, answer "
                            "repetition} on (EM, exact LLM calls). epsilon-dominance: "
                            "EM differences within epsilon are ties; L if some simple "
                            "policy epsilon-dominates the learned point, W if the "
                            "learned point epsilon-dominates the training-selected "
                            "B*, T otherwise. epsilon = 0 is the executive-matrix rule."),
           "fresh_cells": []}
    for tag, fname, label, short in fresh_cells():
        if args.only and tag not in args.only.split(","):
            continue
        print(f"[cell] {tag}", flush=True)
        cell = Cell(tag, fname, label, short)
        res = analyse_cell(cell, seed=args.seed)
        pp = res["pooled_pareto"]
        print(f"   nondominated={pp['nondominated']} domG={pp['dominators']['guarded_value']} "
              f"domV={pp['dominators']['value_only']} "
              f"margins={pp['margin_sensitivity']['guarded_value']['pass_at_margin']} "
              f"eqcalls={res['equal_cost']['calls']['guarded_value']['delta_em_vs_mixture_at_equal_cost']['mean_em_points']:+.2f} "
              f"repro_lambda={res['target_variants']['fbg_refit']['reproduces_logged_lambda']} "
              f"agree={res['target_variants']['fbg_refit']['guarded_stop_agreement_with_logged']}",
              flush=True)
        out["fresh_cells"].append(res)
    out["crossed_backbone_cells"] = []
    for tag, fname, label, short in fresh_cells(extra=True):
        if args.only and tag not in args.only.split(","):
            continue
        print(f"[crossed] {tag}", flush=True)
        cell = Cell(tag, fname, label, short)
        out["crossed_backbone_cells"].append(analyse_cell(cell, seed=args.seed))
    out["historical_cells"] = historical_epsilon()
    # epsilon tallies over all cells (historical + fresh)
    tallies = {}
    for eps in EPSILONS:
        t = {"value_only": {"W": 0, "T": 0, "L": 0}, "guarded_value": {"W": 0, "T": 0, "L": 0}}
        for c in out["historical_cells"].values():
            for pol in t:
                t[pol][c["epsilon_dominance"][str(eps)][pol]["verdict"]] += 1
        for c in out["fresh_cells"]:
            for pol in t:
                t[pol][c["epsilon_dominance"][str(eps)][pol]["verdict"]] += 1
        t["n_cells"] = len(out["historical_cells"]) + len(out["fresh_cells"])
        tallies[str(eps)] = t
    out["epsilon_tallies"] = tallies
    out["summary"] = summarise(out)
    if out["crossed_backbone_cells"]:
        out["crossed_summary"] = summarise({"fresh_cells": out["crossed_backbone_cells"]})
        out["crossed_summary"]["backbones"] = sorted(
            {c["model"] for c in out["crossed_backbone_cells"]})
        write_crossed_table(out["crossed_backbone_cells"])
    dest = MEAS / "round9_analyses.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    write_tables(out)
    print(json.dumps(out["summary"], indent=1))
    print("epsilon tallies:", json.dumps(tallies))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
