"""Canonical reference implementation of the deployed \\system{} controller
(Algorithm 1 of the paper) -- self-contained, matching run_experiment.py's
evaluated policy exactly.

The historical panel uses a value gate: continue searching while the value
head's future-best-gain estimate clears frozen threshold lambda.  The fresh
natural-corpus confirmation uses the optional guarded rule, which waits for
both the value crossing and one consecutive normalized-answer repetition.
The error head attaches a certificate flag (never changing the stopping
time), calibrated by corrected SGR on a dedicated episode split excluded
from fitting and threshold selection.

Two entry points:
  * stop_index(values, lam)         -- offline replay on logged per-prefix
                                       value scores (what every table uses)
  * guarded_stop_index(...)         -- value crossing + repetition guard
  * run_episode(...)                -- live loop for deployment, taking an
                                       `llm` callable and a `search` callable

Threshold provenance: lambda comes from frozen_protocol.select_threshold on
CROSS-FITTED training chains (see frozen_protocol.py); q_alpha comes from
sgr_threshold below. Neither is ever selected on test outcomes.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

from run_experiment import (VoICritic, _norm, prefix_features,  # noqa: F401
                            answer_from, gen_query)


# --------------------------------------------------------------------------- #
# Deployed stopping rule (identical to run_experiment.voi_stop_on_trajectory
# with mode="value", and to frozen_protocol.test_stop_index)
# --------------------------------------------------------------------------- #
def stop_index(values: Sequence[float], lam: float) -> int:
    """First non-terminal prefix whose predicted future-best gain < lam."""
    for t in range(len(values) - 1):
        if values[t] < lam:
            return t
    return len(values) - 1


def guarded_stop_index(values: Sequence[float], answers: Sequence[str],
                       lam: float) -> int:
    """Stop after both first value crossing and first answer repetition."""
    value_crossing = stop_index(values, lam)
    repeat_crossing = len(answers) - 1
    for t in range(1, len(answers)):
        if _norm(answers[t]) == _norm(answers[t - 1]):
            repeat_crossing = t
            break
    return max(value_crossing, repeat_crossing)


# --------------------------------------------------------------------------- #
# Certificate calibration: SGR with the delta/ceil(log2 m) correction
# (Geifman & El-Yaniv 2017, Algorithm 1 / Theorem 3.2)
# --------------------------------------------------------------------------- #
def _cp_upper(k: int, n: int, level: float) -> float:
    from scipy.stats import beta as beta_dist
    if n == 0 or k >= n:
        return 1.0
    return float(beta_dist.ppf(level, k + 1, n - k))


def sgr_threshold(err_hats: Sequence[float], errors: Sequence[int],
                  alpha: float, delta: float = 0.10):
    """Threshold returned by corrected SGR binary search.

    Binary search over selection sizes; each of the <= ceil(log2 m) tests
    uses a Clopper-Pearson bound at level delta/ceil(log2 m); a union bound
    gives simultaneous 1-delta validity. Returns (threshold, n_selected),
    or (None, 0) when nothing can be certified (the fail-safe outcome).

    Calibration units MUST be the stopped states of the deployed
    frozen-lambda rule -- one unit per episode -- so the certified
    distribution matches deployment.
    """
    pairs = sorted(zip(err_hats, errors))
    m = len(pairs)
    if m == 0:
        return None, 0
    steps = max(1, math.ceil(math.log2(m)))
    level = 1 - delta / steps
    cum, tot = [], 0
    for _, e in pairs:
        tot += e
        cum.append(tot)
    # A deployed threshold selects every score <= q, so tied scores are an
    # indivisible candidate.  Searching raw row indices could otherwise
    # report n_selected smaller than the subset actually selected at q.
    ends = [i for i in range(1, m + 1)
            if i == m or pairs[i][0] != pairs[i - 1][0]]
    lo, hi, best = 0, len(ends) - 1, None
    for _ in range(steps):
        if lo > hi:
            break
        mid = (lo + hi + 1) // 2
        n_selected = ends[mid]
        if _cp_upper(cum[n_selected - 1], n_selected, level) <= alpha:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    if best is None:
        return None, 0
    n_selected = ends[best]
    return pairs[n_selected - 1][0], n_selected


# --------------------------------------------------------------------------- #
# Live episode (deployment loop). The environment contract matches
# run_experiment.QuestionEnv: search(query, k) -> list[str].
# --------------------------------------------------------------------------- #
def run_episode(env, model: str, critic: VoICritic, lam: float,
                q_alpha: float | None, max_steps: int = 4, k: int = 3,
                rule: str = "value",
                llm_answer: Callable = answer_from,
                llm_query: Callable = gen_query) -> dict:
    """Algorithm 1: frozen value-gate stop; certificate on the answer."""
    evidence: list[str] = []
    traj = [{"searches": 0, "evidence": [], "answer": llm_answer(env, [], model)}]
    t = 0
    value_crossed = False
    answer_repeated = False
    while t < max_steps:
        feats = prefix_features(env, traj, t)
        value_crossed |= critic.value(feats) < lam
        if t > 0:
            answer_repeated |= (_norm(traj[t]["answer"])
                                == _norm(traj[t - 1]["answer"]))
        if value_crossed and (rule == "value" or answer_repeated):
            break
        q = llm_query(env, evidence, model)
        for h in env.search(q, k=k):
            if h not in evidence:
                evidence.append(h)
        traj.append({"searches": t + 1, "evidence": list(evidence),
                     "answer": llm_answer(env, evidence, model), "query": q})
        t += 1
    feats = prefix_features(env, traj, t)
    certified = (q_alpha is not None
                 and critic.err_prob(feats) <= q_alpha)
    return {"answer": traj[t]["answer"], "searches": traj[t]["searches"],
            "certified": bool(certified), "rule": rule,
            "trajectory": traj}
