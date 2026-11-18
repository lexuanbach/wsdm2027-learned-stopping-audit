"""Stop-RAG (faithful reimplementation, stoprag_run.py) on a fresh corpus-index
run's logged trajectories -- no new LLM calls.

Training states are the 90 policy-training episodes of the run (the same
seed-41 split that the FBG critic, TASR calibration, and fixed-budget
selection use; the 30 certificate episodes are untouched); the last 20 of
those 90 serve as Stop-RAG's margin-validation split exactly as in the
anchor-cell comparison.  Test states are the 400 untouched test
trajectories.  Stop-RAG's DeBERTa Q-network reads the question and the
retrieved sentences; its stop decision is replayed on the logged prefixes,
so calls = 1 + 2 * stop under the shared answer-at-every-prefix loop.

    python3 stoprag_fresh.py --tag corpus_guarded_qwen3b [--cpu]

Writes results/measured/stoprag_traj_<tag>.json (states) and
stoprag_result_<tag>.json (frontier, selected margin, per-question stops).
"""
from __future__ import annotations

import argparse
import json
import pickle
import types

import numpy as np

import run_experiment as rx
import stoprag_run

MEAS = rx.OUT


def build_states(tag):
    run = json.loads((MEAS / f"run_{tag}.json").read_text())
    cache = pickle.loads((MEAS / f"critic_data_{tag}.pkl").read_bytes())
    X = np.asarray(cache["X_val"], float)
    y_err = np.asarray(cache["y_err"], int)
    all_chains = rx.chain_indices(X)
    rng = np.random.default_rng(41)
    order = np.arange(len(all_chains))
    rng.shuffle(order)
    ncal = max(1, int(round(run["config"]["certificate_calibration_fraction"]
                            * len(all_chains))))
    fit_ids = order[:-ncal]
    traces = cache["traces"]
    assert len(traces) == len(all_chains)
    out = []
    for i in fit_ids:
        tr, ch = traces[i], all_chains[i]
        assert len(tr["trajectory"]) == len(ch)
        out.append({"id": tr["id"], "question": tr["question"],
                    "gold": tr["gold_answers"][0], "aliases": tr["gold_answers"][1:],
                    "split": "train",
                    "prefixes": [{"searches": p["searches"], "evidence": p["evidence"],
                                  "answer": p["answer"], "f1": float(tr["corr"][t]),
                                  "em": int(1 - y_err[ch[t]])}
                                 for t, p in enumerate(tr["trajectory"])]})
    n_train = len(out)
    for r in run["per_q"]:
        out.append({"id": r["id"], "question": r["question"],
                    "gold": r["gold_answers"][0], "aliases": r["gold_answers"][1:],
                    "split": "test",
                    "prefixes": [{"searches": p["searches"], "evidence": p["evidence"],
                                  "answer": p["answer"], "f1": float(r["prefix_f1"][t]),
                                  "em": int(r["prefix_em"][t])}
                                 for t, p in enumerate(r["trajectory"])]})
    (MEAS / f"stoprag_traj_{tag}.json").write_text(json.dumps(out))
    return n_train, len(out) - n_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--encoder", default="microsoft/deberta-v3-base")
    ap.add_argument("--fitted_iters", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_train, n_test = build_states(args.tag)
    print(f"[states] {n_train} policy-training episodes (70 train / 20 val), "
          f"{n_test} test trajectories", flush=True)
    targs = types.SimpleNamespace(tag=args.tag, encoder=args.encoder,
                                  n_train=n_train, n_test=n_test,
                                  fitted_iters=args.fitted_iters, batch=args.batch,
                                  cpu=args.cpu, seed=args.seed)
    stoprag_run.train(targs)
    res = json.loads((MEAS / f"stoprag_result_{args.tag}.json").read_text())
    res["source_run"] = args.tag
    res["training_split"] = ("90 policy-training episodes of the run (seed-41 split); "
                             "last 20 used as margin-validation; certificate episodes untouched")
    res["cost_note"] = "exact LLM calls = 1 + 2 * stop (shared answer-at-every-prefix loop)"
    (MEAS / f"stoprag_result_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print("[done]", {k: res[k] for k in ["ref_em", "ref_s", "margin_star", "stoprag_em", "stoprag_s"]})


if __name__ == "__main__":
    main()
