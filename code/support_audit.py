"""Logging-policy support audit: re-roll alternative continuations from the
states where the deployed controller stops.

The FBG target is the best answer quality reached along ONE logged
continuation, so it under-estimates gains the logging policy failed to
realise, and clipping at zero maps "nothing more was gained in this log" and
"continuing was harmful" onto the same target.  This script measures both
directly on a fresh corpus-index run, without touching its results:

  * take the guarded rule's stop state s_T for each test question;
  * stratify the stops by the logged FBG target at that state (zero vs
    positive) and by the stop index, and sample N of them with a fixed seed;
  * from each sampled stop, re-roll ``--rollouts`` independent continuations to
    the step budget with temperature ``--temperature`` and distinct LLM seeds,
    re-using the *same* frozen corpus index, prompts, and answer extraction;
  * compare the best answer quality reached by the alternative continuations
    with the logged continuation's best after the same stop.

Reported quantities (per stratum and overall):
  missed_gain_rate       share of stops where some alternative continuation
                         beats the logged best-after-stop in token-F1;
  mean_missed_gain       mean of max(0, alt_best - logged_best_after_stop);
  clipped_but_reachable  share of stops whose logged FBG target is exactly 0
                         yet an alternative continuation improves on u_t;
  em_flip_rate           share of stops where an alternative continuation
                         reaches EM 1 while the logged continuation stayed 0.

Nothing here changes any reported policy, threshold, or number; it bounds the
bias of the target that the audited controller was trained on.

    python3 support_audit.py --tag corpus_guarded_qwen3b --n 300 --rollouts 2
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time

import numpy as np
import pandas as pd

import run_experiment as rx

MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
ART = pathlib.Path(__file__).resolve().parents[1]


def build_corpus(parquet_name):
    df_all = pd.read_parquet(ART / parquet_name)
    pool, seen = [], set()
    for _, row in df_all.iterrows():
        for sentence in rx.row_sentences(row):
            if sentence not in seen:
                seen.add(sentence)
                pool.append(sentence)
    return rx.CorpusBM25(pool), len(pool), len(df_all)


def env_for(record, row, corpus):
    env = rx.QuestionEnv(row)
    env.sents = []
    env.bm25 = corpus
    assert str(env.qid) == str(record["id"]), (env.qid, record["id"])
    return env


def continue_from(env, evidence, start_step, max_steps, model):
    """Roll the logging policy forward from a given evidence state."""
    out = []
    evidence = list(evidence)
    for t in range(start_step, max_steps):
        query = rx.gen_query(env, evidence, model)
        for hit in env.search(query, k=3):
            if hit not in evidence:
                evidence.append(hit)
        answer = rx.answer_from(env, evidence, model)
        out.append({"searches": t + 1, "answer": answer,
                    "f1": env.f1(answer), "em": env.em(answer)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="corpus_guarded_qwen3b")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=911)
    args = ap.parse_args()

    run = json.loads((MEAS / f"run_{args.tag}.json").read_text())
    records = run["per_q"]
    model = run["config"]["model"]
    max_steps = run["config"]["max_steps"]
    parquet = run["config"]["parquet"]
    assert parquet, "this audit is defined for the parquet corpus-index runs"

    # stops of the deployed guarded rule and the logged FBG target there
    rows = []
    for i, r in enumerate(records):
        stop = int(r["primary"]["searches"])
        f1s = r["prefix_f1"]
        future = f1s[stop + 1:]
        fbg = max(max(future) - f1s[stop], 0.0) if future else 0.0
        rows.append({"idx": i, "id": str(r["id"]), "stop": stop,
                     "u_t": f1s[stop], "logged_fbg": fbg,
                     "logged_best_after": max(future) if future else f1s[stop],
                     "em_t": r["prefix_em"][stop],
                     "logged_em_after": max(r["prefix_em"][stop:])})
    eligible = [r for r in rows if r["stop"] < max_steps]
    strata = {}
    for r in eligible:
        key = (r["logged_fbg"] > 1e-9, r["stop"])
        strata.setdefault(key, []).append(r)
    rng = np.random.default_rng(args.seed)
    total = min(args.n, len(eligible))
    sampled = []
    keys = sorted(strata, key=lambda k: (-len(strata[k]), k))
    # proportional allocation, then top-up from the largest strata
    for key in keys:
        share = int(round(total * len(strata[key]) / len(eligible)))
        pick = rng.choice(len(strata[key]), size=min(share, len(strata[key])),
                          replace=False)
        sampled += [strata[key][int(i)] for i in pick]
    leftover = [r for r in eligible if r not in sampled]
    if len(sampled) < total and leftover:
        pick = rng.choice(len(leftover), size=min(total - len(sampled), len(leftover)),
                          replace=False)
        sampled += [leftover[int(i)] for i in pick]
    sampled = sampled[:total]
    print(f"[audit] {len(sampled)} sampled stops from {len(eligible)} eligible "
          f"({len(records)} test questions); strata "
          f"{ {f'fbg>0={k[0]},stop={k[1]}': len(v) for k, v in strata.items()} }",
          flush=True)

    corpus, n_sent, n_ctx = build_corpus(parquet)
    print(f"[corpus] {n_sent} sentences from {n_ctx} contexts", flush=True)
    df_all = pd.read_parquet(ART / parquet)
    df = df_all.sample(frac=1.0, random_state=run["config"]["data_seed"]).reset_index(drop=True)
    test_rows = df.iloc[run["config"]["n_train"]:
                        run["config"]["n_train"] + run["config"]["n_test"]]
    test_rows = list(test_rows.iterrows())

    t0 = time.time()
    out_rows = []
    for j, r in enumerate(sampled):
        _, row = test_rows[r["idx"]]
        record = records[r["idx"]]
        env = env_for(record, row, corpus)
        evidence = list(record["trajectory"][r["stop"]]["evidence"])
        alts = []
        for k in range(args.rollouts):
            rx._LLM_SEED = args.seed + 1000 * k
            rx._LLM_TEMPERATURE = args.temperature
            try:
                alts.append(continue_from(env, evidence, r["stop"], max_steps, model))
            finally:
                rx._LLM_SEED = run["config"].get("llm_seed", 7)
                rx._LLM_TEMPERATURE = run["config"].get("llm_temperature", 0.0)
        alt_best_f1 = max([p["f1"] for a in alts for p in a] + [r["u_t"]])
        alt_best_em = max([p["em"] for a in alts for p in a] + [r["em_t"]])
        out_rows.append({
            **r,
            "alt_best_f1": alt_best_f1, "alt_best_em": alt_best_em,
            "alt_minus_logged_best": alt_best_f1 - r["logged_best_after"],
            "alt_minus_u_t": alt_best_f1 - r["u_t"],
            "rollouts": [[{k2: v2 for k2, v2 in p.items() if k2 != "answer"}
                          for p in a] for a in alts],
        })
        if (j + 1) % 20 == 0:
            print(f"  [{j+1}/{len(sampled)}] {time.time()-t0:.0f}s", flush=True)
            (MEAS / f"support_audit_{args.tag}.partial.json").write_text(
                json.dumps({"per_stop": out_rows}, indent=1))

    def agg(rowset):
        if not rowset:
            return None
        miss = np.asarray([max(x["alt_minus_logged_best"], 0.0) for x in rowset])
        clipped = [x for x in rowset if x["logged_fbg"] <= 1e-9]
        return {
            "n": len(rowset),
            "missed_gain_rate": round(float(np.mean(miss > 1e-9)), 4),
            "mean_missed_gain_f1": round(float(miss.mean()), 4),
            "mean_positive_missed_gain_f1": (
                round(float(miss[miss > 1e-9].mean()), 4) if (miss > 1e-9).any() else 0.0),
            "n_clipped_zero_target": len(clipped),
            "clipped_but_reachable_rate": (
                round(float(np.mean([x["alt_minus_u_t"] > 1e-9 for x in clipped])), 4)
                if clipped else None),
            "em_flip_rate": round(float(np.mean(
                [x["alt_best_em"] > x["logged_em_after"] for x in rowset])), 4),
            "alt_worse_rate": round(float(np.mean(
                [x["alt_minus_logged_best"] < -1e-9 for x in rowset])), 4),
        }

    result = {
        "status": "alternative_continuation_reroll_from_deployed_stops",
        "tag": args.tag, "model": model,
        "model_metadata": rx.model_metadata(model),
        "n_test_questions": len(records), "n_eligible_stops": len(eligible),
        "n_sampled": len(sampled), "rollouts_per_stop": args.rollouts,
        "temperature": args.temperature, "sampling_seed": args.seed,
        "stratification": "logged FBG target (zero vs positive) x stop index",
        "retrieval_corpus": run.get("retrieval_corpus"),
        "overall": agg(out_rows),
        "by_logged_target": {
            "zero": agg([x for x in out_rows if x["logged_fbg"] <= 1e-9]),
            "positive": agg([x for x in out_rows if x["logged_fbg"] > 1e-9]),
        },
        "by_stop_index": {str(s): agg([x for x in out_rows if x["stop"] == s])
                          for s in sorted({x["stop"] for x in out_rows})},
        "interpretation": (
            "missed_gain_rate bounds how often the logged FBG target under-"
            "states what continuing could have achieved; clipped_but_reachable"
            "_rate is the share of zero-target stops at which an alternative "
            "continuation still improves. Alternative rollouts use temperature "
            f"{args.temperature}, so they also carry decoding noise; the rate "
            "is an upper bound on support bias, not a point estimate."),
        "per_stop": out_rows,
        "wall_seconds": round(time.time() - t0, 1),
    }
    dest = MEAS / f"support_audit_{args.tag}.json"
    dest.write_text(json.dumps(result, indent=1))
    partial = MEAS / f"support_audit_{args.tag}.partial.json"
    if partial.exists():
        partial.unlink()
    print(json.dumps({k: result[k] for k in
                      ["overall", "by_logged_target", "by_stop_index"]}, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
