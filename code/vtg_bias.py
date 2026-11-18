"""Quantify the pessimism of logged-trajectory VTG targets (review Q2).

The VTG target is the best answer quality reached along ONE logged
continuation; the true value-to-go maximises over ALL continuations. This
script bounds the underestimation empirically: run a second, independent
ReAct rollout (higher temperature) per question and measure how often it
reaches a strictly better best-prefix answer than the logging rollout.

  P(alt rollout beats logged best)  -> upper-bounds how often the VTG
  target says "stop" when a different continuation could still improve.

Usage: python vtg_bias.py --parquet ... --model qwen2.5:3b-instruct --n 30
"""
import argparse
import json

import numpy as np
import pandas as pd

import run_experiment as R


def best_f1(env, traj):
    return max(env.f1(p["answer"]) for p in traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=4)
    ap.add_argument("--tag", default="vtg_bias")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    df = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    rows = df.iloc[: args.n]

    beats, ties, worse, gaps = 0, 0, 0, []
    for _, row in rows.iterrows():
        env = R.QuestionEnv(row)
        t1 = R.run_react(env, args.model, max_steps=args.max_steps, k=3)
        # independent alternate continuation: same prefix-0, temp-varied queries
        old_llm = R.llm

        def hot_llm(msgs, model, temperature=0.0, max_tokens=256, retries=4):
            return old_llm(msgs, model, temperature=0.8, max_tokens=max_tokens,
                           retries=retries)
        R.llm = hot_llm
        try:
            t2 = R.run_react(env, args.model, max_steps=args.max_steps, k=3)
        finally:
            R.llm = old_llm
        b1, b2 = best_f1(env, t1), best_f1(env, t2)
        gaps.append(b2 - b1)
        if b2 > b1 + 1e-9:
            beats += 1
        elif b2 < b1 - 1e-9:
            worse += 1
        else:
            ties += 1
        print(f"  b1={b1:.2f} b2={b2:.2f}", flush=True)

    out = {"n": args.n, "alt_beats_logged": beats, "ties": ties,
           "alt_worse": worse,
           "mean_gap": float(np.mean(gaps)),
           "mean_positive_gap": float(np.mean([g for g in gaps if g > 0]) if any(g > 0 for g in gaps) else 0.0)}
    dest = R.OUT / f"{args.tag}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
