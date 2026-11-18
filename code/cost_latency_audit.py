"""Real-systems cost audit from the logged per-prefix ledgers.

Offline: reads only the `answer_usage` / `query_usage` /
`retrieval_latency_seconds` records already in results/measured/run_*.json.
Adds what the manuscript currently lacks -- p50/p95/p99 latency
distributions per LLM call and per question -- across >= 2 backbones
(local ollama and Bedrock), plus per-axis paired cost deltas between the
deployed guarded policy and the full budget.

HONESTY NOTES, carried into the output file:
  * No billed dollar cost exists anywhere in this artifact.  Nothing here is
    money.  Any price would be list-price arithmetic, not a bill.
  * Failures, timeouts and retries were NOT logged: run_experiment.py retries
    inside a bare except and writes no failure code.  The failure rate is
    therefore UNKNOWN, not zero, and is reported as null.

    python3 cost_latency_audit.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import round9_analyses as r9
from analyze_corpus_confirmation import cumulative_cost

MEAS = r9.MEAS


def pct(a, q):
    return None if len(a) == 0 else round(float(np.percentile(a, q)), 4)


def backbone_of(run):
    md = run.get("model_metadata", {})
    return {"requested_model": md.get("requested_model", run["config"]["model"]),
            "backend": md.get("backend"),
            "digest": md.get("digest"),
            "quantization_level": md.get("quantization_level")}


def audit(tag, fname, label, short):
    cell = r9.Cell(tag, fname, label, short)
    run = cell.run
    per_call_latency, per_call_out, per_q_latency, per_q_calls = [], [], [], []
    retrieval_latency = []
    for r in run["per_q"]:
        q_lat = 0.0
        q_calls = 0
        for p in r["trajectory"]:
            for part in ("query_usage", "answer_usage"):
                u = p.get(part, {})
                if not u:
                    continue
                calls = float(u.get("calls", 0))
                lat = float(u.get("latency_seconds", 0.0))
                if calls > 0:
                    per_call_latency.append(lat / calls)
                    per_call_out.append(float(u.get("out", 0)) / calls)
                q_lat += lat
                q_calls += calls
            rl = float(p.get("retrieval_latency_seconds", 0.0))
            retrieval_latency.append(rl)
            q_lat += rl
        per_q_latency.append(q_lat)
        per_q_calls.append(q_calls)
    a = np.asarray(per_call_latency, float)
    b = np.asarray(per_q_latency, float)
    rl = np.asarray(retrieval_latency, float)

    idx = np.arange(cell.n)
    full = cell.cost[idx, cell.last]
    guarded = cell.cost[idx, cell.g_stop]
    axes = ["calls", "input_tokens", "wall_seconds"]
    deltas = {}
    for j, name in enumerate(axes):
        d = guarded[:, j] - full[:, j]
        rng = np.random.default_rng(2029)
        means = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
        deltas[name] = {
            "guarded_mean": round(float(guarded[:, j].mean()), 4),
            "full_budget_mean": round(float(full[:, j].mean()), 4),
            "paired_mean_delta": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(means, 2.5)), 4),
                     round(float(np.percentile(means, 97.5)), 4)],
            "relative_saving_pct": round(
                100 * float(-d.mean() / max(1e-9, full[:, j].mean())), 2),
        }
    return {
        "tag": tag, "label": label, "backbone": backbone_of(run),
        "n_test_questions": int(cell.n),
        "n_llm_calls_logged_full_budget": int(sum(per_q_calls)),
        "per_llm_call_latency_seconds": {
            "mean": round(float(a.mean()), 4), "p50": pct(a, 50),
            "p90": pct(a, 90), "p95": pct(a, 95), "p99": pct(a, 99),
            "max": round(float(a.max()), 4), "n": int(a.size)},
        "per_question_wall_seconds_full_budget": {
            "mean": round(float(b.mean()), 4), "p50": pct(b, 50),
            "p90": pct(b, 90), "p95": pct(b, 95), "p99": pct(b, 99),
            "max": round(float(b.max()), 4), "n": int(b.size)},
        "per_retrieval_latency_seconds": {
            "mean": round(float(rl.mean()), 5), "p50": pct(rl, 50),
            "p95": pct(rl, 95), "n": int(rl.size)},
        "paired_cost_delta_guarded_minus_full": deltas,
        "billed_cost_usd": None,
        "billed_cost_note": "not measured; no invoice or pricing telemetry "
                            "exists for any run in this artifact",
        "failure_or_timeout_count": None,
        "failure_note": "NOT LOGGED. run_experiment.py retries inside a bare "
                        "except and writes no failure code, so the failure "
                        "rate is unknown for every completed cell; it is not "
                        "zero-by-evidence.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cells = r9.fresh_cells() + r9.fresh_cells(extra=True)
    rows = []
    for tag, fname, label, short in cells:
        r = audit(tag, fname, label, short)
        rows.append(r)
        c = r["per_llm_call_latency_seconds"]
        q = r["per_question_wall_seconds_full_budget"]
        print(f"{tag:40s} {r['backbone']['requested_model'][:34]:34s} "
              f"call p50={c['p50']:.3f} p95={c['p95']:.3f} | "
              f"question p50={q['p50']:.2f} p95={q['p95']:.2f}", flush=True)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cost_latency_audit.json").write_text(json.dumps(
        {"what": "p50/p95/p99 latency distributions and paired per-axis cost "
                 "deltas from the logged per-prefix ledgers",
         "backbones_covered": sorted({r["backbone"]["requested_model"] for r in rows}),
         "money": "NOT MEASURED ANYWHERE. No dollar figure is reported.",
         "failures": "NOT LOGGED. See failure_note on every row.",
         "cells": rows}, indent=1))
    print("wrote", outdir / "cost_latency_audit.json")


if __name__ == "__main__":
    main()
