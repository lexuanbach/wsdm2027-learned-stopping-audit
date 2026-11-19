"""The hindsight-oracle truncation ceiling, priced in the paper's own currency.

Section 5 declares "hindsight-oracle truncation" among the simple controls, but
no table in the paper reported it. This script closes that gap.

Convention (this is the whole content of the script, and it must be stated
wherever the number appears): for each question the oracle stops at the FIRST
prefix whose `prefix_em` is 1; when NO prefix of the logged trajectory is
correct, the oracle is charged the FULL budget, exactly as every other policy
is when it never stops early.

The `oracle_searches` field stored in the run registries is deliberately NOT
reused. `run_experiment.py:1116` picks

    best = max(range(len(traj)), key=lambda t: (prefix_em[t], -traj[t]["searches"]))

which, on a question no prefix answers correctly, breaks the all-zero tie in
favour of the CHEAPEST prefix and so charges 0 searches to questions the oracle
never gets right. On the primary cell that yields a mean of 0.3175 searches,
which is not a cost anyone could pay.

Costs are priced with the shipped `analyze_corpus_confirmation.cumulative_cost`
so the row is in the same currency as every other policy in the paper, and the
`searches` column is the stop index, matching `summarize()`.

Usage:  python3 oracle_frontier.py            (writes results/measured/oracle_frontier.json)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from analyze_corpus_confirmation import cumulative_cost

ART = pathlib.Path(__file__).resolve().parents[1]
MEAS = ART / "results" / "measured"
MANU = ART / "manuscript"
if not MANU.exists():
    MANU = pathlib.Path(__file__).resolve().parents[2] / "manuscript"

# the fourteen fresh corpus-index cells, in the order of tab:supp-pooled
CELLS = [
    ("corpus_guarded_qwen3b", "A s7"),
    ("corpus_guarded_qwen3b_seed11", "A s11"),
    ("corpus_guarded_qwen3b_seed19", "A s19"),
    ("corpus_guarded_qwen3b_data17", "B"),
    ("corpus_guarded_qwen3b_data23", "C"),
    ("corpus_guarded_qwen3b_data29", "D"),
    ("corpus_guarded_qwen3b_data31", "E"),
    ("corpus2wiki_guarded_qwen3b_data17", "2Wiki d1"),
    ("corpus2wiki_guarded_qwen3b_data23", "2Wiki d2"),
    ("corpus2wiki_guarded_qwen3b_data29", "2Wiki d3"),
    ("corpusmusique_guarded_qwen3b_data17", "MuSiQue d1"),
    ("corpusmusique_guarded_qwen3b_data23", "MuSiQue d2"),
    ("corpusmusique_guarded_qwen3b_data29", "MuSiQue d3"),
    ("wiki25m_guarded_qwen3b", "Wiki 25M"),
]


def oracle_stop(record):
    """First correct prefix; the full budget when no prefix is correct."""
    prefix_em = record["prefix_em"]
    for t, em in enumerate(prefix_em):
        if em == 1:
            return t, True
    return len(prefix_em) - 1, False


def price(records, stops):
    em = np.asarray([r["prefix_em"][t] for r, t in zip(records, stops)], float)
    costs = [cumulative_cost(r, t) for r, t in zip(records, stops)]
    return {
        "em": round(100 * float(em.mean()), 2),
        "searches": round(float(np.mean(stops)), 3),
        "llm_calls": round(float(np.mean([c["calls"] for c in costs])), 3),
    }


def cell(tag):
    path = MEAS / f"run_{tag}.json"
    if not path.exists():
        return None
    records = json.loads(path.read_text())["per_q"]
    stops, reached = zip(*[oracle_stop(r) for r in records])
    last = [len(r["prefix_em"]) - 1 for r in records]
    row = {
        "n": len(records),
        "oracle": price(records, list(stops)),
        "full_budget": price(records, last),
        "reachable_rate": round(float(np.mean(reached)), 4),
        "stored_oracle_searches_mean": round(float(np.mean(
            [r["oracle_searches"] for r in records])), 4),
    }
    row["headroom_em"] = round(row["oracle"]["em"] - row["full_budget"]["em"], 2)
    row["calls_saved"] = round(
        row["full_budget"]["llm_calls"] - row["oracle"]["llm_calls"], 3)
    return row


def main():
    out = {
        "convention": ("stop at the first prefix with prefix_em == 1; charge "
                       "the full budget when no prefix is correct"),
        "stored_field_rejected": (
            "per_q.oracle_searches is argmax (prefix_em, -searches) and charges "
            "0 searches to never-correct questions; not used here"),
        "cells": {},
    }
    for tag, label in CELLS:
        row = cell(tag)
        if row is None:
            continue
        row["label"] = label
        out["cells"][tag] = row
    heads = [r["headroom_em"] for r in out["cells"].values()]
    out["headroom_em_min"] = round(min(heads), 2)
    out["headroom_em_max"] = round(max(heads), 2)
    out["headroom_em_mean"] = round(float(np.mean(heads)), 2)
    out["cells_where_oracle_beats_full"] = sum(1 for h in heads if h > 0)
    out["n_cells"] = len(heads)
    (MEAS / "oracle_frontier.json").write_text(json.dumps(out, indent=2) + "\n")

    # plain-language row labels for the paper table
    plain = {
        "A s7": "HotpotQA A, seed 7", "A s11": "HotpotQA A, seed 11",
        "A s19": "HotpotQA A, seed 19", "B": "HotpotQA sample B",
        "C": "HotpotQA sample C", "D": "HotpotQA sample D",
        "E": "HotpotQA sample E", "2Wiki d1": "2Wiki draw 1",
        "2Wiki d2": "2Wiki draw 2", "2Wiki d3": "2Wiki draw 3",
        "MuSiQue d1": "MuSiQue draw 1", "MuSiQue d2": "MuSiQue draw 2",
        "MuSiQue d3": "MuSiQue draw 3", "Wiki 25M": "Wikipedia 25M",
    }
    lines = []
    for tag, label in CELLS:
        r = out["cells"].get(tag)
        if r is None:
            continue
        lines.append(
            f"{plain.get(label, label)} & {r['full_budget']['em']:.2f} & {r['oracle']['em']:.2f} "
            f"& {r['headroom_em']:+.2f} & {r['oracle']['searches']:.2f} & "
            f"{r['oracle']['llm_calls']:.2f} \\\\")
    (MANU / "tables" / "oracle_frontier_rows.tex").write_text(
        "\n".join(lines) + "\n")

    for tag, label in CELLS:
        r = out["cells"].get(tag)
        if r is None:
            print(f"{label:12s} MISSING")
            continue
        print(f"{label:12s} n={r['n']:4d} full {r['full_budget']['em']:6.2f} EM "
              f"@ {r['full_budget']['llm_calls']:.3f} calls | oracle "
              f"{r['oracle']['em']:6.2f} EM @ {r['oracle']['searches']:.3f} "
              f"searches / {r['oracle']['llm_calls']:.3f} calls | headroom "
              f"{r['headroom_em']:+.2f} | stored-field mean "
              f"{r['stored_oracle_searches_mean']:.4f}")
    print(f"headroom EM: min {out['headroom_em_min']:+.2f}, "
          f"max {out['headroom_em_max']:+.2f}, "
          f"mean {out['headroom_em_mean']:+.2f} over {out['n_cells']} cells")


if __name__ == "__main__":
    main()
