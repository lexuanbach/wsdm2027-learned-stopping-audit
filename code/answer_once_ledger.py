#!/usr/bin/env python3
"""Second cost currency: the answer-once (conventional-agent) call ledger.

The manuscript's primary accounting charges every policy 1+2T live LLM calls
(one initial answer, then a query and an answer per search), because the FBG
critic reads five features that need an answer at *every* prefix
(answer_changed, answer_length, answer_grounding, stable_two,
distinct_answers).  A conventional agent that generates only once, at the end,
pays 1+T instead.  This script reports the comparison under that convention.

This script re-prices the measured runs under it.  Nothing is re-executed: the
per-prefix search counts and EM come straight from the corpus_confirmation*
registries.  Policies whose stopping signal needs a per-prefix answer keep
1+2T; policies that do not (the fixed budgets) drop to 1+T.

Output: results/measured/answer_once_ledger.json
"""
import json
import pathlib
import sys

ARTIFACT_ROOT = pathlib.Path(__file__).resolve().parent.parent
ROOT = ARTIFACT_ROOT
MEAS = ARTIFACT_ROOT / "results" / "measured"

# Policies whose stopping signal requires an answer at every prefix, so they
# cannot use the answer-once convention.  Justification per policy:
#   guarded_value/value_only/repeat_conditioned_gain -- FBG critic features
#   answer_repeat                                    -- compares consecutive answers
#   semantic_guard                                   -- NLI over consecutive answers
#   tasr                                             -- native answer-token log-probs
NEEDS_PER_PREFIX_ANSWER = {
    "guarded_value", "value_only", "repeat_conditioned_gain",
    "answer_repeat", "semantic_guard", "tasr",
}
# Policies that only need a query per search plus one final answer.
ANSWER_ONCE_CAPABLE = {"full_budget", "fixed_0", "fixed_1", "fixed_2",
                       "fixed_3", "fixed_4", "fixed_5", "fixed_6"}
SKIP = {"gold_title_coverage_diagnostic", "gold_sentence_coverage_diagnostic"}

SIMPLE_FAMILY = lambda name: name.startswith("fixed_") or name == "answer_repeat"


def price(name, m):
    """Return (primary_calls, answer_once_calls) for one measured policy."""
    searches = m["searches"]
    primary = m["llm_calls"]
    if name in NEEDS_PER_PREFIX_ANSWER:
        return primary, primary          # cannot answer once
    if name in ANSWER_ONCE_CAPABLE:
        return primary, 1.0 + searches   # T queries + 1 final answer
    return None


def dominates(a, b, tol=1e-9):
    """a dominates b: no worse on EM and cost, strictly better on one."""
    better_em = a["em"] > b["em"] + tol
    better_cost = a["calls"] < b["calls"] - tol
    worse_em = a["em"] < b["em"] - tol
    worse_cost = a["calls"] > b["calls"] + tol
    return (not worse_em) and (not worse_cost) and (better_em or better_cost)


def analyse_cell(path):
    d = json.loads(path.read_text())
    methods = d.get("methods", {})
    rows = {}
    for name, m in methods.items():
        if name in SKIP or not isinstance(m, dict) or "searches" not in m:
            continue
        p = price(name, m)
        if p is None:
            continue
        primary, once = p
        rows[name] = {
            "em": m["em"],
            "searches": m["searches"],
            "primary_calls": primary,
            "answer_once_calls": once,
            "answer_once_applies": name in ANSWER_ONCE_CAPABLE,
        }
    if "guarded_value" not in rows:
        return None

    out = {"cell": path.stem, "n_test": d.get("n_test"), "policies": rows}
    for currency, key in (("primary", "primary_calls"),
                          ("answer_once", "answer_once_calls")):
        pts = {n: {"em": r["em"], "calls": r[key]} for n, r in rows.items()}
        simple = {n: v for n, v in pts.items() if SIMPLE_FAMILY(n)}
        fixed_only = {n: v for n, v in pts.items() if n.startswith("fixed_")}
        verdicts = {}
        for n, v in pts.items():
            if SIMPLE_FAMILY(n):
                continue
            dom = sorted(s for s, sv in simple.items() if dominates(sv, v))
            verdicts[n] = {"dominated_by": dom, "verdict": "L" if dom else "N"}
        # is repetition itself dominated by a plain fixed budget?
        rep = pts.get("answer_repeat")
        rep_dom = sorted(s for s, sv in fixed_only.items()
                         if rep and dominates(sv, rep)) if rep else []
        max_simple_cost = max((v["calls"] for v in simple.values()), default=None)
        g = pts["guarded_value"]
        max_fixed_cost = max((v["calls"] for v in fixed_only.values()), default=None)
        out[currency] = {
            "verdicts": verdicts,
            "max_fixed_budget_calls": max_fixed_cost,
            "guarded_cost_inside_fixed_range": (
                max_fixed_cost is not None and g["calls"] <= max_fixed_cost + 1e-9),
            "answer_repeat_dominated_by_fixed": rep_dom,
            "max_simple_family_calls": max_simple_cost,
            "guarded_calls": g["calls"],
            # a matched-cost mixture over the simple family only exists if the
            # guard's cost lies inside the family's cost range
            "guarded_cost_inside_simple_range": (
                max_simple_cost is not None and g["calls"] <= max_simple_cost + 1e-9),
        }
    return out


def main():
    # Keep this robustness ledger on the declared 18-cell common family.
    # The prospectively frozen news cell is reported separately and must not
    # silently change the denominator of an already defined analysis family.
    cells = [p for p in sorted(MEAS.glob("corpus_confirmation*.json"))
             if "extnews" not in p.name]
    results = [r for r in (analyse_cell(p) for p in cells) if r]
    if not results:
        print("no cells found", file=sys.stderr)
        return 1
    summary = {}
    for currency in ("primary", "answer_once"):
        n_guard_dom = sum(1 for r in results
                          if r[currency]["verdicts"]["guarded_value"]["verdict"] == "L")
        n_rep_dom = sum(1 for r in results
                        if r[currency]["answer_repeat_dominated_by_fixed"])
        n_outside = sum(1 for r in results
                        if not r[currency]["guarded_cost_inside_simple_range"])
        n_rep_present = sum(1 for r in results if "answer_repeat" in r["policies"])
        summary[currency] = {
            "n_cells": len(results),
            "guarded_dominated_by_simple_family": n_guard_dom,
            "answer_repeat_present": n_rep_present,
            "answer_repeat_dominated_by_a_fixed_budget": n_rep_dom,
            "cells_where_guard_cost_exceeds_simple_family_range": n_outside,
            "cells_where_guard_cost_exceeds_fixed_budget_range": sum(
                1 for r in results
                if not r[currency]["guarded_cost_inside_fixed_range"]),
        }
    primary_cell = next(r for r in results if r["cell"] == "corpus_confirmation")
    out = {
        "description": ("Second cost currency on the declared 18-cell common "
                        "family. Policies whose stopping signal "
                        "needs an answer at every prefix keep 1+2T live calls; "
                        "policies that can defer generation to the end pay 1+T. "
                        "No run was re-executed; EM and search counts are the "
                        "measured values in corpus_confirmation*.json."),
        "answer_once_capable": sorted(ANSWER_ONCE_CAPABLE),
        "needs_per_prefix_answer": sorted(NEEDS_PER_PREFIX_ANSWER),
        "summary": summary,
        "primary_cell": primary_cell,
        "cells": results,
    }
    dest = MEAS / "answer_once_ledger.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")

    # LaTeX rows for the supplement table (primary cell, both currencies)
    label = {"fixed_0": "fixed-0", "fixed_1": "fixed-1", "fixed_2": "fixed-2",
             "fixed_3": "fixed-3", "fixed_4": "full budget",
             "full_budget": "full budget", "answer_repeat": "answer repetition",
             "semantic_guard": "FBG + NLI guard", "value_only": "FBG-only",
             "tasr": "full TASR", "guarded_value": "FBG + repeat guard",
             "repeat_conditioned_gain": "repeat-conditioned gain"}
    pts = {n: {"em": r["em"], "calls": r["answer_once_calls"]}
           for n, r in primary_cell["policies"].items()}
    simple = {n: v for n, v in pts.items() if SIMPLE_FAMILY(n)}
    tex = ROOT / "outputs" / "tables" / "answer_once_rows.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for n, r in sorted(primary_cell["policies"].items(),
                       key=lambda kv: (kv[1]["answer_once_calls"], kv[0])):
        if n == "fixed_4":
            continue          # identical EM and cost to full_budget in this cell
        mark = "" if r["answer_once_applies"] else "$^{\\dagger}$"
        dom = sorted({label.get(x, x) for x, sv in simple.items()
                      if x != n and dominates(sv, pts[n])})
        domtxt = ", ".join(dom) if dom else "NA"
        lines.append(f"{label.get(n, n)}{mark} & {r['em']:.2f} & "
                     f"{r['primary_calls']:.2f} & {r['answer_once_calls']:.2f} & "
                     f"{domtxt} \\\\")
    tex.write_text("\n".join(lines) + "\n")
    print(f"wrote {tex}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
