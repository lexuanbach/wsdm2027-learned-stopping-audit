"""LS11 retrospective check: pooled policy training beyond 90 episodes.

For each of the 14 primary corpus-index cells, the guarded/value FBG pipeline
(cross-fit -> frozen band selection -> refit -> test replay) is rerun with the
target cell's 90 training chains plus training chains (never certificate or
test chains) from donor cells. Leakage control: a donor chain is dropped if its
question id is in the target cell's test set; at most one trajectory is kept per
question id (target cell first, then donors in fixed order), so cross-fitting
folds are question-disjoint. Verdicts use the target cell's own simple reference
family and training-selected fixed budget, exactly as in round9_analyses.

Pools: own (90, fresh refit baseline), env (same retrieval environment), all
(all 14 cells, cross-environment). No endpoint calls; saved outputs only.
"""
from pathlib import Path
import sys, json, collections
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import round9_analyses as r9  # noqa: E402
import run_experiment as rx  # noqa: E402


def env_of(tag):
    if tag.startswith("corpus2wiki"):
        return "2wiki"
    if tag.startswith("corpusmusique"):
        return "musique"
    if tag.startswith("wiki25m"):
        return "wiki25m"
    return "hotpot_corpus"


def fit_block(cell):
    """Training-chain rows of one cell: list of (qid, X rows, y_fbg, y_err)."""
    out = []
    for cid in cell.fit_ids:
        ch = cell.all_chains[cid]
        out.append((str(cell.traces[cid]["id"]), cell.X[ch], cell.y_val[ch], cell.y_err[ch]))
    return out


def pooled_pipeline(cell, blocks, folds=5, xseed=3, head="gbr"):
    X = np.concatenate([b[1] for b in blocks])
    y = np.concatenate([b[2] for b in blocks])
    ye = np.concatenate([b[3] for b in blocks]).astype(int)
    chains, k = [], 0
    for b in blocks:
        chains.append(list(range(k, k + len(b[1]))))
        k += len(b[1])
    oof = r9.crossfit(X, y, chains, folds, xseed, head)
    out = {}
    test_X = cell.feats.reshape(-1, cell.feats.shape[-1])
    scores = r9.fit_predict(X, y, test_X, xseed, head).reshape(cell.n, cell.width)
    for rule in ["guarded", "value"]:
        th, meta = rx.select_frozen_lambda(oof, X, ye, chains, r9.GRID6, rule, band=1.0)
        stops = r9.replay_scores(scores, cell.answers, th, rule)
        out[rule] = {"threshold": th, "reason": meta["reason"], "stops": stops}
    return out, len(chains)


def main():
    cells = [r9.Cell(*s) for s in r9.fresh_cells()]
    blocks = {c.tag: fit_block(c) for c in cells}
    rows = []
    for cell in cells:
        test_ids = {str(r["id"]) for r in cell.records}
        assert not ({b[0] for b in blocks[cell.tag]} & test_ids), cell.tag
        simple = cell.simple_points()
        sn = {k: (v["em"], v["calls"]) for k, v in simple.items()}
        hull = r9.upper_hull({k: {"cost": v["calls"], "em": v["em"], "em_vec": v["em_vec"]}
                              for k, v in simple.items()})
        row = {"tag": cell.tag, "env": env_of(cell.tag), "n_test": cell.n, "pools": {}}
        for pool in ["own", "env", "all"]:
            donors = [c.tag for c in cells if c.tag != cell.tag and (
                pool == "all" or (pool == "env" and env_of(c.tag) == env_of(cell.tag)))]
            seen, chosen, dropped_test, dropped_dup = set(), [], 0, 0
            for tag in [cell.tag] + donors:
                for b in blocks[tag]:
                    if b[0] in test_ids:
                        dropped_test += 1
                        continue
                    if b[0] in seen:
                        dropped_dup += 1
                        continue
                    seen.add(b[0])
                    chosen.append(b)
            fit, n_train = pooled_pipeline(cell, chosen)
            res = {"n_train_episodes": n_train, "donor_cells": len(donors),
                   "dropped_test_overlap": dropped_test, "dropped_duplicate_question": dropped_dup}
            for rule, f in fit.items():
                p = cell.point(f["stops"])
                v, dom = r9.verdict_eps(p["em"], p["calls"], sn, f"fixed_{cell.bstar}", 0.0)
                res[rule] = {"threshold": f["threshold"], "selection_reason": f["reason"],
                             "em": p["em"], "calls": p["calls"],
                             "category": {"W": "I", "T": "F", "L": "D"}[v], "dominated_by": dom}
                mix, n1, n2, w = r9.hull_mixture(hull, p["calls"], simple)
                pc = r9.paired_ci(p["em_vec"] - mix)
                res[rule]["vs_matched_mixture"] = {
                    "mixture_of": [n1, n2], "weight_first": round(w, 4), "delta_em": pc,
                    "interval_verdict": "win" if pc["ci95"][0] > 0 else (
                        "loss" if pc["ci95"][1] < 0 else "unresolved")}
            row["pools"][pool] = res
        rows.append(row)
        print(cell.tag, {k: (v["n_train_episodes"], v["guarded"]["category"], v["guarded"]["em"],
                             v["guarded"]["calls"]) for k, v in row["pools"].items()}, flush=True)
    summary = {}
    for pool in ["own", "env", "all"]:
        summary[pool] = {"n_train_range": [min(r["pools"][pool]["n_train_episodes"] for r in rows),
                                           max(r["pools"][pool]["n_train_episodes"] for r in rows)]}
        for rule in ["guarded", "value"]:
            summary[pool][rule] = {
                "counts_DFI": dict(collections.Counter(r["pools"][pool][rule]["category"] for r in rows)),
                "mean_em": float(np.mean([r["pools"][pool][rule]["em"] for r in rows])),
                "mean_calls": float(np.mean([r["pools"][pool][rule]["calls"] for r in rows])),
                "matched_mixture_interval_verdicts": dict(collections.Counter(
                    r["pools"][pool][rule]["vs_matched_mixture"]["interval_verdict"] for r in rows)),
                "matched_mixture_mean_delta_em": float(np.mean(
                    [r["pools"][pool][rule]["vs_matched_mixture"]["delta_em"]["mean_em_points"] for r in rows]))}
    out = {"description": __doc__.strip(), "cells": rows, "summary": summary}
    (ROOT / "results/retrospective/pooled_training.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def h2(x):
    """Round half up to 2 decimals, matching the manuscript convention."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def write_table():
    """Supplement rows from results/retrospective/pooled_training.json."""
    d = json.loads((ROOT / "results/retrospective/pooled_training.json").read_text())["summary"]
    names = {"own": "Own cell", "env": "Same environment", "all": "All 14 cells"}
    lines = []
    for pool in ["own", "env", "all"]:
        lo, hi = d[pool]["n_train_range"]
        for rule, rname in [("guarded", "Guarded"), ("value", "FBG-only")]:
            s = d[pool][rule]
            c, iv = s["counts_DFI"], s["matched_mixture_interval_verdicts"]
            lines.append(
                f"{names[pool]} & {lo if lo == hi else f'{lo}--{hi}'} & {rname} & "
                f"{c.get('D', 0)}/{c.get('F', 0)}/{c.get('I', 0)} & {h2(s['mean_em'])} & "
                f"{h2(s['mean_calls'])} & ${'+' if s['matched_mixture_mean_delta_em'] >= 0 else ''}{h2(s['matched_mixture_mean_delta_em'])}$ & "
                f"{iv.get('win', 0)}/{iv.get('unresolved', 0)}/{iv.get('loss', 0)} \\\\")
    (ROOT / "outputs/tables/pooled_training_rows.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    if "--table" not in sys.argv:
        main()
    write_table()
