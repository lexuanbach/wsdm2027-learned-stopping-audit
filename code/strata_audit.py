"""Is the draw-to-draw EM swing a question-composition artefact?

Motivation: full-budget EM moves 24.50 -> 34.25 across the five
HotpotQA corpus draws, and the guarded rule's non-inferiority verdict moves with
it. If the draws differed in *composition* -- more comparison questions, easier
questions, better gold-support retrieval -- the disagreement would be an
artefact of sampling the wrong strata rather than genuine question-level
variance.

This script measures that directly, from the logged per-question fields the
runs already carry (`type`, `level`, `support_sentences`, `trajectory`). Nothing
is generated, retrieved or re-run.

Strata:
  * question type      -- bridge vs comparison (HotpotQA's own label)
  * level              -- easy/medium/hard (HotpotQA's own label)
  * gold-support recall -- the fraction of a question's gold support sentences
    that appear anywhere in the union of the trajectory's retrieved evidence,
    matched after stripping the "Title: " prefix and normalising to alphanumeric
    tokens. recall == 1 is used as an answerability proxy.

For each stratum we report its size, its full-budget EM, and the paired
guarded-minus-full EM contrast inside it.

Usage:  python3 strata_audit.py     (writes results/measured/strata_audit.json)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats

from analyze_corpus_confirmation import norm, policy_stop

ART = pathlib.Path(__file__).resolve().parents[1]
MEAS = ART / "results" / "measured"
MANU = ART / "outputs"  # generated tables/ and figures/ (not tracked)
(MANU / "tables").mkdir(parents=True, exist_ok=True)
(MANU / "figures").mkdir(parents=True, exist_ok=True)

DRAWS = [
    ("corpus_guarded_qwen3b", "A"),
    ("corpus_guarded_qwen3b_data17", "B"),
    ("corpus_guarded_qwen3b_data23", "C"),
    ("corpus_guarded_qwen3b_data29", "D"),
    ("corpus_guarded_qwen3b_data31", "E"),
]


def strip_title(sentence):
    """Retrieved evidence is 'Title: sentence'; gold support is bare."""
    head, sep, tail = str(sentence).partition(": ")
    return tail if sep else sentence


def support_recall(record):
    gold = [norm(s) for s in record.get("support_sentences", [])]
    gold = [g for g in gold if g]
    if not gold:
        return None
    pool = set()
    for prefix in record["trajectory"]:
        for ev in prefix.get("evidence", []):
            pool.add(norm(strip_title(ev)))
    joined = " || ".join(sorted(pool))
    hit = sum(1 for g in gold if g in joined)
    return hit / len(gold)


def paired(em_guard, em_full):
    """Point contrast in EM points; no new inference, no bootstrap."""
    if not em_guard:
        return None
    d = np.asarray(em_guard, float) - np.asarray(em_full, float)
    return round(100 * float(d.mean()), 2)


def draw(tag):
    path = MEAS / f"run_{tag}.json"
    if not path.exists():
        return None
    records = json.loads(path.read_text())["per_q"]
    g = [r["prefix_em"][policy_stop(r, "guarded_value")] for r in records]
    f = [r["prefix_em"][policy_stop(r, "full_budget")] for r in records]
    recalls = [support_recall(r) for r in records]

    row = {
        "n": len(records),
        "full_budget_em": round(100 * float(np.mean(f)), 2),
        "guarded_em": round(100 * float(np.mean(g)), 2),
        "guarded_minus_full_em": paired(g, f),
        "type_counts": {},
        "level_counts": {},
        "support_recall_mean": round(float(np.mean(
            [x for x in recalls if x is not None])), 4),
        "strata": {},
    }
    for r in records:
        row["type_counts"][r.get("type")] = \
            row["type_counts"].get(r.get("type"), 0) + 1
        row["level_counts"][r.get("level")] = \
            row["level_counts"].get(r.get("level"), 0) + 1

    def stratum(name, mask):
        idx = [i for i, keep in enumerate(mask) if keep]
        if not idx:
            row["strata"][name] = {"n": 0}
            return
        gg = [g[i] for i in idx]
        ff = [f[i] for i in idx]
        row["strata"][name] = {
            "n": len(idx),
            "share": round(len(idx) / len(records), 4),
            "full_budget_em": round(100 * float(np.mean(ff)), 2),
            "guarded_minus_full_em": paired(gg, ff),
        }

    stratum("bridge", [r.get("type") == "bridge" for r in records])
    stratum("comparison", [r.get("type") == "comparison" for r in records])
    stratum("recall_1", [x is not None and x >= 1.0 for x in recalls])
    stratum("recall_lt_1", [x is not None and x < 1.0 for x in recalls])
    return row


def main():
    out = {
        "question": ("does question-type mixture or gold-support retrieval "
                     "recall track the draw-to-draw full-budget EM swing?"),
        "draws": {},
    }
    for tag, label in DRAWS:
        row = draw(tag)
        if row is None:
            continue
        row["label"] = label
        out["draws"][tag] = row

    ems = [r["full_budget_em"] for r in out["draws"].values()]
    bridge_share = [r["strata"]["bridge"]["share"] for r in out["draws"].values()]
    recall = [r["support_recall_mean"] for r in out["draws"].values()]
    r1_share = [r["strata"]["recall_1"]["share"] for r in out["draws"].values()]

    def corr(x):
        """Pearson r against full-budget EM, with its two-sided p (n draws)."""
        if len(set(x)) < 2 or len(x) < 3:
            return None
        r = float(np.corrcoef(x, ems)[0, 1])
        n = len(x)
        if abs(r) >= 1.0:
            return {"r": round(r, 3), "p": 0.0, "n": n}
        t = abs(r) * np.sqrt((n - 2) / (1 - r * r))
        p = float(2 * stats.t.sf(t, n - 2))
        return {"r": round(r, 3), "p": round(p, 3), "n": n}

    def standardise():
        """Direct standardisation of each draw's full-budget EM onto the
        POOLED composition over type x gold-support-recall cells.

        If the draw-to-draw swing were a composition artefact, equalising the
        composition would collapse it. This is the decisive statistic; the
        correlations above are five-point and cannot resolve anything.
        """
        cells, weights = {}, {}
        for tag, _ in DRAWS:
            path = MEAS / f"run_{tag}.json"
            if not path.exists():
                continue
            recs = json.loads(path.read_text())["per_q"]
            rec = [support_recall(r) for r in recs]
            f = [r["prefix_em"][policy_stop(r, "full_budget")] for r in recs]
            per = {}
            for r, x, y in zip(recs, rec, f):
                key = (r.get("type"),
                       "r1" if (x is not None and x >= 1.0) else "r<1")
                per.setdefault(key, []).append(y)
                weights[key] = weights.get(key, 0) + 1
            cells[tag] = per
        total = sum(weights.values())
        share = {k: v / total for k, v in weights.items()}
        out_std = {}
        for tag, per in cells.items():
            num = den = 0.0
            for key, w in share.items():
                if key in per and per[key]:
                    num += w * float(np.mean(per[key]))
                    den += w
            out_std[tag] = round(100 * num / den, 2) if den else None
        vals = [v for v in out_std.values() if v is not None]
        return {"pooled_cell_shares": {f"{k[0]}|{k[1]}": round(v, 4)
                                       for k, v in share.items()},
                "standardised_full_budget_em": out_std,
                "standardised_swing": round(max(vals) - min(vals), 2),
                "raw_swing": round(max(ems) - min(ems), 2)}

    levels = set()
    for r in out["draws"].values():
        levels |= set(r["level_counts"])
    out["summary"] = {
        "n_draws": len(ems),
        "full_budget_em_range": [round(min(ems), 2), round(max(ems), 2)],
        "full_budget_em_swing": round(max(ems) - min(ems), 2),
        "bridge_share_range": [round(min(bridge_share), 4),
                               round(max(bridge_share), 4)],
        "support_recall_range": [round(min(recall), 4), round(max(recall), 4)],
        "recall1_share_range": [round(min(r1_share), 4), round(max(r1_share), 4)],
        "pearson_bridge_share_vs_em": corr(bridge_share),
        "pearson_support_recall_vs_em": corr(recall),
        "pearson_recall1_share_vs_em": corr(r1_share),
        "level_values": sorted(levels),
        "level_is_degenerate": len(levels) == 1,
        "direct_standardisation": standardise(),
    }
    (MEAS / "strata_audit.json").write_text(json.dumps(out, indent=2) + "\n")

    lines = []
    for tag, label in DRAWS:
        r = out["draws"].get(tag)
        if r is None:
            continue
        b, c = r["strata"]["bridge"], r["strata"]["comparison"]
        r1, r0 = r["strata"]["recall_1"], r["strata"]["recall_lt_1"]
        lines.append(
            f"{label} & {r['full_budget_em']:.2f} & {b['n']}/{c['n']} & "
            f"{b['full_budget_em']:.1f}/{c['full_budget_em']:.1f} & "
            f"{r['support_recall_mean']:.3f} & {r1['n']} & "
            f"{r1['full_budget_em']:.1f} & {r0['full_budget_em']:.1f} & "
            f"{r1['guarded_minus_full_em']:+.2f} & "
            f"{r0['guarded_minus_full_em']:+.2f} \\\\")
    (MANU / "tables" / "strata_audit_rows.tex").write_text(
        "\n".join(lines) + "\n")

    for tag, label in DRAWS:
        r = out["draws"].get(tag)
        if r is None:
            print(f"{label} MISSING")
            continue
        print(f"{label}: n={r['n']} full={r['full_budget_em']:.2f} "
              f"G-F={r['guarded_minus_full_em']:+.2f} "
              f"types={r['type_counts']} levels={r['level_counts']} "
              f"recall={r['support_recall_mean']:.3f} "
              f"recall1_n={r['strata']['recall_1']['n']} "
              f"({r['strata']['recall_1']['share']:.3f}) "
              f"recall1_EM={r['strata']['recall_1']['full_budget_em']:.2f} "
              f"recall<1_EM={r['strata']['recall_lt_1']['full_budget_em']:.2f}")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
