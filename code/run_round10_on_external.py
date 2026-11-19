"""Run the round-10 offline audits on the external cell only.

The external cell is deliberately NOT a member of the frozen 26-cell matrix,
so `frontier_analyses.FRESH` does not list it.  This driver appends it to
that list in-process (never on disk) and reruns the round-10 audits restricted
to it, writing to a separate output directory.

    python3 run_round10_on_external.py --tag extnews_guarded_qwen3b --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import frontier_analyses as mra

CELL = ("extnews_guarded_qwen3b",
        "corpus_confirmation_extnews_guarded_qwen3b.json",
        "MultiHopRAG news / Qwen2.5-3B (external)", "extnews")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=CELL[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mra.FRESH = list(mra.FRESH) + [CELL]

    import sufficiency_classifier as sc
    import calibration_audit as ca
    import cost_latency_audit as cl
    import strata_audit_extra as se

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = {}
    payload["sufficiency_classifier"] = sc.run_cell(*CELL)
    payload["calibration"] = ca.run_cell(*CELL)
    payload["cost_latency"] = cl.audit(*CELL)
    payload["strata"] = se.cell_strata(*CELL)
    (outdir / f"external_audits_{args.tag}.json").write_text(
        json.dumps({"what": "round-10 offline audits restricted to the "
                            "external non-Wikipedia cell",
                    "cell": CELL[0], **payload}, indent=1, default=str))

    s = payload["sufficiency_classifier"]["variants"]["same9"]
    c = payload["calibration"]
    print("sufficiency classifier same9: tau=%s EM=%.2f calls=%.2f AUC=%s "
          "base=%.3f" % (s["tau"], s["test_em"], s["test_calls"],
                         s["crossfit_auc"], s["label_base_rate"]))
    print("  equal-cost vs mixture (calls):",
          s["equal_cost"]["calls"]["delta_em_vs_mixture_at_equal_cost"],
          s["equal_cost"]["calls"]["verdict"])
    e = c["error_head"]
    print("error head: ECE_train=%.4f Brier_train=%.4f ECE_test=%.4f "
          "Brier_test=%.4f" % (e["train_crossfit"]["ece_15_equal_width"],
                               e["train_crossfit"]["brier"],
                               e["test_prefixes"]["ece_15_equal_width"],
                               e["test_prefixes"]["brier"]))
    r = c["selective_risk_at_guarded_stop"]
    print("AURC fbg=%.4f err=%.4f random=%.4f" % (
        r["aurc_fbg_value_head"], r["aurc_error_head"],
        r["aurc_random_ordering_mean_of_20"]))
    q = payload["cost_latency"]
    print("latency per call p50=%.3f p95=%.3f | per question p50=%.2f p95=%.2f"
          % (q["per_llm_call_latency_seconds"]["p50"],
             q["per_llm_call_latency_seconds"]["p95"],
             q["per_question_wall_seconds_full_budget"]["p50"],
             q["per_question_wall_seconds_full_budget"]["p95"]))
    st = payload["strata"]["strata"]
    print("hop depth:", st["hop_depth"] if st["hop_depth"] else
          payload["strata"].get("hop_depth_note"))
    print("ambiguity:", {k: (v["mean_em_points"], v["verdict"])
                         for k, v in st["retrieval_ambiguity_jaccard"].items()})
    print("wrote", outdir / f"external_audits_{args.tag}.json")


if __name__ == "__main__":
    sys.exit(main())
