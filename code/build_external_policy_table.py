"""Rows of the external news-cell table (main paper Table 6, supplement Table 3).

Reads results/round10/external_cell_extnews_guarded_qwen3b.json and writes
outputs/tables/external_policy_rows.tex: for each learned policy, test EM,
exact LLM calls, and the paired EM difference (with 95% interval) against the
randomized simple-policy mixture at equal mean calls.
"""
import json
import pathlib
from decimal import ROUND_HALF_UP, Decimal

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "round10" / "external_cell_extnews_guarded_qwen3b.json"
OUT = ROOT / "outputs" / "tables" / "external_policy_rows.tex"
POLICIES = [("FBG-only", "value_only"), ("Guarded FBG", "guarded_value"),
            ("Future-improvement clf.", "stop_classifier"),
            ("Error-head gate", "sufficiency_gate")]


def r2(x):
    """Round half up to two decimals, as typeset in the paper."""
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def signed(x):
    v = r2(x)
    return f"+{v}" if v >= 0 else f"{v}"


def main():
    cell = json.loads(SRC.read_text())["round9_cell_analysis"]
    rows = []
    for label, key in POLICIES:
        pt = cell["pooled_pareto"]["points"][key]
        d = cell["equal_cost"]["calls"][key]["delta_em_vs_mixture_at_equal_cost"]
        lo, hi = d["ci95"]
        rows.append(f"{label} & {r2(pt['em'])} & {r2(pt['calls'])} & "
                    f"${signed(d['mean_em_points'])}$ & $[{r2(lo)},{r2(hi)}]$ \\\\")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(rows) + "\n")
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
