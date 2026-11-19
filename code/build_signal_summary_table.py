"""Build the main cross-signal summary from the reconciled round-10 outputs.

Point categories come from extended_decision_matrix.json. Cellwise and
family-wise matched-cost counts come from mixture_wins_multiplicity.json.
The gold-support classifier has cellwise matched-cost intervals but no retained
per-question vectors for simultaneous correction, so those entries are NA.
"""
from __future__ import annotations

import json
import pathlib


ART = pathlib.Path(__file__).resolve().parents[1]
MANU = ART / "outputs"  # generated tables/ (not tracked)
(MANU / "tables").mkdir(parents=True, exist_ok=True)
ROUND10 = ART / "results" / "round10"

ROWS = [
    ("FBG-only", "value_only", "value_only"),
    ("Guarded FBG", "guarded_value", "guarded_value"),
    ("Future-improvement clf.", "stop_classifier", "stop_classifier"),
    ("Error-head gate", "sufficiency_gate", "sufficiency_gate"),
    ("Gold-support sufficiency", "gold_support_sufficiency_clf_same9", None),
    ("Full TASR", "tasr", "tasr"),
    ("Stop-RAG", "stoprag", "stoprag"),
]


def main() -> None:
    matrix = json.loads((ROUND10 / "extended_decision_matrix.json").read_text())
    point = matrix["summary"]
    inference = json.loads(
        (ROUND10 / "mixture_wins_multiplicity.json").read_text()
    )["policies"]
    lines = []
    for label, point_key, inference_key in ROWS:
        p = point[point_key]
        d = p["dominance"]
        cellwise = p["vs_matched_mixture"]
        n_matched = sum(cellwise.values())
        if inference_key is None:
            bonf, maxt = "NA", "NA"
        else:
            q = inference[inference_key]
            assert q["n_cells"] == n_matched, (label, q["n_cells"], n_matched)
            counts = q["counts"]
            assert counts["marginal_win"] == cellwise["win"]
            assert counts["marginal_loss"] == cellwise["loss"]
            bonf = str(counts["bonferroni_win"])
            maxt = str(counts["maxt_win"])
        effects = [
            cell["policies"][point_key]["vs_matched_mixture_on_calls"]
            ["delta_em"]["mean_em_points"]
            for cell in matrix["cells"]
            if point_key in cell["policies"]
            and cell["policies"][point_key].get("vs_matched_mixture_on_calls")
        ]
        effect_range = f"${min(effects):.2f},{max(effects):.2f}$"
        lines.append(
            f"{label} & {p['cells_covered']}/{n_matched} & "
            f"{d['D']}/{d['F']}/{d['I']} & {effect_range} & "
            f"{cellwise['win']}/{cellwise['loss']} & "
            f"{bonf} & {maxt} \\\\"
        )
    out = MANU / "tables" / "signal_summary_rows.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} rows)")


if __name__ == "__main__":
    main()
