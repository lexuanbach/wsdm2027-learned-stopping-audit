"""Emit outputs/tables/whowins_rows.tex (supplement Table 5, frontier summary).

Every cell is read from results/measured/round9_analyses.json; nothing is
typed in.  Columns:

  Nondom.   summary.cells_<family>_nondominated over the cells that family was
            actually run on. When the round-10 matrix is available, Stop-RAG
            uses its expanded 14-cell evaluation rather than the earlier
            five-cell three-currency subset.
  Band      summary.band_winner_counts["1.0"/"2.0"/"3.0"]: how often the family
            is the CHEAPEST policy within that EM band of the full budget.
            The single FBG win in the 1-EM band is semantic_guard, the
            exploratory NLI-augmented guard, and is reported on its own row.
  dEM       summary.equal_{calls,input_tokens,wall_seconds}_<policy>.mean_delta_em
            against the randomized simple-family mixture at matched cost.
"""
from __future__ import annotations

import json
import pathlib

MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
ART = pathlib.Path(__file__).resolve().parents[1]
MANU = ART / "outputs"  # generated tables/ and figures/ (not tracked)
(MANU / "tables").mkdir(parents=True, exist_ok=True)
(MANU / "figures").mkdir(parents=True, exist_ok=True)

# label, nondominated-count key, denominator key, band member policies,
# equal-cost policy key (None = not computed for this family)
ROWS = [
    ("Fixed budgets", "cells_fixed2_nondominated", "n_fresh_cells",
     ["fixed_0", "fixed_1", "fixed_2", "fixed_3", "fixed_4"], None),
    ("Answer repetition", "cells_repeat_nondominated", "n_fresh_cells",
     ["answer_repeat"], None),
    ("FBG-only", "cells_value_nondominated", "n_fresh_cells",
     ["value_only"], "value_only"),
    ("FBG + repeat guard", "cells_guarded_nondominated", "n_fresh_cells",
     ["guarded_value", "guarded"], "guarded_value"),
    ("FBG + NLI guard", None, None, ["semantic_guard"], None),
    ("Stop classifier", "cells_stop_classifier_nondominated", "n_fresh_cells",
     ["stop_classifier"], "stop_classifier"),
    ("Full TASR", "cells_tasr_nondominated", "cells_with_tasr", ["tasr"], "tasr"),
    ("Stop-RAG", "cells_stoprag_nondominated", "cells_stoprag_present",
     ["stoprag"], "stoprag"),
    ("Error-head gate", "cells_sufficiency_nondominated", "n_fresh_cells",
     ["sufficiency"], "sufficiency_gate"),
    ("Verifier-inspired coverage", "cells_halt_nondominated", "cells_halt_present",
     ["halt"], None),
]
RULE_AFTER = {"Answer repetition", "FBG + NLI guard"}


def main() -> None:
    s = json.loads((MEAS / "round9_analyses.json").read_text())["summary"]
    extended_path = ART / "results" / "round10" / "extended_decision_matrix.json"
    extended = json.loads(extended_path.read_text()) if extended_path.exists() else None
    bands = s["band_winner_counts"]
    checks = {b: 0 for b in ("1.0", "2.0", "3.0")}
    lines = []
    for label, nkey, dkey, members, cost in ROWS:
        nd = "NA" if nkey is None else f"{s[nkey]} of {s[dkey]}"
        band = []
        for b in ("1.0", "2.0", "3.0"):
            v = sum(bands[b].get(m, 0) for m in members)
            checks[b] += v
            band.append(str(v))
        if cost is None:
            cells = ["NA", "NA", "NA"]
        else:
            cells = [(f"${s[f'equal_{k}_{cost}']['mean_delta_em']:+.2f}$"
                      if f"equal_{k}_{cost}" in s else "NA")
                     for k in ("calls", "input_tokens", "wall_seconds")]
        if label == "Stop-RAG" and extended is not None:
            stoprag = extended["summary"]["stoprag"]
            nondom = stoprag["dominance"]["F"] + stoprag["dominance"]["I"]
            nd = f"{nondom} of {stoprag['cells_covered']}"
            deltas = [
                cell["policies"]["stoprag"]["vs_matched_mixture_on_calls"]
                    ["delta_em"]["mean_em_points"]
                for cell in extended["cells"]
                if "stoprag" in cell["policies"]
            ]
            band = ["NA", "NA", "NA"]
            cells = [f"${sum(deltas) / len(deltas):+.2f}$", "NA", "NA"]
        lines.append(f"{label} & {nd} & {'/'.join(band)} & "
                     + " & ".join(cells) + " \\\\")
        if label in RULE_AFTER:
            lines.append("\\midrule")
    for b, total in checks.items():
        assert total == s["n_fresh_cells"], (b, total)
    out = MANU / "tables" / "whowins_rows.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len([l for l in lines if l != chr(92) * 2])} rows)")


if __name__ == "__main__":
    main()
