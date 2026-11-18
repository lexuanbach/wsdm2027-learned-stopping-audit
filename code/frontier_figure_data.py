"""Machine-readable data for a three-axis frontier figure.

The current fig_frontier_main.pdf plots EM versus LLM calls only, and does not
draw the simple-family hull or the matched randomized mixture.  Everything
needed for the token and wall-time panels and for the hull/mixture overlay is
already computed in results/measured/round9_analyses.json; this script simply
extracts it into one flat file so the figure can be rebuilt without recomputing
anything.

    python3 frontier_figure_data.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib

MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
AXES = ["calls", "input_tokens", "wall_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r9 = json.loads((MEAS / "round9_analyses.json").read_text())
    cells = []
    for cell in r9["fresh_cells"]:
        eq = cell.get("equal_cost", {})
        entry = {"tag": cell.get("tag"), "label": cell.get("label"),
                 "pooled_pareto": cell.get("pooled_pareto"), "axes": {}}
        for ax in AXES:
            block = eq.get(ax)
            if not block:
                continue
            entry["axes"][ax] = {
                "hull_vertices": block.get("hull_vertices"),
                "learned_points": {k: v for k, v in block.items()
                                   if k != "hull_vertices"},
            }
        cells.append(entry)
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "frontier_figure_data.json").write_text(json.dumps(
        {"what": "hull vertices, matched randomized-mixture points and learned "
                 "policy points on all three measured cost axes, extracted "
                 "from round9_analyses.json for a three-panel frontier figure",
         "axes": AXES,
         "note": "no number is recomputed here; this is an extraction",
         "cells": cells}, indent=1))
    print("wrote", outdir / "frontier_figure_data.json",
          "with", len(cells), "cells and axes", AXES)


if __name__ == "__main__":
    main()
