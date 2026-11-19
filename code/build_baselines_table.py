"""Emit the adaptive-retrieval baselines table from rev2 measured runs.

Reads run_rev2_*.json (which carry selfrag/flare/skr/adaptive fields) and
writes outputs/tables/baselines_rows.tex with one row block per
(dataset, backbone): the four prompted baselines, ReAct, and VoI at its
matched-accuracy point.
"""
import json
import pathlib

import numpy as np

MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
ART = pathlib.Path(__file__).resolve().parents[1]
MANU = ART / "outputs"  # generated tables/ and figures/ (not tracked)
(MANU / "tables").mkdir(parents=True, exist_ok=True)
(MANU / "figures").mkdir(parents=True, exist_ok=True)
TABDIR = MANU / "tables"

RUNS = [
    ("rev2_hotpot_qwen3b", "HotpotQA", "Qwen2.5-3B"),
    ("rev2_hotpot_llama3b", "HotpotQA", "Llama-3.2-3B"),
    ("rev2_musique_qwen3b", "MuSiQue", "Qwen2.5-3B"),
    ("rev2_musique_llama3b", "MuSiQue", "Llama-3.2-3B"),
]
BASE = [("Self-RAG (prompted)", "selfrag"), ("FLARE-instruct", "flare"),
        ("SKR-prompt", "skr"), ("Adaptive-RAG (prompted)", "adaptive")]
REAL = [("Self-RAG (released 7B)", "selfrag_released"),
        ("FLARE-direct", "flare_direct"),
        ("SKR$_{\\mathrm{knn}}$", "skr_knn"),
        ("Adaptive-RAG (trained)", "adaptive_rag_trained")]


def real_block(tag):
    """Rows from real_baselines_<short>.json when it exists."""
    short = tag.replace("rev2_", "")
    p = MEAS / f"real_baselines_{short}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    # merge the raw-format Self-RAG correction if present
    fix = MEAS / "real_baselines_selfrag_fix.json"
    if fix.exists():
        d["selfrag_released"] = json.loads(fix.read_text())["selfrag_released"]
    rows = []
    for name, key in REAL:
        if key in d:
            rows.append((name, d[key]["em"], d[key]["s"]))
    return rows or None


def stats(Q, key):
    em = 100 * float(np.mean([r[f"{key}_em"] for r in Q]))
    s = float(np.mean([r[f"{key}_searches"] for r in Q]))
    return em, s


def voi_frozen(tag, d):
    """VoI at the FROZEN lambda* (cross-fitted selection on training chains
    only; see frozen_protocol.py) -- no test-set operating-point selection."""
    import frozen_protocol as fp
    Q = d["per_q"]
    ref = 100 * float(np.mean([r["prefix_em"][-1] for r in Q]))
    X, y_vtg, y_err, chains = fp.load_chains(tag)
    vscores, _ = fp.oof_scores(X, y_vtg, y_err, chains)
    grid = [float(x) for x in d["lam_grid"]]
    lam, _meta = fp.select_threshold(vscores, X, y_err, chains, grid, "lt")
    stops = [fp.test_stop_index(r, lam) for r in Q]
    em = 100 * float(np.mean([r["prefix_em"][t] for r, t in zip(Q, stops)]))
    s = float(np.mean([r["prefix_searches"][t] for r, t in zip(Q, stops)]))
    return ref, (em, s), lam


def main():
    lines = []
    for tag, ds, bb in RUNS:
        p = MEAS / f"run_{tag}.json"
        if not p.exists():
            p = MEAS / f"run_{tag}.partial.json"
            if not p.exists():
                continue
        d = json.loads(p.read_text())
        Q = d["per_q"]
        ref, (vem, vs), _lam = voi_frozen(tag, d)
        n = len(Q)
        rb = real_block(tag)
        if rb is None:
            continue          # real-methods table only; skip prompted-only blocks
        # (cell identity lives in the table caption; a \multicolumn header
        # here breaks under acmart v2.20's tabular patches)
        for name, em, s in rb:
            lines.append(f"\\;{name} & {em:.1f} & {s:.2f} & {round(100*(1-s/4.0))}\\% \\\\")
        lines.append(f"\\;ReAct (full budget) & {ref:.1f} & 4.00 & 0\\% \\\\")
        lines.append(f"\\;\\textbf{{FBG-only}} & \\textbf{{{vem:.1f}}} & "
                     f"\\textbf{{{vs:.2f}}} & "
                     f"\\textbf{{{round(100*(1-vs/4.0))}\\%}} \\\\")
        lines.append("\\addlinespace")
    TABDIR.mkdir(exist_ok=True)
    (TABDIR / "baselines_rows.tex").write_text("\n".join(lines))
    print(f"[wrote] {TABDIR/'baselines_rows.tex'} ({len(lines)} lines)")
    for ln in lines:
        print(" ", ln)


if __name__ == "__main__":
    main()
