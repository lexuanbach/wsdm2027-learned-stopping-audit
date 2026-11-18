"""Build the manuscript figures for the FROZEN-selection protocol.

Replaces build_real_figures.py's test-selected views:
  * fig_success_cost.pdf : primary-backbone frontier with a chain-bootstrap
    uncertainty band on the VoI sweep (Pareto-front uncertainty, not just
    point uncertainty) and the two FROZEN operating points (per-cell
    lambda*, global lambda_g) marked -- the sweep itself is labelled as a
    diagnostic sensitivity curve, since no test-selected point is reported.
  * fig_risk.pdf : corrected-SGR risk/coverage (delta-splitting binary
    search, calibrated on the deployed value-stop units at lambda*).
  * fig_difficulty.pdf : per-type search allocation at the frozen lambda*.

Inputs: run_<tag>.json + frozen_protocol.json (run frozen_protocol.py first).
"""
import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import paperfig as P

MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
ART = pathlib.Path(__file__).resolve().parents[1]
MANU = ART / "manuscript"
if not MANU.exists():
    MANU = pathlib.Path(__file__).resolve().parents[2] / "manuscript"
FIGDIR = MANU / "figures"
P.apply_style()

PRETTY = {
    "final_haiku": "Claude Haiku 4.5", "final_sonnet45": "Claude Sonnet 4.5",
    "final_gptoss120b": "GPT-oss-120B", "final_novalite": "Amazon Nova Lite",
    "final_qwen3b": "Qwen2.5-3B", "final_llama3b": "Llama-3.2-3B",
    "final_commandr35b": "Command-R-35B", "final_kimik25": "Kimi K2.5",
    "final_minimaxm25": "MiniMax M2.5",
}


def load_run(tag):
    return json.loads((MEAS / f"run_{tag}.json").read_text())


def load_frozen():
    return json.loads((MEAS / "frozen_protocol.json").read_text())


def cell(fz, tag):
    return next(c for c in fz["hotpot_cells"] if c["tag"] == tag)


def fig_frontier(primary, fz, n_boot=2000):
    d = load_run(primary)
    Q = d["per_q"]
    c = cell(fz, primary)
    lam_star = c["lambda"]
    lam_g = fz["global_lambda"]["lambda"]

    max_pref = max(len(r["prefix_em"]) for r in Q)
    fx, fy = [], []
    for b in range(max_pref):
        fy.append(100 * np.mean([r["prefix_em"][b] for r in Q]))
        fx.append(np.mean([r["prefix_searches"][b] for r in Q]))

    lams = [float(x) for x in d["lam_grid"]]
    em_mat = np.array([[r["voi"][str(l)]["em"] for r in Q] for l in lams], float)
    s_mat = np.array([[r["voi"][str(l)]["searches"] for r in Q] for l in lams],
                     float)
    vx = s_mat.mean(1)
    vy = 100 * em_mat.mean(1)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(Q), (n_boot, len(Q)))
    boots = 100 * em_mat[:, idx].mean(2)          # (lam, boot)
    lo = np.percentile(boots, 2.5, axis=1)
    hi = np.percentile(boots, 97.5, axis=1)

    def point_at(lam):
        stops = []
        for r in Q:
            t = len(r["prefix_value"]) - 1
            for u in range(len(r["prefix_value"]) - 1):
                if r["prefix_value"][u] < lam:
                    t = u
                    break
            stops.append(t)
        em = 100 * np.mean([r["prefix_em"][t] for r, t in zip(Q, stops)])
        s = np.mean([r["prefix_searches"][t] for r, t in zip(Q, stops)])
        return s, em

    fig, ax = plt.subplots(figsize=(3.34, 2.5))
    order = np.argsort(vx)
    ax.fill_between(vx[order], lo[order], hi[order], color=P.BLUE, alpha=0.13,
                    linewidth=0)
    ax.plot(fx, fy, color=P.ORANGE, marker="s", ls="--",
            label="ReAct (fixed budget)")
    ax.plot(vx[order], vy[order], color=P.BLUE, marker="o",
            label="VoI sweep (diagnostic)")
    sx, sy = point_at(lam_star)
    gx, gy = point_at(lam_g)
    ax.scatter([sx], [sy], s=80, marker="*", color=P.BLUE, zorder=6)
    ax.scatter([gx], [gy], s=52, marker="D", facecolor="white",
               edgecolor=P.BLUE, zorder=6, linewidth=1.1)
    ax.annotate(f"frozen $\\lambda^*$", xy=(sx, sy),
                xytext=(sx - 1.45, sy + 3.0), fontsize=6.0, color=P.BLUE,
                ha="left",
                arrowprops=dict(arrowstyle="-", color=P.BLUE, lw=0.6,
                                shrinkA=2, shrinkB=5))
    ax.annotate(f"global $\\lambda_g$", xy=(gx, gy),
                xytext=(gx + 0.65, gy - 11.0), fontsize=6.0, color=P.BLUE,
                ha="left",
                arrowprops=dict(arrowstyle="-", color=P.BLUE, lw=0.6,
                                shrinkA=2, shrinkB=5))
    ax.annotate("ReAct", (fx[-1], fy[-1]), xytext=(-6, -10),
                textcoords="offset points", fontsize=6.4, color=P.ORANGE)
    ax.set_xlabel("searches / question")
    ax.set_ylabel("exact match (%)")
    ax.set_title(PRETTY.get(primary, primary))
    ax.grid(axis="y")
    ax.legend(loc="lower right", fontsize=6.0, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_success_cost.pdf")
    plt.close(fig)


def fig_risk(fz, primary, extra=()):
    series = [(primary, P.BLUE, "o", "-")]
    palette = [(P.ORANGE, "s", "--"), (P.AQUA, "v", "-.")]
    for tag, (col, m, ls) in zip(extra, palette):
        series.append((tag, col, m, ls))
    fig, ax = plt.subplots(figsize=(3.34, 2.5))
    amax = 0.6
    ax.plot([0, amax], [0, amax], color=P.GRAY, ls="--", lw=0.9)
    ax.annotate("y = x", (amax * 0.8, amax * 0.85), fontsize=6.2, color=P.GRAY,
                rotation=35)
    for si, (tag, color, marker, ls) in enumerate(series):
        dy = [-9, 7, -17][si % 3]
        curve = fz["rq4"][tag]
        a = [x["alpha"] for x in curve if x["fired"]]
        rr = [x["risk"] for x in curve if x["fired"]]
        cov = [x["coverage"] for x in curve if x["fired"]]
        refused = [x["alpha"] for x in curve if not x["fired"]]
        ax.plot(a, rr, color=color, marker=marker, ls=ls,
                label=PRETTY.get(tag, tag))
        if refused:
            ax.scatter(refused, [0.0] * len(refused), marker="x", s=18,
                       color=color, alpha=0.65)
        for x, y, cv in zip(a, rr, cov):
            ax.annotate(f"{round(cv * 100)}%", (x, y), xytext=(2, dy),
                        textcoords="offset points", fontsize=5.4, color=color)
    ax.annotate("$\\times$ = certificate refused (fail-safe)",
                (0.24, 0.035), fontsize=5.8, color=P.GRAY)
    ax.set_xlabel(r"target risk $\alpha$")
    ax.set_ylabel("empirical certified-stop error")
    ax.set_title("Corrected SGR certification (deployed stops)")
    ax.set_xlim(0, amax + 0.02)
    ax.set_ylim(-0.02, amax + 0.05)
    ax.grid(axis="y")
    ax.legend(loc="upper left", fontsize=6.0, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_risk.pdf")
    plt.close(fig)


def fig_difficulty(primary, fz):
    d = load_run(primary)
    Q = d["per_q"]
    lam_star = cell(fz, primary)["lambda"]
    types = sorted({r["type"] for r in Q})
    fig, ax = plt.subplots(figsize=(3.34, 2.4))
    labels, react, voi, orac, ns = [], [], [], [], []
    for ty in types:
        sub = [r for r in Q if r["type"] == ty]
        if len(sub) < 8:
            continue
        stops = []
        for r in sub:
            t = len(r["prefix_value"]) - 1
            for u in range(len(r["prefix_value"]) - 1):
                if r["prefix_value"][u] < lam_star:
                    t = u
                    break
            stops.append(t)
        labels.append(ty)
        ns.append(len(sub))
        react.append(np.mean([r["prefix_searches"][-1] for r in sub]))
        voi.append(np.mean([r["prefix_searches"][t]
                            for r, t in zip(sub, stops)]))
        orac.append(np.mean([r["oracle_searches"] for r in sub]))
    x = np.arange(len(labels))
    w = 0.26
    b1 = ax.bar(x - w, react, w, color=P.ORANGE, label="ReAct",
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x, voi, w, color=P.BLUE, label="VoI ($\\lambda^*$)",
                edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + w, orac, w, color=P.GRAY, label="Oracle",
                edgecolor="white", linewidth=0.5)
    for b in (b1, b2, b3):
        ax.bar_label(b, fmt="%.1f", fontsize=6.0, padding=1.5, color=P.GRAY)
    ax.set_xticks(x, [f"{l}\n(n={n})" for l, n in zip(labels, ns)])
    ax.set_ylabel("searches / question")
    ax.set_ylim(0, max(react) * 1.28)
    ax.grid(axis="y")
    ax.legend(loc="upper center", ncol=3, fontsize=6.4, handlelength=1.2,
              frameon=False, columnspacing=1.0)
    ax.set_title("Adaptive budget by question type")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_difficulty.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="final_haiku")
    ap.add_argument("--risk-extra", nargs="*",
                    default=["final_novalite", "final_qwen3b"])
    args = ap.parse_args()
    fz = load_frozen()
    fig_frontier(args.primary, fz)
    if "rq4" in fz:
        fig_risk(fz, args.primary, extra=args.risk_extra)
    fig_difficulty(args.primary, fz)
    print(f"[figures] written to {FIGDIR}")


if __name__ == "__main__":
    main()
