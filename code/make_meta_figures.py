"""Figures for the full audit, regenerated from measured JSON only.

  fig_samples.pdf   forest plot: per-run paired contrasts (guarded - full,
                    guarded - fixed-3, guarded - fixed-2, guarded - repetition)
                    with 95% bootstrap CIs for every corpus-index run, plus
                    the question-clustered pooled estimate (from
                    meta_review_analyses.json, hierarchical block).
  fig_nested.pdf    nested refit/reselect bootstrap: per instrumented run, the share
                    of replicates reselecting the logged threshold, the share
                    in which a fixed budget or repetition Pareto-dominates the
                    refit guarded rule, and the share in which the guard
                    strictly beats the reselected B* (nested block).
  fig_frontier.pdf  accuracy-cost frontier on the corpus runs: EM vs
                    exact LLM calls for fixed budgets, repetition, TASR, and
                    the learned policies, one panel per question sample
                    (corpus_confirmation*.json).

Every number is read from results/measured/; nothing is typed in.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

# Camera-ready style: text set by LaTeX with the paper's Libertine fonts, a
# muted palette shared with Figure 1, light grids, and no top/right spines.
INK, MUTE, GRIDC = "#2B3441", "#8A919C", "#E3E6EA"
NAVY, TEAL, RUST, OCHRE, PLUM = "#2F4B66", "#2F7470", "#B4583C", "#A8781F", "#6A4C6B"
matplotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[tt=false,type1=true]{libertine}"
                           r"\usepackage[libertine]{newtxmath}",
    "font.family": "serif",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": "#9AA1AB", "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#5A616C", "ytick.color": "#5A616C",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "xtick.major.pad": 1.6, "ytick.major.pad": 1.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.titlepad": 3, "axes.labelpad": 1.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
})

MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
ART = pathlib.Path(__file__).resolve().parents[1]
MANU = ART / "manuscript"
if not MANU.exists():
    MANU = pathlib.Path(__file__).resolve().parents[2] / "manuscript"
FIGS = MANU / "figures"

RUN_LABELS = {
    "corpus_guarded_qwen3b": "A s7",
    "corpus_guarded_qwen3b_seed11": "A s11",
    "corpus_guarded_qwen3b_seed19": "A s19",
    "corpus_guarded_qwen3b_data17": "B",
    "corpus_guarded_qwen3b_data23": "C",
    "corpus_guarded_qwen3b_data29": "D",
    "corpus_guarded_qwen3b_data31": "E",
    "wiki25m_guarded_qwen3b": "Wiki",
    "corpus2wiki_guarded_qwen3b_data17": "2Wiki d1",
    "corpus2wiki_guarded_qwen3b_data23": "2Wiki d2",
    "corpus2wiki_guarded_qwen3b_data29": "2Wiki d3",
    "corpusmusique_guarded_qwen3b_data17": "MuSiQue d1",
    "corpusmusique_guarded_qwen3b_data23": "MuSiQue d2",
    "corpusmusique_guarded_qwen3b_data29": "MuSiQue d3",
}
# plain-language names used in figure text
PLAIN_RUN = {
    "A s7": "HotpotQA A, seed 7", "A s11": "HotpotQA A, seed 11",
    "A s19": "HotpotQA A, seed 19", "B": "HotpotQA sample B",
    "C": "HotpotQA sample C", "D": "HotpotQA sample D", "E": "HotpotQA sample E",
    "Wiki": "Wikipedia 25M", "2Wiki d1": "2Wiki draw 1", "2Wiki d2": "2Wiki draw 2",
    "2Wiki d3": "2Wiki draw 3", "MuSiQue d1": "MuSiQue draw 1",
    "MuSiQue d2": "MuSiQue draw 2", "MuSiQue d3": "MuSiQue draw 3",
}
CONF_FILES = {
    "corpus_guarded_qwen3b": "corpus_confirmation.json",
}


def conf_path(tag):
    return MEAS / CONF_FILES.get(tag, f"corpus_confirmation_{tag}.json")


def fig_samples(meta):
    hv = meta["hierarchical_variance"]
    contrasts = [("guarded_minus_full", r"vs.\ full search"),
                 ("guarded_minus_fixed3", r"vs.\ three fixed searches"),
                 ("guarded_minus_fixed2", r"vs.\ two fixed searches"),
                 ("guarded_minus_repeat", r"vs.\ answer repetition")]
    short = {"A s7": "A, seed 7", "A s11": "A, seed 11", "A s19": "A, seed 19",
             "B": "sample B", "C": "sample C", "D": "sample D", "E": "sample E"}
    runs = hv["question_sample_A_runs"] + list(hv["independent_question_samples"].values())
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 2.75), sharey=True)
    axes = axes.flat
    for ax, (key, title) in zip(axes, contrasts):
        c = hv["contrasts"][key]
        ys, labels = [], []
        for i, tag in enumerate(runs):
            pr = c["per_run"][tag]
            lo, hi = pr["ci95"]
            ax.errorbar(pr["mean_em_points"], i, xerr=[[pr["mean_em_points"] - lo], [hi - pr["mean_em_points"]]],
                        fmt="o", color=NAVY if tag.startswith("corpus_guarded_qwen3b_data") or tag.endswith("qwen3b") and "seed" not in tag and "data" not in tag else MUTE,
                        ms=3.5, capsize=2, lw=1)
            labels.append(short.get(RUN_LABELS.get(tag, tag), tag))
        plo, phi = c["pooled_question_clustered_ci95"]
        pm = c["pooled_all_run_mean_em_points"]
        i = len(runs)
        ax.errorbar(pm, i, xerr=[[pm - plo], [phi - pm]], fmt="D", color=RUST, ms=4.5, capsize=2, lw=1.4)
        labels.append("pooled (q-clustered)")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        if key == "guarded_minus_full":
            ax.axvline(-3, color=RUST, lw=0.6, ls="--")
        ax.set_title(title, fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l.replace("pooled (q-clustered)", "pooled") for l in labels], fontsize=6)
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    for ax in list(axes)[2:]:
        ax.set_xlabel("EM difference (points)", fontsize=6.5)
    fig.axes[0].invert_yaxis()   # shared y: invert once, runs top, pooled last
    fig.tight_layout(w_pad=0.4, h_pad=0.5)
    fig.savefig(FIGS / "fig_samples.pdf")
    plt.close(fig)


def fig_samples_main(meta):
    """Single-column 1x2 forest plot for the main text.

    Two contrasts only (guard - full budget, guard - fixed-2); the remaining
    two stay in the 2x2 supplement figure.  Each row carries the two-sided
    95% bootstrap CI (bar) AND the one-sided 95% lower bound (tick), because
    the non-inferiority verdicts are decided on the one-sided bound and the
    two differ.  Same-question reruns are grey, independent question draws
    navy, the question-clustered pooled estimate rust.
    """
    hv = meta["hierarchical_variance"]
    contrasts = [("guarded_minus_full", r"vs.\ full search"),
                 ("guarded_minus_fixed2", r"vs.\ two fixed searches")]
    plain = {"A s7": "A, seed 7", "A s11": "A, seed 11", "A s19": "A, seed 19",
             "B": "sample B", "C": "sample C", "D": "sample D", "E": "sample E"}
    runs = hv["question_sample_A_runs"] + list(
        hv["independent_question_samples"].values())
    nA = len(hv["question_sample_A_runs"])
    fig, axes = plt.subplots(1, 2, figsize=(3.34, 1.40), sharey=True)
    n_pts = 0
    for ax, (key, title) in zip(axes, contrasts):
        c = hv["contrasts"][key]
        labels = []
        ax.axhspan(nA - 0.5, len(runs) - 0.5, color="#EEF2F6", lw=0, zorder=0)
        for i, tag in enumerate(runs):
            pr = c["per_run"][tag]
            lo, hi = pr["ci95"]
            m = pr["mean_em_points"]
            indep = tag.startswith("corpus_guarded_qwen3b_data")
            col = NAVY if indep else MUTE
            ax.plot([lo, hi], [i, i], "-", color=col, lw=1.1,
                    solid_capstyle="round", zorder=3)
            ax.plot([m], [i], "o", color=col, ms=3.4, mec="white", mew=0.5,
                    zorder=4)
            ax.plot([pr["lower95_one_sided"]], [i], marker="|", ms=4.5,
                    mew=0.9, color=INK, lw=0, zorder=4)
            labels.append(plain.get(RUN_LABELS.get(tag, tag), tag))
            n_pts += 1
        plo, phi = c["pooled_question_clustered_ci95"]
        pm = c["pooled_all_run_mean_em_points"]
        i = len(runs)
        ax.plot([plo, phi], [i, i], "-", color=RUST, lw=1.6,
                solid_capstyle="round", zorder=3)
        ax.plot([pm], [i], "D", color=RUST, ms=3.8, mec="white", mew=0.5,
                zorder=4)
        labels.append(r"\textbf{pooled}")
        n_pts += 1
        ax.axvline(0, color=INK, lw=0.6, ls=(0, (1.2, 1.2)), zorder=2)
        if key == "guarded_minus_full":
            ax.axvspan(-50, -3, color=RUST, alpha=0.08, lw=0, zorder=0)
            ax.axvline(-3, color=RUST, lw=0.7, ls=(0, (3, 1.5)), zorder=2)
            ax.text(-3.25, -0.95, r"$-3$ margin", color=RUST, fontsize=5.8,
                    ha="right", va="center")
        xs = [v for t in runs for v in c["per_run"][t]["ci95"]] + [plo, phi, 0]
        if key == "guarded_minus_full":
            xs.append(-3)
        pad = 0.06 * (max(xs) - min(xs))
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_title(title, color=INK)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("EM difference (points)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    axes[0].set_ylim(len(runs) + 0.6, -1.4)   # runs top -> pooled last
    fig.tight_layout(w_pad=0.5)
    fig.savefig(FIGS / "fig_samples_main.pdf")
    plt.close(fig)
    print(f"fig_samples_main.pdf: {n_pts} plotted estimates")


def fig_nested(meta):
    """Compact annotated heatmap: rows are the four nested-bootstrap shares,
    columns the 14 fully instrumented runs, cells the percentage of 400
    training-chain resamples (values read from nested_refit_select)."""
    import numpy as np
    from decimal import Decimal, ROUND_HALF_UP
    from matplotlib.colors import LinearSegmentedColormap
    ns = meta["nested_refit_select"]
    tags = [t for t in RUN_LABELS if t in ns]
    labels = [RUN_LABELS[t] for t in tags]
    rows = [("logged threshold kept", "lambda_logged_frequency"),
            ("guarded FBG beaten", "p_guarded_dominated_by_simple"),
            ("FBG-only beaten", "p_value_dominated_by_simple"),
            ("guarded FBG wins", "p_guarded_strictly_beats_bstar")]
    data = np.array([[100 * ns[t][k] for t in tags] for _, k in rows])
    cmap = LinearSegmentedColormap.from_list("muted", ["#FBFAF7", "#BFD0DF", "#5F7A94", NAVY])
    fig, ax = plt.subplots(figsize=(3.34, 1.10))
    ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            pct = (Decimal(str(ns[tags[c]][rows[r][1]])) * 100).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP)   # same rounding as text
            ax.text(c, r, f"{pct}", ha="center", va="center", fontsize=5.9,
                    color="white" if v >= 62 else INK)
    short = {"A s7": "7", "A s11": "11", "A s19": "19", "Wiki": "25M"}
    short = [short.get(lab, lab.split()[-1].lstrip("d")) for lab in labels]
    subgroups = [("seeds of A", 0, 2), ("samples", 3, 6), ("draws", 8, 10),
                 ("draws", 11, 13)]
    for name, lo, hi in subgroups:
        ax.text((lo + hi) / 2, -0.215, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=5.6, color=MUTE)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(short, fontsize=6.0)
    groups = {}
    for i, t in enumerate(tags):
        g = ("HotpotQA" if t.startswith("corpus_guarded") else
             "Wiki" if t.startswith("wiki25m") else
             "2Wiki" if t.startswith("corpus2wiki") else "MuSiQue")
        groups.setdefault(g, []).append(i)
    for g, idx in groups.items():
        lo, hi = min(idx), max(idx)
        ax.annotate("", xy=(lo - 0.38, -0.385), xytext=(hi + 0.38, -0.385),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.6))
        ax.text((lo + hi) / 2, -0.41, g, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.3, color=INK)
        if hi < len(tags) - 1:
            ax.axvline(hi + 0.5, color="white", lw=2.2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r for r, _ in rows], fontsize=6.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(tags)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_title(r"\% of 400 training resamples", fontsize=6.5, color=MUTE,
                 loc="right", pad=2)
    fig.subplots_adjust(left=0.315, right=0.995, top=0.90, bottom=0.33)
    fig.savefig(FIGS / "fig_nested.pdf")
    plt.close(fig)


MAIN_PANELS = ["corpus_guarded_qwen3b", "corpus_guarded_qwen3b_data17",
               "corpus_guarded_qwen3b_data23", "corpus_guarded_qwen3b_data29",
               "corpus_guarded_qwen3b_data31", "wiki25m_guarded_qwen3b"]


def fig_frontier(filename="fig_frontier.pdf", only=None, ncol_max=3,
                 figwidth=7.2, rowheight=1.9, fontscale=1.0, legend_ncol=5):
    tags = [t for t in RUN_LABELS if conf_path(t).exists()]
    # one panel per question sample; sample A uses the seed-7 primary run
    panels = [t for t in tags if "seed11" not in t and "seed19" not in t]
    if only is not None:
        panels = [t for t in panels if t in only]
    ncol = min(ncol_max, len(panels))
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(figwidth, rowheight * nrow),
                             squeeze=False)
    styles = {
        "fixed": (MUTE, "s", "fixed budget"),
        "answer_repeat": (OCHRE, "^", "repetition"),
        "tasr": (PLUM, "v", "TASR"),
        "value_only": (NAVY, "o", "FBG-only"),
        "guarded_value": (RUST, "D", "guarded FBG"),
    }
    for ax, tag in zip(axes.flat, panels):
        m = json.loads(conf_path(tag).read_text())["methods"]
        fx = [m[f"fixed_{k}"]["llm_calls"] for k in range(5) if f"fixed_{k}" in m]
        fy = [m[f"fixed_{k}"]["em"] for k in range(5) if f"fixed_{k}" in m]
        ax.plot(fx, fy, "-", color=styles["fixed"][0], lw=0.8, marker=styles["fixed"][1], ms=3, label=styles["fixed"][2])
        for key in ["answer_repeat", "tasr", "value_only", "guarded_value"]:
            if key in m:
                col, mk, lab = styles[key]
                ax.plot([m[key]["llm_calls"]], [m[key]["em"]], mk, color=col, ms=5, label=lab)
        name = RUN_LABELS.get(tag, tag)
        ax.set_title(PLAIN_RUN.get(name, name), fontsize=8 * fontscale)
        ax.tick_params(labelsize=6.5 * fontscale)
        ax.grid(lw=0.3, alpha=0.5)
        ax.set_xlim(0.5, 9.5)
    for ax in axes[-1]:
        ax.set_xlabel("LLM calls per question", fontsize=7 * fontscale)
    for row in axes:
        row[0].set_ylabel(r"EM (\%)", fontsize=7 * fontscale)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=legend_ncol,
               fontsize=6.5 * fontscale, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGS / filename)
    plt.close(fig)


MAIN_FRONTIER_PANELS = [
    ("corpus_guarded_qwen3b", "(a) HotpotQA, 269k sentences"),
    ("wiki25m_guarded_qwen3b", "(b) Wikipedia, 25M sentences"),
]


def fig_frontier_main():
    """Single-column two-panel accuracy-cost frontier for the main text.

    Reads methods.<policy>.em and methods.<policy>.llm_calls from
    corpus_confirmation.json and corpus_confirmation_wiki25m_guarded_qwen3b.json;
    nine policies per panel (fixed-0..4, repetition, TASR, FBG value-only,
    FBG + repeat guard) = 18 plotted points.

    Each point carries a vertical bar built from the STORED paired 95% interval
    methods.<policy>.paired_vs_full.ci95 (policy minus full budget, question
    bootstrap): the bar spans full_budget_EM + ci95, so it is the uncertainty
    the paper's domination claims rest on, not a marginal per-point interval.
    No new inference is computed here. The shaded band is the admissible region
    EM >= full_budget_EM - 1, the same band round9_analyses.select_le uses to
    pick "the cheapest policy within 1 EM of the full budget".
    """
    styles = {
        "answer_repeat": (OCHRE, "^", "repetition"),
        "tasr": (PLUM, "v", "TASR"),
        "value_only": (NAVY, "o", "FBG-only"),
        "guarded_value": (RUST, "D", "guarded FBG"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(3.34, 1.26), sharey=True)
    n_pts = 0
    n_bars = 0
    for ax, (tag, title) in zip(axes, MAIN_FRONTIER_PANELS):
        m = json.loads(conf_path(tag).read_text())["methods"]
        full_em = m["full_budget"]["em"]

        def bar(key):
            """[lo, hi] in EM units from the stored paired-vs-full ci95."""
            ci = m[key].get("paired_vs_full", {}).get("ci95")
            if ci is None:
                return None
            return full_em + ci[0], full_em + ci[1]

        # admissible region: EM >= full-budget EM - 1 (select_le's band rule)
        ax.axhspan(full_em - 1.0, 100.0, color=TEAL, alpha=0.10, lw=0, zorder=0)
        ax.axhline(full_em - 1.0, color=TEAL, lw=0.6, ls=(0, (2.5, 1.5)),
                   zorder=1)

        def errbar(x, b, col, lw):
            ax.plot([x] * 2, list(b), "-", color=col, lw=lw, alpha=0.55,
                    zorder=2, solid_capstyle="round")

        ks = [k for k in range(5) if f"fixed_{k}" in m]
        for k in ks:
            b = bar(f"fixed_{k}")
            if b is not None:
                errbar(m[f"fixed_{k}"]["llm_calls"], b, MUTE, 0.9)
                n_bars += 1
        ax.plot([m[f"fixed_{k}"]["llm_calls"] for k in ks],
                [m[f"fixed_{k}"]["em"] for k in ks], "-",
                color=MUTE, lw=1.0, marker="s", ms=2.8, mec="white", mew=0.4,
                label="fixed budget", zorder=3)
        n_pts += len(ks)
        for key in ["answer_repeat", "tasr", "value_only", "guarded_value"]:
            col, mk, lab = styles[key]
            b = bar(key)
            if b is not None:
                errbar(m[key]["llm_calls"], b, col, 1.0)
                n_bars += 1
            ax.plot([m[key]["llm_calls"]], [m[key]["em"]], mk, color=col,
                    ms=4.0 if mk != "D" else 3.6, mec="white", mew=0.5,
                    label=lab, zorder=5)
            n_pts += 1
        if tag == "wiki25m_guarded_qwen3b":     # highlight boundary-cell domination
            fx2, fy2 = m["fixed_2"]["llm_calls"], m["fixed_2"]["em"]
            ax.fill_between([fx2, 9.7], 0, fy2, color=RUST, alpha=0.06, lw=0,
                            zorder=0)
            ax.text(9.5, 5.0, "worse than two\nfixed searches", fontsize=6,
                    ha="right", va="bottom", color=RUST, zorder=4,
                    linespacing=0.95)
        ax.set_title(title)
        ax.set_xlim(0.3, 9.7)
        ax.set_ylim(4.0, 27.5)   # widened to show the fixed-0 paired bar
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.set_xlabel("LLM calls per question")
    axes[0].set_ylabel(r"EM (\%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=0.5, pad=0.2)
    fig.legend(handles, labels, loc="lower center", ncol=5,
               handletextpad=0.15, columnspacing=0.75, handlelength=1.3,
               borderpad=0.0, bbox_to_anchor=(0.5, 0.955))
    fig.savefig(FIGS / "fig_frontier_main.pdf")
    plt.close(fig)
    print(f"fig_frontier_main.pdf: {n_pts} plotted points, "
          f"{n_bars} stored paired 95% bars")


def main():
    FIGS.mkdir(exist_ok=True)
    meta = json.loads((MEAS / "meta_review_analyses.json").read_text())
    fig_samples(meta)
    fig_nested(meta)
    fig_frontier("fig_frontier.pdf", only=MAIN_PANELS, ncol_max=3)
    fig_frontier("fig_frontier_all.pdf", only=None, ncol_max=4)
    # main-text single-column frontier: primary corpus run + 25M Wikipedia
    fig_frontier_main()
    fig_samples_main(meta)
    print(f"wrote fig_samples.pdf, fig_samples_main.pdf, fig_nested.pdf, "
          f"fig_frontier.pdf, fig_frontier_main.pdf, fig_frontier_all.pdf in {FIGS}")


if __name__ == "__main__":
    main()
