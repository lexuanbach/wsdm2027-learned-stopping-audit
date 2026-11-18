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
CONF_FILES = {
    "corpus_guarded_qwen3b": "corpus_confirmation.json",
}


def conf_path(tag):
    return MEAS / CONF_FILES.get(tag, f"corpus_confirmation_{tag}.json")


def fig_samples(meta):
    hv = meta["hierarchical_variance"]
    contrasts = [("guarded_minus_full", "guard $-$ full budget"),
                 ("guarded_minus_fixed3", "guard $-$ fixed-3"),
                 ("guarded_minus_fixed2", "guard $-$ fixed-2"),
                 ("guarded_minus_repeat", "guard $-$ repetition")]
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
                        fmt="o", color="#1f77b4" if tag.startswith("corpus_guarded_qwen3b_data") or tag.endswith("qwen3b") and "seed" not in tag and "data" not in tag else "#7f7f7f",
                        ms=3.5, capsize=2, lw=1)
            labels.append(RUN_LABELS.get(tag, tag))
        plo, phi = c["pooled_question_clustered_ci95"]
        pm = c["pooled_all_run_mean_em_points"]
        i = len(runs)
        ax.errorbar(pm, i, xerr=[[pm - plo], [phi - pm]], fmt="D", color="#d62728", ms=4.5, capsize=2, lw=1.4)
        labels.append("pooled (q-clustered)")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        if key == "guarded_minus_full":
            ax.axvline(-3, color="#d62728", lw=0.6, ls="--")
        ax.set_title(title, fontsize=7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l.replace("pooled (q-clustered)", "pooled") for l in labels], fontsize=6)
        ax.tick_params(axis="x", labelsize=6)
        ax.invert_yaxis()
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    for ax in list(axes)[2:]:
        ax.set_xlabel("paired EM points", fontsize=6.5)
    fig.tight_layout(w_pad=0.4, h_pad=0.5)
    fig.savefig(FIGS / "fig_samples.pdf")
    plt.close(fig)


def fig_samples_main(meta):
    """Single-column 1x2 forest plot for the main text.

    Two contrasts only (guard - full budget, guard - fixed-2); the remaining
    two stay in the 2x2 supplement figure.  Each row carries the two-sided
    95% bootstrap CI (bar) AND the one-sided 95% lower bound (caret), because
    the non-inferiority verdicts are decided on the one-sided bound and the
    two differ.
    """
    hv = meta["hierarchical_variance"]
    contrasts = [("guarded_minus_full", "guard $-$ full budget"),
                 ("guarded_minus_fixed2", "guard $-$ fixed-2")]
    runs = hv["question_sample_A_runs"] + list(
        hv["independent_question_samples"].values())
    fig, axes = plt.subplots(1, 2, figsize=(3.34, 1.62), sharey=True)
    n_pts = 0
    for ax, (key, title) in zip(axes, contrasts):
        c = hv["contrasts"][key]
        labels = []
        for i, tag in enumerate(runs):
            pr = c["per_run"][tag]
            lo, hi = pr["ci95"]
            m = pr["mean_em_points"]
            indep = tag.startswith("corpus_guarded_qwen3b_data")
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o",
                        color="#1f77b4" if indep else "#7f7f7f",
                        ms=3, capsize=1.6, lw=0.9)
            ax.plot([pr["lower95_one_sided"]], [i], marker="|", ms=5,
                    color="#333333", lw=0)
            labels.append(RUN_LABELS.get(tag, tag))
            n_pts += 1
        plo, phi = c["pooled_question_clustered_ci95"]
        pm = c["pooled_all_run_mean_em_points"]
        i = len(runs)
        ax.errorbar(pm, i, xerr=[[pm - plo], [phi - pm]], fmt="D",
                    color="#d62728", ms=3.6, capsize=1.6, lw=1.2)
        labels.append("pooled")
        n_pts += 1
        ax.axvline(0, color="k", lw=0.6, ls=":")
        if key == "guarded_minus_full":
            ax.axvline(-3, color="#d62728", lw=0.7, ls="--")
        ax.set_title(title, fontsize=6.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=5.6)
        ax.tick_params(axis="x", labelsize=5.6)
        ax.set_xlabel("paired EM points", fontsize=6)
        ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].invert_yaxis()      # sharey: invert once, runs top -> pooled last
    fig.tight_layout(w_pad=0.4)
    fig.savefig(FIGS / "fig_samples_main.pdf")
    plt.close(fig)
    print(f"fig_samples_main.pdf: {n_pts} plotted estimates")


def fig_nested(meta):
    ns = meta["nested_refit_select"]
    tags = [t for t in RUN_LABELS if t in ns]
    labels = [RUN_LABELS[t] for t in tags]
    kept = [100 * ns[t]["lambda_logged_frequency"] for t in tags]
    dom = [100 * ns[t]["p_guarded_dominated_by_simple"] for t in tags]
    beat = [100 * ns[t]["p_guarded_strictly_beats_bstar"] for t in tags]
    vdom = [100 * ns[t]["p_value_dominated_by_simple"] for t in tags]
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    x = range(len(tags))
    w = 0.2
    ax.bar([i - 1.5 * w for i in x], kept, w, label="logged threshold reselected", color="#7f7f7f")
    ax.bar([i - 0.5 * w for i in x], dom, w, label="guard dominated by simple policy", color="#d62728")
    ax.bar([i + 0.5 * w for i in x], vdom, w, label="FBG-only dominated", color="#ff9896")
    ax.bar([i + 1.5 * w for i in x], beat, w, label="guard strictly beats $B^*$", color="#2ca02c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("% of 400 training resamples", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=5.5, loc="lower center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.tight_layout()
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
        "fixed": ("#7f7f7f", "s", "fixed-$k$"),
        "answer_repeat": ("#9467bd", "^", "repetition"),
        "tasr": ("#8c564b", "v", "TASR"),
        "value_only": ("#1f77b4", "o", "FBG-only"),
        "guarded_value": ("#d62728", "D", "FBG + guard"),
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
        ax.set_title(RUN_LABELS.get(tag, tag), fontsize=8 * fontscale)
        ax.tick_params(labelsize=6.5 * fontscale)
        ax.grid(lw=0.3, alpha=0.5)
        ax.set_xlim(0.5, 9.5)
    for ax in axes[-1]:
        ax.set_xlabel("exact LLM calls / question", fontsize=7 * fontscale)
    for row in axes:
        row[0].set_ylabel("EM (%)", fontsize=7 * fontscale)
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
    ("corpus_guarded_qwen3b", "(a) HotpotQA, 269k-sentence index"),
    ("wiki25m_guarded_qwen3b", "(b) Wikipedia, 25M-sentence hybrid"),
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
        "answer_repeat": ("#9467bd", "^", "repetition"),
        "tasr": ("#8c564b", "v", "TASR"),
        "value_only": ("#1f77b4", "o", "FBG-only"),
        "guarded_value": ("#d62728", "D", "FBG + guard"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(3.34, 1.34))
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
        ax.axhspan(full_em - 1.0, 100.0, color="#2ca02c", alpha=0.10, lw=0,
                   zorder=0)
        ax.axhline(full_em - 1.0, color="#2ca02c", lw=0.5, ls=(0, (2.5, 1.5)),
                   zorder=1)

        ks = [k for k in range(5) if f"fixed_{k}" in m]
        for k in ks:
            b = bar(f"fixed_{k}")
            if b is not None:
                x = m[f"fixed_{k}"]["llm_calls"]
                ax.plot([x] * 2, list(b), "-", color="#7f7f7f", lw=0.7,
                        alpha=0.9, zorder=2, solid_capstyle="butt")
                for yy in b:
                    ax.plot([x - 0.16, x + 0.16], [yy, yy], "-",
                            color="#7f7f7f", lw=0.7, alpha=0.9, zorder=2)
                n_bars += 1
        ax.plot([m[f"fixed_{k}"]["llm_calls"] for k in ks],
                [m[f"fixed_{k}"]["em"] for k in ks], "-",
                color="#7f7f7f", lw=0.8, marker="s", ms=2.2, label="fixed-$k$",
                zorder=3)
        n_pts += len(ks)
        for key in ["answer_repeat", "tasr", "value_only", "guarded_value"]:
            col, mk, lab = styles[key]
            b = bar(key)
            if b is not None:
                x = m[key]["llm_calls"]
                ax.plot([x] * 2, list(b), "-", color=col, lw=0.8, alpha=0.95,
                        zorder=4, solid_capstyle="butt")
                for yy in b:
                    ax.plot([x - 0.16, x + 0.16], [yy, yy], "-", color=col,
                            lw=0.8, alpha=0.95, zorder=4)
                n_bars += 1
            ax.plot([m[key]["llm_calls"]], [m[key]["em"]], mk, color=col,
                    ms=3.2, label=lab, zorder=5)
            n_pts += 1
        if tag == "wiki25m_guarded_qwen3b":     # highlight boundary-cell domination
            fx2, fy2 = m["fixed_2"]["llm_calls"], m["fixed_2"]["em"]
            ax.axhspan(0, fy2, xmin=(fx2 - 0.3) / 9.4, xmax=1.0,
                       color="#d62728", alpha=0.07, lw=0, zorder=0)
            ax.text(9.4, 8.6, "dominated by fixed-2", fontsize=4.8,
                    ha="right", va="bottom", color="#a02020", zorder=4)
        ax.set_title(title, fontsize=5.8, pad=2)
        ax.tick_params(labelsize=5.4, pad=1.5, length=2)
        ax.grid(lw=0.3, alpha=0.5)
        ax.set_xlim(0.3, 9.7)
        ax.set_ylim(4.0, 27.5)   # widened to show the fixed-0 paired bar
        ax.set_xticks([1, 3, 5, 7, 9])
        ax.set_xlabel("exact LLM calls / question", fontsize=5.8, labelpad=1)
    axes[0].set_ylabel("EM (%)", fontsize=5.8, labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=5.4,
               frameon=False, handletextpad=0.25, columnspacing=0.8,
               borderpad=0.0, bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(rect=(0, 0.085, 1, 1), w_pad=0.6)
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
