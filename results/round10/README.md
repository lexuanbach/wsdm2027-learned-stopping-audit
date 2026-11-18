# Round 10 — external validity and verifier baselines (2026-08-24)

Everything in this directory was produced in one session against the frozen
artifact. **No frozen file was overwritten.** New per-cell run files written
into `artifact/results/measured/` carry new tags only (`extnews_*`, plus
`stoprag_result_*` for cells that had none); every derived analysis lands here.

Read `frozen-external-endpoint-2026-08-24.md` first: it is the pre-registration
for the external cell and was written before any policy was fit or any test
question was answered.

---

## What this round changes about what the paper can claim

1. **A genuinely non-Wikipedia environment now exists.** MultiHopRAG news
   articles (609 articles, 49 outlets, 53,943 retrieval units), frozen
   endpoint, selection on that cell's own training chains only. See
   `external_cell_extnews_guarded_qwen3b.json`.
2. **A trained sufficiency verifier now exists as a baseline** — supervised on
   gold evidence-sufficiency, not on the paper's own error signal, and it is
   genuinely predictive (cross-fit AUC 0.65–0.95). It does **not** escape the
   fixed-budget boundary. See `sufficiency_classifier.json`.
3. **The "no resolved gain over a randomized mixture" claim can be made
   precise and multiplicity-aware** rather than left as an unqualified
   sentence. See `mixture_wins_multiplicity.py` output — and the correction
   note below, which the manuscript agent must act on.
4. **A mechanism for the negative result.** The FBG value head carries almost
   no selective-prediction signal (it beats a random coverage ordering in only
   7 of 18 cells), while the error head beats random in 18 of 18. See
   `calibration_audit.json`.

---

## Files

| File | What it is |
|---|---|
| `frozen-external-endpoint-2026-08-24.md` | Pre-registration for the external cell. Written before any test evaluation. |
| `adapter_report_multihoprag.json` | Provenance for the MultiHopRAG → HotpotQA-schema conversion. |
| `run_extnews_guarded_qwen3b.log` | Raw log of the external 520-question run. |
| `external_cell_extnews_guarded_qwen3b.json` | The external cell's confirmatory endpoint and full frontier. |
| `sufficiency_classifier.json` | The new gold-support-supervised sufficiency verifier, 18 cells, 2 declared signal sets, with the matched-mixture control. |
| `extended_decision_matrix.json` | Every policy × every cell, with dominance verdict (D/F/I) and matched-mixture interval verdict (win/loss/**unresolved**). |
| `mixture_wins_multiplicity.json` | Whether any equal-cost gain survives Bonferroni / max-t correction. |
| `simultaneous_intervals.json` | Family-wise intervals for the guarded-minus-full contrast. |
| `calibration_audit.json` | ECE / Brier / reliability bins for the error head; risk–coverage for the FBG value head. |
| `cost_latency_audit.json` | p50/p95/p99 latency per call and per question on three backbones; paired per-axis cost deltas. |
| `strata_audit_extra.json` | Hop-depth, evidence-length and retrieval-ambiguity strata. |
| `policy_cards.json` / `.tex` | One configuration card per cell (supplement table). |
| `prompt_and_seed_ledger.json` | sha256 of every prompt constant + consolidated seed ledger. |
| `per_question_paired.csv` | 59,400 rows: one per (cell, question, policy) with stop depth, calls, tokens, wall seconds, EM. |
| `frontier_figure_data.json` | Hull vertices + matched-mixture points on all three cost axes, for a three-panel frontier figure. |
| `stoprag_extend.log` | Stop-RAG extension to the cells that had none. |

New scripts live in `artifact/code/`: `prepare_multihoprag_parquet.py`,
`sufficiency_classifier.py`, `calibration_audit.py`, `cost_latency_audit.py`,
`strata_audit_extra.py`, `policy_cards.py`, `simultaneous_intervals.py`,
`extended_decision_matrix.py`, `mixture_wins_multiplicity.py`,
`frontier_figure_data.py`, `external_cell_report.py`.

---

## Correction the manuscript agent must make

`main.tex:61-62`, `sections/06-results.tex:135-136` and
`sections/07-conclusion.tex:8-9` say, without qualification, that **no cell**
shows a statistically resolved gain over a randomized fixed-budget mixture.

That is exactly true for the FBG rules (guarded and value-only: 0 marginal wins
in 18 cells). It is **not** true as written for two other learned policies that
are in the same table: `results/measured/round9_analyses.json` already contains
three cells with a marginal 95% CI entirely above zero
(`stop_classifier` on corpus2wiki_data17 +1.62 [0.00, 3.20], on haiku45_data17
+2.61 [0.14, 4.89], on qwen3b_data29 +0.11 [0.05, 0.16]; `sufficiency_gate` on
qwen3b_data23 +0.14 [0.04, 0.27]).

The honest fix costs one clause, and the multiplicity analysis makes it
*stronger*, not weaker:

> No learned policy achieves a gain over a matched randomized fixed-budget
> mixture that survives a family-wise correction across cells (max-t: 0 of 18
> for every policy family). The two gains that survive Bonferroni are
> $+0.11$ and $+0.14$ EM points.

Numbers: `mixture_wins_multiplicity.json`.
