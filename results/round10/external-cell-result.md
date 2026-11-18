# External (non-Wikipedia) cell — result

Frozen endpoint: `frozen-external-endpoint-2026-08-24.md` (written before any
policy was fit or any test question was answered).
Raw run: `artifact/results/measured/run_extnews_guarded_qwen3b.json`
(4,579 s, 5,080 LLM calls). Analysis: `external_cell_extnews_guarded_qwen3b.json`,
`external_pairwise_vs_simple.json`, `external_audits_extnews_guarded_qwen3b.json`.

## The cell

MultiHopRAG `inference_query`, 609 English news articles from 49 outlets,
53,943 pooled FTS5/BM25 retrieval units. **No Wikipedia text.** 120 training /
400 test questions, `qwen2.5:3b-instruct` (ollama, digest 357c53fb, Q4_K_M),
temperature 0, `--max_steps 4 --k 3 --alpha 0.10 --data_seed 13 --llm_seed 7
--corpus_index --primary_rule guarded`. theta = 0.010 and B* = 4 were selected
on this cell's own 96 policy-training episodes. **One draw.**

## Pre-registered primary endpoint: MET

| | EM | LLM calls |
|---|---|---|
| Full budget | 72.75 | 9.00 |
| Guarded FBG | **73.25** | **7.40** |

Paired (guarded − full budget) = **+0.50 EM, 95% CI [0.00, +1.25]**, one-sided
lower bound 0.00 > −3.0, and calls fall 9.00 → 7.40. Both halves of the
pre-registered criterion hold.

## But the guarded rule still resolves nothing

Paired per-question EM differences against each simple policy
(`external_pairwise_vs_simple.json`) — every interval contains or touches zero:

| Guarded FBG vs | ΔEM | 95% CI | Δcalls |
|---|---|---|---|
| fixed-2 | +2.25 | [−0.25, +5.00] | +2.40 |
| fixed-3 | +1.25 | [−0.25, +2.75] | +0.40 |
| fixed-4 (= B*) | +0.50 | [0.00, +1.25] | −1.60 |
| answer repetition | +1.00 | [−0.50, +2.75] | +2.21 |
| randomized mixture at matched calls | +0.71 | [−0.07, +1.57] | 0 (matched) |
| randomized mixture at matched tokens | +0.72 | [−0.08, +1.61] | 0 (matched) |
| randomized mixture at matched wall | +0.72 | [−0.08, +1.60] | 0 (matched) |

Under the frozen ε-dominance rule the "W" verdict is fragile: it holds at
ε = 0 and 0.5 EM, and flips to **L** at ε = 1 (dominated by answer repetition)
and at ε = 2 (fixed-3 and answer repetition).

## What IS resolved here — and it is not the paper's controller

In this environment a *different*, simpler learned rule — thresholding the
paper's own **error head** (the "sufficiency gate") — does resolve gains:

| Sufficiency gate (73.75 EM @ 7.64 calls) vs | ΔEM | 95% CI |
|---|---|---|
| fixed-2 | +2.75 | [+0.25, +5.50] |
| fixed-3 | +1.75 | [+0.25, +3.50] |
| fixed-4 | +1.00 | [+0.25, +2.00] |
| randomized mixture, matched calls | +1.18 | [+0.41, +2.06] |
| randomized mixture, matched tokens | +1.18 | [+0.41, +2.06] |
| randomized mixture, matched wall | +1.17 | [+0.41, +2.06] |

The same-feature stop classifier also beats the matched mixture (+2.10
[+0.12, +4.10] on calls) but only because it sits at 4.35 calls where the hull
interpolates; it is *below* fixed-2 in EM (70.50 vs 71.00).

**These are exploratory in this cell.** The freeze document named the guarded
FBG rule as the primary policy; the gate and the classifier were not
pre-registered here, and this is a single 400-question draw.

## The honest reading

The external run does **not** reverse the paper's headline and does **not**
manufacture a win for the audited controller. It does change the scope of the
conclusion in one specific, defensible way:

> The FBG controller resolves no gain in the non-Wikipedia news environment
> either. What changes off Wikipedia is the *regime*: accuracy saturates at
> 71.0 EM by two searches with only 1.75 EM of headroom left to the full
> budget, and in that regime a one-parameter threshold on a well-calibrated
> error head — not the future-gain critic — is the learned rule that resolves
> a gain over a matched randomized fixed-budget mixture (+1.18 EM
> [+0.41, +2.06]). The audited boundary is therefore a property of the
> future-gain signal, not of learned stopping as such.

## Two things that did NOT work in this environment, reported as such

1. **The gold-support-supervised sufficiency classifier is not applicable
   here.** Its label — "the retrieved evidence at this prefix contains every
   annotated supporting fact" — has a base rate of **0.008** on this cell
   (news facts are individual sentences scattered across 2–4 articles; k = 3
   retrieval units almost never cover them all). The classifier degenerates to
   the full budget and its result on this cell must not be reported as a
   verifier comparison.
2. **No easy/hard difficulty contrast exists** here either; MultiHopRAG
   carries no difficulty label. A hop-depth contrast does exist and is
   reported (2/3/4 gold articles: n = 72/177/151, all guarded-minus-full
   contrasts unresolved).

## Calibration in the new environment

Error head: ECE 0.077 (train, cross-fitted) / 0.052 (test prefixes),
Brier 0.148 / 0.157. Selective risk at the guarded stop: AURC 0.121 for the
error head, 0.207 for the FBG value head, 0.274 for a random ordering — the
same ordering as the Wikipedia cells, and the same explanation: the error
signal carries information the future-gain signal does not.

Latency: per LLM call p50 0.574 s, p95 1.285 s; per question at the full
budget p50 6.95 s, p95 10.00 s.
