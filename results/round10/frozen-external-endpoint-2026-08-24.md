# Frozen endpoint protocol -- external (non-Wikipedia) evaluation cell

Written 2026-08-24T10:05:19Z, BEFORE any policy was fit, any threshold was
selected, or any test question was answered in this environment. Nothing below
may be revised after test outcomes are read; if it is revised, the cell is
demoted to exploratory and this file records the revision.

## 1. Why this cell exists

The panel's stated scientific gap is external validity: all 26 cells are
Wikipedia-derived. This cell moves the evidence corpus off Wikipedia entirely.

## 2. The single confirmatory external cell

| Field | Value |
|---|---|
| Tag | `extnews_guarded_qwen3b` |
| Question source | yixuantt/MultiHopRAG, config `MultiHopRAG`, split `train`, `question_type == "inference_query"` only (816 rows of 2,556) |
| Evidence corpus | yixuantt/MultiHopRAG, config `corpus`, split `train` -- 609 English news articles from 49 outlets (technology, sports, entertainment, business, science, health). No Wikipedia text of any kind. |
| Corpus parquet sha256 | 8729d87c81dfe9d17f628abdaef861acf8d41c5e7be631f17b949e8be36907e2 |
| QA parquet sha256 | edb671d7640d83a42792fe23a6c2f8126c03dc62d11aecd2a16feb27f2a6b100 |
| Retrieval | `--corpus_index`: one shared SQLite FTS5/BM25 index over every sentence pooled from the converted parquet (the full news corpus), identical code path to the primary cell `corpus_guarded_qwen3b` |
| Backbone | `qwen2.5:3b-instruct` on local ollama, digest 357c53fb659c, Q4_K_M, temperature 0.0, `--llm_seed 7` |
| Split | `--data_seed 13 --n_train 120 --n_test 400`, identical to the primary cell |
| Depth | `--max_steps 4 --k 3 --alpha 0.10`, identical to the primary cell |
| Rule | `--primary_rule guarded` |

## 3. Primary endpoint (one, and only one)

* Primary quality metric: **exact match (EM)** on the 400 untouched test
  questions, with the paper's existing normalization (`run_experiment._norm`).
* Primary cost metric: **exact LLM calls**, reconstructed as `1 + 2T` from the
  logged per-prefix ledger.
* Primary comparison: **guarded FBG vs. the full budget** (the paper's own
  confirmatory contrast, `analyze_corpus_confirmation.py --tag extnews_guarded_qwen3b`).
* Predeclared success criterion, copied unchanged from the internal protocol:
  the paired-bootstrap 95% one-sided lower bound on (guarded - full budget) EM
  must exceed **-3.0 EM points**, AND guarded stopping must reduce mean LLM calls.
* Secondary, also frozen here: the same-frontier comparison against the
  **randomized fixed-budget mixture at matched expected cost** on calls, input
  tokens, and wall seconds (`round9_analyses.analyse_cell`), read as
  win / tie / loss / **unresolved** by whether the paired 95% CI excludes zero.

## 4. Selection rule (train-only)

theta (VoI threshold) and B* (best fixed budget) are selected **only** on this
cell's own 120 training chains under the frozen rule already in
`frozen_protocol.json`: "cheapest grid point with train EM >= train full-budget
EM - 1.0 band; fallback argmax train EM; ties broken by fewest searches". The
frozen theta from any Wikipedia cell is NOT reused. No test question
participates in any selection.

## 5. Pre-registered exclusions and their reasons (frozen before running)

* `comparison_query` (856) and `temporal_query` (583): gold answers are
  "Yes"/"no". EM on a binary label is dominated by prior-matching, not by
  evidence sufficiency, so these are **not** the confirmatory endpoint. They
  are declared here as an **exploratory** stratum and will be reported as such
  if run.
* `null_query` (301): gold answer is the literal string "Insufficient
  information."; the paper's answer prompt offers no abstention option, so
  every policy would score 0 by construction. **Excluded**, not reported as a
  result.
* This leaves `inference_query`, whose answers are short entities
  (median 2 tokens; e.g. "Sam Bankman-Fried", "OpenAI"), i.e. the same answer
  form the paper's EM normalization was built for. 816 rows >= 120 + 400.

## 6. Schema conversion, fixed before the run

`prepare_multihoprag_parquet.py` converts to the HotpotQA struct schema:
sentence-split article bodies (regex on sentence-final punctuation, drop
fragments under 4 tokens), `context = {title, sentences}`, `supporting_facts =
{title, sent_id}` located by verbatim match of each `evidence_list[i].fact`
inside the split article (measured before freezing: 2,721 of 2,743 facts are
exactly one split sentence, 4 more are contained in one; the remaining 18 fall
back to maximum token-overlap and are flagged in the adapter's report).
Distractor articles are assigned round-robin so the pooled corpus index covers
all 609 articles.

## 7. What had been inspected at freeze time (full disclosure)

The dataset file was downloaded and profiled at the SCHEMA level before this
document was written: row counts, `question_type` counts, gold-answer token
lengths, evidence-title coverage, and fact-to-sentence alignment rates. Those
are the numbers quoted in sections 2, 5 and 6. **No** model output, retrieval
result, policy, threshold, or EM value existed at freeze time.

## 8. Outcome commitment

A negative or null result is reported as-is, in the same table as the
Wikipedia cells. There is no post-hoc dataset swap, no re-tuning after seeing
the test fold, and no re-declaration of the endpoint. If the run fails for an
infrastructure reason, that fact is reported and the cell is dropped, not
replaced by a different one.

## 9. Single draw

Budget permits one 400-question draw at `data_seed 13`. This cell is therefore
a **single draw**, not a five-draw replication like the primary HotpotQA cell,
and must be described that way.
