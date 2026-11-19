# A Selection-Aware Audit of Learned Stopping in Short-Horizon Iterative Retrieval

Code, stored trajectories, and analysis scripts for the WSDM 2027 paper

> Xuan-Bach Le, Phat T. Tran-Truong, Son Ha Xuan, and Duy Nguyen. 2027.
> *A Selection-Aware Audit of Learned Stopping in Short-Horizon Iterative Retrieval.*
> In Proceedings of the Twentieth ACM International Conference on Web Search and
> Data Mining (WSDM 2027), Hong Kong. https://doi.org/10.1145/3862322.3863667

The online supplement is included as [`supplement.pdf`](supplement.pdf).

## Overview

An iterative retrieval agent alternates between searching and answering, and a
stopping rule decides when to stop searching and commit to an answer. The paper
audits learned stopping rules (future-best gain, binary future improvement,
current-answer error, and gold-support coverage) against simple policies (fixed
search budgets and answer repetition) at matched cost. Policies are fitted with
five-fold cross-fitting, thresholds are selected on development questions and
frozen, and every comparison is evaluated on untouched test questions.

This repository contains everything needed to recompute the tables and figures
of the paper and the supplement from stored trajectories, on a CPU, without any
LLM or retrieval service. It also contains the code that collected the
trajectories, for readers who want to run new experiments.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Experiment runner, stopping controller, analysis scripts, and `make_all.py`, which regenerates every table and figure |
| `results/measured/` | Stored trajectories (`run_<cell>.json`), critic caches (`critic_data_<cell>.pkl`), baseline runs, and derived analyses for all cells |
| `results/round10/` | External news cell, calibration, sufficiency verifier, multiplicity correction, policy cards, and the per-question paired outcomes (`per_question_paired.csv`) |
| `results/retrospective/` | Pooled training, expanded-reference controls, and the precision checks they use |
| `results/long_horizon/` | Frozen protocol and 320 fresh eight-search trajectories for the longer-horizon check |
| `tests/` | Unit tests, data-consistency tests, and checks that regenerated tables match the paper |
| `outputs/` | Created by `make_all.py`: `tables/` (LaTeX table rows as typeset) and `figures/` (PDF figures). Not tracked |
| `SHA256SUMS` | SHA-256 of every file in `results/` |
| `supplement.pdf` | Online supplement |

A *cell* fixes the dataset, retrieval index, backbone, question sample, and
decoding seed. Cell names encode these, for example
`corpus_guarded_qwen3b_data17` is the shared HotpotQA corpus index, Qwen2.5-3B,
question sample with data seed 17, and the guarded rule as the predeclared
policy. `final_<model>` cells are the nine call-only HotpotQA backbone cells.

## Requirements

* Python 3.11 or newer (tested with 3.12 and 3.14) on Linux or macOS.
* CPU only, no GPU. The repository needs about 150 MB of disk.
* Optional: a LaTeX installation with the `libertine` and `newtxmath` packages.
  With it the figures are typeset in the paper's font. Without it the same
  figures are drawn with matplotlib's default serif font.

## Installation

```sh
git clone https://github.com/lexuanbach/wsdm2027-learned-stopping-audit.git
cd wsdm2027-learned-stopping-audit
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Or with Docker, which runs the full regeneration and the tests:

```sh
docker build -t learned-stopping .
docker run --rm learned-stopping
```

## Quick check

```sh
python code/make_all.py --check   # verify results/ against SHA256SUMS
python -m pytest -q               # unit and data-consistency tests (seconds)
```

## Reproducing the paper

```sh
python code/make_all.py           # full recomputation, about 20 to 30 minutes on a laptop CPU
python -m pytest -q               # now also checks every regenerated table
```

`make_all.py --quick` reuses the stored nested bootstrap (400 resamples per
run) and the stored training-size curve, and finishes in a few minutes.

`make_all.py` writes the table rows to `outputs/tables/` and the figures to
`outputs/figures/`, and recomputes the derived JSON files in `results/`.
`tests/test_reproduction.py` then checks that all 27 regenerated tables are
byte-identical to the ones typeset in the camera-ready paper and supplement.
With LaTeX installed, the regenerated figures are also pixel-identical to the
paper's.

We verified this from a fresh clone in a clean Python 3.12 environment on
macOS, and the CI workflow repeats the table check on Linux. The nested
bootstrap (`frontier_analyses.json`) comes back byte-identical. The round-9
analysis (`round9_analyses.json`) is byte-identical on macOS and differs only
in floating-point detail on Linux, with every table built from it unchanged.
The other derived files listed in `tests/recomputed.txt` are rewritten with
different JSON formatting. Their values are unchanged, with three exceptions
that do not enter any table or figure: CPU timing fields (`deterministic_controls.json`,
`training_size_curve.json`), small numerical differences in two unreported
diagnostic blocks (the training-frontier diagnostics and the
Command-R to Llama transfer row of `frozen_protocol.json`, and the
out-of-fold calibration bins of `revision_experiments.json`), and the TASR
replay of the crossed Llama cell (`corpus_confirmation_corpus_guarded_llama3b_data17.json`,
`answer_once_ledger.json`), which was collected after those files were archived.
`git checkout results/` restores the archived versions.

### Main paper

| Item | Content | Script (in `code/`) | Output | Main inputs |
|---|---|---|---|---|
| Figure 1 | Overview diagram and logged example | not generated | | example question `5a7df1095542990b8f503afe` in `results/measured/run_corpus_guarded_qwen3b.json` |
| Table 1 | Retrieval settings and populations | descriptive | | run configurations in `results/measured/run_*.json` |
| Table 2 | Claim-to-evidence hierarchy | descriptive | | |
| Figure 2 | EM versus exact LLM calls | `make_figures.py` | `figures/fig_frontier_main.pdf` | `corpus_confirmation*.json` |
| Table 3 | Complete FBG point audit (26 cells) | `frontier_analyses.py` | `tables/decision_matrix_rows.tex` | `run_*.json`, `critic_data_*.pkl`, `frozen_protocol.json` |
| Table 4 | Hindsight-oracle ceiling | `oracle_frontier.py` | `tables/oracle_frontier_rows.tex` | corpus-cell trajectories |
| Table 5 | Cross-signal frontier and matched-cost summary | `build_signal_summary_table.py` | `tables/signal_summary_rows.tex` | `round10/extended_decision_matrix.json`, `round10/mixture_wins_multiplicity.json` |
| Figure 3 | Guarded FBG minus full search or two fixed searches | `make_figures.py` | `figures/fig_samples_main.pdf` | `frontier_analyses.json` |
| Table 6 | External news-cell policies | `build_external_policy_table.py` | `tables/external_policy_rows.tex` | `round10/external_cell_extnews_guarded_qwen3b.json` |
| Table 7 | Signal and trajectory diagnostics | stored values | | `round10/calibration_audit.json` (risk ordering, ECE), `round10/sufficiency_classifier.json` (AUROC), `measured/frozen_protocol.json` (`flips.reachability`, late answer reversals), `measured/support_audit_corpus_guarded_qwen3b.json` (`overall`, rerolled futures) |
| Figure 4 | Nested refit and reselection | `make_figures.py` | `figures/fig_nested.pdf` | `frontier_analyses.json` (nested block) |
| Table 8 | Deployment-matched SGR certificate | `analyze_corpus_confirmation.py` | `dedicated_calibration_sgr` block of `corpus_confirmation*.json` | corpus-cell trajectories |

### Supplement

| Item | Content | Script | Output |
|---|---|---|---|
| Tables 1, 2, 4 | Hierarchy, regime guidance, related signals | descriptive | |
| Table 3 | External-cell policies | `build_external_policy_table.py` | `external_policy_rows.tex` |
| Table 5 | Fourteen-cell frontier summary | `build_whowins_table.py` | `whowins_rows.tex` |
| Tables 6 to 11 | ε-dominance, targets, equal cost, crossed draws, pooled frontiers | `round9_analyses.py` | `round9_epsilon_rows.tex`, `round9_target_rows.tex`, `round9_target_value_rows.tex`, `round9_equalcost_rows.tex`, `round9_crossed_rows.tex`, `round9_pooled_rows.tex` |
| Table 12 | Question-type and support strata | `strata_audit.py` | `strata_audit_rows.tex` |
| Table 13 | Answer-once cost ledger | `answer_once_ledger.py` | `answer_once_rows.tex` |
| Table 14 | Verifier-inspired substitutes | `round9_analyses.py` | `round9_halt_rows.tex` |
| Table 15 | Predicting learned-rule survival | `when_learned_wins.py` | `when_wins_rows.tex` |
| Table 16 | Pooled policy training | `pooled_training.py` | `pooled_training_rows.tex` |
| Table 17 | Training-size sensitivity | `training_size_curve.py` | `training_size_rows.tex` |
| Tables 18, 19 | Selection sensitivity, feature ablations | `round9_analyses.py` | `round9_sensitivity_rows.tex`, `round9_ablation_rows.tex` |
| Table 20, nested figure | Nested refit and reselection | `frontier_analyses.py`, `make_figures.py` | `nested_uncertainty_rows.tex`, `fig_nested.pdf` |
| Table 21, samples figure | Paired contrasts across reruns and draws | `frontier_analyses.py`, `make_figures.py` | `simple_policy_paired_table.tex`, `fig_samples.pdf` |
| Frontier figure | Frontiers for all corpus cells | `make_figures.py` | `fig_frontier_all.pdf` |
| Table 22 | Primary corpus-wide confirmation | `analyze_corpus_confirmation.py` | `corpus_confirmation_rows.tex` |
| Tables 23, 24 | Nine backbones, dataset replication | `frozen_protocol.py` | `backbones_frozen_rows.tex`, `datasets_frozen_rows.tex` |
| Table 25 | Adaptive-retrieval baselines | `build_baselines_table.py` | `baselines_rows.tex` |
| Table 26 | Expanded-reference sensitivity | `deterministic_controls.py` | `deterministic_controls_rows.tex` |
| Table 27 | Fresh longer-horizon check | `analyze_long_horizon.py` | `long_horizon_rows.tex` |

The analyses in `results/round10/` (external cell, calibration, sufficiency
verifier, multiplicity correction, simultaneous intervals, cost and latency,
policy cards) are stored outputs that `make_all.py` reads but does not rewrite.
Each can be recomputed into a separate folder in a few seconds. The recomputed
files contain the same values as the stored ones (only JSON whitespace differs,
and the external-cell report records its protocol path under `results/round10/`):

```sh
O=outputs/round10
python code/calibration_audit.py         --out $O
python code/sufficiency_classifier.py    --out $O
python code/extended_decision_matrix.py  --out $O --suffclf $O/sufficiency_classifier.json
python code/mixture_wins_multiplicity.py --out $O
python code/simultaneous_intervals.py    --out $O
python code/cost_latency_audit.py        --out $O
python code/strata_audit_extra.py        --out $O
python code/policy_cards.py              --out $O   # also prompt ledger and per_question_paired.csv
python code/frontier_figure_data.py      --out $O
python code/external_cell_report.py --tag extnews_guarded_qwen3b \
    --label "MultiHopRAG news / Qwen2.5-3B (external)" --out $O
```

## What the stored results contain

* `run_<cell>.json`: one record per test question with the full trajectory
  (queries, retrieved evidence, the answer after every search, token counts and
  latency for the fully instrumented cells), per-prefix EM and F1, the stopping
  features, and the cell configuration (model, seeds, split sizes).
* `critic_data_<cell>.pkl`: the training trajectories and features used to fit
  the critics. These are Python pickles. Load them only from this repository.
* `corpus_confirmation*.json`, `frozen_protocol.json`, `round9_analyses.json`,
  `frontier_analyses.json` and the other JSON files are derived analyses and are
  recomputed by `make_all.py`.

## Running new experiments

New trajectories need the datasets, a model endpoint, and
`pip install -r requirements-experiments.txt`. New runs are separate from the
offline reproduction above. Provider models and services change over time, so
a rerun reproduces the protocol, not the exact archived outputs.

Backbones. Local models run through [Ollama](https://ollama.com) at
`http://localhost:11434` (`qwen2.5:3b-instruct`, `llama3.2:3b`,
`command-r:35b`). The fully instrumented cells record Ollama 0.30.10 and the
Qwen2.5-3B digest `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`
(Q4_K_M). Model names starting with `claude-` use the Anthropic API
(`ANTHROPIC_API_KEY`). Amazon Bedrock model IDs (`us.anthropic.claude-haiku-4-5-20251001-v1:0`,
`us.anthropic.claude-sonnet-4-5-20250929-v1:0`, `us.amazon.nova-lite-v1:0`,
`moonshotai.kimi-k2.5`, `minimax.minimax-m2.5`, `openai.gpt-oss-120b-1:0`)
use Bedrock (`AWS_BEARER_TOKEN_BEDROCK` or standard AWS credentials).

Datasets (place them in `data/`, which is not tracked):

```sh
mkdir -p data
# HotpotQA distractor validation (7,405 questions)
curl -L -o data/hotpot_val.parquet \
  "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet?download=true"
```

* 2WikiMultihopQA dev (12,576 questions), from the Hugging Face mirror
  `xanhho/2WikiMultihopQA`, converted with
  `python code/prepare_2wiki_parquet.py --src data/2wiki_dev.parquet --dst data/2wiki_dev_hotpot_schema.parquet`.
* MuSiQue-Answerable dev, `musique_ans_v1.0_dev.jsonl` (2,417 questions), from
  https://github.com/StonyBrookNLP/musique.
* MultiHop-RAG (`yixuantt/MultiHopRAG`, configs `MultiHopRAG` and `corpus`),
  converted with `python code/prepare_multihoprag_parquet.py --qa ... --corpus ... --dst ...`.
  The frozen protocol for this cell is in
  `results/round10/frozen-external-endpoint-2026-08-24.md`.
* FEVER 2017 Wikipedia `wiki-pages`, indexed with
  `python code/build_wikipedia_index.py --wiki-pages data/wiki-pages --output data/fever_wiki.sqlite`.

One fully instrumented cell (520 questions with local Qwen2.5-3B on Apple
silicon or a consumer GPU: about one to one and a half hours):

```sh
python code/run_experiment.py --parquet data/hotpot_val.parquet \
  --model qwen2.5:3b-instruct --n_train 120 --n_test 400 --max_steps 4 \
  --certificate_calibration_fraction 0.25 --corpus_index --primary_rule guarded \
  --llm_seed 7 --llm_temperature 0.0 --data_seed 17 \
  --tag corpus_guarded_qwen3b_data17
python code/evaluate_tasr.py --tag corpus_guarded_qwen3b_data17
python code/evaluate_semantic_guard.py --tag corpus_guarded_qwen3b_data17
python code/analyze_corpus_confirmation.py --tag corpus_guarded_qwen3b_data17
```

The settings of every stored cell (model, `data_seed`, `llm_seed`, split sizes,
index) are in the `config` block of its `run_<cell>.json`. The primary cell
`corpus_guarded_qwen3b` uses `--data_seed 13 --llm_seed 7`. Same-question
reruns change `--llm_seed` (11, 19). Independent question draws change
`--data_seed` (17, 23, 29, 31). 2Wiki and MuSiQue use `--parquet` or `--jsonl`
with the files above. The Wikipedia cell adds

```sh
  --sqlite_corpus data/fever_wiki.sqlite \
  --sqlite_corpus_sha256 e903b6bd72337f236ae6667642597565f5f5b3b95a0748275b4459eef2671149 \
  --sqlite_corpus_source "official FEVER Wikipedia 2017 wiki-pages" \
  --sqlite_reranker cross-encoder/ms-marco-MiniLM-L6-v2 \
  --sqlite_reranker_revision 233902d25c440f23af6f7d6e94d2946bac0bee0a \
  --sqlite_candidate_k 50 --tag wiki25m_guarded_qwen3b
```

Other learned baselines are fitted on stored trajectories. Stop-RAG
(`stoprag_fresh.py --tag <cell>`) fine-tunes DeBERTa-v3-base, about 9 minutes
per cell on Apple silicon or a GPU. The verifier-inspired substitute
(`evaluate_halt.py --tag <cell>`) uses two public cross-encoders. The support
audit (`support_audit.py --tag corpus_guarded_qwen3b --n 300 --rollouts 2`)
re-rolls 600 continuations with the local Qwen2.5-3B, about 25 minutes.

The longer-horizon check (supplement Table 27) collected 320 eight-search
HotpotQA trajectories with Claude Haiku 4.5 (`claude-haiku-4-5-20251001`,
temperature 0). Its frozen protocol, question IDs, and dataset hash are in
`results/long_horizon/protocol.json`. To collect a new set, download
`data/hotpot_val.parquet`, set `ANTHROPIC_API_KEY`, and run
`LONG_HORIZON_OUT=<empty folder> python code/long_horizon.py`. The script stops
at an estimated USD 25 at list prices. The archived collection cost an
estimated USD 2.03.

## Data sources and licenses

The code is released under the MIT License (`LICENSE`). The MIT License does
not apply to third-party text in the stored trajectories. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and `LICENSES/`:

* HotpotQA (Yang et al., 2018), CC BY-SA 4.0.
* 2WikiMultihopQA (Ho et al., 2020), Apache 2.0 for the repository, with
  Wikipedia and Wikidata content under their own terms.
* MuSiQue (Trivedi et al., 2022), CC BY 4.0.
* Wikipedia text (FEVER 2017 snapshot), CC BY-SA.
* MultiHop-RAG (Tang and Yang, 2024), ODC-BY 1.0. News passages are removed
  from the stored external-cell files. Questions, short answers, IDs, features,
  and results remain.

No model weights, credentials, or full datasets are included.

## Citation

```bibtex
@inproceedings{le2027learnedstopping,
  author    = {Le, Xuan-Bach and Tran-Truong, Phat T. and Ha Xuan, Son and Nguyen, Duy},
  title     = {A Selection-Aware Audit of Learned Stopping in Short-Horizon Iterative Retrieval},
  booktitle = {Proceedings of the Twentieth ACM International Conference on Web Search and Data Mining},
  series    = {WSDM '27},
  year      = {2027},
  address   = {Hong Kong},
  publisher = {ACM},
  doi       = {10.1145/3862322.3863667}
}
```

## Contact

Xuan-Bach Le (corresponding author). Please open an issue in this repository
for questions about the code or data.
