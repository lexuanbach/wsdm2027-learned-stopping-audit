# Learned Stopping Audit — WSDM 2027

Public research artifact for **A Selection-Aware Audit of Learned Stopping in Short-Horizon Iterative Retrieval**, by Xuan-Bach Le, Phat T. Tran-Truong, Son Ha Xuan, and Duy Nguyen. Accepted to WSDM 2027 Main.

Paper DOI (assigned; may not resolve before publication): https://doi.org/10.1145/3862322.3863667

This release contains code, stored experimental outputs, manuscript source and offline revision analyses. The manuscript is a camera-ready working revision; ACM rights completion and production approval are still pending. Historical four-search results from nine frozen backbones are preserved. Offline reanalysis does not mean the LLM endpoints have been rerun. A separate post-review extension collected 320 fresh eight-search HotpotQA trajectories with Claude Haiku 4.5 (120 train, 200 test). It does not replace the original core. All 12 exploratory conditional simultaneous matched-cost intervals contain zero. Collection cost is estimated at USD2.034058, not an invoice.

## Reproduction

Use Python 3.12+ and install the numerical dependencies from `requirements-lock.txt`. For offline replay the required core packages are numpy, pandas, scipy, scikit-learn, rank-bm25, matplotlib and pytest. The complete lock also lists optional model-provider packages.

```sh
python -m pytest tests -q
python revision/analyze_revision.py
python revision/pooled_training.py
python revision/deterministic_controls.py
python revision/analyze_long_horizon.py
cd manuscript
latexmk -pdf main.tex
latexmk -pdf supplement.tex
```

Run analysis commands from the repository root. The offline scripts load locally supplied pickle caches; use only the checksum-verified files from this release. The checksum manifest binds every released input. CPU timings may differ; the wall-time control extension is explicitly a mixed-hardware sensitivity analysis.

`revision/numerical-audit.json` records attribution, signed-target and reference-control checks. `revision/pooled-training.json` excludes target test questions from all training pools. `revision/deterministic-controls.json` reports the expanded reference family and an exploratory 84-contrast conditional simultaneous analysis. See the supplement source for uncertainty and scope.

`revision/ls13/protocol.json` freezes the new split and generation settings, `trajectories/` holds every completed run, `analysis.json` records every result, and `spend.json` records provider usage. `revision/analyze_long_horizon.py` reproduces the extension offline. To regenerate the costly trajectories, download the official HotpotQA distractor validation Parquet from https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet?download=true into `revision/ls13-data/hotpot_val.parquet`, verify its SHA-256 against the protocol, configure ANTHROPIC_API_KEY, and use `revision/long_horizon.py`. Existing completed trajectories are reused, so use a separate fresh workspace for a new collection. The runner has a USD25 list-price budget guard. Do not mistake a rerun with newer data/model service behavior for the exact archived experiment.

Fresh model experiments require dataset downloads and a configured Ollama/Anthropic/Bedrock endpoint. The script `code/run_experiment.py --help` describes the runner. They are separate from offline reproduction and may incur provider costs.

## License and provenance

The authors' software is MIT licensed; see `LICENSE`. Dataset-derived text is **not** relicensed as MIT. See `THIRD_PARTY_NOTICES.md` and `LICENSES/` for source attribution and separate terms. Raw full datasets, model weights, credentials, peer-review reports, author correspondence and tokenized production links are not included.
