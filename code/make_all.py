"""Regenerate every table and figure of the paper and its supplement.

    python code/make_all.py           # full offline recomputation (CPU only)
    python code/make_all.py --quick   # reuse the stored nested bootstrap and
                                      # training-size curve (a few minutes)
    python code/make_all.py --check   # only verify results/ against SHA256SUMS

Inputs are the stored trajectories and critic caches in results/. No LLM,
retrieval service, or GPU is used. Outputs:

    outputs/tables/*.tex     table rows exactly as typeset in the paper
    outputs/figures/*.pdf    plotted figures

Derived JSON files in results/ (for example frozen_protocol.json) are
recomputed in place. On a clean checkout they come back with the same content,
so `git status results/` stays clean apart from the wall-clock fields noted in
the README.
"""
import argparse
import hashlib
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MEAS = ROOT / "results" / "measured"

CONFIRMATION_TAGS = [
    "corpus_guarded_qwen3b_seed11",
    "corpus_guarded_qwen3b_seed19",
    "corpus_guarded_qwen3b_data17",
    "corpus_guarded_qwen3b_data23",
    "corpus_guarded_qwen3b_data29",
    "corpus_guarded_qwen3b_data31",
    "corpus2wiki_guarded_qwen3b_data17",
    "corpus2wiki_guarded_qwen3b_data23",
    "corpus2wiki_guarded_qwen3b_data29",
    "corpusmusique_guarded_qwen3b_data17",
    "corpusmusique_guarded_qwen3b_data23",
    "corpusmusique_guarded_qwen3b_data29",
    "wiki25m_guarded_qwen3b",
    # crossed backbone x population draws
    "corpus_guarded_llama3b_data17",
    "corpus_guarded_llama3b_data23",
    "corpus_guarded_llama3b_data29",
    "corpus_guarded_haiku45_data17",
    "corpus_guarded_haiku45_data23",
    "corpus_guarded_haiku45_data29",
]


def steps(quick):
    """Ordered pipeline. Each entry is [script, *arguments]."""
    out = [
        ["frozen_protocol.py"],             # Supp. Tables 23-24
        ["revision_experiments.py"],        # diagnostics used by later steps
        ["build_baselines_table.py"],       # Supp. Table 25
        ["analyze_corpus_confirmation.py"],  # Supp. Table 22 (sample A, seed 7)
    ]
    out += [["analyze_corpus_confirmation.py", "--tag", t]
            for t in CONFIRMATION_TAGS if (MEAS / f"run_{t}.json").exists()]
    out += [
        ["aggregate_corpus_robustness.py"],
        # Table 3, Supp. Tables 20-21, Figures 3-4 data
        ["frontier_analyses.py"] + (["--reuse-nested"] if quick else []),
        ["make_figures.py"],                # Figures 2-4, Supp. figures
        ["round9_analyses.py"],             # Supp. Tables 6-11, 14, 18-19
        ["build_whowins_table.py"],         # Supp. Table 5
        ["when_learned_wins.py", "--reuse"],  # Supp. Table 15
        ["answer_once_ledger.py"],          # Supp. Table 13
        ["oracle_frontier.py"],             # Table 4
        ["strata_audit.py"],                # Supp. Table 12
        # Supp. Table 17 (about 4 minutes when recomputed)
        ["training_size_curve.py"] + (["--emit-only"] if quick else []),
        ["build_signal_summary_table.py"],  # Table 5
        ["build_external_policy_table.py"],  # Table 6, Supp. Table 3
        ["retrospective_checks.py"],        # inputs to the next two steps
        ["pooled_training.py"],             # Supp. Table 16
        ["deterministic_controls.py"],      # Supp. Table 26
        ["analyze_long_horizon.py"],        # Supp. Table 27
    ]
    return out


def check_sums():
    """Verify the shipped results against SHA256SUMS.

    Stored inputs must match exactly. Files that this script recomputes
    (tests/recomputed.txt) are only reported, because a previous run may have
    rewritten them.
    """
    recomputed = set((ROOT / "tests" / "recomputed.txt").read_text().split())
    bad, rewritten, n = [], [], 0
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = ROOT / name
        n += 1
        if (path.exists()
                and hashlib.sha256(path.read_bytes()).hexdigest() == digest):
            continue
        (rewritten if name in recomputed else bad).append(name)
    if bad:
        sys.exit(f"[check] {len(bad)} stored files differ from SHA256SUMS: "
                 + ", ".join(bad[:10]))
    print(f"[check] {n - len(rewritten)} of {n} files match SHA256SUMS")
    if rewritten:
        print(f"[check] {len(rewritten)} recomputed files differ from the "
              "archived versions: " + ", ".join(rewritten))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true",
                    help="reuse the stored nested bootstrap and training-size curve")
    ap.add_argument("--check", action="store_true",
                    help="only verify results/ against SHA256SUMS")
    args = ap.parse_args()
    check_sums()
    if args.check:
        return
    start = time.time()
    for argv in steps(args.quick):
        print(f"[run] {' '.join(argv)}", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / argv[0]), *argv[1:]],
                           cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            sys.exit(f"{' '.join(argv)} failed")
        print(f"      done in {time.time() - t0:.0f} s", flush=True)
    print(f"[done] {time.time() - start:.0f} s. Tables: outputs/tables, "
          "figures: outputs/figures")


if __name__ == "__main__":
    main()
