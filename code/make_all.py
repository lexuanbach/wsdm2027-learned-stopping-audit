"""One command regenerates every reported table and figure from the
immutable measured inputs, then compiles the PDF.

    python3 make_all.py [--no-pdf]

Inputs (never modified): results/measured/run_<tag>.json,
critic_data_<tag>.pkl, real_baselines_*.json.
Outputs: results/measured/frozen_protocol.json, manuscript/tables/*.tex,
manuscript/figures/*.pdf, manuscript/main.pdf.

The manifest below lists the exact measured files each artifact consumes;
tests/test_consistency.py checks the emitted numbers against the JSON.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
MEAS = HERE.parent / "results" / "measured"
LOCAL_MANU = HERE.parent / "manuscript"
WORKTREE_MANU = HERE.parents[1] / "manuscript"
MANU = LOCAL_MANU if LOCAL_MANU.exists() else WORKTREE_MANU

CONFIRMATION_TAGS = [
    "corpus_guarded_qwen3b",
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
]

# Crossed backbone x population draws (round 9). Present only if the runs
# were executed; every analysis and table treats them as an optional block.
CROSSED_TAGS = [
    "corpus_guarded_llama3b_data17",
    "corpus_guarded_llama3b_data23",
    "corpus_guarded_llama3b_data29",
    "corpus_guarded_haiku45_data17",
    "corpus_guarded_haiku45_data23",
    "corpus_guarded_haiku45_data29",
]
EXTERNAL_TAGS = ["extnews_guarded_qwen3b"]

MANIFEST = {
    "frozen_protocol.py": [
        "run_final_haiku.json", "run_final_sonnet45.json",
        "run_final_gptoss120b.json", "run_final_novalite.json",
        "run_final_qwen3b.json", "run_final_llama3b.json",
        "run_final_commandr35b.json", "run_final_kimik25.json",
        "run_final_minimaxm25.json", "run_final2wiki_qwen3b.json",
        "run_rev2_musique_qwen3b.json", "run_rev2_musique_llama3b.json",
    ] + [f"critic_data_{t}.pkl" for t in [
        "final_haiku", "final_sonnet45", "final_gptoss120b", "final_novalite",
        "final_qwen3b", "final_llama3b", "final_commandr35b", "final_kimik25",
        "final_minimaxm25", "final2wiki_qwen3b", "rev2_musique_qwen3b",
        "rev2_musique_llama3b"]],
    "build_baselines_table.py": ["run_rev2_hotpot_qwen3b.json",
                                 "real_baselines_hotpot_qwen3b.json",
                                 "real_baselines_selfrag_fix.json"],
    "rq6_analysis.py": ["run_power_qwen3b.json", "run_shared_qwen3b.json",
                        "critic_data_power_qwen3b.pkl",
                        "critic_data_shared_qwen3b.pkl"],
    "analyze_corpus_confirmation.py": [
        "run_corpus_guarded_qwen3b.json",
        "critic_data_corpus_guarded_qwen3b.pkl",
        "run_corpus_guarded_qwen3b_seed11.json",
        "critic_data_corpus_guarded_qwen3b_seed11.pkl",
        "run_corpus_guarded_qwen3b_seed19.json",
        "critic_data_corpus_guarded_qwen3b_seed19.pkl",
        "run_corpus_guarded_qwen3b_data17.json",
        "critic_data_corpus_guarded_qwen3b_data17.pkl",
        "tasr_corpus_guarded_qwen3b.json",
        "semantic_features_corpus_guarded_qwen3b.json",
        "semantic_guard_corpus_guarded_qwen3b.json",
    ],
    "aggregate_corpus_robustness.py": [],
    "meta_review_analyses.py": [],
    "make_meta_figures.py": [],
    "round9_analyses.py": [],
    "build_whowins_table.py": ["round9_analyses.json"],
    "when_learned_wins.py": [],
    "answer_once_ledger.py": [],
    "training_size_curve.py": [],
    "evaluate_halt.py": [],
    "support_audit.py": ["support_audit_corpus_guarded_qwen3b.json"],
    "stoprag_fresh.py": [],
    "open_corpus_confirmation": [
        "run_wiki25m_guarded_qwen3b.json",
        "critic_data_wiki25m_guarded_qwen3b.pkl",
    ],
    "review_analyses.py": ["review_analyses.json (derived)"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="fail closed if current input/code hashes differ; "
                         "do not regenerate any result, table, figure, or PDF")
    ap.add_argument("--manifest-only", action="store_true",
                    help="rewrite input_manifest.json from the current inputs, "
                         "code, and tests without regenerating anything; use "
                         "only after edits that do not affect generated outputs "
                         "(e.g. the test suite itself)")
    args = ap.parse_args()

    digest = {}
    for step, files in MANIFEST.items():
        for f in files:
            p = MEAS / f
            if p.exists():
                digest[f"results/measured/{f}"] = hashlib.sha256(
                    p.read_bytes()).hexdigest()
    # Optional confirmatory baselines are measured inputs whenever present.
    # Keep derived corpus_confirmation*.json out: including an output as its
    # own input would make --verify-only fail after every regeneration.
    for tag in CONFIRMATION_TAGS + CROSSED_TAGS + EXTERNAL_TAGS:
        for f in [f"run_{tag}.json", f"critic_data_{tag}.pkl",
                  f"tasr_{tag}.json", f"semantic_features_{tag}.json",
                  f"semantic_guard_{tag}.json",
                  f"stoprag_result_{tag}.json", f"halt_{tag}.json",
                  f"halt_claims_{tag}.json", f"halt_scores_{tag}.json"]:
            p = MEAS / f
            if p.exists():
                digest[f"results/measured/{f}"] = hashlib.sha256(
                    p.read_bytes()).hexdigest()
    for script in sorted(HERE.glob("*.py")):
        digest[f"code/{script.name}"] = hashlib.sha256(
            script.read_bytes()).hexdigest()
    artifact_root = HERE.parent
    for path in [artifact_root / "requirements-lock.txt",
                 artifact_root / "Dockerfile",
                 artifact_root / "pytest.ini"]:
        digest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((artifact_root / "tests").glob("test_*.py")):
        digest[f"tests/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    local_workflow = artifact_root / ".github" / "workflows" / "artifact.yml"
    worktree_workflow = (artifact_root.parent / ".github" / "workflows"
                         / "artifact.yml")
    workflow = local_workflow if local_workflow.exists() else worktree_workflow
    if workflow.exists():
        digest[".github/workflows/artifact.yml"] = hashlib.sha256(
            workflow.read_bytes()).hexdigest()
    manifest_path = MEAS / "input_manifest.json"
    if args.verify_only:
        expected = json.loads(manifest_path.read_text())
        if expected != digest:
            missing = sorted(set(expected) - set(digest))
            added = sorted(set(digest) - set(expected))
            changed = sorted(k for k in set(expected) & set(digest)
                             if expected[k] != digest[k])
            raise SystemExit(f"manifest mismatch: missing={missing}, "
                             f"added={added}, changed={changed}")
        r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"],
                           cwd=HERE.parent)
        raise SystemExit(r.returncode)
    manifest_path.write_text(json.dumps(digest, indent=1))
    print(f"[manifest] {len(digest)} measured inputs hashed -> "
          f"{manifest_path}")
    if args.manifest_only:
        return

    scripts = ["frozen_protocol.py", "revision_experiments.py",
               "build_frozen_figures.py", "build_baselines_table.py",
               "rq6_analysis.py"]
    if (MEAS / "run_corpus_guarded_qwen3b.json").exists():
        scripts.append("analyze_corpus_confirmation.py")
    for tag in CONFIRMATION_TAGS[1:] + CROSSED_TAGS:
        if (MEAS / f"run_{tag}.json").exists():
            scripts.append(["analyze_corpus_confirmation.py", "--tag", tag])
    if (MEAS / "corpus_confirmation.json").exists():
        scripts.append("aggregate_corpus_robustness.py")
        scripts.append("meta_review_analyses.py")
        scripts.append("make_meta_figures.py")
        scripts.append("round9_analyses.py")
        scripts.append("build_whowins_table.py")
        scripts.append(["when_learned_wins.py", "--reuse"])
        scripts.append("answer_once_ledger.py")
        # supplement-only tables added in round 10; both are pure replays of
        # the logged prefixes and cost seconds, no generation or retrieval
        scripts.append("oracle_frontier.py")
        scripts.append("strata_audit.py")
        # 14 cells x 5 sizes x 20 draws; ~250 s on CPU
        scripts.append("training_size_curve.py")
    for command in scripts:
        argv = command if isinstance(command, list) else [command]
        print(f"[run] {' '.join(argv)}")
        r = subprocess.run([sys.executable, str(HERE / argv[0]), *argv[1:]],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            sys.exit(f"{' '.join(argv)} failed")

    if not args.no_pdf:
        print("[run] latexmk")
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                            "main.tex"], cwd=MANU, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            sys.exit("latexmk failed; see manuscript/main.log")
        print(f"[done] {MANU / 'main.pdf'}")
        if (MANU / "supplement.tex").exists():
            print("[run] latexmk supplement")
            r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                                "supplement.tex"], cwd=MANU, capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                sys.exit("latexmk supplement failed; see manuscript/supplement.log")
            print(f"[done] {MANU / 'supplement.pdf'}")


if __name__ == "__main__":
    main()
