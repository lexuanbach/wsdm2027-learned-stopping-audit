"""Automated consistency tests: paper numbers == measured files == one
deployed policy. Run from artifact/: python3 -m pytest tests/ -q"""
import json
import math
import pathlib
import pickle
import sys

import numpy as np
import pytest

CODE = pathlib.Path(__file__).resolve().parents[1] / "code"
MEAS = pathlib.Path(__file__).resolve().parents[1] / "results" / "measured"
ART = pathlib.Path(__file__).resolve().parents[1]
WORKTREE_MANU = ART.parent / "manuscript"
MANU = WORKTREE_MANU if WORKTREE_MANU.exists() else ART / "manuscript"
PROJECT = WORKTREE_MANU.parent
ROUND10_WORKTREE = PROJECT / "results" / "round10-external-2026-08-24"
ROUND10_RELEASE = ART / "results" / "round10"
ROUND10 = ROUND10_WORKTREE if ROUND10_WORKTREE.exists() else ROUND10_RELEASE
sys.path.insert(0, str(CODE))

import frozen_protocol as fp                      # noqa: E402
from voi_controller import (guarded_stop_index, stop_index,
                            sgr_threshold)   # noqa: E402
from run_experiment import CorpusBM25, SQLiteCorpusBM25, file_sha256  # noqa: E402
from evaluate_tasr import first_answer_margin  # noqa: E402
from build_wikipedia_index import parse_lines  # noqa: E402

HOTPOT = fp.HOTPOT_TAGS


def test_manuscript_style_constraints_and_verdict_vocabulary():
    sources = [MANU / "main.tex", MANU / "supplement.tex"]
    sources.extend(sorted((MANU / "sections").glob("*.tex")))
    text = "\n".join(p.read_text() for p in sources)
    assert "--" not in text
    assert ";" not in text.replace(r"\;", "")
    assert "W/T/L" not in text
    assert "win/tie/loss" not in text
    matrix = (MANU / "tables" / "decision_matrix_rows.tex").read_text()
    for line in matrix.splitlines():
        if not line.strip() or "&" not in line:
            continue
        cells = [x.strip() for x in line.split("&")]
        assert cells[7] in {"D", "F", "I"}
        assert cells[10] in {"D", "F", "I"}


def test_intro_motivating_example_matches_logged_trajectory():
    """The illustrative answer reversal in the introduction is a logged trace."""
    path = MEAS / "run_corpus_guarded_qwen3b.json"
    if not path.exists():
        pytest.skip("primary run registry absent from this bundle")
    records = json.loads(path.read_text())["per_q"]
    item = next(r for r in records if r["id"] == "5a7df1095542990b8f503afe")
    answers = [prefix["answer"] for prefix in item["trajectory"]]

    assert item["question"] == "Who was born first out of  Thomas Carr and Joyce Wieland?"
    assert item["prefix_em"] == [0, 0, 1, 0, 0]
    assert answers[1:4] == ["Joyce Wieland", "Thomas Carr", "Joyce Wieland"]
    assert item["primary"]["searches"] == 3 and item["primary"]["em"] == 0
    assert item["prefix_em"][2] == 1 and item["prefix_em"][-1] == 0

    intro = (MANU / "sections" / "01-introduction.tex").read_text()
    assert "Who was born first, Thomas Carr or Joyce Wieland?" in intro
    assert "Thomas Carr after two searches" in intro
    assert "learned guard stops after that third search" in intro
    assert "fixed two-search budget" in intro


def test_round10_signal_summary_and_external_claims_when_present():
    extended_path = ROUND10 / "extended_decision_matrix.json"
    external_path = ROUND10 / "external_cell_extnews_guarded_qwen3b.json"
    if not extended_path.exists() or not external_path.exists():
        pytest.skip("round 10 analysis package is absent")
    extended = json.loads(extended_path.read_text())
    summary = extended["summary"]
    rows = (MANU / "tables" / "signal_summary_rows.tex").read_text()
    names = {
        "FBG-only": "value_only",
        "Guarded FBG": "guarded_value",
        "Stop classifier": "stop_classifier",
        "Error-head gate": "sufficiency_gate",
        "Gold-support sufficiency": "gold_support_sufficiency_clf_same9",
        "Full TASR": "tasr",
        "Stop-RAG": "stoprag",
    }
    for label, key in names.items():
        s = summary[key]
        d = s["dominance"]
        matched = s["vs_matched_mixture"]
        n_matched = sum(matched.values())
        effects = [
            cell["policies"][key]["vs_matched_mixture_on_calls"]
                ["delta_em"]["mean_em_points"]
            for cell in extended["cells"]
            if key in cell["policies"]
            and cell["policies"][key].get("vs_matched_mixture_on_calls")
        ]
        assert (f"{label} & {s['cells_covered']}/{n_matched} & "
                f"{d['D']}/{d['F']}/{d['I']} & "
                f"${min(effects):.2f},{max(effects):.2f}$ & "
                f"{matched['win']}/{matched['loss']}") in rows

    multiplicity_path = ROUND10 / "mixture_wins_multiplicity.json"
    if multiplicity_path.exists():
        multiplicity = json.loads(multiplicity_path.read_text())["policies"]
        for label, key in names.items():
            if key == "gold_support_sufficiency_clf_same9":
                line = next(x for x in rows.splitlines()
                            if x.startswith(label + " &"))
                assert line.endswith("NA & NA \\\\"), line
                continue
            q = multiplicity[key]
            line = next(x for x in rows.splitlines()
                        if x.startswith(label + " &"))
            counts = q["counts"]
            assert line.endswith(
                f"& {counts['bonferroni_win']} & {counts['maxt_win']} \\\\"
            ), line
    manuscript = ((MANU / "main.tex").read_text() + "\n" +
                  "\n".join(p.read_text() for p in
                            (MANU / "sections").glob("*.tex")))
    ext = json.loads(external_path.read_text())["primary_endpoint"]
    assert f"{ext['guarded_em']:.2f} EM" in manuscript
    assert f"{ext['guarded_calls']:.2f} calls" in manuscript
    assert "matched-cost advantage remains unresolved" in manuscript


@pytest.fixture(scope="module")
def fz():
    return json.loads((MEAS / "frozen_protocol.json").read_text())


def test_replay_matches_logged_voi_sweep():
    """The single deployed rule reproduces every logged voi[lam] entry."""
    for tag in ["final_haiku", "final_qwen3b"]:
        d = fp.load_run(tag)
        for r in d["per_q"]:
            for lam in d["lam_grid"]:
                t = stop_index(r["prefix_value"], float(lam))
                assert r["voi"][str(lam)]["em"] == r["prefix_em"][t]
                assert (r["voi"][str(lam)]["searches"]
                        == r["prefix_searches"][t])


def test_stop_rules_agree():
    rng = np.random.default_rng(0)
    for _ in range(200):
        vals = rng.uniform(0, 0.3, 5).tolist()
        lam = float(rng.uniform(0, 0.3))
        assert stop_index(vals, lam) == fp.test_stop_index(
            {"prefix_value": vals}, lam)


def test_guarded_stop_is_max_of_first_crossings():
    values = [0.2, 0.03, 0.01, 0.0]
    answers = ["A", "B", "B", "B"]
    assert stop_index(values, 0.04) == 1
    assert guarded_stop_index(values, answers, 0.04) == 2
    assert guarded_stop_index(values, ["A", "A", "C", "D"], 0.04) == 1


def test_frozen_protocol_shape(fz):
    assert len(fz["hotpot_cells"]) == 9
    grid = [0.01, 0.02, 0.04, 0.07, 0.12, 0.2]
    for c in fz["hotpot_cells"]:
        assert c["lambda"] in grid
        assert 0 <= c["voi_frozen"]["save_pct"] <= 100
        lo, hi = c["paired_voi_vs_react"]["ci95"]
        assert lo <= c["paired_voi_vs_react"]["mean_diff"] <= hi
    assert fz["global_lambda"]["lambda"] in grid


def test_tex_rows_match_json(fz):
    rows = (MANU / "tables" / "backbones_frozen_rows.tex").read_text()
    assert rows.count("\\\\") == 9
    for c in fz["hotpot_cells"]:
        assert f"{c['voi_frozen']['em']:.1f}" in rows
        assert f"{c['voi_frozen']['save_pct']}\\%" in rows
    trows = (MANU / "tables" / "transfer_frozen_rows.tex").read_text()
    assert trows.count("\\\\") == len(fz["transfer"]) == 4


def test_sgr_corrected_is_no_looser_than_uncorrected():
    """Corrected SGR never certifies a larger set than a pointwise scan."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        m = 80
        eh = np.sort(rng.uniform(0, 1, m))
        err = (rng.uniform(0, 1, m) < eh).astype(int)
        for alpha in [0.2, 0.4, 0.6]:
            th, nsel = sgr_threshold(eh, err, alpha, delta=0.1)
            # uncorrected scan at full delta
            from scipy.stats import beta as beta_dist
            best_n = 0
            k = 0
            for i in range(m):
                k += err[i]
                ub = (beta_dist.ppf(0.9, k + 1, i + 1 - k)
                      if k < i + 1 else 1.0)
                if ub <= alpha:
                    best_n = i + 1
            assert nsel <= max(best_n, 1) or nsel == 0 or best_n == 0


def test_sgr_refuses_infeasible():
    th, nsel = sgr_threshold([0.5] * 30, [1] * 30, alpha=0.1, delta=0.1)
    assert th is None and nsel == 0


def test_sgr_reference_implementations_are_equivalent():
    """Canonical deployment and analysis copies agree exactly."""
    rng = np.random.default_rng(17)
    for m in [1, 2, 7, 31, 90]:
        for _ in range(20):
            scores = rng.uniform(0, 1, m).tolist()
            errors = rng.integers(0, 2, m).tolist()
            for alpha in [0.1, 0.3, 0.5, 0.7]:
                assert sgr_threshold(scores, errors, alpha, 0.1) == fp.sgr_threshold(
                    scores, errors, alpha, 0.1)


def test_sgr_handles_ties_and_nonmonotone_empirical_risk():
    scores = [0.1, 0.1, 0.2, 0.3, 0.3, 0.4]
    errors = [0, 1, 0, 1, 0, 1]
    for alpha in [0.1, 0.4, 0.8, 0.99]:
        threshold, n_selected = sgr_threshold(scores, errors, alpha, 0.1)
        assert 0 <= n_selected <= len(scores)
        assert (threshold is None) == (n_selected == 0)
        if threshold is not None:
            # The reported selection must equal the deployed <= threshold
            # subset, including every tied score, and its queried corrected
            # Clopper--Pearson bound must actually pass.
            selected_errors = [e for s, e in zip(scores, errors)
                               if s <= threshold]
            assert n_selected == len(selected_errors)
            steps = max(1, math.ceil(math.log2(len(scores))))
            assert fp.cp_upper(sum(selected_errors), n_selected,
                               1 - 0.1 / steps) <= alpha


def test_rq4_no_violations(fz):
    assert "rq4" not in fz
    assert "score-mismatched" in fz["certificate_status"]
    fresh = MEAS / "corpus_confirmation.json"
    if fresh.exists():
        result = json.loads(fresh.read_text())
        for cell in result["dedicated_calibration_sgr"]:
            if cell["fired"] and cell["n_test_selected"] > 0:
                assert cell["test_risk"] <= cell["alpha"], cell


def test_readme_backbone_count():
    readme = (pathlib.Path(__file__).resolve().parents[1]
              / "README.md").read_text()
    assert "five frozen backbones" not in readme
    assert "nine" in readme.lower()


def test_manuscript_terminology_is_purged():
    banned = ["Snell envelope", "value-to-go", "no human labels",
              "label-free"]
    for tex in (MANU / "sections").glob("*.tex"):
        body = tex.read_text()
        for term in banned:
            assert term not in body, (tex.name, term)


def test_portal_title_claims_and_core_numbers_match_manuscript():
    import re
    main = (MANU / "main.tex").read_text()
    portal_path = PROJECT / "WSDM-submission.md"
    if not portal_path.exists():
        pytest.skip("portal file is intentionally excluded from anonymous artifact")
    portal = portal_path.read_text()
    pdf_title = re.search(r"pdftitle=\{([^}]+)\}", main).group(1)
    portal_title = re.search(r"## Title\s+([^\n]+)", portal).group(1).strip()
    assert portal_title == pdf_title
    banned = ["yielding a conservative rule", "valid as implemented",
              "two backbones improve significantly", "27–63% fewer"]
    for phrase in banned:
        assert phrase not in portal
    compiled_sources = main + "".join(
        p.read_text() for p in (MANU / "sections").glob("*.tex"))
    for number in ["24.75", "24.50", "9.00", "6.75", "3.25", "0.54",
                   "0.32", "23.00", "21.25", "5.90"]:
        assert number in compiled_sources, number
    # The portal should carry the manuscript's final problem, evidence, and
    # deployment standard rather than a superseded revision summary.
    for phrase in ["Pareto-dominates guarded FBG in 15",
                   "18-cell matched-cost family",
                   "available simultaneous analysis",
                   "independent population under simultaneous inference"]:
        assert phrase in portal, phrase


def test_corpus_table_and_rq6_numbers_match_analysis_when_present():
    path = MEAS / "corpus_confirmation.json"
    if not path.exists():
        return
    result = json.loads(path.read_text())
    table = (MANU / "tables" / "corpus_confirmation_rows.tex").read_text()
    rq6 = (MANU / "sections" / "06-results-rq6.tex").read_text()
    for method in ["full_budget", "fixed_2", "answer_repeat",
                   "value_only", "guarded_value"]:
        row = result["methods"][method]
        assert f"{row['em']:.1f}" in table
        assert f"{row['llm_calls']:.2f}" in table
    primary = result["predeclared_primary"]
    assert f"{primary['mean_em_points']:+.2f}" in rq6
    assert f"lower bound of ${primary['lower95_one_sided']:.2f}$" in rq6
    assert "passes the declared 3-EM" in rq6
    assert all(not cell["fired"]
               for cell in result["dedicated_calibration_sgr"])


def test_fresh_run_configs_are_anonymous_when_present():
    paths = list(MEAS.glob("run_corpus_guarded_qwen3b*.json"))
    paths += list(MEAS.glob("run_wiki25m_guarded_qwen3b*.json"))
    for path in paths:
        if ".partial." in path.name:
            continue
        body = path.read_text()
        assert "/" + "Users" + "/" not in body
        assert "/private/" + "tmp/" not in body
        config = json.loads(body)["config"]
        assert not str(config.get("parquet", "")).startswith("/")


def test_corpus_index_is_cross_document_and_ranked():
    index = CorpusBM25([
        "Alpha: Paris is the capital of France.",
        "Beta: Berlin is the capital of Germany.",
        "Gamma: France won the match.",
    ])
    hits = index.search("France capital", 2)
    assert len(hits) == 2
    assert hits[0].startswith("Alpha:")
    assert len(set(hits)) == len(hits)


def test_sqlite_corpus_is_read_only_ranked_and_title_prefixed(tmp_path):
    import sqlite3
    path = tmp_path / "wiki.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE sentences(page_id TEXT, line_number INTEGER, text TEXT)")
    db.execute("CREATE VIRTUAL TABLE sentences_fts USING fts5(page_id UNINDEXED, "
               "line_number UNINDEXED, text, content='sentences', content_rowid='rowid')")
    rows = [("City_of_Paris", 0, "Paris is the capital of France."),
            ("Berlin", 0, "Berlin is the capital of Germany."),
            ("Football", 0, "France won the football match.")]
    db.executemany("INSERT INTO sentences VALUES (?, ?, ?)", rows)
    db.execute("INSERT INTO sentences_fts(sentences_fts) VALUES ('rebuild')")
    db.commit(); db.close()
    index = SQLiteCorpusBM25(path)
    hits = index.search("France capital", 2)
    assert index.sentence_count == 3
    assert hits[0].startswith("City of Paris:")


def test_streamed_file_hash(tmp_path):
    import hashlib
    path = tmp_path / "input.bin"
    payload = b"full-corpus-provenance" * 1024
    path.write_bytes(payload)
    assert file_sha256(path, chunk_bytes=17) == hashlib.sha256(payload).hexdigest()


def test_tasr_answer_token_margin_with_and_without_literal_marker():
    def lp(token, first, second):
        return {"token": token, "logprob": first,
                "top_logprobs": [{"token": token, "logprob": first},
                                  {"token": "x", "logprob": second}]}
    marked = [lp("Answer", -0.1, -2), lp(":", -0.1, -2),
              lp(" Paris", -0.2, -1.7)]
    bare = [lp("Paris", -0.3, -1.4)]
    assert first_answer_margin("Answer: Paris", marked) == pytest.approx(1.5)
    assert first_answer_margin("Paris", bare) == pytest.approx(1.1)


def test_completed_tasr_replay_is_aligned_when_present():
    for path in MEAS.glob("tasr_*.json"):
        result = json.loads(path.read_text())
        run = json.loads((MEAS / f"run_{result['tag']}.json").read_text())
        expected_ids = {str(row["id"]) for row in run["per_q"]}
        assert {row["id"] for row in result["per_q"]} == expected_ids
        assert result["isotonic_calibration_episodes"] == 90
        assert result["excluded_certificate_episodes"] == 30
        for row in result["per_q"]:
            assert row["deployment_cost"]["calls"] == 1 + 2 * row["stop"]
            assert 0 <= row["stop"] <= run["config"]["max_steps"]
            assert row["em"] in [0, 1]
            assert 0 <= row["f1"] <= 1


def test_fever_wikipedia_line_parser_deduplicates_line_numbers():
    lines = "0\tParis is in France.\tFrance\n0\tduplicate\n1\tCapital city."
    parsed = list(parse_lines("Paris", lines))
    assert [(row[1], row[2]) for row in parsed] == [
        (0, "Paris is in France."), (1, "Capital city.")]


def test_fresh_split_threshold_and_cost_provenance_when_present():
    run_paths = list(MEAS.glob("run_corpus_guarded_qwen3b*.json"))
    run_paths += list(MEAS.glob("run_wiki25m_guarded_qwen3b*.json"))
    for run_path in run_paths:
        if ".partial." in run_path.name:
            continue
        tag = run_path.stem.removeprefix("run_")
        cache_path = MEAS / f"critic_data_{tag}.pkl"
        assert cache_path.exists(), tag
        run = json.loads(run_path.read_text())
        cache = pickle.loads(cache_path.read_bytes())
        train_ids = {r["id"] for r in cache["traces"]}
        test_ids = {r["id"] for r in run["per_q"]}
        assert train_ids.isdisjoint(test_ids)
        assert len(train_ids) == run["config"]["n_train"]
        assert len(test_ids) == run["config"]["n_test"]
        assert run["config"]["primary_rule"] == "guarded"
        assert run["lam_main"] in {
            row["lambda"] for row in run["selection"]["frontier"]}
        assert run["input_sha256"] == (
            "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6")
        assert len(run["model_metadata"].get("digest", "")) == 64
        if tag.startswith("wiki25m"):
            assert run["retrieval_corpus"].get("sha256") == (
                "e903b6bd72337f236ae6667642597565f5f5b3b95a0748275b4459eef2671149")
            assert run["retrieval_corpus"]["sentence_count"] > 25_000_000
            assert "FEVER" in run["retrieval_corpus"].get("source_label", "")
        else:
            assert run["retrieval_corpus"]["sentence_count"] == 269_558
            assert run["retrieval_corpus"]["question_context_count"] == 7_405
        for record in run["per_q"]:
            assert {"id", "question", "gold_answers", "support_titles",
                    "support_sentences", "trajectory"} <= set(record)
            for t, prefix in enumerate(record["trajectory"]):
                calls = sum(p.get(part, {}).get("calls", 0)
                            for p in record["trajectory"][:t + 1]
                            for part in ["query_usage", "answer_usage"])
                assert calls == 1 + 2 * prefix["searches"]
                assert {"answer", "evidence", "answer_usage",
                        "retrieval_latency_seconds"} <= set(prefix)
                if t:
                    assert {"query", "query_usage"} <= set(prefix)


def test_meta_review_tables_match_json_when_present():
    """Decision matrix, hierarchical, and nested rows regenerate from JSON."""
    path = MEAS / "meta_review_analyses.json"
    if not path.exists():
        pytest.skip("meta_review_analyses.json not generated")
    result = json.loads(path.read_text())
    matrix = (MANU / "tables" / "decision_matrix_rows.tex").read_text()
    cells = result["decision_matrix"]["cells"]
    display_verdict = {"W": "I", "T": "F", "L": "D"}
    assert matrix.count("\\\\") == len(cells) == 26
    for c in cells:
        for pol in ["value", "guarded"]:
            lo, hi = c[f"{pol}_ci95_vs_full"]
            assert lo <= c[f"{pol}_delta_vs_full"] <= hi
            assert (f"{c[f'{pol}_em']:.1f}/{c[f'{pol}_calls']:.2f} & "
                    f"{c[f'{pol}_delta_vs_full']:+.1f}\\,[{lo:.1f},{hi:.1f}] & "
                    f"{display_verdict[c[f'{pol}_verdict']]}") in matrix
        assert c["value_verdict"] in {"W", "T", "L"}
        assert c["guarded_verdict"] in {"W", "T", "L"}
        if c["setting"] == "fresh":
            assert c["certificate"] == "refuse"
    tally = result["decision_matrix"]["tally"]
    for pol in ["value", "guarded"]:
        assert sum(tally[pol].values()) == len(cells)
    hier = (MANU / "tables" / "simple_policy_paired_rows.tex").read_text()
    contrasts = result["hierarchical_variance"]["contrasts"]
    for key in ["guarded_minus_full", "guarded_minus_fixed3",
                "guarded_minus_fixed2", "value_minus_fixed2"]:
        c = contrasts[key]
        assert f"{c['pooled_four_run_mean_em_points']:+.2f}" in hier
        lo, hi = c["pooled_question_clustered_ci95"]
        assert f"[{lo:.2f},{hi:.2f}]" in hier
        assert lo <= c["pooled_four_run_mean_em_points"] <= hi
    nested = (MANU / "tables" / "nested_uncertainty_rows.tex").read_text()
    for tag, res in result["nested_refit_select"].items():
        assert res["logged_guarded_replay_match_rate"] == 1.0, tag
        assert sum(res["lambda_frequency"].values()) == res["n_boot"]
        g = res["guarded_minus_full_em"]
        assert f"{g['mean']:+.2f}\\,[{g['p2.5']:.2f},{g['p97.5']:.2f}]" in nested
        assert 0.0 <= res["p_guarded_dominated_by_simple"] <= 1.0


def test_meta_review_numbers_quoted_in_manuscript_when_present():
    path = MEAS / "meta_review_analyses.json"
    if not path.exists():
        pytest.skip("meta_review_analyses.json not generated")
    result = json.loads(path.read_text())
    body = "".join(p.read_text() for p in (MANU / "sections").glob("*.tex"))
    abstract = (MANU / "main.tex").read_text()
    tally = result["decision_matrix"]["tally"]
    assert f"FBG-only is dominated in {tally['value']['L']} cells" in body
    assert (f"dominated in {tally['guarded']['L']} of 26 cells, "
            f"is frontier-only in {tally['guarded']['T']}") in body
    hv = result["hierarchical_variance"]["contrasts"]["guarded_minus_full"]
    icc = hv["sample_A_variance_components"]["icc_question"]
    assert "do not estimate a stable variance decomposition" in body
    assert f"{round(100 * icc):.0f}\\%" not in body
    g3 = result["hierarchical_variance"]["contrasts"]["guarded_minus_fixed3"]
    assert f"{g3['pooled_all_run_mean_em_points']:+.2f}" in body
    gf = result["hierarchical_variance"]["contrasts"]["guarded_minus_full"]
    assert f"{gf['pooled_all_run_mean_em_points']:+.2f}" in body
    lo, hi = gf["pooled_question_clustered_ci95"]
    assert f"[{lo:.2f},{hi:.2f}]" in body.replace("$", "")
    passes = result["hierarchical_variance"]["independent_sample_margin_passes"]
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    assert passes["of"] == 5
    assert f"{words[passes['passed']]} of five" in body


def test_cross_dataset_draws_provenance_when_present():
    """2Wiki / MuSiQue corpus-index draws use the documented inputs and split."""
    expected = {
        "corpus2wiki": ("555cab3b1cdded5a853a451c3123f9633d61dbb1d2e46f5ddc50c04f11819650",
                        "2WikiMultihopQA", 206_743, 12_576),
        "corpusmusique": ("15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
                          "MuSiQue", 21_100, 2_417),
    }
    seen = 0
    for prefix, (sha, kind, n_units, n_questions) in expected.items():
        for run_path in MEAS.glob(f"run_{prefix}_guarded_qwen3b_data*.json"):
            if ".partial." in run_path.name:
                continue
            seen += 1
            run = json.loads(run_path.read_text())
            assert run["input_sha256"] == sha, run_path.name
            assert run["retrieval_corpus"]["kind"].startswith(kind), run_path.name
            assert run["retrieval_corpus"]["sentence_count"] == n_units
            assert run["retrieval_corpus"]["question_context_count"] == n_questions
            cfg = run["config"]
            assert cfg["primary_rule"] == "guarded"
            assert cfg["n_train"] == 120 and cfg["n_test"] == 400
            assert cfg["certificate_calibration_fraction"] == 0.25
            assert cfg["llm_seed"] == 7 and cfg["llm_temperature"] == 0.0
            assert run["selection"]["n_policy_training_episodes"] == 90
            assert run["selection"]["n_certificate_calibration_episodes"] == 30
            assert len(run["model_metadata"].get("digest", "")) == 64
            cache = pickle.loads((MEAS / f"critic_data_{run_path.stem.removeprefix('run_')}.pkl").read_bytes())
            assert {r["id"] for r in cache["traces"]}.isdisjoint(
                {r["id"] for r in run["per_q"]})
    if not seen:
        pytest.skip("no cross-dataset draws present")


def test_round9_tables_match_json_when_present():
    """Pooled-frontier, target, sensitivity, and ablation rows regenerate."""
    path = MEAS / "round9_analyses.json"
    if not path.exists():
        pytest.skip("round9_analyses.json not generated")
    result = json.loads(path.read_text())
    cells = result["fresh_cells"]
    assert len(cells) == 14
    pooled = (MANU / "tables" / "round9_pooled_rows.tex").read_text()
    assert pooled.count("\\\\") == len(cells)
    for c in cells:
        pp = c["pooled_pareto"]
        # the frontier is internally consistent: a dominated learned policy is
        # never listed as nondominated, and every dominator really dominates.
        for pol in ["guarded_value", "value_only"]:
            dom = pp["dominators"][pol]
            assert (pol in pp["nondominated"]) == (not dom), (c["tag"], pol)
            em, calls = pp["points"][pol]["em"], pp["points"][pol]["calls"]
            for other in dom:
                o = pp["points"][other]
                assert o["em"] >= em - 1e-9 and o["calls"] <= calls + 1e-9
        mg = pp["margin_sensitivity"]["guarded_value"]
        lo, hi = mg["ci95"]
        assert lo <= mg["mean_em_points"] <= hi
        # a looser margin can never be harder to pass
        assert (mg["pass_at_margin"]["3.0"] >= mg["pass_at_margin"]["2.0"]
                >= mg["pass_at_margin"]["1.0"])
        assert f"{mg['lower95_one_sided']:+.2f}" in pooled
        # epsilon-dominance is monotone in epsilon for the "loss" verdict
        eps = c["epsilon_dominance"]
        losses = [eps[str(e)]["guarded_value"]["verdict"] == "L"
                  for e in [0.0, 0.5, 1.0, 2.0]]
        assert losses == sorted(losses), c["tag"]
        # equal-cost mixtures are convex combinations of hull policies
        for coord in ["calls", "input_tokens", "wall_seconds"]:
            entry = c["equal_cost"][coord]["guarded_value"]
            assert 0.0 - 1e-9 <= entry["mixture"]["weight_first"] <= 1.0 + 1e-9
            d = entry["delta_em_vs_mixture_at_equal_cost"]
            assert d["ci95"][0] <= d["mean_em_points"] <= d["ci95"][1]
    sens = (MANU / "tables" / "round9_sensitivity_rows.tex").read_text()
    assert sens.count("\\\\") == len(cells)
    for c in cells:
        summary = c["selection_sensitivity_summary"]
        assert summary["n_configs"] == len(c["selection_sensitivity"])
        assert summary["guarded_em_range"][0] <= summary["guarded_em_range"][1]
    targets = (MANU / "tables" / "round9_target_rows.tex").read_text()
    assert targets.count("\\\\") == len(cells)
    for c in cells:
        # the refit FBG pipeline must reproduce the logged policy whenever it
        # reselects the logged threshold
        tv = c["target_variants"]["fbg_refit"]
        if tv["reproduces_logged_lambda"]:
            assert tv["guarded_stop_agreement_with_logged"] == 1.0, c["tag"]
        assert c["target_variants"]["fbg_target_recomputation_max_abs_dev"] < 1e-9
    tallies = result["epsilon_tallies"]
    for eps in ["0.0", "0.5", "1.0", "2.0"]:
        for pol in ["value_only", "guarded_value"]:
            assert sum(tallies[eps][pol].values()) == tallies[eps]["n_cells"] == 26
    # epsilon = 0 must agree with the executive matrix tally
    meta = json.loads((MEAS / "meta_review_analyses.json").read_text())
    assert tallies["0.0"] == {
        "value_only": meta["decision_matrix"]["tally"]["value"],
        "guarded_value": meta["decision_matrix"]["tally"]["guarded"],
        "n_cells": 26}


def test_round9_numbers_quoted_in_manuscript_when_present():
    path = MEAS / "round9_analyses.json"
    if not path.exists():
        pytest.skip("round9_analyses.json not generated")
    result = json.loads(path.read_text())
    s = result["summary"]
    body = "".join(p.read_text() for p in (MANU / "sections").glob("*.tex"))
    abstract = (MANU / "main.tex").read_text()
    supplement = (MANU / "supplement.tex").read_text()
    flat = (body + abstract + supplement).replace("$", "").replace("~", " ")
    flat = " ".join(flat.split())

    # The generated supplement table preserves the exact 14-cell frontier
    # counts even when the main-text discussion compresses the comparison.
    legacy = (MANU / "tables" / "whowins_legacy_rows.tex").read_text()
    assert f"FBG-only & {s['cells_value_nondominated']} of 14" in legacy
    assert f"FBG + repeat guard & {s['cells_guarded_nondominated']} of 14" in legacy
    assert f"Fixed budgets & {s['cells_fixed2_nondominated']} of 14" in legacy
    assert (f"Stop classifier & {s['cells_stop_classifier_nondominated']} "
            f"of 14") in legacy
    assert f"Full TASR & {s['cells_tasr_nondominated']} of {s['cells_with_tasr']}" in legacy

    # How often an FBG-family rule is cheapest within the 1-EM band remains
    # tied to the registry, including the exploratory semantic guard.
    winners = s["band_winner_counts"]["1.0"]
    fbg_family = {"value_only", "guarded_value", "guarded", "semantic_guard"}
    n_fbg = sum(v for k, v in winners.items() if k in fbg_family)
    assert f"only {n_fbg} of 14 cells" in flat

    # Equal-cost, sensitivity, and power claims remain present without forcing
    # the polished prose to reproduce every table caption verbatim.
    g = s["equal_calls_guarded_value"]
    assert f"{g['mean_delta_em']:+.2f}" in flat
    assert g["n_ci_above_zero"] == 0
    eps_rows = (MANU / "tables" / "round9_epsilon_rows.tex").read_text()
    for eps in (0.0, 0.5, 1.0, 2.0):
        t = result["epsilon_tallies"][str(eps)]
        expected = (f"{eps:g} & {t['value_only']['W']}/{t['value_only']['T']}/"
                    f"{t['value_only']['L']} & {t['guarded_value']['W']}/"
                    f"{t['guarded_value']['T']}/{t['guarded_value']['L']}")
        assert expected in eps_rows
    sen = s["sensitivity"]
    assert f"{sen['total_configs']}" in flat
    assert (f"Across {sen['total_configs']} protocol variants, simple policies "
            f"dominate {sen['total_guarded_dominated']}") in flat
    mp = s["margin_power_primary"]
    assert f"{mp['margin_with_80pct_power_em']:.2f}" in flat

    for banned in ["appears in every fresh run", "matched on every local fresh cell",
                   "cells where logits exist", "cells with logits"]:
        assert banned not in flat, banned

    # Descriptive D/F/I counts and the primary-cell interpretation remain
    # coupled to the 26-cell decision matrix.
    meta = json.loads((MEAS / "meta_review_analyses.json").read_text())
    tally = meta["decision_matrix"]["tally"]
    assert (f"dominated in {tally['guarded']['L']} of 26 cells, is frontier-only "
            f"in {tally['guarded']['T']}, and improves B^\\star in "
            f"{tally['guarded']['W']}") in flat
    primary = next(c for c in meta["decision_matrix"]["cells"]
                   if c["cell"] == "corpus_guarded_qwen3b")
    if primary["guarded_verdict"] != "W":
        assert "primary HotpotQA cell" in flat
    if primary["value_verdict"] == "W":
        assert "FBG-only is cheaper" in flat


def test_stoprag_fresh_replay_is_aligned_when_present():
    """Stop-RAG results on fresh cells cover the run's own test questions."""
    seen = 0
    for path in MEAS.glob("stoprag_result_corpus*.json"):
        result = json.loads(path.read_text())
        if "test_per_q" not in result:
            continue
        tag = path.stem.removeprefix("stoprag_result_")
        run_path = MEAS / f"run_{tag}.json"
        if not run_path.exists():
            continue
        seen += 1
        run = json.loads(run_path.read_text())
        assert result["n_test"] == len(run["per_q"])
        assert {r["id"] for r in result["test_per_q"]} == {
            str(r["id"]) for r in run["per_q"]}
        assert result["n_train_episodes"] + result["n_val_episodes"] == 90
        max_steps = run["config"]["max_steps"]
        by_id = {str(r["id"]): r for r in run["per_q"]}
        for row in result["test_per_q"]:
            assert 0 <= row["stop"] <= max_steps
            # replayed EM must match the logged prefix EM at that stop
            assert row["em"] == by_id[row["id"]]["prefix_em"][row["stop"]]
            assert row["searches"] == row["stop"]
    if not seen:
        pytest.skip("no fresh Stop-RAG replays present")


def test_support_audit_is_consistent_when_present():
    """Re-rolled continuations line up with the run they audit."""
    for path in MEAS.glob("support_audit_*.json"):
        if ".partial." in path.name:
            continue
        result = json.loads(path.read_text())
        run = json.loads((MEAS / f"run_{result['tag']}.json").read_text())
        records = {str(r["id"]): r for r in run["per_q"]}
        assert result["n_test_questions"] == len(run["per_q"])
        assert result["n_sampled"] <= result["n_eligible_stops"]
        max_steps = run["config"]["max_steps"]
        for row in result["per_stop"]:
            r = records[row["id"]]
            # the audited stop is the deployed guarded stop of that question
            assert row["stop"] == int(r["primary"]["searches"]) < max_steps
            assert row["u_t"] == r["prefix_f1"][row["stop"]]
            # the logged FBG target is recomputed from the logged prefixes
            future = r["prefix_f1"][row["stop"] + 1:]
            assert abs(row["logged_fbg"] - max(max(future) - row["u_t"], 0.0)) < 1e-9
            assert len(row["rollouts"]) == result["rollouts_per_stop"]
            for alt in row["rollouts"]:
                assert len(alt) == max_steps - row["stop"]
                for prefix in alt:
                    assert 0.0 <= prefix["f1"] <= 1.0 and prefix["em"] in (0, 1)
            assert row["alt_best_f1"] >= row["u_t"] - 1e-9
        o = result["overall"]
        assert 0.0 <= o["missed_gain_rate"] <= 1.0
        assert o["n"] == len(result["per_stop"]) == result["n_sampled"]
        body = "".join(p.read_text() for p in (MANU / "sections").glob("*.tex"))
        quoted = " ".join(body.replace("\\%", "%").split())
        assert f"Beat logged best in {100 * o['missed_gain_rate']:.1f}%" in quoted
        assert f"changed correctness in {100 * o['em_flip_rate']:.1f}%" in quoted
        assert "reachable gains outside the logged path" in quoted


def test_halt_matched_condition_is_consistent_when_present():
    """Matched generated-claim HALT lines up with the run and the manuscript."""
    seen = 0
    for path in MEAS.glob("halt_*.json"):
        if "claims" in path.name or "scores" in path.name:
            continue
        result = json.loads(path.read_text())
        tag = result["tag"]
        run = json.loads((MEAS / f"run_{tag}.json").read_text())
        seen += 1
        assert result["n_test"] == len(run["per_q"])
        selection = run["selection"]
        assert result["n_policy_training_episodes"] == selection["n_policy_training_episodes"]
        assert (selection["n_policy_training_episodes"]
                + selection["n_certificate_calibration_episodes"]
                == run["config"]["n_train"])
        assert result["claim_generation_calls_per_question"] == 1
        by_id = {str(r["id"]): r for r in run["per_q"]}
        max_steps = run["config"]["max_steps"]
        for name, variant in result["variants"].items():
            assert variant["tau"] in result["tau_grid"]
            assert len(variant["per_q"]) == result["n_test"]
            for row in variant["per_q"]:
                rec = by_id[row["id"]]
                assert 0 <= row["stop"] <= max_steps
                # replayed EM is the logged prefix EM at the coverage stop
                assert row["em"] == rec["prefix_em"][row["stop"]]
                cost = row["deployment_cost"]
                assert cost["calls_without_claim_generation"] == 1 + 2 * row["stop"]
                assert cost["calls"] == cost["calls_without_claim_generation"] + 1
                # the stop is the first prefix whose coverage clears tau
                below = [c for c in row["coverage"][:row["stop"]]]
                assert all(c < variant["tau"] for c in below), (tag, name, row["id"])
    if not seen:
        pytest.skip("no matched HALT runs present")
    # the manuscript's HALT claims must match the registry
    r9 = MEAS / "round9_analyses.json"
    if r9.exists():
        r9_data = json.loads(r9.read_text())
        s = r9_data["summary"]
        body = "".join(p.read_text() for p in (MANU / "sections").glob("*.tex"))
        flat = " ".join((body + (MANU / "main.tex").read_text()
                         + (MANU / "supplement.tex").read_text()).split())
        words = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                 6: "six", 7: "seven"}
        expected = sum("halt_nli" in c["pooled_pareto"]["candidates"]
                       for c in r9_data["fresh_cells"])
        assert s["cells_halt_present"] == expected
        assert f"all {words[expected]}" in flat, expected
        # "dominated in all six cells" may only be claimed if that is true
        if s["cells_halt_nondominated"] == 0:
            assert "dominated in \\emph{all six} cells" in flat or \
                "dominated in all six cells" in flat
        else:
            assert "dominated in all six cells" not in flat
        assert f"by fixed-2 in {words[s['cells_halt_dominated_by_fixed2']]}" in flat
        # the run uses public verifiers in place of HALT's CUDA-only trained
        # one, so the manuscript must present it as a substitute, never as a
        # reproduction of HALT
        assert "verifier-inspired" in flat
        assert "substitute" in flat
        assert "does not evaluate HALT's trained verifier" in flat


def test_when_learned_wins_is_consistent_when_present():
    """The prospective-prediction analysis matches its quoted numbers."""
    path = MEAS / "when_learned_wins.json"
    if not path.exists():
        pytest.skip("when_learned_wins.json not generated")
    result = json.loads(path.read_text())
    assert result["n_cells"] == 26
    # predictors are training-chain properties only; outcomes come from the
    # frozen decision matrix, so the two must agree cell by cell
    meta = json.loads((MEAS / "meta_review_analyses.json").read_text())
    verdicts = {c["cell"]: c for c in meta["decision_matrix"]["cells"]}
    for row in result["cells"]:
        v = verdicts[row["tag"]]
        assert row["guarded_not_dominated"] == int(v["guarded_verdict"] != "L")
        assert row["value_not_dominated"] == int(v["value_verdict"] != "L")
        assert 0.0 <= row["early_correct_rate"] <= 1.0
        assert 0.0 <= row["threshold_stability"] <= 1.0
        assert sum(row["threshold_frequency"].values()) == \
            result["n_bootstrap_refits_per_cell"]
    guarded = result["leave_one_cell_out"]["guarded_not_dominated"]
    rule = guarded["predeclared_rule"]
    assert rule["tp"] + rule["fp"] + rule["fn"] + rule["tn"] == 26
    body = "".join(p.read_text() for p in (MANU / "sections").glob("*.tex"))
    supplement = (MANU / "supplement.tex").read_text()
    flat = " ".join((body + (MANU / "main.tex").read_text() + supplement).split())
    # The moderator was moved out of the main claims and explicitly labeled
    # retrospective. Its accuracy, base rate, and p-value remain auditable.
    assert f"{100 * rule['accuracy']:.1f}\\%" in flat
    assert f"{100 * rule['base_rate_accuracy']:.1f}\\% majority-class base rate" in flat
    assert f"p{{=}}{rule['permutation_p']:.3f}" in flat
    assert "retrospective exploratory analysis" in flat
    assert "No selection among these variants was used to choose" in flat
    # the two null outcomes must be reported as nulls, not as wins
    for outcome in ["value_not_dominated", "any_fbg_not_dominated"]:
        p_null = result["leave_one_cell_out"][outcome]["predeclared_rule"]["permutation_p"]
        assert p_null > 0.05, outcome
        assert f"p{{=}}{p_null:.2f}" in flat, outcome
    # the deployable-frontier attempt quoted in the body: it must be reported
    # with its base rate, so a failure cannot read as a success
    fresh = result.get("fresh_cell_outcomes", {})
    if "fbg_on_pooled_frontier" in fresh and "predeclared_rule" in fresh["fbg_on_pooled_frontier"]:
        front = fresh["fbg_on_pooled_frontier"]["predeclared_rule"]
        assert (f"{100 * front['accuracy']:.1f}\\% against an "
                f"{100 * front['base_rate_accuracy']:.1f}\\% base rate") in flat
        assert front["accuracy"] < front["base_rate_accuracy"], (
            "the frontier rule now beats its base rate; the manuscript calls it a failure")
    # the sensitivity variant quoted in the body
    sens = result["rule_sensitivity"]["guarded_not_dominated"]
    alt = sens["early + threshold stability"]
    assert f"{100 * alt['accuracy']:.1f}\\%" in flat
    solo = sens["repeat precision alone"]
    assert f"{100 * solo['accuracy']:.1f}\\%" in flat


def test_answer_once_ledger_matches_manuscript_when_present():
    """The second cost currency (answer-once) is quoted exactly as measured.

    Guards the claim added in the 2026-08-22 revision: repricing the fresh runs
    so that only policies needing no per-prefix answer pay 1+T sharpens rather
    than softens the negative result.  The ledger must also stay internally
    honest: a policy that reads an answer at every prefix may never be given a
    cheaper answer-once price than its measured 1+2T cost.
    """
    path = MEAS / "answer_once_ledger.json"
    if not path.exists():
        pytest.skip("answer_once_ledger.json not generated")
    result = json.loads(path.read_text())
    body = "".join(p.read_text() for p in (MANU / "sections").glob("*.tex"))
    supplement = (MANU / "supplement.tex").read_text()
    flat = " ".join((body + supplement).replace("$", "").replace("~", " ").split())

    once = result["summary"]["answer_once"]
    primary = result["summary"]["primary"]
    n = once["n_cells"]

    # a policy that needs an answer at every prefix cannot answer once
    for cell in result["cells"]:
        for name, row in cell["policies"].items():
            if name in result["needs_per_prefix_answer"]:
                assert row["answer_once_calls"] == row["primary_calls"], (cell["cell"], name)
                assert not row["answer_once_applies"], (cell["cell"], name)
            else:
                assert row["answer_once_calls"] == pytest.approx(1.0 + row["searches"]), \
                    (cell["cell"], name)
                assert row["answer_once_calls"] <= row["primary_calls"] + 1e-9

    # the counts quoted in the body clause and the supplement section
    assert f"dominated in {once['guarded_dominated_by_simple_family']} of {n}" in flat
    assert f"against {primary['guarded_dominated_by_simple_family']} of {n}" in flat
    assert f"costs more than every fixed budget in all {n}" in flat
    assert once["cells_where_guard_cost_exceeds_fixed_budget_range"] == n, (
        "the manuscript says the guard is outside the fixed-budget cost range in "
        "every cell; the ledger no longer agrees")
    assert (f"the entire simple family in "
            f"{once['cells_where_guard_cost_exceeds_simple_family_range']}") in flat

    # the knock-on the revision is required to disclose: under this currency
    # answer repetition is dominated by a plain fixed budget everywhere, so the
    # manuscript may not present repetition as a cheap add-on without scoping
    assert once["answer_repeat_dominated_by_a_fixed_budget"] == n
    assert f"Answer repetition is dominated by a plain fixed budget in all {n}" in flat
    assert (f"against {primary['answer_repeat_dominated_by_a_fixed_budget']} of {n}") in flat

    # the re-pricing must not be sold as an improvement for the learned rule
    assert once["guarded_dominated_by_simple_family"] >= \
        primary["guarded_dominated_by_simple_family"]
    assert "the simple-policy advantage grows" in flat


def test_training_set_size_axis_matches_registry_when_present():
    """The 80/90/96 training-size axis is quoted exactly, including its limits.

    The certificate-split fraction is the only training-set-size axis that can
    be varied without new backbone generation.  The manuscript reports it as a
    bounded null; this pins the tally and forbids the stronger claim.
    """
    path = MEAS / "round9_analyses.json"
    if not path.exists():
        pytest.skip("round9_analyses.json not generated")
    result = json.loads(path.read_text())
    cells = result["fresh_cells"]
    body = "".join(p.read_text() for p in (MANU / "sections").glob("*.tex"))
    supplement = (MANU / "supplement.tex").read_text()
    flat = " ".join((body + supplement).replace("$", "").replace("~", " ").split())

    # config -> training episodes, and the guarded-verdict tally over the cells
    axis = {}
    for cfg in ("cal0.20", "base", "cal0.33"):
        sizes = {c["selection_sensitivity"][cfg]["n_policy_training_episodes"]
                 for c in cells}
        assert len(sizes) == 1, (cfg, sizes)
        n_train = sizes.pop()
        rows = [c["selection_sensitivity"][cfg] for c in cells]
        axis[n_train] = {
            "dominated": sum(1 for r in rows if r["guarded_verdict"] == "L"),
            "mean_em": sum(r["guarded_em"] for r in rows) / len(rows),
            "mean_calls": sum(r["guarded_calls"] for r in rows) / len(rows),
        }
    assert sorted(axis) == [80, 90, 96], sorted(axis)
    n_cells = len(cells)

    assert "leave 96/90/80 policy-training episodes" in flat

    # the supplement's fuller statement, to 2dp
    assert (f"in {axis[96]['dominated']}, {axis[90]['dominated']}, and "
            f"{axis[80]['dominated']} of {n_cells} cells at 96, 90, and 80 episodes") in flat
    def triple(key, fmt="{:.2f}"):
        a, b, c = (fmt.format(axis[k][key]) for k in (96, 90, 80))
        return f"{a}, {b}, and {c}"
    ems = triple("mean_em")
    assert f"mean guarded EM {ems}" in flat, ems
    calls = triple("mean_calls")
    assert f"at {calls} mean calls" in flat, calls

    # honesty guard: the axis only goes down from 90 without new generation, so
    # the manuscript may not claim saturation, only a bounded null
    for banned in ["training data is sufficient", "training size saturates",
                   "enough training episodes", "more training data would not help",
                   "not a data-starvation artifact"]:
        assert banned not in flat, banned
    assert "cannot exclude" in flat
    assert "bounded null" in flat
    assert "do not claim saturation" in flat


def test_training_size_curve_matches_manuscript_when_present():
    """The 30--90 episode subsample curve is quoted exactly as measured.

    The direction was pre-committed: a curve still climbing at n=90 is evidence
    against the paper's reading and must be reported as such.  These assertions
    pin the pooled numbers and force the manuscript to disclose both the trend
    and the fact that the sweep cannot go above the logged 90 episodes.
    """
    path = MEAS / "training_size_curve.json"
    if not path.exists():
        pytest.skip("training_size_curve.json not generated")
    result = json.loads(path.read_text())
    supplement = (MANU / "supplement.tex").read_text()
    body = "".join(p.read_text() for p in (MANU / "sections").glob("*.tex"))
    flat = " ".join((body + supplement).replace("$", "").replace("~", " ").split())

    sizes = result["sizes"]
    pooled = result["pooled"]
    assert sizes == [30, 45, 60, 75, 90], sizes
    assert result["n_seeds_per_size"] == 20

    # every draw must have actually run: a silently-skipped cell would flatter
    # the curve by dropping whichever cells the subsample breaks
    total = sum(pooled[str(n)]["n_draws_total"] for n in sizes)
    n_failed = sum(c["sizes"][str(n)]["n_failed"]
                   for c in result["cells"].values() for n in sizes)
    assert n_failed == 0, n_failed
    assert f"{total:,}".replace(",", "{,}") + " refits" in flat, total
    for n in sizes:
        assert pooled[str(n)]["n_cells"] == 14
        assert pooled[str(n)]["n_draws_total"] == 280

    lo, hi = pooled["90"]["dominated_rate"], pooled["30"]["dominated_rate"]
    assert f"{100 * hi:.1f}\\% to {100 * lo:.1f}\\%" in flat
    worst = max(pooled[str(n)]["dominated_rate"] for n in sizes)
    worst_n = [n for n in sizes if pooled[str(n)]["dominated_rate"] == worst][0]
    assert f"rises to {100 * worst:.1f}\\% at n{{=}}{worst_n}" in flat
    ems = [pooled[str(n)]["mean_em_over_cells"] for n in sizes]
    assert f"({ems[0]:.2f} to {ems[-1]:.2f})" in flat, ems
    assert f"by {100 * (hi - lo):.1f} points" in flat
    full_range = max(pooled[str(n)]["dominated_rate"] for n in sizes) - \
        min(pooled[str(n)]["dominated_rate"] for n in sizes)
    assert f"a {100 * full_range:.1f}-point full range" in flat

    # the manuscript must not sell a non-monotone curve as monotone
    rates = [pooled[str(n)]["dominated_rate"] for n in sizes]
    monotone = all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))
    assert not monotone, "curve is now monotone; the supplement calls it non-monotonic"
    assert "trend is not monotone" in flat

    # the EM sweep must stay below the loss the design can actually resolve
    assert max(ems) - min(ems) < 1.4
    assert "1.4-EM loss" in flat


def _supp_related_rows():
    """Rows of supplement.tex's positioning table, keyed by leading method."""
    text = (MANU / "supplement.tex").read_text()
    start = text.index("\\label{tab:supp-related}")
    body = text[start:text.index("\\end{tabular}", start)]
    rows = {}
    for line in body.splitlines():
        line = line.strip()
        if "&" not in line or line.startswith("%"):
            continue
        head = line.split("&", 1)[0].strip()
        name = head.split("~\\cite")[0].split("\\cite")[0].strip()
        if name:
            rows.setdefault(name, line)
    return rows


def test_supp_related_table_reports_experiments_that_were_actually_run():
    """The positioning table may not call an executed condition ``not run''.

    ``not run'' is the legend for a declared-missing method and stays defined
    in the caption, but a method whose matched condition IS in the registry
    must report that condition and its measured outcome instead -- otherwise a
    reviewer meeting this table first is told the decision-critical experiment
    was never done.
    """
    r9 = MEAS / "round9_analyses.json"
    if not r9.exists():
        pytest.skip("round9_analyses.json not generated")
    s = json.loads(r9.read_text())["summary"]
    rows = _supp_related_rows()
    words = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven"}

    # the legend itself must survive: it is still needed for genuinely
    # undone methods, so this guard is scoped to rows, never to the file
    caption = (MANU / "supplement.tex").read_text()
    assert "identifies a declared missing condition" in caption

    halt = rows["HALT"]
    if s["cells_halt_present"] > 0:
        assert "not run" not in halt, halt
        assert words[s["cells_halt_present"]] in halt, halt
        assert "substitute" in halt, halt
        assert f"fixed-2 in {words[s['cells_halt_dominated_by_fixed2']]}" in halt
    else:
        assert "not run" in halt

    stoprag = rows["Stop-RAG"]
    extended = ROUND10 / "extended_decision_matrix.json"
    if extended.exists():
        ss = json.loads(extended.read_text())["summary"]["stoprag"]
        assert f"{ss['cells_covered']} common-family cells" in stoprag, stoprag
        assert f"dominated in {ss['dominance']['D']} cells" in stoprag, stoprag
        assert f"frontier-only in {ss['dominance']['F']}" in stoprag, stoprag
    elif s["cells_stoprag_present"] > 0:
        n = s["cells_stoprag_present"]
        assert f"{words[n]} cells" in stoprag, stoprag
        assert f"{s['cells_stoprag_nondominated']} of {n}" in stoprag, stoprag

    # transplanted baselines are priced in searches, not exact LLM calls;
    # per-question call counts were never logged for them, so the caption
    # must say so rather than let the reader read searches as calls
    for name in ("Self-RAG", "FLARE"):
        assert "searches on the baseline-implementation cell" in rows[name], rows[name]
    assert "price transplanted methods in \\emph{searches}" in caption


def test_hindsight_oracle_ceiling_matches_trajectories_and_manuscript():
    """B5: the oracle row is recomputed here from the raw prefix grades.

    Guards three separable failure modes:
    (1) the quoted ceiling drifts from the logged trajectories;
    (2) the manuscript silently reverts to the stored `oracle_searches`
        field, which bills 0 searches to never-correct questions;
    (3) the convention ("full budget when no prefix is correct") disappears
        from the supplement, letting a reviewer read the ceiling as
        achievable.
    """
    path = MEAS / "oracle_frontier.json"
    if not path.exists():
        pytest.skip("oracle_frontier.json not generated")
    stored = json.loads(path.read_text())

    run = MEAS / "run_corpus_guarded_qwen3b.json"
    if not run.exists():
        pytest.skip("primary run registry absent from this bundle")
    records = json.loads(run.read_text())["per_q"]

    sys.path.insert(0, str(CODE))
    from analyze_corpus_confirmation import cumulative_cost  # noqa: E402

    stops, full_stops = [], []
    for r in records:
        prefix_em = r["prefix_em"]
        last = len(prefix_em) - 1
        hit = next((t for t, em in enumerate(prefix_em) if em == 1), last)
        stops.append(hit)
        full_stops.append(last)

    def em_of(idx):
        return round(100 * float(np.mean(
            [r["prefix_em"][t] for r, t in zip(records, idx)])), 2)

    def calls_of(idx):
        return round(float(np.mean(
            [cumulative_cost(r, t)["calls"] for r, t in zip(records, idx)])), 3)

    oracle_em, oracle_calls = em_of(stops), calls_of(stops)
    oracle_searches = round(float(np.mean(stops)), 3)
    full_em, full_calls = em_of(full_stops), calls_of(full_stops)

    cell = stored["cells"]["corpus_guarded_qwen3b"]
    assert cell["oracle"]["em"] == oracle_em
    assert cell["oracle"]["llm_calls"] == oracle_calls
    assert cell["oracle"]["searches"] == oracle_searches
    assert cell["full_budget"]["em"] == full_em
    assert cell["full_budget"]["llm_calls"] == full_calls
    assert cell["headroom_em"] == round(oracle_em - full_em, 2)

    # (2) the buggy stored field must stay rejected: it is far cheaper than
    # the honest convention, so quoting it would understate the ceiling's cost
    buggy = round(float(np.mean([r["oracle_searches"] for r in records])), 4)
    assert buggy == cell["stored_oracle_searches_mean"]
    assert buggy < oracle_searches / 2, (buggy, oracle_searches)

    body = "".join(p.read_text() for p in (MANU / "sections").glob("*.tex"))
    supplement = (MANU / "supplement.tex").read_text()
    flat = " ".join((body + supplement).replace("$", "")
                    .replace("~", " ").split())
    assert (f"{oracle_em:.2f} EM at {oracle_searches:.2f} searches / "
            f"{oracle_calls:.2f} exact calls") in flat
    assert f"{oracle_searches:.2f} searches" in flat

    # (3) the convention must be printed where the number is
    supp_flat = " ".join(supplement.replace("\\emph{", "")
                         .replace("}", "").replace("---", " ").split())
    assert "charges the full budget when no prefix is correct" in supp_flat
    assert "ceiling, not a deployable policy" in supp_flat

    # the row is headroom, never evidence that stopping works
    assert "not evidence that stopping works" in supp_flat

    lo = min(c["headroom_em"] for c in stored["cells"].values())
    hi = max(c["headroom_em"] for c in stored["cells"].values())
    assert f"{lo:+.2f} to {hi:+.2f} EM" in flat
    assert stored["cells_where_oracle_beats_full"] == stored["n_cells"]


def test_strata_audit_matches_logs_and_keeps_the_failing_draws_prominent():
    """D5: the composition null, recomputed from the logged strata fields.

    Guards (1) that the standardised and raw EM swings quoted in the
    supplement are the measured ones, (2) that `level` is still degenerate so
    the stated limitation stays true, and (3) -- the one that matters -- that
    the strata result is never used to demote the two failing independent
    draws, whose lower bounds must stay in the body verbatim.
    """
    path = MEAS / "strata_audit.json"
    if not path.exists():
        pytest.skip("strata_audit.json not generated")
    d = json.loads(path.read_text())
    summary = d["summary"]
    std = summary["direct_standardisation"]

    body = "".join(p.read_text() for p in (MANU / "sections").glob("*.tex"))
    supplement = (MANU / "supplement.tex").read_text()
    flat = " ".join((body + supplement).replace("$", "")
                    .replace("~", " ").replace("\\%", "%").split())

    # (1) the swing numbers must be the measured ones, in both documents
    assert (f"swing at {std['standardised_swing']:.2f} EM against a raw "
            f"{std['raw_swing']:.2f} EM") in flat
    assert f"{std['standardised_swing']:.2f}" in supplement
    assert f"{std['raw_swing']:.2f}" in supplement
    # composition explains a minority of the swing; if that ever flips, the
    # supplement's "not a composition artifact" sentence must be rewritten
    assert std["standardised_swing"] > 0.5 * std["raw_swing"], std

    # recompute the raw swing straight from the run registries
    ems = [r["full_budget_em"] for r in d["draws"].values()]
    assert round(max(ems) - min(ems), 2) == std["raw_swing"]
    assert summary["n_draws"] == 5

    # (2) the degenerate level stratum is a stated limitation, not a null
    assert summary["level_is_degenerate"] is True
    assert summary["level_values"] == ["hard"]
    assert "degenerate" in supplement and "cannot be tested" in " ".join(
        supplement.split())

    # (3) the failing draws keep their prominence: the strata section must
    # disclaim, and the body must still carry both lower bounds and the count
    assert "not evidence that the failures are artifacts" in " ".join(
        supplement.replace("\\emph{", "").replace("}", "").split())
    body_flat = " ".join(body.replace("$", "").replace("~", " ").split())
    assert "lower bounds are −3.25, −5.00" in body_flat or \
           "lower bounds are -3.25, -5.00" in body_flat, body_flat[:0]
    assert "Across all five question samples, three of five pass" in body_flat

    # the answerable subgroup is where the guard's loss lands in the two
    # failing draws; that qualification must not silently disappear
    fail = [d["draws"][t] for t in ("corpus_guarded_qwen3b_data17",
                                    "corpus_guarded_qwen3b_data23")]
    for row in fail:
        r1 = row["strata"]["recall_1"]["guarded_minus_full_em"]
        r0 = row["strata"]["recall_lt_1"]["guarded_minus_full_em"]
        assert r1 < r0, (row["label"], r1, r0)
        assert f"{r1:+.2f}".replace("+", "") in supplement.replace("$", "")
