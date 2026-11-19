"""Unit and data-consistency tests for the stored results and the shipped code.

Run from the repository root:  python -m pytest -q
"""
import json
import math
import pathlib
import pickle
import sys

import numpy as np
import pytest

ART = pathlib.Path(__file__).resolve().parents[1]
CODE = ART / "code"
MEAS = ART / "results" / "measured"
ROUND10 = ART / "results" / "round10"
sys.path.insert(0, str(CODE))

import frozen_protocol as fp                      # noqa: E402
from voi_controller import (guarded_stop_index, stop_index,
                            sgr_threshold)   # noqa: E402
from run_experiment import CorpusBM25, SQLiteCorpusBM25, file_sha256  # noqa: E402
from evaluate_tasr import first_answer_margin  # noqa: E402
from build_wikipedia_index import parse_lines  # noqa: E402

HOTPOT = fp.HOTPOT_TAGS


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


def test_fresh_run_configs_contain_no_local_paths():
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
