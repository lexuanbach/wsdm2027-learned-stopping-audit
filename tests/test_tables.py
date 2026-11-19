"""Checks on the generated table rows (run python code/make_all.py first).

Each test compares rows in outputs/tables/ with the JSON they were built from.
Tests are skipped when the outputs have not been generated yet.
"""
import json
import pathlib

import pytest

ART = pathlib.Path(__file__).resolve().parents[1]
MEAS = ART / "results" / "measured"
TABLES = ART / "outputs" / "tables"

if not (TABLES / "decision_matrix_rows.tex").exists():
    pytest.skip("outputs/tables not generated; run python code/make_all.py",
                allow_module_level=True)


@pytest.fixture(scope="module")
def fz():
    return json.loads((MEAS / "frozen_protocol.json").read_text())


def test_tex_rows_match_json(fz):
    rows = (TABLES / "backbones_frozen_rows.tex").read_text()
    assert rows.count("\\\\") == 9
    for c in fz["hotpot_cells"]:
        assert f"{c['voi_frozen']['em']:.1f}" in rows
        assert f"{c['voi_frozen']['save_pct']}\\%" in rows
    trows = (TABLES / "transfer_frozen_rows.tex").read_text()
    assert trows.count("\\\\") == len(fz["transfer"]) == 4


def test_frontier_tables_match_json_when_present():
    """Decision matrix, hierarchical, and nested rows regenerate from JSON."""
    path = MEAS / "frontier_analyses.json"
    if not path.exists():
        pytest.skip("frontier_analyses.json not generated")
    result = json.loads(path.read_text())
    matrix = (TABLES / "decision_matrix_rows.tex").read_text()
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
    hier = (TABLES / "simple_policy_paired_rows.tex").read_text()
    contrasts = result["hierarchical_variance"]["contrasts"]
    for key in ["guarded_minus_full", "guarded_minus_fixed3",
                "guarded_minus_fixed2", "value_minus_fixed2"]:
        c = contrasts[key]
        assert f"{c['pooled_four_run_mean_em_points']:+.2f}" in hier
        lo, hi = c["pooled_question_clustered_ci95"]
        assert f"[{lo:.2f},{hi:.2f}]" in hier
        assert lo <= c["pooled_four_run_mean_em_points"] <= hi
    nested = (TABLES / "nested_uncertainty_rows.tex").read_text()
    for tag, res in result["nested_refit_select"].items():
        assert res["logged_guarded_replay_match_rate"] == 1.0, tag
        assert sum(res["lambda_frequency"].values()) == res["n_boot"]
        g = res["guarded_minus_full_em"]
        assert f"{g['mean']:+.2f}\\,[{g['p2.5']:.2f},{g['p97.5']:.2f}]" in nested
        assert 0.0 <= res["p_guarded_dominated_by_simple"] <= 1.0


def test_round9_tables_match_json_when_present():
    """Pooled-frontier, target, sensitivity, and ablation rows regenerate."""
    path = MEAS / "round9_analyses.json"
    if not path.exists():
        pytest.skip("round9_analyses.json not generated")
    result = json.loads(path.read_text())
    cells = result["fresh_cells"]
    assert len(cells) == 14
    pooled = (TABLES / "round9_pooled_rows.tex").read_text()
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
    sens = (TABLES / "round9_sensitivity_rows.tex").read_text()
    assert sens.count("\\\\") == len(cells)
    for c in cells:
        summary = c["selection_sensitivity_summary"]
        assert summary["n_configs"] == len(c["selection_sensitivity"])
        assert summary["guarded_em_range"][0] <= summary["guarded_em_range"][1]
    targets = (TABLES / "round9_target_rows.tex").read_text()
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
    meta = json.loads((MEAS / "frontier_analyses.json").read_text())
    assert tallies["0.0"] == {
        "value_only": meta["decision_matrix"]["tally"]["value"],
        "guarded_value": meta["decision_matrix"]["tally"]["guarded"],
        "n_cells": 26}
