"""Regression checks for audit rounding, interval provenance, and configuration counts."""
import json, sys
from decimal import Decimal
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from display_numbers import display, percent

@pytest.mark.parametrize('value, expected', [('5.895','5.90'),('25.25','25.25'),('-0.0001','0.00')])
def test_decimal_half_up_and_zero(value, expected):
    assert display(Decimal(value)) == expected

@pytest.mark.parametrize('proportion, expected', [('0.925','93'),('0.505','51'),('0.725','73')])
def test_nested_percent_ties(proportion, expected):
    assert percent(Decimal(proportion)) == expected

def test_full_search_plot_uses_frozen_endpoint_intervals():
    from make_figures import displayed_contrast
    meta = json.loads((ROOT / 'results/measured/frontier_analyses.json').read_text())
    # Historical hierarchical and endpoint bootstraps differ on this run.
    result = displayed_contrast(meta, 'guarded_minus_full', 'corpus_guarded_qwen3b')
    assert result['ci95'] == [-0.75, 1.5]
    assert result['mean_em_points'] == 0.25
    assert result['lower95_one_sided'] == -0.75
    assert meta['hierarchical_variance']['contrasts']['guarded_minus_full']['per_run']['corpus_guarded_qwen3b']['ci95'] == [-0.75, 1.25]

def test_sensitivity_denominators_have_eleven_configurations():
    data = json.loads((ROOT / 'results/measured/round9_analyses.json').read_text())
    cells = data['fresh_cells']
    assert len(cells) == 14
    for cell in cells:
        assert len(cell['selection_sensitivity']) == cell['selection_sensitivity_summary']['n_configs'] == 11
        assert {'base','refit_s3'} <= cell['selection_sensitivity'].keys()
