"""Render archived audit summaries without inference, refitting, or input mutation."""
from pathlib import Path
import argparse, json, hashlib

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs'))
    p.add_argument('--figures', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    tables = out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    import frontier_analyses as meta
    import round9_analyses as round9
    import when_learned_wins as when
    import build_signal_summary_table as signal
    import build_whowins_table as wins
    import training_size_curve as training
    meta.TABLES = round9.TABLES = when.TABLES = tables
    signal.MANU = wins.MANU = out
    inputs = ['frontier_analyses.json', 'round9_analyses.json',
              'when_learned_wins.json', 'training_size_curve.json']
    paths = [root / 'results/measured' / n for n in inputs]
    paths += [root / 'results/round10' / n for n in
              ['extended_decision_matrix.json', 'mixture_wins_multiplicity.json']]
    before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    load = lambda name: json.loads((root / 'results/measured' / name).read_text())
    saved = load(inputs[0])
    meta.write_tables(saved)
    round9.write_tables(load(inputs[1]))
    when.write_table(load(inputs[2]))
    training.emit_rows(load(inputs[3]), tables / 'training_size_rows.tex')
    signal.main()
    wins.main()
    if args.figures:
        import make_figures as figures
        figures.FIGS = out / 'figures'
        figures.FIGS.mkdir(exist_ok=True)
        figures.fig_samples(saved)
        figures.fig_samples_main(saved)
        figures.fig_nested(saved)
    assert all(hashlib.sha256((root / k).read_bytes()).hexdigest() == v for k,v in before.items())
    (out / 'INPUTS.sha256.json').write_text(json.dumps(before, indent=2) + '\n')
    print('Archived inputs unchanged; rendered outputs:', out)

if __name__ == '__main__':
    main()
