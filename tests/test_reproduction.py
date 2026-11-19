"""Regenerated tables must match the camera-ready paper byte for byte.

tests/expected_tables.json holds the SHA-256 of every table body typeset in the
paper and the supplement. Run python code/make_all.py first; the tests are
skipped until outputs/tables exists.
"""
import hashlib
import json
import pathlib

import pytest

ART = pathlib.Path(__file__).resolve().parents[1]
TABLES = ART / "outputs" / "tables"
EXPECTED = json.loads((ART / "tests" / "expected_tables.json").read_text())["tables"]


def test_stored_inputs_match_sha256sums():
    """Trajectories, critic caches and stored analyses are unmodified.

    Files that make_all.py recomputes are listed in tests/recomputed.txt and
    are checked through the regenerated tables instead.
    """
    recomputed = set((ART / "tests" / "recomputed.txt").read_text().split())
    bad = []
    for line in (ART / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name in recomputed:
            continue
        path = ART / name
        if (not path.exists()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            bad.append(name)
    assert not bad, f"{len(bad)} stored files changed: {bad[:5]}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_regenerated_table_matches_paper(name):
    path = TABLES / name
    if not path.exists():
        pytest.skip("run python code/make_all.py to generate outputs/tables")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    assert got == EXPECTED[name]["sha256"], (
        f"{name} ({EXPECTED[name]['paper']}) differs from the camera-ready table")
