"""Build the read-only Wikipedia FTS5 index used by --sqlite_corpus.

Input is the official FEVER wiki-pages JSONL directory.  Each row has ``id``
and tab-separated ``lines``.  The builder refuses to overwrite a destination
and atomically renames a temporary database only after FTS construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sqlite3


def parse_lines(page_id, lines):
    seen = set()
    for raw in str(lines).splitlines():
        fields = raw.split("\t")
        if len(fields) < 2:
            continue
        try:
            number = int(fields[0])
        except ValueError:
            continue
        text = fields[1].strip()
        if text and number not in seen:
            seen.add(number)
            yield page_id, number, text, json.dumps(fields[2:], ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-pages", required=True, type=pathlib.Path,
                        help="directory containing official FEVER wiki-*.jsonl files")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    files = sorted(args.wiki_pages.glob("*.jsonl"))
    if not files:
        raise SystemExit("no JSONL files found")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    temp = args.output.with_suffix(args.output.suffix + ".building")
    if temp.exists():
        raise SystemExit(f"remove stale incomplete build first: {temp}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(temp)
    db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE sentences(
          page_id TEXT NOT NULL, line_number INTEGER NOT NULL,
          text TEXT NOT NULL, links_json TEXT NOT NULL,
          PRIMARY KEY(page_id, line_number));
        CREATE VIRTUAL TABLE sentences_fts USING fts5(
          page_id UNINDEXED, line_number UNINDEXED, text,
          content='sentences', content_rowid='rowid',
          tokenize='porter unicode61');
        CREATE TABLE provenance(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    digest = hashlib.sha256()
    count = 0
    for path in files:
        digest.update(path.name.encode()); digest.update(b"\0")
        with path.open("rb") as raw:
            for line in raw:
                digest.update(line)
                row = json.loads(line)
                records = list(parse_lines(str(row["id"]), row.get("lines", "")))
                db.executemany("INSERT OR IGNORE INTO sentences VALUES (?,?,?,?)", records)
                count += len(records)
        db.commit()
        print(f"[{path.name}] parsed sentences={count}", flush=True)
    db.execute("INSERT INTO sentences_fts(sentences_fts) VALUES ('rebuild')")
    db.execute("INSERT INTO sentences_fts(sentences_fts) VALUES ('optimize')")
    db.executemany("INSERT INTO provenance VALUES (?,?)", [
        ("source", "official FEVER Wikipedia 2017 wiki-pages JSONL"),
        ("source_file_count", str(len(files))),
        ("source_aggregate_sha256", digest.hexdigest()),
        ("sentence_count", str(count)),
    ])
    db.commit(); db.close()
    temp.replace(args.output)
    print(f"wrote {args.output}: {count} sentences")


if __name__ == "__main__":
    main()
