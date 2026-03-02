"""Remove all warehouse rows containing 'gia đình' / 'gia_đình' (streaming).

Use cases:
- Count how many records are contaminated by the 'gia đình' artifact.
- Remove those records and optionally reassign IDs sequentially.

Default behavior is safe:
- Creates a timestamped backup next to the input file.
- Writes to a temporary file and atomically replaces the original.

Usage (from repo root):
  python -m pipeline.tools.remove_giadinh_rows --count-only
  python -m pipeline.tools.remove_giadinh_rows --apply --reid

Options:
  --path pipeline/warehouse.csv
  --count-only
  --apply
  --dry-run
  --reid

Notes:
- Matches both forms and minor variations: 'gia đình', 'gia_đình', multiple spaces/underscores.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path


# Match both stored variants:
# - "gia đình" (one or more spaces)
# - "gia_đình" (underscore)
# We NFC-normalize per row to handle decomposed diacritics.
_GIA_DINH_RE = re.compile(r"gia(?:\s+|_)+đình", re.IGNORECASE | re.UNICODE)


def _default_warehouse_path() -> Path:
    return Path(__file__).resolve().parents[1] / "warehouse.csv"


def _count_giadinh_occurrences(text: str) -> int:
    if not text:
        return 0
    text = unicodedata.normalize("NFC", text)
    return sum(1 for _ in _GIA_DINH_RE.finditer(text))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(_default_warehouse_path()), help="Path to warehouse.csv")
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count matching rows; do not write changes.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletion (writes output and replaces the file).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan and report; do not write changes.")
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Rewrite id column sequentially from 1..N while writing.",
    )
    args = parser.parse_args(argv)

    src_path = Path(args.path)
    if not src_path.exists():
        print(f"File not found: {src_path}", file=sys.stderr)
        return 2

    # count-only and dry-run are both non-writing modes.
    write_mode = bool(args.apply) and not bool(args.dry_run) and not bool(args.count_only)

    if not args.count_only and not args.dry_run and not args.apply:
        print("Nothing to do: choose one of --count-only, --dry-run, or --apply", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_path = src_path.with_suffix(src_path.suffix + f".bak_{ts}")
    tmp_path = src_path.with_suffix(src_path.suffix + f".tmp_{ts}")

    total_rows = 0
    matched_rows = 0
    matched_occurrences = 0
    kept_rows = 0

    with src_path.open("r", encoding="utf-8-sig", newline="") as rf:
        reader = csv.DictReader(rf)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            print("warehouse.csv has no header/fieldnames", file=sys.stderr)
            return 2

        # Ensure an id column exists if reid is requested.
        if args.reid and "id" not in [f.lower() for f in fieldnames]:
            fieldnames = ["id", *fieldnames]

        if not write_mode:
            for row in reader:
                total_rows += 1
                text = row.get("text", "")
                occ = _count_giadinh_occurrences(text)
                if occ:
                    matched_rows += 1
                    matched_occurrences += occ
                if total_rows % 200000 == 0:
                    print(
                        f"... scanned {total_rows:,} rows | matched_rows {matched_rows:,} | matched_occ {matched_occurrences:,}"
                    )

            print(
                f"DONE. rows={total_rows:,} matched_rows={matched_rows:,} matched_occurrences={matched_occurrences:,}"
            )
            return 0

        with tmp_path.open("w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for row in reader:
                total_rows += 1
                text = row.get("text", "")

                occ = _count_giadinh_occurrences(text)
                if occ:
                    matched_rows += 1
                    matched_occurrences += occ
                else:
                    kept_rows += 1
                    if args.reid:
                        row["id"] = str(kept_rows)
                    writer.writerow(row)

                if total_rows % 200000 == 0:
                    print(
                        f"... processed {total_rows:,} rows | matched_rows {matched_rows:,} | matched_occ {matched_occurrences:,} | kept {kept_rows:,}"
                    )

    # Backup then replace atomically.
    os.replace(src_path, bak_path)
    os.replace(tmp_path, src_path)

    print(
        f"DONE. rows={total_rows:,} matched_rows={matched_rows:,} matched_occurrences={matched_occurrences:,} kept={kept_rows:,}"
    )
    print(f"Backup: {bak_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
