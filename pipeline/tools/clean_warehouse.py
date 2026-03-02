"""Clean warehouse.csv in a streaming (low-memory) way.

Fixes two common historical artifacts:
1) Spaced/broken emoji tokens produced by some tokenizers:
   ": mặt _ tan _ chảy :" -> ":mặt_tan_chảy:"
2) Threads crawler leak: stray "gia đình" / "gia_đình" placed right before an emoji token.
   "... òi gia_đình :mặt_tan_chảy:" -> "... òi :mặt_tan_chảy:"

This script is intentionally conservative:
- It removes "gia đình" only when it is immediately followed by a :token: emoji label.
- It does not try to "normalize" the whole text beyond these issues.

Usage (from repo root):
  venv/Scripts/python.exe -m pipeline.tools.clean_warehouse

Optional args:
  --path pipeline/warehouse.csv
  --dry-run
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


_ALLOWED_NAME_CHARS = r"a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"

_EMOJI_TOKEN_SPACED_RE = re.compile(
    rf":\s*(?P<name>[{_ALLOWED_NAME_CHARS}]+(?:\s*_\s*[{_ALLOWED_NAME_CHARS}]+)*)\s*:",
    re.UNICODE | re.IGNORECASE,
)

# Remove "gia đình" or "gia_đình" only when it directly precedes a :token:
# Example: "... gia_đình :mặt_tan_chảy:" -> "... :mặt_tan_chảy:"
_GIA_DINH_BEFORE_TOKEN_RE = re.compile(
    rf"\s*gia(?:\s+|_)+đình\s*(?=:[{_ALLOWED_NAME_CHARS}]+:)",
    re.UNICODE | re.IGNORECASE,
)

# Remove "gia đình" / "gia_đình" when it appears immediately before/after ANY tag like <NUM>, <DATE>, ...
# (conservative adjacency rule; does not remove 'gia đình' in normal sentences).
_TAG_RE_STR = r"<\s*[A-Za-z0-9_]+\s*>"
_GIA_DINH_WORD_RE_STR = r"gia(?:\s+|_)+đình"

# Some stored texts contain separators like "%" between tokens (e.g. "<NUM>% gia đình").
# Allow optional percent-separators while keeping the rule conservative (adjacent only).
_TAG_ADJ_SEP_STR = r"(?:\s*%+\s*)?"

_GIA_DINH_BEFORE_TAG_RE = re.compile(
    rf"\s*{_GIA_DINH_WORD_RE_STR}\s*{_TAG_ADJ_SEP_STR}(?={_TAG_RE_STR})",
    re.UNICODE | re.IGNORECASE,
)
_TAG_THEN_GIA_DINH_RE = re.compile(
    rf"(?P<tag>{_TAG_RE_STR})\s*{_TAG_ADJ_SEP_STR}{_GIA_DINH_WORD_RE_STR}\s*",
    re.UNICODE | re.IGNORECASE,
)


def _canonicalize_emoji_tokens(text: str) -> str:
    if not text:
        return text

    def _fix(m: re.Match) -> str:
        name = m.group("name")
        name = re.sub(r"\s*_\s*", "_", name.strip())
        return f":{name}:"

    return _EMOJI_TOKEN_SPACED_RE.sub(_fix, text)


def clean_text(text: str) -> tuple[str, bool]:
    """Return (cleaned_text, changed)."""

    if text is None:
        return "", text is not None

    original = text

    # Keep unicode stable
    text = unicodedata.normalize("NFC", text)

    # 1) Fix spaced/broken :token: formats
    text = _canonicalize_emoji_tokens(text)

    # 2) Remove stray "gia đình" right before :token:
    text = _GIA_DINH_BEFORE_TOKEN_RE.sub(" ", text)

    # 3) Remove stray "gia đình" adjacent to placeholder tags (<NUM>, <DATE>, ...)
    text = _GIA_DINH_BEFORE_TAG_RE.sub(" ", text)
    text = _TAG_THEN_GIA_DINH_RE.sub(lambda m: m.group("tag") + " ", text)

    # Minimal whitespace cleanup after removals
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    return text, text != original


def _default_warehouse_path() -> Path:
    return Path(__file__).resolve().parents[1] / "warehouse.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(_default_warehouse_path()), help="Path to warehouse.csv")
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes; just report.")
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak_path = src_path.with_suffix(src_path.suffix + f".bak_{ts}")
    tmp_path = src_path.with_suffix(src_path.suffix + f".tmp_{ts}")

    changed_rows = 0
    total_rows = 0

    # warehouse.csv is utf-8; handle potential BOM
    with src_path.open("r", encoding="utf-8-sig", newline="") as rf:
        reader = csv.DictReader(rf)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            print("warehouse.csv has no header/fieldnames", file=sys.stderr)
            return 2

        if args.reid and "id" not in [f.lower() for f in fieldnames]:
            # Ensure we have an id column to rewrite.
            fieldnames = ["id", *fieldnames]

        if args.dry_run:
            for row in reader:
                total_rows += 1
                text = row.get("text", "")
                cleaned, changed = clean_text(text)
                if changed:
                    changed_rows += 1
                if args.reid:
                    # Resetting ids always changes semantic content; count it.
                    # We don't attempt to compare old/new id values in dry-run.
                    changed_rows += 0
                if total_rows % 200000 == 0:
                    print(f"... scanned {total_rows:,} rows, changed {changed_rows:,}")
            print(f"DONE (dry-run). scanned={total_rows:,} changed={changed_rows:,}")
            return 0

        with tmp_path.open("w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for row in reader:
                total_rows += 1
                text = row.get("text", "")
                cleaned, changed = clean_text(text)
                if changed:
                    changed_rows += 1
                    row["text"] = cleaned

                if args.reid:
                    row["id"] = str(total_rows)

                writer.writerow(row)

                if total_rows % 200000 == 0:
                    print(f"... processed {total_rows:,} rows, changed {changed_rows:,}")

    # Backup then replace
    os.replace(src_path, bak_path)
    os.replace(tmp_path, src_path)

    print(f"DONE. rows={total_rows:,} changed={changed_rows:,}")
    print(f"Backup: {bak_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
