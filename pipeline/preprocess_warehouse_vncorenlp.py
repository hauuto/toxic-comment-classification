"""CLI: Scan + preprocess warehouse.csv using VnCoreNLP.

This script is intentionally CLI-only (no GUI/web).
It reads a warehouse CSV (potentially large) in chunks, preprocesses the text
using the existing `VietnameseCommentPreprocessor` pipeline, and writes an
output CSV with extra columns:

- cleaned_text
- is_valid
- filter_reason

Example:
  python pipeline/preprocess_warehouse_vncorenlp.py \
    --input pipeline/warehouse.csv \
    --output pipeline/warehouse_preprocessed.csv \
    --backend vncorenlp \
    --auto-download-model

Notes:
- VnCoreNLP requires Java. If initialization fails, the script will error by
  default (to ensure you are truly using VnCoreNLP).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from tqdm import tqdm


def _import_pipeline():
    """Import pipeline classes with a resilient import strategy."""
    try:
        from pipeline.nlp_pipeline import VietnameseCommentPreprocessor  # type: ignore
        from pipeline.nlp_pipeline.word_segmentor import WordSegmentor  # type: ignore
        return VietnameseCommentPreprocessor, WordSegmentor
    except Exception:
        # When running as `python pipeline/<script>.py`, `pipeline` might not be importable
        # because sys.path[0] becomes the `pipeline/` folder.
        from nlp_pipeline import VietnameseCommentPreprocessor  # type: ignore
        from nlp_pipeline.word_segmentor import WordSegmentor  # type: ignore
        return VietnameseCommentPreprocessor, WordSegmentor


VietnameseCommentPreprocessor, WordSegmentor = _import_pipeline()


_TEXT_COL_CANDIDATES = [
    "text",
    "comment",
    "content",
    "message",
    "body",
    "raw_text",
]


def _detect_encoding(csv_path: Path) -> str:
    """Best-effort encoding detection for Vietnamese CSVs."""
    # Try a small set of common encodings; prefer UTF-8.
    for enc in ("utf-8-sig", "utf-8", "cp1258", "cp1252"):
        try:
            pd.read_csv(csv_path, nrows=5, encoding=enc)
            return enc
        except UnicodeDecodeError:
            continue
        except Exception:
            # Parsing errors are not necessarily encoding errors.
            # If decoding succeeded, treat encoding as acceptable.
            return enc
    return "utf-8-sig"


def _resolve_default_models_dir() -> Path:
    # Default to pipeline/models (matches VietnameseCommentPreprocessor default behavior)
    script_dir = Path(__file__).resolve().parent
    return (script_dir / "models").resolve()


def _pick_text_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(f"[ERR] --text-col '{explicit}' not found. Columns: {list(df.columns)}")
        return explicit

    cols_lower = {c.lower(): c for c in df.columns}
    for cand in _TEXT_COL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]

    raise SystemExit(
        "[ERR] Could not auto-detect text column. "
        "Use --text-col. Available columns: "
        f"{list(df.columns)}"
    )


def _build_preprocessor(
    backend: str,
    vncorenlp_dir: Optional[Path],
    auto_download_model: bool,
    require_vncorenlp: bool,
    *,
    vncorenlp_heap: str,
    max_seg_chars: Optional[int],
) -> Any:
    backend = (backend or "").strip().lower()
    if backend != "vncorenlp":
        raise SystemExit("[ERR] This CLI is intended for VnCoreNLP. Use --backend vncorenlp.")

    models_dir = (vncorenlp_dir or _resolve_default_models_dir()).resolve()

    segmentor = WordSegmentor(
        backend="vncorenlp",
        vncorenlp_dir=str(models_dir),
        auto_download=auto_download_model,
        vncorenlp_max_heap=vncorenlp_heap,
        vncorenlp_annotators=["wseg"],
        max_input_chars=max_seg_chars,
    )

    if require_vncorenlp and getattr(segmentor, "backend_name", "").lower() != "vncorenlp":
        raise SystemExit(
            "[ERR] Failed to initialize VnCoreNLP backend (it fell back to another backend).\n"
            "- Ensure Java is installed and accessible (JAVA_HOME / PATH).\n"
            "- Ensure `py_vncorenlp` and `pyjnius` are installed.\n"
            "- Try: --auto-download-model or specify --vncorenlp-dir.\n"
        )

    return VietnameseCommentPreprocessor(
        segmentor=segmentor,
        segmentor_backend="vncorenlp",
        vncorenlp_dir=str(models_dir),
    )


def preprocess_warehouse(
    *,
    input_csv: Path,
    output_csv: Path,
    text_col: Optional[str],
    chunk_size: int,
    limit: Optional[int],
    drop_invalid: bool,
    backend: str,
    vncorenlp_dir: Optional[Path],
    auto_download_model: bool,
    require_vncorenlp: bool,
    vncorenlp_heap: str,
    max_seg_chars: Optional[int],
    overwrite: bool,
) -> None:
    # Resolve to absolute paths early because some VnCoreNLP/JVM wrappers may
    # change the current working directory during initialization.
    input_csv = input_csv.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()

    if not input_csv.exists():
        raise SystemExit(f"[ERR] Input not found: {input_csv}")

    if output_csv.exists() and not overwrite:
        raise SystemExit(f"[ERR] Output already exists: {output_csv} (use --overwrite)")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    encoding = _detect_encoding(input_csv)
    print(f"[INFO] Reading: {input_csv}")
    print(f"[INFO] Encoding: {encoding}")

    pre = _build_preprocessor(
        backend=backend,
        vncorenlp_dir=vncorenlp_dir,
        auto_download_model=auto_download_model,
        require_vncorenlp=require_vncorenlp,
        vncorenlp_heap=vncorenlp_heap,
        max_seg_chars=max_seg_chars,
    )

    # We stream by chunks to avoid loading a potentially huge CSV into RAM.
    reader = pd.read_csv(
        input_csv,
        encoding=encoding,
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    total_done = 0
    wrote_header = False

    pbar_total = limit if limit is not None else None
    pbar = tqdm(total=pbar_total, desc="preprocess", unit="row")

    try:
        for chunk in reader:
            if chunk.empty:
                continue

            chosen_text_col = _pick_text_column(chunk, text_col)

            cleaned_texts: list[str] = []

            for raw in chunk[chosen_text_col].tolist():
                if limit is not None and total_done >= limit:
                    break

                res = pre.process_comment(raw)
                cleaned_texts.append(res.get("cleaned_text", ""))

                total_done += 1
                pbar.update(1)

            # If we broke early due to limit, trim chunk to match
            out_chunk = chunk.iloc[: len(cleaned_texts)].copy()
            out_chunk["cleaned_text"] = cleaned_texts
            # Keep output tidy: drop original text column if present.
            out_chunk.drop(columns=["text"], inplace=True, errors="ignore")

            out_chunk.to_csv(
                output_csv,
                index=False,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                encoding="utf-8-sig",
            )
            wrote_header = True

            if limit is not None and total_done >= limit:
                break

    finally:
        pbar.close()

    print(f"[DONE] Wrote: {output_csv}")
    print(f"[DONE] Processed rows: {total_done}")


def build_arg_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Scan + preprocess warehouse CSV using VnCoreNLP")
    parser.add_argument(
        "--input",
        default=str(script_dir / "warehouse.csv"),
        help="Input warehouse CSV path (default: pipeline/warehouse.csv)",
    )
    parser.add_argument(
        "--output",
        default=str(script_dir / "warehouse_preprocessed.csv"),
        help="Output CSV path (default: pipeline/warehouse_preprocessed.csv)",
    )
    parser.add_argument(
        "--text-col",
        default=None,
        help="Name of the text column to preprocess (default: auto-detect)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Rows per chunk (default: 2000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N rows (for quick tests)",
    )
    parser.add_argument(
        "--drop-invalid",
        action="store_true",
        help="Drop rows that fail filters (default: keep all rows and mark is_valid)",
    )

    parser.add_argument(
        "--backend",
        default="vncorenlp",
        choices=["vncorenlp"],
        help="Segmentation backend (only vncorenlp is supported here)",
    )
    parser.add_argument(
        "--vncorenlp-dir",
        default=None,
        help="Directory containing VnCoreNLP models/jar (default: pipeline/models)",
    )
    parser.add_argument(
        "--auto-download-model",
        action="store_true",
        help="Auto-download VnCoreNLP model files if missing",
    )
    parser.add_argument(
        "--vncorenlp-heap",
        default=os.environ.get("VNCORENLP_MAX_HEAP", "-Xmx2g"),
        help="Java heap size for VnCoreNLP (default: -Xmx2g). Example: -Xmx1g",
    )
    parser.add_argument(
        "--max-seg-chars",
        type=int,
        default=int(os.environ.get("VNCORENLP_MAX_INPUT_CHARS", "8000")),
        help="Skip VnCoreNLP segmentation if text length exceeds this (default: 8000).",
    )
    parser.add_argument(
        "--no-require-vncorenlp",
        action="store_true",
        help="Allow fallback if VnCoreNLP cannot initialize (default: strict)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it exists",
    )

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    preprocess_warehouse(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        text_col=args.text_col,
        chunk_size=args.chunk_size,
        limit=args.limit,
        drop_invalid=args.drop_invalid,
        backend=args.backend,
        vncorenlp_dir=Path(args.vncorenlp_dir) if args.vncorenlp_dir else None,
        auto_download_model=bool(args.auto_download_model),
        require_vncorenlp=not bool(args.no_require_vncorenlp),
        vncorenlp_heap=str(args.vncorenlp_heap),
        max_seg_chars=int(args.max_seg_chars) if args.max_seg_chars else None,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
