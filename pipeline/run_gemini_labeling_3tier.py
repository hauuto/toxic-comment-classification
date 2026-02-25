"""run_gemini_labeling_3tier.py – Gán nhãn bình luận 3-tier bằng Gemini (Google AI Studio).

Usage:
  python pipeline/run_gemini_labeling_3tier.py
  python pipeline/run_gemini_labeling_3tier.py --input pipeline/warehouse.csv --output pipeline/labeled_data.csv

Env:
  GEMINI_API_KEY=...
  GEMINI_MODEL=gemini-2.0-flash (optional)
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict, List

import pandas as pd
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return False
from tqdm import tqdm

from gemini_hierarchical_classifier import GeminiHierarchicalClassifier


POSTFIX_EVERY = 50


def _load_existing_ids(output_path: str) -> set[int]:
    if not os.path.exists(output_path):
        return set()

    ids: set[int] = set()
    try:
        with open(output_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ids.add(int(row.get("id", "0")))
                except Exception:
                    continue
    except Exception:
        return set()

    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Labeling 3-tier với Gemini API (streaming)")
    parser.add_argument("--input", default=os.path.join("pipeline", "warehouse.csv"))
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--output", default=os.path.join("pipeline", "labeled_data.csv"))
    parser.add_argument("--model", default="", help="Gemini model name (default from GEMINI_MODEL or gemini-2.0-flash)")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size (default: 5, max: 20)")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--resume", action="store_true", help="Skip rows already in output (by id)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    if not args.dry_run and not os.getenv("GEMINI_API_KEY", "").strip():
        raise RuntimeError("Missing GEMINI_API_KEY (set in .env or environment)")

    df = pd.read_csv(args.input)

    batch_size = max(1, min(int(args.batch_size), 20))
    model = (args.model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")).strip() or "gemini-2.0-flash"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    file_exists = os.path.exists(args.output)

    labeled_ids: set[int] = set()
    if args.resume:
        labeled_ids = _load_existing_ids(args.output)

    classifier = None if args.dry_run else GeminiHierarchicalClassifier(model=model, timeout=args.timeout)

    label_counts: dict[str, int] = {}
    processed = 0

    buffer_rows: List[Dict[str, Any]] = []
    buffer_tasks: List[Dict[str, Any]] = []

    total = len(df)
    if args.resume and labeled_ids:
        total_pending = sum(1 for _, r in df.iterrows() if int(r.get(args.id_col, 0)) not in labeled_ids)
    else:
        total_pending = total

    print("=" * 60)
    print("🚀 BẮT ĐẦU PHÂN LOẠI (GEMINI - 3 TIER)")
    print(f"   - Model: {model}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Resume: {'ON' if args.resume else 'OFF'}")
    print("   - Ctrl + C để dừng an toàn")
    print("=" * 60)

    pbar = tqdm(total=total_pending, desc="Đang xử lý", unit=" dòng")

    def flush_batch() -> None:
        nonlocal processed, file_exists

        if args.dry_run:
            preds = [
                {"tier1_spam": "Not Spam", "tier2_toxic": "Clean", "tier3_labels": ["Neutral"]}
                for _ in buffer_tasks
            ]
        else:
            preds = classifier.predict(buffer_tasks)

        rows_to_write = []
        for row_i, pred in zip(buffer_rows, preds):
            t1 = pred["tier1_spam"]
            t2 = pred["tier2_toxic"]
            t3_list = pred.get("tier3_labels", []) or []
            t3 = "|".join(t3_list)

            label_counts[t1] = label_counts.get(t1, 0) + 1
            label_counts[t2] = label_counts.get(t2, 0) + 1
            for lbl in t3_list:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

            processed += 1
            rows_to_write.append(
                {
                    "id": row_i[args.id_col],
                    "text": row_i[args.text_col],
                    "tier1_spam": t1,
                    "tier2_toxic": t2,
                    "tier3_labels": t3,
                }
            )

            pbar.update(1)
            if processed % POSTFIX_EVERY == 0:
                pbar.set_postfix(label_counts)

        pd.DataFrame(rows_to_write).to_csv(
            args.output,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8-sig",
        )
        file_exists = True

    try:
        for _, row in df.iterrows():
            rid = row.get(args.id_col, 0)
            try:
                rid_int = int(rid)
            except Exception:
                rid_int = 0

            if args.resume and rid_int in labeled_ids:
                continue

            buffer_rows.append(row)
            buffer_tasks.append({"data": {"text": str(row[args.text_col])}})

            if len(buffer_tasks) < batch_size:
                continue

            flush_batch()
            buffer_rows.clear()
            buffer_tasks.clear()

        if buffer_tasks:
            flush_batch()

    except KeyboardInterrupt:
        tqdm.write("\n🛑 Ctrl + C — dừng an toàn, dữ liệu đã được lưu.")

    finally:
        pbar.close()

    print(f"\n{'✅ KẾT QUẢ':^60}")
    print("-" * 60)
    for lbl, count in label_counts.items():
        print(f"{lbl:<20} | {count:<10}")
    print("-" * 60)
    print(f"Tổng: {processed} dòng")
    print(f"📂 File: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
