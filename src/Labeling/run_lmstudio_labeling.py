"""
run_lmstudio_labeling.py – Gán nhãn bình luận bằng LM Studio (CLI streaming mode).

Usage:
    python -m src.Labeling.run_lmstudio_labeling --input warehouse.csv --output labeled_data.csv
    python -m src.Labeling.run_lmstudio_labeling --endpoint http://localhost:1234 --batch-size 5
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm

from .lmstudio_classifier import LMStudioClassifier


POSTFIX_EVERY = 50


def main():
    parser = argparse.ArgumentParser(description="Labeling với LM Studio API - Streaming")
    parser.add_argument("--input", default=os.path.join("pipeline", "warehouse.csv"))
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--output", default=os.path.join("pipeline", "labeled_data.csv"))
    parser.add_argument("--endpoint", default="http://localhost:1234",
                        help="LM Studio base URL (default: http://localhost:1234)")
    parser.add_argument("--model", default="", help="Model name (empty = use loaded model)")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size (default: 5)")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Test connection first
    print("🔌 Kiểm tra kết nối tới LM Studio...")
    result = LMStudioClassifier.test_connection(args.endpoint)
    if not result["ok"]:
        print(f"❌ {result['error']}")
        return
    print(f"✅ Đã kết nối. Models: {', '.join(result['models'])}")

    df = pd.read_csv(args.input)

    chat_endpoint = f"{args.endpoint.rstrip('/')}/v1/chat/completions"
    classifier = None if args.dry_run else LMStudioClassifier(
        endpoint=chat_endpoint,
        model=args.model,
        timeout=args.timeout,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    file_exists = os.path.exists(args.output)

    label_counts: dict[str, int] = {}
    processed = 0
    bulk_size = args.batch_size

    print("=" * 60)
    print("🚀 BẮT ĐẦU PHÂN LOẠI (LM STUDIO - STREAMING MODE)")
    print(f"   - Endpoint: {chat_endpoint}")
    print(f"   - Model: {args.model or '(auto)'}")
    print(f"   - Batch size: {bulk_size}")
    print("   - Ghi file ngay sau mỗi batch")
    print("   - Ctrl + C để dừng an toàn")
    print("=" * 60)

    buffer_rows = []
    buffer_tasks = []

    pbar = tqdm(total=len(df), desc="Đang xử lý", unit=" dòng")

    try:
        for _, row in df.iterrows():
            buffer_rows.append(row)
            buffer_tasks.append({"data": {"text": str(row[args.text_col])}})

            if len(buffer_tasks) < bulk_size:
                continue

            if args.dry_run:
                predictions = [
                    {"result": [{"value": {"choices": ["Clean"]}}]}
                    for _ in buffer_tasks
                ]
            else:
                predictions = classifier.predict(buffer_tasks)

            rows_to_write = []
            for row_i, pred in zip(buffer_rows, predictions):
                label = pred["result"][0]["value"]["choices"][0]
                label_counts[label] = label_counts.get(label, 0) + 1
                processed += 1
                rows_to_write.append({
                    "id": row_i[args.id_col],
                    "text": row_i[args.text_col],
                    "label": label,
                })
                pbar.update(1)
                if processed % POSTFIX_EVERY == 0:
                    pbar.set_postfix(label_counts)

            pd.DataFrame(rows_to_write).to_csv(
                args.output, mode="a", header=not file_exists,
                index=False, encoding="utf-8-sig",
            )
            file_exists = True
            buffer_rows.clear()
            buffer_tasks.clear()

        # Remainder
        if buffer_tasks:
            if args.dry_run:
                predictions = [
                    {"result": [{"value": {"choices": ["Clean"]}}]}
                    for _ in buffer_tasks
                ]
            else:
                predictions = classifier.predict(buffer_tasks)

            rows_to_write = []
            for row_i, pred in zip(buffer_rows, predictions):
                label = pred["result"][0]["value"]["choices"][0]
                label_counts[label] = label_counts.get(label, 0) + 1
                processed += 1
                rows_to_write.append({
                    "id": row_i[args.id_col],
                    "text": row_i[args.text_col],
                    "label": label,
                })
                pbar.update(1)

            pd.DataFrame(rows_to_write).to_csv(
                args.output, mode="a", header=not file_exists,
                index=False, encoding="utf-8-sig",
            )

    except KeyboardInterrupt:
        tqdm.write("\n🛑 Ctrl + C — dừng an toàn, dữ liệu đã được lưu.")

    finally:
        pbar.close()

    print(f"\n{'✅ KẾT QUẢ':^60}")
    print("-" * 60)
    print(f"{'NHÃN':<20} | {'SỐ LƯỢNG':<10}")
    print("-" * 60)
    for lbl, count in label_counts.items():
        print(f"{lbl:<20} | {count:<10}")
    print("-" * 60)
    print(f"Tổng: {processed} dòng")
    print(f"📂 File: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
