import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from .gemini_classifier import GeminiClassifier


BULK_SIZE = 20          # 🔥 tăng batch
POSTFIX_EVERY = 100     # cập nhật thống kê mỗi 100 dòng


def main():
    parser = argparse.ArgumentParser(description="Labeling với Gemini API - Streaming")
    parser.add_argument("--input", default=os.path.join("src", "Preprocess2", "combined.csv"))
    parser.add_argument("--id-col", default="no")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--output", default=os.path.join("reports", "labeled_data.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    df = pd.read_csv(args.input)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    classifier = None if args.dry_run else GeminiClassifier()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    file_exists = os.path.exists(args.output)

    label_counts: dict[str, int] = {}
    processed = 0

    print("=" * 60)
    print("🚀 BẮT ĐẦU PHÂN LOẠI (STREAMING MODE)")
    print("   - Batch size:", BULK_SIZE)
    print("   - Ghi file ngay sau mỗi batch")
    print("   - Ctrl + C để dừng an toàn")
    print("=" * 60)

    buffer_rows = []
    buffer_tasks = []

    pbar = tqdm(total=len(df), desc="Đang xử lý", unit=" dòng")

    try:
        for _, row in df.iterrows():
            buffer_rows.append(row)
            buffer_tasks.append({
                "data": {"text": str(row[args.text_col])}
            })

            if len(buffer_tasks) < BULK_SIZE:
                continue

            predictions = classifier.predict(buffer_tasks)

            rows_to_write = []

            for row_i, pred in zip(buffer_rows, predictions):
                label = pred["result"][0]["value"]["choices"][0]
                label_counts[label] = label_counts.get(label, 0) + 1
                processed += 1

                rows_to_write.append({
                    "id": row_i[args.id_col],
                    "text": row_i[args.text_col],
                    "label": label
                })

                pbar.update(1)

                if processed % POSTFIX_EVERY == 0:
                    pbar.set_postfix(label_counts)

            pd.DataFrame(rows_to_write).to_csv(
                args.output,
                mode="a",
                header=not file_exists,
                index=False,
                encoding="utf-8-sig"
            )
            file_exists = True

            buffer_rows.clear()
            buffer_tasks.clear()

        # xử lý phần dư (< BULK_SIZE)
        if buffer_tasks:
            predictions = classifier.predict(buffer_tasks)
            rows_to_write = []

            for row_i, pred in zip(buffer_rows, predictions):
                label = pred["result"][0]["value"]["choices"][0]
                label_counts[label] = label_counts.get(label, 0) + 1
                processed += 1

                rows_to_write.append({
                    "id": row_i[args.id_col],
                    "text": row_i[args.text_col],
                    "label": label
                })

                pbar.update(1)

            pd.DataFrame(rows_to_write).to_csv(
                args.output,
                mode="a",
                header=not file_exists,
                index=False,
                encoding="utf-8-sig"
            )

    except KeyboardInterrupt:
        tqdm.write("\n🛑 Ctrl + C — dừng an toàn, dữ liệu đã được lưu.")

    finally:
        pbar.close()

    print(f"\n{'✅ KẾT QUẢ HIỆN TẠI':^60}")
    print("-" * 60)
    print(f"{'NHÃN':<20} | {'SỐ LƯỢNG':<10}")
    print("-" * 60)
    for lbl, count in label_counts.items():
        print(f"{lbl:<20} | {count:<10}")
    print("-" * 60)
    print(f"📂 File: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
