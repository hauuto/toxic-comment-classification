import os
import argparse
import pandas as pd
from dotenv import load_dotenv


def normalize_to_text_only(input_path: str, output_path: str, text_col: str = "text") -> None:
    # Load env just in case future processing needs it
    load_dotenv()

    # Read with pandas; handle potential messy CSVs
    df = pd.read_csv(input_path)

    if text_col not in df.columns:
        # Try to infer: pick first object/string-like column
        object_cols = [c for c in df.columns if df[c].dtype == object]
        if not object_cols:
            raise ValueError(
                f"Cannot find '{text_col}' or any text-like column in {input_path}"
            )
        text_col = object_cols[0]

    # Select, clean
    s = df[text_col].astype(str).str.strip()
    s = s.replace({"nan": ""})
    s = s.str.replace(r"\s+", " ", regex=True)

    # Drop empties and duplicates, keep order
    cleaned = s[s != ""].drop_duplicates()

    # Ensure output dir exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Write two-column CSV: 'no' (sequential index) and 'text'
    texts = cleaned.tolist()
    out_df = pd.DataFrame({
        "no": list(range(len(texts))),
        "text": texts,
    })
    out_df.to_csv(output_path, index=False)
    print(f"[OK] Wrote text-only CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Normalize CSV to text-only column")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument(
        "--output",
        default=os.path.join("data", "combined_text_only.csv"),
        help="Output CSV path (default: data/combined_text_only.csv)",
    )
    parser.add_argument(
        "--text-col", default="text", help="Column name for text (default: text)"
    )
    args = parser.parse_args()

    normalize_to_text_only(args.input, args.output, args.text_col)


if __name__ == "__main__":
    main()
