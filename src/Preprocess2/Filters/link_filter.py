import re
import pandas as pd


class LinkFilter:
    def __init__(self):
        self.url_pattern = re.compile(
            r"(https?://\S+|www\.[^\s]+|\b[\w-]+\.(?:com|net|org|io|vn|info|co)(?:/\S*)?)",
            re.IGNORECASE,
        )

    def filt(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        if text_col not in df.columns:
            return df
        before = len(df)
        mask = ~df[text_col].astype(str).str.contains(self.url_pattern)
        out = df.loc[mask].copy()
        print(f"Còn lại {len(out)}/{before} sau LinkFilter")
        return out
