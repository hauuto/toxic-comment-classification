import pandas as pd
from langdetect import detect_langs, DetectorFactory, LangDetectException


class EnglishFilter:
    def __init__(self):
        # đảm bảo kết quả langdetect deterministic
        DetectorFactory.seed = 0

    def analyze(self, text: str):
        """
        Phân tích 1 text:
        - Trả về (is_vi: bool, vi_chance: float)
        - Chỉ gọi langdetect 1 lần
        """
        if not isinstance(text, str) or not text.strip():
            return False, 0.0

        try:
            res = detect_langs(text)
            for r in res:
                if r.lang == "vi":
                    return True, r.prob
            return False, 0.0
        except LangDetectException:
            return False, 0.0

    def filt(self, df: pd.DataFrame, text_col: str):
        """
        Lọc DataFrame, chỉ giữ lại comment tiếng Việt
        Thêm cột: vi_chance
        """
        old_cnt = len(df)

        mask = []
        chances = []

        for text in df[text_col]:
            is_vi, chance = self.analyze(text)
            mask.append(is_vi)
            if is_vi:
                chances.append(chance)

        result_df = df.loc[mask].copy()
        result_df["vi_chance"] = chances

        print(f"Còn lại {len(result_df)}/{old_cnt} sau EnglishFilter")
        return result_df
