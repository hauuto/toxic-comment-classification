import re


class ElongationNormalizer:
    @staticmethod
    def normalize(text: str, max_repeat: int = 1) -> str:
        if not isinstance(text, str):
            return ""

        def _repl(m):
            ch = m.group(1)
            return ch * max_repeat

        # Áp dụng cho chữ cái Latin + tiếng Việt có dấu
        return re.sub(r"([A-Za-zÀ-ỹ])\1{1,}", _repl, text)
