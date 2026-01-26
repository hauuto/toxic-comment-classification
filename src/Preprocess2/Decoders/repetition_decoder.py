import re


class RepetitionDecoder:
    def __init__(self):
        # Precompile regex: match any character repeated 2+ times
        self._repeat_re = re.compile(r"(.)\1+", flags=re.UNICODE)

    def _collapse(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        def repl(m: re.Match) -> str:
            ch = m.group(1)
            # Only collapse repeated letters; keep repeated punctuation/emoji intact
            return ch if ch.isalpha() else m.group(0)

        return self._repeat_re.sub(repl, text)

    def normalize(self, text: str) -> str:
        """
        Ví dụ:
        - "tôiiiiiiiiii" -> "tôi"
        - "mẹeeeeeee" -> "mẹ"
        Giữ nguyên các chuỗi không phải chữ.
        """
        return self._collapse(text)
