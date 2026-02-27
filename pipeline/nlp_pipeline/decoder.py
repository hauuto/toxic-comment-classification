"""
Module 1: Decoder
Treats a raw comment as an encoded sequence and decodes it into a cleaner form.
- Converts Unicode emoji → :token: labels (using emoji_vi.json for Vietnamese names)
- Truncates repeated icon characters (e.g. :)))))) → :)))  )
- Removes encoding errors and broken Unicode
- Normalizes special characters (fullwidth → ASCII)
- Replaces IPv4 addresses with <IP> token
"""
import json
import re
import unicodedata

# IPv4 address pattern — matches e.g. 192.168.1.1
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# --- Threads-specific artifact patterns (safety net for NLP pipeline) ---
# Trailing "Translate" / "Dịch" button text stuck to comment
_TRAILING_TRANSLATE = re.compile(r"(?:[Tt]ranslate|Dịch)\s*$")
# "gia đình" repeated anomaly from emoji <img alt="gia đình"> leak.
# Some crawlers/UI layers may convert whitespace to underscores early, so handle both
# "gia đình" and "gia_đình" (case-insensitive).
_GIA_DINH_WORD = r"gia(?:\s+|_)+đình"
_GIA_DINH_SPAM = re.compile(rf"(?:\s*{_GIA_DINH_WORD}\s*){{2,}}:?", re.IGNORECASE)
_GIA_DINH_NEAR_EMOJI = re.compile(rf"\s*{_GIA_DINH_WORD}\s*:?(?=:|$)", re.IGNORECASE)


class Decoder:
    """Decode raw Vietnamese social media comments."""

    def __init__(self, emoji_mapping_path: str):
        """
        Args:
            emoji_mapping_path: Path to emoji_vi.json
                Format: { "😊": "cười_đỏ_má", "😂": "cười_ra_nước_mắt", ... }
                (emoji character → Vietnamese name)
        """
        with open(emoji_mapping_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Build lookup: emoji_char → :vietnamese_name:
        # Supports both formats:
        #   New format: { "😊": "cười_đỏ_má" }  (emoji_vi.json)
        #   Old format: { ":token:": ["😊", "🙂"] }  (emoji_mapping.json)
        self.emoji_to_token = {}

        if raw:
            first_key = next(iter(raw))
            first_val = raw[first_key]

            if isinstance(first_val, str):
                # New format: emoji → name
                for emoji_char, name in raw.items():
                    token = f":{name}:"
                    self.emoji_to_token[emoji_char] = token
            elif isinstance(first_val, list):
                # Old format: token → [emoji, ...]
                for token, emojis in raw.items():
                    for em in emojis:
                        self.emoji_to_token[em] = token

        # Sort by length descending so multi-char emoji (e.g. flags, ZWJ sequences)
        # are matched before their components
        if self.emoji_to_token:
            self._emoji_pattern = re.compile(
                "|".join(
                    re.escape(e)
                    for e in sorted(self.emoji_to_token.keys(), key=len, reverse=True)
                )
            )
        else:
            self._emoji_pattern = re.compile(r"(?!)")  # never matches

        # Pattern for remaining unmapped emoji (Unicode emoji ranges)
        # NOTE: Removed U+24C2-U+1F251 (too broad, catches enclosed alphanumerics ①②③)
        self._unmapped_emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Misc Symbols & Pictographs
            "\U0001F680-\U0001F6FF"  # Transport & Map
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002702-\U000027B0"  # Dingbats
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols Extended-A
            "\U00002600-\U000026FF"  # Misc symbols
            "\U0000FE00-\U0000FE0F"  # Variation Selectors
            "\U0000200D"             # Zero Width Joiner
            "]+",
            flags=re.UNICODE,
        )

        # Icon patterns: emoticons like :) :)) :>>> =(( etc.
        # Captures : or = followed by repeated character
        self._icon_patterns = [
            # Smiley variants: :) :)) :)))
            (re.compile(r"(:\){2,})"), ":)", ")"),
            # Sad variants: :( :(( :((( 
            (re.compile(r"(:\({2,})"), ":(", "("),
            # Arrow smiley: :> :>> :>>>
            (re.compile(r"(:{2,})"), ":>", ">"),
            # Laugh: =)) =)))
            (re.compile(r"(=\){2,})"), "=)", ")"),
            # Sad: =(( =((( 
            (re.compile(r"(=\({2,})"), "=(", "("),
            # Pacman: :v :V (no truncation)
            (re.compile(r"(:(?:v|V))(?!:)"), ":v", None),
            # Cute: :3 (no truncation)
            (re.compile(r"(:3)(?!:)"), ":3", None),
            # Big grin: :D :DD
            (re.compile(r"(:D{2,})"), ":D", "D"),
            # Tongue: :P :p (no truncation)
            (re.compile(r"(:(?:P|p))(?!:)"), ":P", None),
        ]

        # Literal text emoticons: fixed string → token
        # Each tuple: (compiled_pattern, replacement_token)
        self._literal_emoticons = [
            # Heart: <3  <33  <333 → :tim:
            (re.compile(r"<3+"), ":tim:"),
            # Broken heart: </3 → :tim_vo:
            (re.compile(r"</3"), ":tim_vo:"),
        ]

    def decode_emoji(self, text: str) -> str:
        """Replace known Unicode emoji with :token: labels."""
        text = self._emoji_pattern.sub(
            lambda m: " " + self.emoji_to_token[m.group()] + " ", text
        )
        # Replace any remaining unmapped emoji with :emoji:
        text = self._unmapped_emoji_pattern.sub(" :emoji: ", text)
        return text

    def decode_icons(self, text: str, max_repeat: int = 3) -> str:
        """Replace and truncate text emoticons.

        Literal replacements (e.g. <3 → :tim:) are applied first,
        then repeated-char truncation (e.g. :)))))) → :)))  ).
        """
        # 1. Literal emoticons (fixed substitution)
        for pattern, token in self._literal_emoticons:
            text = pattern.sub(f" {token} ", text)

        # 1.5 Protect fixed-form emoticons that shouldn't be truncated.
        # Convert them into :token: form so later normalizer steps (e.g. abbreviations)
        # cannot rewrite their inner text (e.g. ":v" -> ":vậy").
        for pattern, base, repeat_char in self._icon_patterns:
            if repeat_char is not None:
                continue

            token = (base or "").strip()
            if token.startswith(":") and not token.endswith(":"):
                token = token.lower() + ":"
            text = pattern.sub(f" {token} ", text)

        # 2. Repeated-char truncation (:))))  → :)))  etc.)
        for pattern, base, repeat_char in self._icon_patterns:
            if repeat_char is None:
                # No repeat truncation needed (e.g. :v, :3)
                continue

            def _truncate(m, _base=base, _rc=repeat_char, _max=max_repeat):
                # m.group() is like ":)))))" - we want to keep base + max_repeat-1 of repeat_char
                # Since base already has 1 repeat_char, we add max_repeat-1 more
                matched_len = len(m.group())
                base_len = len(_base)
                repeat_count = matched_len - base_len + 1  # Total ) chars
                keep_count = min(repeat_count, _max)
                return _base[:-1] + _rc * keep_count  # base without last char + repeated chars

            text = pattern.sub(_truncate, text)
        return text

    def remove_encoding_errors(self, text: str) -> str:
        """Remove broken Unicode, surrogate characters, and null bytes."""
        # Remove null bytes
        text = text.replace("\x00", "")
        # Remove surrogates
        text = re.sub(r"[\ud800-\udfff]", "", text)
        # Remove other control chars except newline/tab
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text

    def decode_special_chars(self, text: str) -> str:
        """Normalize fullwidth characters to ASCII equivalents."""
        result = []
        for char in text:
            # Fullwidth ASCII variants (Ａ-Ｚ, ａ-ｚ, ０-９, etc.)
            cp = ord(char)
            if 0xFF01 <= cp <= 0xFF5E:
                result.append(chr(cp - 0xFEE0))
            elif cp == 0x3000:  # Ideographic space
                result.append(" ")
            else:
                result.append(char)
        return "".join(result)

    def replace_ip(self, text: str) -> str:
        """Replace IPv4 addresses with <IP> token."""
        return _IP_PATTERN.sub("<IP>", text)

    def clean_threads_artifacts(self, text: str) -> str:
        """Remove Threads-specific scraping artifacts (trailing Translate, gia đình spam).

        This is a safety net: even if the crawler didn't strip these, the NLP
        pipeline will catch them here before filtering/normalizing.
        """
        text = _TRAILING_TRANSLATE.sub("", text)
        text = _GIA_DINH_SPAM.sub(" ", text)
        text = _GIA_DINH_NEAR_EMOJI.sub(" ", text)
        return text.strip()

    def decode(self, text: str) -> str:
        """Run full decode pipeline on a comment."""
        # NFC normalization first: handles decomposed Unicode from copy-paste/editors
        text = unicodedata.normalize("NFC", text)
        text = self.remove_encoding_errors(text)
        text = self.decode_special_chars(text)
        text = self.replace_ip(text)          # IPv4 → <IP>
        text = self.decode_emoji(text)
        text = self.decode_icons(text)
        text = self.clean_threads_artifacts(text)
        return text
