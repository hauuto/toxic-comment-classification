"""
Module 3: Normalizer
Standardizes text into a stable, consistent form while preserving emotional cues.
- Character elongation reduction
- Abbreviation expansion
- Case normalization
- URL/email/mention/hashtag/date replacement
- Numerical expression normalization (2k → <NUM> nghìn, 5 triệu → <NUM> triệu)
- Via signature removal
- Whitespace and punctuation normalization
"""
import json
import re
import unicodedata
from .config import MAX_REPEAT_CHARS, MAX_REPEAT_PUNCTUATION, VIA_SIGNATURES

try:
    from underthesea import text_normalize as _underthesea_text_normalize
except Exception:  # underthesea may be missing or unsupported on some Python versions
    _underthesea_text_normalize = None


class Normalizer:
    """Normalize Vietnamese social media comments."""

    def __init__(self, abbreviations_path: str):
        """
        Args:
            abbreviations_path: Path to abbreviations.json
                Format: { "target_word": ["abbrev1", "abbrev2", ...], ... }
        """
        with open(abbreviations_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Build inverted lookup: abbreviation → full form
        self.abbrev_map = {}
        for target, variants in raw.items():
            for v in variants:
                self.abbrev_map[v.lower()] = target

        # Sort by length descending to match longer abbreviations first
        sorted_abbrevs = sorted(self.abbrev_map.keys(), key=len, reverse=True)

        # Build regex: match abbreviations as whole words, but do NOT match
        # if touching an apostrophe (protects English contractions like don't)
        escaped = [re.escape(a) for a in sorted_abbrevs]
        self._abbrev_pattern = re.compile(
            r"(?<![\w'’])(" + "|".join(escaped) + r")(?![\w'’])",
            re.IGNORECASE,
        )

        # Via signature patterns
        self._via_patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in VIA_SIGNATURES]

        # Email pattern - must come BEFORE URL pattern
        self._email_pattern = re.compile(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
            re.IGNORECASE,
        )

        # IP address pattern — must come BEFORE URL and number patterns
        self._ip_pattern = re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        )

        # URL pattern - matches:
        # 1. http://... or https://...
        # 2. www.example.com
        # 3. domain.com (with common TLDs, supports Vietnamese characters via \w)
        self._url_pattern = re.compile(
            r"https?://[^\s<>\"']+|"  # http:// or https://
            r"www\.[^\s<>\"']+|"  # www.domain...
            r"[\w-]+\.(?:com|vn|net|org|edu|gov|io|co|info|biz|me|tv|asia|online|site|tech|store|app|dev|blog|news|media|xyz|top|club|live|today|world|space)(?=\s|$|[^\w])",
            re.IGNORECASE | re.UNICODE,
        )

        # Mention: @username
        self._mention_pattern = re.compile(r"@[\w.]+")

        # Hashtag: #topic
        self._hashtag_pattern = re.compile(r"#[\w]+", re.UNICODE)

        # Date patterns (various Vietnamese/international formats)
        self._date_patterns = [
            # dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy (with 2 or 4 digit year)
            re.compile(r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b"),
            # yyyy/mm/dd or yyyy-mm-dd
            re.compile(r"\b\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}\b"),
            # Vietnamese text dates: "ngày 30 tháng 12 năm 2025", "tháng 12/2025"
            re.compile(r"(?:ngày\s+)?\d{1,2}\s+(?:tháng|thg)\s+\d{1,2}(?:\s+(?:năm\s+)?\d{2,4})?", re.IGNORECASE),
            # Compact: T12/2025, T6/14/11/25
            re.compile(r"\bT\d{1,2}/\d{2,4}\b", re.IGNORECASE),
            # Standalone year with context: "năm 2025", "2025 rồi"
            # (Only match 4-digit years 19xx-20xx when preceded by common context)
            re.compile(r"(?:năm\s+)(?:19|20)\d{2}\b", re.IGNORECASE),
        ]

        # Short date formats like dd/mm (no year).
        # Example: "12/10" should be treated as <date> rather than <num>/<num>.
        # We avoid matching dd/mm/yyyy by requiring NOT followed by "/yyyy".
        self._short_date_slash = re.compile(
            r"\b(?P<d>\d{1,2})\s*/\s*(?P<m>\d{1,2})\b(?!\s*/\s*\d{2,4}\b)",
            re.UNICODE,
        )

        # If the text already contains placeholders like <NUM>/<NUM>, treat it as <DATE>.
        # This helps when older data or upstream systems already normalized numbers.
        self._short_date_placeholder = re.compile(
            r"<\s*num\s*>\s*/\s*<\s*num\s*>",
            re.IGNORECASE | re.UNICODE,
        )

        # Timestamp pattern: 12:30, 1:45:00 (from 2-Preprocessing.ipynb)
        self._timestamp_pattern = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

        # Numerical expression patterns
        # Order matters: longer units first
        # Vietnamese number units: tỷ/tỉ (billion), triệu (million), nghìn/ngàn/k (thousand)
        self._num_patterns = [
            # e.g. "5 tỷ", "1.5tỷ", "2,5 tỉ"
            (re.compile(
                r"(\d+(?:[.,]\d+)?)\s*(?:tỷ|tỉ)\b",
                re.IGNORECASE | re.UNICODE,
            ), r"<num> tỷ"),
            # e.g. "5 triệu", "1.5triệu"
            (re.compile(
                r"(\d+(?:[.,]\d+)?)\s*triệu\b",
                re.IGNORECASE | re.UNICODE,
            ), r"<num> triệu"),
            # e.g. "5 củ" (VN slang: 1 củ = 1 triệu đồng)
            (re.compile(
                r"(\d+(?:[.,]\d+)?)\s*củ\b",
                re.UNICODE,
            ), r"<num> triệu"),
            # e.g. "2k5" = 2500 (compound: digit + k + digit)
            (re.compile(
                r"(\d+)[kK](\d+)\b",
                re.UNICODE,
            ), r"<num> nghìn"),
            # e.g. "2k", "5k", "10K", "2 nghìn", "3 ngàn"
            (re.compile(
                r"(\d+(?:[.,]\d+)?)\s*(?:k|K|nghìn|ngàn)\b",
                re.UNICODE,
            ), r"<num> nghìn"),
            # e.g. "100 đồng", "50 đ", "200 vnd", "5 usd", "3 $"
            (re.compile(
                r"(\d+(?:[.,]\d+)?)\s*(?:đồng|đ\b|vnd|usd|\$|€|£)",
                re.IGNORECASE | re.UNICODE,
            ), r"<num>"),
            # Standalone numbers (integers and decimals) not already replaced
            # Negative lookbehind for < and > to avoid matching <3 or >3 (emoji/emoticons)
            (re.compile(
                r"(?<![\w<>])\d+(?:[.,]\d+)?(?![\w>])",
                re.UNICODE,
            ), r"<num>"),
        ]

        # Character elongation: 3+ consecutive same chars → max MAX_REPEAT_CHARS
        self._elongation_pattern = re.compile(r"(.)\1{2,}")

        # Punctuation elongation: !!!! → !!! , ???? → ???, .... → ...
        # Use backreference \1 to match runs of the SAME char only (not mixed !?.)
        self._punct_elongation = re.compile(r"([!?.])\1{2,}")

        # Multiple whitespace
        self._whitespace_pattern = re.compile(r"[ \t]+")
        self._newline_pattern = re.compile(r"\n{3,}")

        # Placeholder protection: tokens like <url>, <mention>, <hashtag>, <email>,
        # <date>, <num> should not be split or lowercased internally.
        # We protect them by temporarily replacing with safe tokens during processing.
        # Also protect emoji/icon tokens in the form :token: so abbreviation expansion
        # cannot rewrite inside them (e.g. ":v:" should NOT become ":vậy:").
        # Also protect custom special-icon tokens wrapped in @@...@@ (and @@ alone).
        self._protected_token_pattern = re.compile(
            r"<(?:url|mention|hashtag|email|date|time|num|ip)>|:[\w]+:|@@[^\s@]{1,64}@@|@@+",
            re.IGNORECASE | re.UNICODE,
        )

    @staticmethod
    def _base26(n: int) -> str:
        """Encode non-negative int to a-z base26 string (no digits).

        We avoid digits in protection markers because some normalizers/tokenizers
        (e.g. Underthesea) can split alnum sequences in unexpected ways.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        if n == 0:
            return "a"
        out = []
        while n:
            n, r = divmod(n, 26)
            out.append(alphabet[r])
        return "".join(reversed(out))

    def _protect_tokens(self, text: str):
        tokens = {}
        counter = [0]

        def _protect(m):
            key = f"xptx{self._base26(counter[0])}xptx"  # already lowercase, no digits
            tokens[key] = m.group()
            counter[0] += 1
            return key

        protected = self._protected_token_pattern.sub(_protect, text)
        return protected, tokens

    def normalize_char_elongation(self, text: str) -> str:
        """Reduce character elongation: hayyyyyyy → hayyy (max 3 repeats)."""
        def _reduce(m):
            char = m.group(1)
            return char * MAX_REPEAT_CHARS
        return self._elongation_pattern.sub(_reduce, text)

    def normalize_punctuation(self, text: str) -> str:
        """Cap repeated punctuation: !!!! → !!!, ???? → ???"""
        def _reduce(m):
            char = m.group(1)
            return char * MAX_REPEAT_PUNCTUATION
        return self._punct_elongation.sub(_reduce, text)

    def normalize_abbreviations(self, text: str) -> str:
        """Expand common Vietnamese abbreviations using word-boundary matching."""
        protected, tokens = self._protect_tokens(text)

        def _replace(m):
            word = m.group(1)
            return self.abbrev_map.get(word.lower(), word)

        out = self._abbrev_pattern.sub(_replace, protected)
        for key, val in tokens.items():
            out = out.replace(key, val)
        return out

    def normalize_case(self, text: str) -> str:
        """Convert text to lowercase, preserving placeholder tokens."""
        # Protect placeholders/tokens from lowercasing using unique lowercase markers.
        protected, placeholders = self._protect_tokens(text)
        lowered = protected.lower()

        # Restore original placeholders (markers survived lowercasing)
        for key, val in placeholders.items():
            lowered = lowered.replace(key, val)
        return lowered

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces and excessive newlines."""
        text = self._whitespace_pattern.sub(" ", text)
        text = self._newline_pattern.sub("\n\n", text)
        return text.strip()

    def normalize_ips(self, text: str) -> str:
        """Replace IP addresses with <IP> token."""
        return self._ip_pattern.sub("<ip>", text)

    def normalize_emails(self, text: str) -> str:
        """Replace email addresses with <email> token."""
        return self._email_pattern.sub("<email>", text)

    def normalize_urls(self, text: str) -> str:
        """Replace URLs with <url> token."""
        return self._url_pattern.sub("<url>", text)

    def normalize_mentions(self, text: str) -> str:
        """Replace @mentions with <mention> token."""
        return self._mention_pattern.sub("<mention>", text)

    def normalize_hashtags(self, text: str) -> str:
        """Replace #hashtags with <hashtag> token."""
        return self._hashtag_pattern.sub("<hashtag>", text)

    def normalize_dates(self, text: str) -> str:
        """Replace date expressions with <date> token."""
        # Fix legacy placeholder form first
        text = self._short_date_placeholder.sub("<date>", text)

        # Handle short dd/mm (no year)
        def _short_date_repl(m: re.Match) -> str:
            try:
                d = int(m.group("d"))
                mo = int(m.group("m"))
                if 1 <= d <= 31 and 1 <= mo <= 12:
                    return "<date>"
            except Exception:
                pass
            return m.group(0)

        text = self._short_date_slash.sub(_short_date_repl, text)

        for pattern in self._date_patterns:
            text = pattern.sub("<date>", text)
        return text

    def normalize_timestamps(self, text: str) -> str:
        """Replace time expressions (HH:MM or HH:MM:SS) with <TIME> token."""
        # Use lowercase placeholder for consistency with other normalizer placeholders;
        # the pipeline canonicalizes it to <TIME> after segmentation.
        return self._timestamp_pattern.sub("<time>", text)

    def normalize_numbers(self, text: str) -> str:
        """Normalize numerical expressions.

        Examples:
            "5 triệu" → "<num> triệu"
            "2k" → "<num> nghìn"
            "3 tỷ" → "<num> tỷ"
            "100 đồng" → "<num>"
            "42" → "<num>"
        """
        for pattern, replacement in self._num_patterns:
            text = pattern.sub(replacement, text)
        return text

    def normalize_via_signatures(self, text: str) -> str:
        """Remove common forum/app via signatures."""
        for pattern in self._via_patterns:
            text = pattern.sub("", text)
        return text

    def normalize_unicode(self, text: str) -> str:
        """Apply Unicode NFC normalization to handle decomposed characters from copy-paste."""
        return unicodedata.normalize("NFC", text)

    def normalize(self, text: str) -> str:
        """Run full normalization pipeline."""
        # Special-case legacy placeholder form early. If we protect <num> tokens
        # too soon, the <num>/<num> -> <date> rule would never fire.
        text = self._short_date_placeholder.sub("<date>", text)

        # Protect placeholders/icon tokens early so they survive mention handling,
        # punctuation/whitespace normalization, and (optionally) Underthesea.
        # This is critical for custom tokens like @@...@@.
        protected, protected_tokens = self._protect_tokens(text)

        # NFC first: handle decomposed Unicode from copy-paste/editors
        protected = self.normalize_unicode(protected)
        # IPs first (before URLs and numbers to avoid mangling)
        protected = self.normalize_ips(protected)
        # Order matters: emails before URLs (emails contain @, URLs don't)
        protected = self.normalize_emails(protected)
        protected = self.normalize_urls(protected)
        protected = self.normalize_mentions(protected)
        protected = self.normalize_hashtags(protected)
        protected = self.normalize_dates(protected)
        protected = self.normalize_timestamps(protected)
        protected = self.normalize_via_signatures(protected)
        protected = self.normalize_char_elongation(protected)
        protected = self.normalize_punctuation(protected)
        protected = self.normalize_numbers(protected)
        protected = self.normalize_case(protected)
        protected = self.normalize_abbreviations(protected)
        protected = self.normalize_whitespace(protected)
        if _underthesea_text_normalize is not None:
            protected = _underthesea_text_normalize(protected)
            protected = self.normalize_whitespace(protected)

        # Restore protected tokens at the end so nothing mutates them.
        for key, val in protected_tokens.items():
            protected = protected.replace(key, val)
        return protected
