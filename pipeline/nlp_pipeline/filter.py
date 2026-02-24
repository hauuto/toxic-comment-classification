"""
Module 2: Filter
Determines whether a comment should be kept or discarded based on content quality.
- Too short comments (< 20 chars BEFORE normalize)
- Spam character repetition
- Non-Latin/Vietnamese script (Chinese, Arabic, Korean, etc.)
- More than 80% English words
- Emoji-only comments
- Empty comments
"""
import re
import unicodedata
from .config import (
    MIN_CHAR_LENGTH,
    MIN_VIETNAMESE_RATIO,
    MAX_SPAM_REPEAT,
    VIETNAMESE_CHARS,
    ADMIN_JUNK_PATTERNS,
)

# Pure ASCII letters that are common English-only (not used in Vietnamese)
_ASCII_LETTERS = re.compile(r"[a-zA-Z]+")
# Vietnamese-specific diacritical characters (quick check set from config)
_PLACEHOLDER_RE = re.compile(r"<(?:url|mention|hashtag|email|date|NUM|IP)>|:[a-z_]+:")


class Filter:
    """Filter out noisy or uninformative comments."""

    def __init__(
        self,
        min_char_length: int = MIN_CHAR_LENGTH,
        min_viet_ratio: float = MIN_VIETNAMESE_RATIO,
        max_spam_repeat: int = MAX_SPAM_REPEAT,
        min_raw_chars: int = 20,
        max_english_ratio: float = 0.80,
    ):
        self.min_char_length = min_char_length
        self.min_viet_ratio = min_viet_ratio
        self.max_spam_repeat = max_spam_repeat
        self.min_raw_chars = min_raw_chars
        self.max_english_ratio = max_english_ratio

        # Pattern for spam: same char repeated excessively (e.g., aaaaaaaaaa)
        self._spam_pattern = re.compile(r"(.)\1{" + str(max_spam_repeat) + r",}")

        # Pattern matching :token: emoji placeholders and <placeholder> tokens
        self._token_pattern = re.compile(r":[a-z_]+:")

        # VOZ admin/mod junk pattern (ban notices, system msgs, sent-from footers)
        self._admin_junk_pattern = re.compile(
            "|".join(ADMIN_JUNK_PATTERNS), re.IGNORECASE | re.MULTILINE
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def is_empty(self, text: str) -> bool:
        """Empty or whitespace-only."""
        return not text or not text.strip()

    def is_too_short_raw(self, text: str) -> bool:
        """< 20 non-whitespace chars BEFORE normalize (raw decoded text)."""
        return len(text.replace(" ", "").replace("\n", "").replace("\t", "")) < self.min_raw_chars

    def is_too_short(self, text: str) -> bool:
        """Check if comment is below minimum character length (post-processing)."""
        cleaned = self._token_pattern.sub("", text).strip()
        return len(cleaned) < self.min_char_length

    def is_spam_chars(self, text: str) -> bool:
        """Check if comment is primarily spam character repetition."""
        matches = self._spam_pattern.findall(text)
        if not matches:
            return False
        spam_len = sum(m.end() - m.start() for m in self._spam_pattern.finditer(text))
        return spam_len > len(text) * 0.5

    def is_non_latin_script(self, text: str) -> bool:
        """
        True if >40% of alphabetic characters come from non-Latin scripts
        (Chinese, Japanese, Korean, Arabic, Cyrillic, etc.).
        Vietnamese uses Latin Extended, so it's fine.
        """
        # Remove placeholders first
        cleaned = _PLACEHOLDER_RE.sub("", text)
        alpha_chars = [c for c in cleaned if c.isalpha()]
        if len(alpha_chars) < 5:
            return False
        non_latin = sum(
            1 for c in alpha_chars
            if unicodedata.category(c).startswith("L")
            and not (
                "\u0000" <= c <= "\u024F"  # Basic Latin + Latin Extended A/B
                or "\u1E00" <= c <= "\u1EFF"  # Latin Extended Additional (Vietnamese)
                or "\u0300" <= c <= "\u036F"  # Combining diacritics
            )
        )
        return non_latin / len(alpha_chars) > 0.40

    def is_mostly_english(self, text: str) -> bool:
        """
        True if >80% of words are purely ASCII-letter (English-style) AND
        the text has NO Vietnamese diacritical characters.
        This avoids filtering Vietnamese text that mixes in English brand names.
        """
        # If there are Vietnamese-specific chars, keep it
        if any(c in VIETNAMESE_CHARS for c in text):
            return False
        # Remove placeholders
        cleaned = _PLACEHOLDER_RE.sub("", text)
        words = cleaned.split()
        if len(words) < 4:
            return False
        ascii_words = sum(1 for w in words if _ASCII_LETTERS.fullmatch(w.strip(".,!?\"'()-")))
        return ascii_words / len(words) > self.max_english_ratio

    def has_enough_vietnamese(self, text: str) -> bool:
        """Check if comment has sufficient Vietnamese characters."""
        cleaned = self._token_pattern.sub("", text)
        cleaned = re.sub(r"<(?:url|mention|hashtag|date|NUM|IP)>", "", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)

        if len(cleaned) == 0:
            return False

        alpha_chars = sum(1 for c in cleaned if c.isalpha())
        viet_chars = sum(1 for c in cleaned if c in VIETNAMESE_CHARS)

        if alpha_chars == 0:
            return False

        if viet_chars > 0:
            return True

        return alpha_chars / max(len(cleaned), 1) >= self.min_viet_ratio

    def is_only_emoji(self, text: str) -> bool:
        """Check if comment contains only emoji tokens and whitespace."""
        cleaned = self._token_pattern.sub("", text).strip()
        return len(cleaned) == 0 or all(not c.isalnum() for c in cleaned)

    def is_admin_junk(self, text: str) -> bool:
        """Check if comment is VOZ admin/mod action or system metadata."""
        return bool(self._admin_junk_pattern.search(text))

    # ------------------------------------------------------------------
    # Main filter entry point
    # ------------------------------------------------------------------

    def filter_comment(self, text: str, raw_text: str | None = None) -> tuple:
        """
        Determine if a comment should be kept.

        Args:
            text: The decoded text (after decoder, before normalize).
            raw_text: Original raw text (optional, used for raw char count).

        Returns:
            (keep: bool, reason: str)
        """
        if self.is_empty(text):
            return False, "empty"

        # Admin junk check (VOZ mod actions, ban notices, sent-from footers)
        if self.is_admin_junk(text):
            return False, "admin_junk"

        # Raw length check (before normalize) — uses text if raw_text not given
        check_raw = raw_text if raw_text is not None else text
        if self.is_too_short_raw(check_raw):
            return False, "too_short_raw"

        if self.is_only_emoji(text):
            return False, "only_emoji"

        if self.is_spam_chars(text):
            return False, "spam_chars"

        if self.is_non_latin_script(text):
            return False, "non_latin_script"

        if self.is_mostly_english(text):
            return False, "mostly_english"

        if not self.has_enough_vietnamese(text):
            return False, "insufficient_vietnamese"

        return True, "ok"
