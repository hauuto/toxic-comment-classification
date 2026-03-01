"""Lightweight worker for pandarallel – does NOT import WordSegmentor / py_vncorenlp.

Each spawned child process imports only this module, which pulls in Decoder,
Filter, Normalizer, and VietnameseNormalizer (pure-Python, no JVM).
"""

import os
import re

from .config import EMOJI_MAPPING_PATH, ABBREVIATIONS_PATH
from .decoder import Decoder
from .filter import Filter
from .normalizer import Normalizer
from .vietnamese_typing_normalizer import VietnameseNormalizer


_CANON_PLACEHOLDER_RE = re.compile(
    r"<\s*(url|mention|hashtag|email|date|time|num|ip)\s*>",
    re.IGNORECASE,
)


def _canonicalize_placeholders(text: str) -> str:
    if not text:
        return text
    return _CANON_PLACEHOLDER_RE.sub(lambda m: f"<{m.group(1).upper()}>", text)


# ── Per-process singleton (lazily created once per child process) ──
_decoder: Decoder | None = None
_filter: Filter | None = None
_normalizer: Normalizer | None = None
_vn_normalizer: VietnameseNormalizer | None = None


def _ensure_components():
    global _decoder, _filter, _normalizer, _vn_normalizer
    if _decoder is not None:
        return

    emoji_path = EMOJI_MAPPING_PATH
    abbrev_path = ABBREVIATIONS_PATH
    if not os.path.exists(emoji_path):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        emoji_path = os.path.join(current_dir, "mappings", "emoji_vi.json")
        abbrev_path = os.path.join(current_dir, "mappings", "abbreviations.json")

    _decoder = Decoder(emoji_path)
    _filter = Filter()
    _normalizer = Normalizer(abbrev_path)
    _vn_normalizer = VietnameseNormalizer()


def preprocess_no_segment(
    text: str,
    use_decoder: bool = True,
    use_filter: bool = True,
    use_normalizer: bool = True,
) -> str | None:
    """Run Decode → Filter → Normalize. Return cleaned text or None if invalid.

    This is the function passed to ``parallel_apply``.  It is deliberately
    simple (returns ``str | None``) so pandas can build the result Series
    directly without dict overhead.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    _ensure_components()

    current_text = text

    # 1. Decode
    if use_decoder:
        try:
            current_text = _decoder.decode(current_text)
        except Exception:
            return None

    # 2. Filter
    if use_filter:
        keep, _ = _filter.filter_comment(current_text, raw_text=text)
        if not keep:
            return None

    # 3. Normalize
    if use_normalizer:
        try:
            current_text = _normalizer.normalize(current_text)
        except Exception:
            return None

        if not current_text.strip():
            return None

        # 3.5 Vietnamese tone placement
        try:
            current_text = _vn_normalizer.normalize(current_text)
        except Exception:
            pass

    # Canonicalize placeholders
    current_text = _canonicalize_placeholders(current_text)

    return current_text
