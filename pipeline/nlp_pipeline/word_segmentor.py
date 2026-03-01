"""
Module 4: WordSegmentor

Vietnamese word segmentation wrapper supporting:
- py_vncorenlp (recommended, most accurate, no server required)
- underthesea (fast, lightweight)
- whitespace fallback

Designed for robust production use.
Placeholders like <url>, <mention>, <hashtag>, <email>, <date>, <NUM>
are protected from being split during segmentation.
"""

import os
import re
import threading
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from .config import VNCORENLP_MAX_CHAR_LENGTH

# One persistent thread for VnCoreNLP calls (JVM is not thread-safe anyway)
_vncore_executor = ThreadPoolExecutor(max_workers=1)

# ===== Try importing underthesea =====
try:
    from underthesea import word_tokenize as _underthesea_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False

# ===== Try importing py_vncorenlp =====
try:
    import py_vncorenlp
    HAS_VNCORENLP = True
except ImportError:
    HAS_VNCORENLP = False


# ============================================================================
#  Shared VnCoreNLP model cache (heavy JVM init) – reuse across the app
# ============================================================================

_vncore_model_lock = threading.Lock()
_vncore_model_cache: dict[str, Any] = {}


def _get_or_create_vncorenlp_model(vncorenlp_dir: str, auto_download: bool) -> Any:
    """Return a cached py_vncorenlp.VnCoreNLP instance for *vncorenlp_dir*."""
    key = os.path.abspath(vncorenlp_dir)
    with _vncore_model_lock:
        if key in _vncore_model_cache:
            return _vncore_model_cache[key]

        # Create under the same lock to ensure single JVM init.
        if auto_download:
            try:
                py_vncorenlp.download_model(save_dir=vncorenlp_dir)
            except Exception:
                pass

        model = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_dir)
        _vncore_model_cache[key] = model
        return model


# Regex to canonicalize spaced emoji tokens: ": tên_liền :" → ":tên_liền:"
# Must run BEFORE placeholder protection so _PLACEHOLDER_RE can match them.
_EMOJI_CANON_RE = re.compile(
    r":\s*([a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+"
    r"(?:\s*_\s*[a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+)*)\s*:",
    re.UNICODE | re.IGNORECASE,
)

# Placeholder pattern: matches placeholder tags like <url>, <mention>, <num>, <ip>...
# and emoji tokens like :cười_ra_nước_mắt: (NO spaces around colons),
# as well as English contractions like don't, we're.
_PLACEHOLDER_RE = re.compile(
    r"<(?:url|mention|hashtag|email|date|time|num|ip)>|"
    r":[a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+:|"
    r"(?<!\w)[a-zA-Z]+[''][a-zA-Z]+(?!\w)",
    re.UNICODE | re.IGNORECASE,
)


class WordSegmentor:
    """Vietnamese word segmentation with multiple backend support."""

    def __init__(
        self,
        backend: str = "underthesea",
        vncorenlp_dir: Optional[str] = None,
        vncorenlp_model: Any = None,
        auto_download: bool = True,
    ):
        """
        Args:
            backend:
                - "vncorenlp"  → most accurate
                - "underthesea" → default, fast
                - "whitespace" → fallback
            vncorenlp_dir:
                Directory containing VnCoreNLP model files
            auto_download:
                Automatically download VnCoreNLP if missing
        """

        self.backend_name = backend.lower()
        self.vncorenlp_model = vncorenlp_model
        self._vncore_timeout_streak = 0   # consecutive timeouts; triggers fallback after 3
        self._vncore_dead = False          # set True when JVM is considered unresponsive

        # =========================================================
        # VNCORENLP BACKEND (BEST ACCURACY)
        # =========================================================
        if self.backend_name == "vncorenlp":

            # If a model is injected, trust it and skip heavy init.
            if self.vncorenlp_model is not None:
                print("[WordSegmentor] Using injected VnCoreNLP model")
            
            elif not HAS_VNCORENLP:
                print(
                    "[WordSegmentor] py_vncorenlp not installed. "
                    "Install with: pip install py_vncorenlp"
                )
                self.backend_name = "underthesea"

            elif not vncorenlp_dir:
                print(
                    "[WordSegmentor] vncorenlp_dir not provided. "
                    "Falling back to underthesea."
                )
                self.backend_name = "underthesea"

            else:
                try:
                    self.vncorenlp_model = _get_or_create_vncorenlp_model(
                        vncorenlp_dir=vncorenlp_dir,
                        auto_download=auto_download,
                    )

                    print(f"[WordSegmentor] Using py_vncorenlp at: {vncorenlp_dir}")

                except Exception as e:
                    print(f"[WordSegmentor] Failed to load VnCoreNLP: {e}")
                    print("[WordSegmentor] Falling back to underthesea")
                    self.backend_name = "underthesea"

        # =========================================================
        # UNDERTHESEA BACKEND
        # =========================================================
        if self.backend_name == "underthesea":

            if not HAS_UNDERTHESEA:
                print(
                    "[WordSegmentor] underthesea not installed. "
                    "Install with: pip install underthesea"
                )
                self.backend_name = "whitespace"
            else:
                print("[WordSegmentor] Using underthesea backend")

        # =========================================================
        # WHITESPACE FALLBACK
        # =========================================================
        if self.backend_name == "whitespace":
            print("[WordSegmentor] Using whitespace segmentor")

    # =============================================================
    # PLACEHOLDER PROTECTION
    # =============================================================
    def _protect_placeholders(self, text: str):
        """
        Extract placeholders from text, replace with safe markers.
        Returns (protected_text, {marker: original_placeholder}).

        Steps:
          1. Canonicalize spaced emoji tokens: ": tên :" → ":tên:"
          2. Protect canonical tokens (:tên:, <URL>, contractions) with markers.
        """
        store = {}
        counter = [0]

        def _replace(m):
            marker = f"XPHX{counter[0]}XPHX"
            store[marker] = m.group()
            counter[0] += 1
            return marker

        # Step 1: canonicalize ": name :" → ":name:"
        def _canon_emoji(m):
            name = m.group(1)
            name = re.sub(r"\s*_\s*", "_", name.strip())
            return f":{name}:"

        canonicalized = _EMOJI_CANON_RE.sub(_canon_emoji, text)
        # Multi-pass for shared colon boundaries: ": a : b :" → ":a: b :" → ":a: :b:"
        for _ in range(5):
            new = _EMOJI_CANON_RE.sub(_canon_emoji, canonicalized)
            if new == canonicalized:
                break
            canonicalized = new

        # Step 2: protect canonical tokens
        protected = _PLACEHOLDER_RE.sub(_replace, canonicalized)
        return protected, store

    def _restore_placeholders(self, text: str, store: dict) -> str:
        """Restore original placeholders from markers."""
        for marker, original in store.items():
            text = text.replace(marker, original)
        return text

    # =============================================================
    # SEGMENT FUNCTION
    # =============================================================
    def segment(self, text: str) -> str:
        """
        Segment Vietnamese text into words.

        Returns:
            String with compound words joined by underscores.
            Example:
                "hôm nay trời đẹp"
                → "hôm_nay trời đẹp"

        Placeholders like <url>, <mention>, <email>, etc. are preserved intact.
        """

        if not text or not text.strip():
            return text

        try:
            # Protect placeholders before segmentation
            protected, store = self._protect_placeholders(text)

            # ---------- VnCoreNLP ----------
            if self.backend_name == "vncorenlp" and self.vncorenlp_model:
                # Guard: skip VnCoreNLP for very long texts to prevent JVM crash
                if len(text) > VNCORENLP_MAX_CHAR_LENGTH:
                    print(f"[WordSegmentor] Skipping VnCoreNLP segmentation: "
                          f"text too long ({len(text)} chars > {VNCORENLP_MAX_CHAR_LENGTH})")
                    return text

                _model = self.vncorenlp_model
                _text  = protected
                if self._vncore_dead:
                    # JVM is unresponsive — skip segmentation to keep pipeline moving
                    segmented = protected
                else:
                    future = _vncore_executor.submit(_model.annotate_text, _text)
                    try:
                        result = future.result(timeout=30)   # 30s per-row hard limit
                        self._vncore_timeout_streak = 0      # reset on success
                    except _FuturesTimeout:
                        self._vncore_timeout_streak += 1
                        print(f"[WordSegmentor] VnCoreNLP timeout #{self._vncore_timeout_streak} "
                              f"(len={len(text)})")
                        if self._vncore_timeout_streak >= 3:
                            self._vncore_dead = True
                            print("[WordSegmentor] VnCoreNLP appears dead — "
                                  "falling back to whitespace for remaining rows")
                        return None
                    tokens = []

                    # result is dict: {sent_id: [word_info, ...]}
                    for sentence in result.values():
                        for word_info in sentence:
                            tokens.append(word_info["wordForm"])
                    segmented = " ".join(tokens)

            # ---------- Underthesea ----------
            elif self.backend_name == "underthesea":
                segmented = _underthesea_tokenize(protected, format="text")

            # ---------- Whitespace ----------
            else:
                segmented = protected

            # Restore placeholders
            result = self._restore_placeholders(segmented, store)
            # Clean up VnCoreNLP's internal XPHX number markers (e.g., "XPHX 20 XPHX" → "<NUM>")
            result = re.sub(r"XPHX\s*\d+\s*XPHX", "<NUM>", result)
            return result

        except Exception as e:
            err = str(e)
            # ArrayIndexOutOfBounds: text already has underscores from prior segmentation
            if "ArrayIndexOutOfBoundsException" in err:
                return text
            # Java heap space → text too long, silently discard
            if "OutOfMemoryError" in err or "heap space" in err or "JVM exception" in err:
                return None
            print(f"[WordSegmentor] Segmentation error: {e}")
            return text

    # =============================================================
    # UTILITIES
    # =============================================================
    @property
    def backend(self) -> str:
        """Return active backend name."""
        return self.backend_name

    def __repr__(self):
        return f"WordSegmentor(backend='{self.backend_name}')"


# Backward-compatible alias
Tokenizer = WordSegmentor
