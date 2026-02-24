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

import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout

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


# Placeholder pattern: matches <url>, <mention>, <hashtag>, <email>, <date>, <NUM>
# and emoji tokens like :cười_ra_nước_mắt:, as well as English contractions like don't, we're
_PLACEHOLDER_RE = re.compile(
    r"<(?:url|mention|hashtag|email|date|NUM)>|"
    r":[a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+:|"
    r"(?<!\w)[a-zA-Z]+['’][a-zA-Z]+(?!\w)",
    re.UNICODE,
)


class WordSegmentor:
    """Vietnamese word segmentation with multiple backend support."""

    def __init__(
        self,
        backend: str = "underthesea",
        vncorenlp_dir: Optional[str] = None,
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
        self.vncorenlp_model = None
        self._vncore_timeout_streak = 0   # consecutive timeouts; triggers fallback after 3
        self._vncore_dead = False          # set True when JVM is considered unresponsive

        # =========================================================
        # VNCORENLP BACKEND (BEST ACCURACY)
        # =========================================================
        if self.backend_name == "vncorenlp":

            if not HAS_VNCORENLP:
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
                    # Auto-download model if requested
                    if auto_download:
                        try:
                            py_vncorenlp.download_model(save_dir=vncorenlp_dir)
                        except Exception:
                            # Already exists → ignore
                            pass

                    self.vncorenlp_model = py_vncorenlp.VnCoreNLP(
                        save_dir=vncorenlp_dir
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
        """
        store = {}
        counter = [0]

        def _replace(m):
            marker = f"XPHX{counter[0]}XPHX"
            store[marker] = m.group()
            counter[0] += 1
            return marker

        protected = _PLACEHOLDER_RE.sub(_replace, text)
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
