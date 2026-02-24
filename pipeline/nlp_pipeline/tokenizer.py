"""
Module 4: Tokenizer

Vietnamese word segmentation wrapper supporting:
- py_vncorenlp (recommended, most accurate, no server required)
- underthesea (fast, lightweight)
- whitespace fallback

Designed for robust production use.
Placeholders like <url>, <mention>, <hashtag>, <email>, <date>, <NUM>
are protected from being split during tokenization.
"""

import re
from typing import Optional

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
# and emoji tokens like :cười_ra_nước_mắt:
_PLACEHOLDER_RE = re.compile(
    r"<(?:url|mention|hashtag|email|date|NUM)>|:[a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+:",
    re.UNICODE,
)


class Tokenizer:
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

        # =========================================================
        # VNCORENLP BACKEND (BEST ACCURACY)
        # =========================================================
        if self.backend_name == "vncorenlp":

            if not HAS_VNCORENLP:
                print(
                    "[Tokenizer] py_vncorenlp not installed. "
                    "Install with: pip install py_vncorenlp"
                )
                self.backend_name = "underthesea"

            elif not vncorenlp_dir:
                print(
                    "[Tokenizer] vncorenlp_dir not provided. "
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

                    print(f"[Tokenizer] Using py_vncorenlp at: {vncorenlp_dir}")

                except Exception as e:
                    print(f"[Tokenizer] Failed to load VnCoreNLP: {e}")
                    print("[Tokenizer] Falling back to underthesea")
                    self.backend_name = "underthesea"

        # =========================================================
        # UNDERTHESEA BACKEND
        # =========================================================
        if self.backend_name == "underthesea":

            if not HAS_UNDERTHESEA:
                print(
                    "[Tokenizer] underthesea not installed. "
                    "Install with: pip install underthesea"
                )
                self.backend_name = "whitespace"
            else:
                print("[Tokenizer] Using underthesea backend")

        # =========================================================
        # WHITESPACE FALLBACK
        # =========================================================
        if self.backend_name == "whitespace":
            print("[Tokenizer] Using whitespace tokenizer")

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
    # TOKENIZE FUNCTION
    # =============================================================
    def tokenize(self, text: str) -> str:
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
            # Protect placeholders before tokenization
            protected, store = self._protect_placeholders(text)

            # ---------- VnCoreNLP ----------
            if self.backend_name == "vncorenlp" and self.vncorenlp_model:
                result = self.vncorenlp_model.annotate_text(protected)
                tokens = []

                # result is dict: {sent_id: [word_info, ...]}
                for sentence in result.values():
                    for word_info in sentence:
                        tokens.append(word_info["wordForm"])
                tokenized = " ".join(tokens)

            # ---------- Underthesea ----------
            elif self.backend_name == "underthesea":
                tokenized = _underthesea_tokenize(protected, format="text")

            # ---------- Whitespace ----------
            else:
                tokenized = protected

            # Restore placeholders
            result = self._restore_placeholders(tokenized, store)
            # Clean up VnCoreNLP's internal XPHX number markers (e.g., "XPHX 20 XPHX" → "<NUM>")
            result = re.sub(r"XPHX\s*\d+\s*XPHX", "<NUM>", result)
            return result

        except Exception as e:
            print(f"[Tokenizer] Tokenization error: {e}")
            return text

    # =============================================================
    # UTILITIES
    # =============================================================
    @property
    def backend(self) -> str:
        """Return active backend name."""
        return self.backend_name

    def __repr__(self):
        return f"Tokenizer(backend='{self.backend_name}')"
