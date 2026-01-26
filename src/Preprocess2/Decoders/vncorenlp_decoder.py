import os
import sys
import re
import json
from typing import Dict
import py_vncorenlp
from Normalizers.elongation_normalizer import ElongationNormalizer
from Normalizers.icon_normalizer import IconNormalizer


# ======================================================
# VNCORE NLP DECODER
# ======================================================

class VnCoreNLPDecoder:
    def __init__(self, model_dir: str, icon_json: str, java_home: str = None):
        """
        :param model_dir: thư mục chứa VnCoreNLP (.jar + models)
        :param icon_json: đường dẫn icons.json
        :param java_home: JAVA_HOME nếu chưa set
        """

        # ---------------- JAVA SETUP ----------------
        if java_home:
            os.environ["JAVA_HOME"] = java_home
            if sys.platform == "win32":
                os.environ["PATH"] = f"{java_home}\\bin;" + os.environ["PATH"]

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Không tìm thấy VnCoreNLP tại {model_dir}")

        try:
            self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg", "pos"],
                save_dir=model_dir,
                max_heap_size="-Xmx2g",
            )
        except Exception as e:
            raise RuntimeError(f"Lỗi khởi tạo VnCoreNLP: {e}")

        # ---------------- NORMALIZERS ----------------
        self.icon_normalizer = IconNormalizer(icon_json)

        # Icon / emoji / emoticon (để tách placeholder)
        self.icon_pattern = re.compile(
            r"(<\/?3+|[:=]\)+|[^\w\s,.<>?/;:\"'\[\]{}\\|`~!@#$%^&*()\-=_+])"
        )

    # ==================================================
    # SEGMENT TEXT
    # ==================================================
    def segment_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        # ---------- 0. TYPO + ELONGATION ----------
        text = self.typing_normalizer.replace(text)
        text = ElongationNormalizer.normalize(text, max_repeat=1)

        # ---------- 1. ICON → PLACEHOLDER ----------
        icons = []

        def _icon_repl(m):
            icons.append(m.group())
            return f" ICONPLACEHOLDER{len(icons)-1} "

        tmp_text = self.icon_pattern.sub(_icon_repl, text)

        # ---------- 2. VNCORE SEGMENT ----------
        try:
            segmented = self.rdrsegmenter.word_segment(tmp_text)
        except Exception as e:
            print(f"[WARN] Segment error: {e}")
            return text

        out = " ".join(segmented)

        # ---------- 3. RESTORE ICON ----------
        for i, icon in enumerate(icons):
            out = out.replace(f"ICONPLACEHOLDER{i}", icon)

        # ---------- 4. ENGLISH CONTRACTION ----------
        out = re.sub(r"(\w+)\s+'\s+(\w+)", r"\1'\2", out)

        # ---------- 5. NORMALIZE ICON ----------
        out = self.icon_normalizer.normalize(out)

        # ---------- 6. DASH WORD ----------
        out = re.sub(r"(\w)\s*-\s*(\w)", r"\1-\2", out)

        # ---------- 7. SPLIT PUNCTUATION ----------
        out = re.sub(r"([,.:?!])", r" \1 ", out)
        out = re.sub(r"\s+", " ", out).strip()

        return out


    # ==================================================
    # ANNOTATE
    # ==================================================
    def annotate_text(self, text: str) -> Dict:
        if not text:
            return {}
        return self.rdrsegmenter.annotate_text(text)
