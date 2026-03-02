"""
Toxic Comment Classifier – GUI Prediction Tool (63.11% Model)
Uses the trained C-LSTM model with the full Vietnamese NLP preprocessing pipeline.
Supports both CPU and GPU inference.
Toxic threshold: 0.38 (tuned for this model checkpoint).
"""

import os
import sys
import json
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import customtkinter as ctk
from tkinter import messagebox

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(os.path.dirname(MODEL_DIR), "pipeline")

# Add pipeline directory to sys.path so we can import nlp_pipeline
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

# Model asset paths
MODEL_PATH = os.path.join(MODEL_DIR, "best_c_lstm_model.pth")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
W2V_PATH = os.path.join(MODEL_DIR, "custom_word2vec.npy")

# ---------------------------------------------------------------------------
# Toxic threshold (tuned for 63.11% model)
# ---------------------------------------------------------------------------
TOXIC_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Device selection – works on CPU, CUDA, and MPS (Apple Silicon)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# C-LSTM Model (must match training definition exactly)
# ---------------------------------------------------------------------------
class C_LSTM_Model(nn.Module):
    def __init__(self, vocab_size, w2v_path):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 300, padding_idx=0)
        self.emb.weight.data.copy_(torch.from_numpy(np.load(w2v_path)))
        self.lstm = nn.LSTM(300, 128, batch_first=True, bidirectional=True)
        self.convs = nn.ModuleList([nn.Conv1d(256, 100, k) for k in [3, 4, 5]])
        self.drop = nn.Dropout(0.5)
        self.h_tox = nn.Linear(300, 1)
        self.h_sent = nn.Linear(300, 3)
        self.h_tdet = nn.Linear(300, 3)

    def forward(self, x):
        lstm_out, _ = self.lstm(self.emb(x))
        cnn_in = lstm_out.permute(0, 2, 1)
        feat = self.drop(
            torch.cat(
                [
                    F.max_pool1d(F.relu(c(cnn_in)), c(cnn_in).shape[2]).squeeze(2)
                    for c in self.convs
                ],
                1,
            )
        )
        return self.h_tox(feat).squeeze(1), self.h_sent(feat), self.h_tdet(feat)


# ---------------------------------------------------------------------------
# Predictor – wraps model + preprocessing
# ---------------------------------------------------------------------------
class ToxicCommentPredictor:
    """Loads the C-LSTM model and the NLP preprocessing pipeline."""

    def __init__(self):
        # 1. Load vocabulary
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            self.vocab: dict[str, int] = json.load(f)

        # 2. Build model and load checkpoint (map to current device)
        self.model = C_LSTM_Model(len(self.vocab), W2V_PATH).to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # 3. Initialize the NLP preprocessing pipeline
        self.preprocessor = self._init_preprocessor()

    # ----- helpers --------------------------------------------------------

    @staticmethod
    def _init_preprocessor():
        """Try to import and initialize VietnameseCommentPreprocessor.

        Falls back gracefully if heavy backends (VnCoreNLP, underthesea)
        are not available.
        """
        try:
            from nlp_pipeline import VietnameseCommentPreprocessor

            # Try VnCoreNLP first, then underthesea, then whitespace
            for backend in ("vncorenlp", "underthesea", "whitespace"):
                try:
                    pre = VietnameseCommentPreprocessor(segmentor_backend=backend)
                    return pre
                except Exception:
                    continue
        except ImportError:
            pass
        return None

    def preprocess(self, raw_text: str) -> str:
        """Run the NLP pipeline (decode → normalize → segment).

        If the pipeline is unavailable, fall back to simple lowercasing + split.
        """
        if self.preprocessor is not None:
            try:
                result = self.preprocessor.process_comment(
                    raw_text,
                    use_decoder=True,
                    use_filter=False,  # We never filter user input in the GUI
                    use_normalizer=True,
                    use_segmentor=True,
                )
                cleaned = result.get("cleaned_text", "")
                if cleaned.strip():
                    return cleaned
            except Exception:
                pass
        # Fallback: lowercase only
        return raw_text.lower()

    def encode(self, text: str) -> torch.Tensor:
        """Convert preprocessed text to padded integer sequence tensor."""
        words = text.split()
        seq = [self.vocab.get(w, self.vocab.get("<UNK>", 1)) for w in words]
        seq = (seq[:50] + [self.vocab.get("<PAD>", 0)] * 50)[:50]
        return torch.tensor([seq], dtype=torch.long).to(DEVICE)

    def predict(self, raw_text: str) -> dict:
        """Run full prediction pipeline and return structured results."""
        cleaned = self.preprocess(raw_text)
        input_tensor = self.encode(cleaned)

        with torch.no_grad():
            l_tox, l_sent, l_tdet = self.model(input_tensor)

            prob_tox = torch.sigmoid(l_tox).item()
            is_toxic = prob_tox >= TOXIC_THRESHOLD

            # Tier 1
            tier1 = "Toxic" if is_toxic else "Clean"
            tier1_conf = prob_tox if is_toxic else (1 - prob_tox)

            # Tier 2A – Sentiment (only meaningful for Clean)
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            sent_probs = torch.softmax(l_sent, dim=1)[0].cpu().numpy()
            if not is_toxic:
                sent_idx = int(np.argmax(sent_probs))
                tier2a = sentiment_map[sent_idx]
                tier2a_conf = float(sent_probs[sent_idx])
            else:
                tier2a = "N/A (Toxic)"
                tier2a_conf = None

            # Tier 2B – Toxic details (only meaningful for Toxic)
            tdet_probs = torch.sigmoid(l_tdet)[0].cpu().numpy()
            tier2b_labels = []
            if is_toxic:
                if tdet_probs[0] >= 0.5:
                    tier2b_labels.append(f"Harassment ({tdet_probs[0]:.0%})")
                if tdet_probs[1] >= 0.5:
                    tier2b_labels.append(f"Obscene ({tdet_probs[1]:.0%})")
                if tdet_probs[2] >= 0.4:  # Lower threshold for Hate Speech
                    tier2b_labels.append(f"Hate Speech ({tdet_probs[2]:.0%})")
                if not tier2b_labels:
                    tier2b_labels.append("Toxic (không rõ chi tiết)")
            else:
                tier2b_labels = ["N/A"]

        return {
            "cleaned_text": cleaned,
            "tier1": tier1,
            "tier1_conf": tier1_conf,
            "tier2a": tier2a,
            "tier2a_conf": tier2a_conf,
            "tier2b": tier2b_labels,
            "prob_toxic": prob_tox,
            "sent_probs": {sentiment_map[i]: float(sent_probs[i]) for i in range(3)},
            "tdet_probs": {
                "Harassment": float(tdet_probs[0]),
                "Obscene": float(tdet_probs[1]),
                "Hate Speech": float(tdet_probs[2]),
            },
        }


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class PredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Toxic Comment Classifier – Dự đoán bình luận độc hại (63.11%)")
        self.geometry("900x720")
        self.minsize(750, 600)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        self.predictor: ToxicCommentPredictor | None = None
        self._loading = False

        # ---- Header ----
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="🔍  Toxic Comment Classifier (63.11%)",
            font=("Arial", 20, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(10, 0))

        self.device_label = ctk.CTkLabel(
            header,
            text=f"Device: {DEVICE}  |  Threshold: {TOXIC_THRESHOLD}  |  Đang tải mô hình...",
            text_color="gray",
            font=("Arial", 12),
        )
        self.device_label.grid(row=1, column=0, sticky="w", padx=15, pady=(2, 10))

        # ---- Input area ----
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=1, column=0, padx=15, pady=5, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(input_frame, text="Nhập bình luận:", font=("Arial", 13, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 0)
        )

        self.input_textbox = ctk.CTkTextbox(input_frame, height=90, font=("Arial", 13))
        self.input_textbox.grid(row=1, column=0, sticky="ew", padx=10, pady=(5, 10))
        self.input_textbox.bind("<Control-Return>", lambda e: self._on_predict())

        # ---- Buttons ----
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=15, pady=5, sticky="ew")

        self.predict_btn = ctk.CTkButton(
            btn_frame, text="Dự đoán  (Ctrl+Enter)", command=self._on_predict,
            font=("Arial", 13, "bold"), height=36, width=200,
        )
        self.predict_btn.pack(side="left")

        self.clear_btn = ctk.CTkButton(
            btn_frame, text="Xoá", command=self._on_clear,
            font=("Arial", 13), height=36, width=80, fg_color="gray",
        )
        self.clear_btn.pack(side="left", padx=(10, 0))

        self.status_label = ctk.CTkLabel(btn_frame, text="", text_color="gray", font=("Arial", 12))
        self.status_label.pack(side="left", padx=(15, 0))

        # ---- Results area ----
        result_frame = ctk.CTkFrame(self)
        result_frame.grid(row=3, column=0, padx=15, pady=(5, 15), sticky="nsew")
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(result_frame, text="Kết quả:", font=("Arial", 13, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 0)
        )

        self.result_textbox = ctk.CTkTextbox(
            result_frame, font=("Consolas", 13), state="disabled", wrap="word",
        )
        self.result_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))

        # ---- Batch section (expandable) ----
        batch_frame = ctk.CTkFrame(self)
        batch_frame.grid(row=4, column=0, padx=15, pady=(0, 15), sticky="ew")
        batch_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            batch_frame,
            text="Dự đoán hàng loạt (mỗi dòng 1 bình luận):",
            font=("Arial", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))

        self.batch_input = ctk.CTkTextbox(batch_frame, height=80, font=("Arial", 12))
        self.batch_input.grid(row=1, column=0, sticky="ew", padx=10, pady=(5, 5))

        batch_btn_row = ctk.CTkFrame(batch_frame, fg_color="transparent")
        batch_btn_row.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 10))

        self.batch_btn = ctk.CTkButton(
            batch_btn_row, text="Dự đoán hàng loạt", command=self._on_batch_predict,
            font=("Arial", 12), height=32, width=180,
        )
        self.batch_btn.pack(side="left")

        self.batch_status = ctk.CTkLabel(batch_btn_row, text="", text_color="gray", font=("Arial", 11))
        self.batch_status.pack(side="left", padx=(10, 0))

        # ---- Load model in background ----
        self.predict_btn.configure(state="disabled")
        self.batch_btn.configure(state="disabled")
        threading.Thread(target=self._load_model, daemon=True).start()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            predictor = ToxicCommentPredictor()
            self.predictor = predictor
            backend = "N/A"
            if predictor.preprocessor is not None:
                backend = getattr(predictor.preprocessor, "segmentor_backend", "unknown")
            self.after(0, lambda: self._on_model_loaded(backend))
        except Exception as e:
            self.after(0, lambda: self._on_model_error(str(e)))

    def _on_model_loaded(self, backend: str):
        self.device_label.configure(
            text=f"Device: {DEVICE}  |  Threshold: {TOXIC_THRESHOLD}  |  Segmentor: {backend}  |  ✅ Sẵn sàng",
            text_color="green",
        )
        self.predict_btn.configure(state="normal")
        self.batch_btn.configure(state="normal")
        self.status_label.configure(text="")

    def _on_model_error(self, msg: str):
        self.device_label.configure(text=f"❌ Lỗi tải mô hình: {msg}", text_color="red")
        messagebox.showerror("Lỗi", f"Không thể tải mô hình:\n{msg}")

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------
    def _on_predict(self):
        if self.predictor is None:
            messagebox.showinfo("Chờ", "Mô hình đang được tải, vui lòng đợi...")
            return
        raw = self.input_textbox.get("0.0", "end").strip()
        if not raw:
            messagebox.showinfo("Thông báo", "Vui lòng nhập bình luận để dự đoán.")
            return

        self.predict_btn.configure(state="disabled")
        self.status_label.configure(text="Đang xử lý...", text_color="orange")

        def _work():
            result = self.predictor.predict(raw)
            self.after(0, lambda: self._show_result(raw, result))

        threading.Thread(target=_work, daemon=True).start()

    def _show_result(self, raw: str, res: dict):
        self.predict_btn.configure(state="normal")
        self.status_label.configure(text="Hoàn tất ✓", text_color="green")

        tier1_icon = "🔴" if res["tier1"] == "Toxic" else "🟢"
        lines = [
            f"📝  Bình luận gốc : {raw}",
            f"🔧  Sau tiền xử lý: {res['cleaned_text']}",
            "",
            f"{'='*60}",
            f"  {tier1_icon}  TIER 1 – Phân loại: {res['tier1']}  "
            f"(Độ tin cậy: {res['tier1_conf']:.1%})  [Ngưỡng: {TOXIC_THRESHOLD}]",
            f"{'='*60}",
            "",
        ]

        if res["tier1"] == "Clean":
            lines.append(f"  💬  TIER 2A – Cảm xúc: {res['tier2a']}  "
                         f"(Độ tin cậy: {res['tier2a_conf']:.1%})")
            lines.append("")
            lines.append("     Chi tiết xác suất cảm xúc:")
            for label, prob in res["sent_probs"].items():
                bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
                lines.append(f"       {label:>10s}  {bar}  {prob:.1%}")
        else:
            lines.append(f"  ⚠️   TIER 2B – Vi phạm: {', '.join(res['tier2b'])}")
            lines.append("")
            lines.append("     Chi tiết xác suất vi phạm:")
            for label, prob in res["tdet_probs"].items():
                bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
                lines.append(f"       {label:>12s}  {bar}  {prob:.1%}")

        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("0.0", "end")
        self.result_textbox.insert("0.0", "\n".join(lines))
        self.result_textbox.configure(state="disabled")

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------
    def _on_batch_predict(self):
        if self.predictor is None:
            messagebox.showinfo("Chờ", "Mô hình đang được tải, vui lòng đợi...")
            return
        raw_lines = self.batch_input.get("0.0", "end").strip().splitlines()
        raw_lines = [l.strip() for l in raw_lines if l.strip()]
        if not raw_lines:
            messagebox.showinfo("Thông báo", "Vui lòng nhập ít nhất 1 bình luận (mỗi dòng 1 câu).")
            return

        self.batch_btn.configure(state="disabled")
        self.batch_status.configure(text=f"Đang xử lý {len(raw_lines)} câu...", text_color="orange")

        def _work():
            results = []
            for text in raw_lines:
                res = self.predictor.predict(text)
                results.append((text, res))
            self.after(0, lambda: self._show_batch_results(results))

        threading.Thread(target=_work, daemon=True).start()

    def _show_batch_results(self, results: list[tuple[str, dict]]):
        self.batch_btn.configure(state="normal")
        self.batch_status.configure(text=f"Hoàn tất {len(results)} câu ✓", text_color="green")

        lines = []
        for i, (text, res) in enumerate(results, 1):
            icon = "🔴" if res["tier1"] == "Toxic" else "🟢"
            tier2_info = (
                f"Vi phạm: {', '.join(res['tier2b'])}"
                if res["tier1"] == "Toxic"
                else f"Cảm xúc: {res['tier2a']}"
            )
            lines.append(
                f"{i:>3}. {icon} [{res['tier1']:>5s} {res['tier1_conf']:>5.0%}]  "
                f"{tier2_info:40s}  │  {text}"
            )

        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("0.0", "end")
        self.result_textbox.insert(
            "0.0",
            f"{'='*80}\n  KẾT QUẢ DỰ ĐOÁN HÀNG LOẠT ({len(results)} câu)\n{'='*80}\n\n"
            + "\n".join(lines),
        )
        self.result_textbox.configure(state="disabled")

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------
    def _on_clear(self):
        self.input_textbox.delete("0.0", "end")
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("0.0", "end")
        self.result_textbox.configure(state="disabled")
        self.status_label.configure(text="")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = PredictorApp()
    app.mainloop()
