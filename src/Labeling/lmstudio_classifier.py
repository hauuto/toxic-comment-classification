"""
lmstudio_classifier.py – Phân loại bình luận toxic bằng LM Studio (OpenAI-compatible local API).

LM Studio chạy tại http://localhost:1234 và cung cấp endpoint /v1/chat/completions
tương thích OpenAI.  Classifier này dùng `requests` thuần để gọi API.
"""

import requests
from typing import List, Dict, Any, Optional


LABELS = [
    "Clean",
    "Spam",
    "Hate Speech",
    "Harassment",
    "Obscene",
]

SYSTEM_PROMPT = (
    "You are an expert Vietnamese text classifier. "
    "Classify each comment into exactly ONE of these labels:\n"
    "  Clean, Spam, Hate Speech, Harassment, Obscene\n\n"
    "Rules:\n"
    "- Clean: normal, non-toxic comments.\n"
    "- Spam: repetitive, promotional, or irrelevant comments.\n"
    "- Hate Speech: comments attacking groups based on race, religion, gender, etc.\n"
    "- Harassment: personal attacks, bullying, or threatening individuals.\n"
    "- Obscene: vulgar, sexually explicit, or profane language.\n\n"
    "Reply with ONLY the label name, one per line, in the same order as the input texts. "
    "Do NOT add numbering, explanations, or extra text."
)


class LMStudioClassifier:
    """Gọi LM Studio local API để phân loại batch bình luận."""

    def __init__(
        self,
        endpoint: str = "http://localhost:1234/v1/chat/completions",
        model: str = "",
        timeout: int = 120,
        temperature: float = 0.1,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.labels = LABELS
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------
    @staticmethod
    def test_connection(base_url: str = "http://localhost:1234") -> Dict[str, Any]:
        """Gọi GET /v1/models để kiểm tra LM Studio đang chạy.

        Returns
        -------
        dict  {"ok": bool, "models": list[str], "error": str}
        """
        base_url = base_url.rstrip("/")
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            r.raise_for_status()
            data = r.json()
            model_ids = [m.get("id", "?") for m in data.get("data", [])]
            return {"ok": True, "models": model_ids, "error": ""}
        except requests.ConnectionError:
            return {"ok": False, "models": [], "error": "Không thể kết nối tới LM Studio. Hãy chắc chắn LM Studio đang chạy."}
        except Exception as e:
            return {"ok": False, "models": [], "error": str(e)}

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    def _build_user_message(self, texts: List[str]) -> str:
        joined = "\n".join(
            f"Text {i + 1}: {text}" for i, text in enumerate(texts)
        )
        return joined

    # ------------------------------------------------------------------
    # Parse response
    # ------------------------------------------------------------------
    def _parse_labels(self, content: str, n: int) -> List[str]:
        """Parse model response into a list of labels."""
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

        labels: List[str] = []
        for i in range(n):
            if i < len(lines):
                raw = lines[i]
                # Strip common prefixes like "1.", "Text 1:", "- ", etc.
                for prefix in [f"Text {i+1}:", f"{i+1}.", f"{i+1})", "- "]:
                    if raw.startswith(prefix):
                        raw = raw[len(prefix):].strip()
                        break
                # Match to nearest valid label
                matched = self._match_label(raw)
                labels.append(matched)
            else:
                labels.append("Clean")
        return labels

    def _match_label(self, raw: str) -> str:
        """Fuzzy-match a raw string to one of the valid labels."""
        raw_lower = raw.lower().strip()
        for lbl in self.labels:
            if lbl.lower() == raw_lower:
                return lbl
        # Partial match
        for lbl in self.labels:
            if lbl.lower() in raw_lower or raw_lower in lbl.lower():
                return lbl
        return "Clean"

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, tasks: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Phân loại batch bình luận qua LM Studio API.

        Parameters
        ----------
        tasks : list[dict]
            Mỗi dict có dạng ``{"data": {"text": "..."}}``.

        Returns
        -------
        list[dict]
            Cùng format với ``GeminiClassifier.predict()``.
        """
        texts = [t.get("data", {}).get("text", "") for t in tasks]
        user_msg = self._build_user_message(texts)

        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
            "max_tokens": len(texts) * 20,  # ~20 tokens per label line
        }
        if self.model:
            payload["model"] = self.model

        try:
            r = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
            r.raise_for_status()
            result = r.json()
            content = result["choices"][0]["message"]["content"]
            labels = self._parse_labels(content, len(tasks))
        except Exception:
            labels = ["Clean"] * len(tasks)

        predictions = []
        for label in labels:
            predictions.append(
                {
                    "result": [
                        {
                            "from_name": "category",
                            "to_name": "text",
                            "type": "choices",
                            "value": {"choices": [label]},
                        }
                    ],
                    "model_version": "lmstudio_local",
                }
            )
        return predictions
