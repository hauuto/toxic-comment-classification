import os
import requests
from typing import List, Dict, Any

try:
    from label_studio_ml.model import LabelStudioMLBase  # type: ignore
except Exception:
    class LabelStudioMLBase:
        def __init__(self, **kwargs):
            pass


class GeminiClassifier(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.labels = [
            "Clean",
            "Spam",
            "Hate Speech",
            "Harassment",
            "Obscene",
        ]

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")

        self.endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={api_key}"
        )

        self.session = requests.Session()

    def _build_bulk_payload(self, texts: List[str]) -> Dict[str, Any]:
        joined = "\n".join(
            f"Text {i+1}: {text}" for i, text in enumerate(texts)
        )

        prompt = (
            "Classify each text below into exactly one of these labels:\n"
            + ", ".join(self.labels)
            + "\n\n"
            + joined
            + "\n\nReply with one label per line, in the same order."
        )

        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }

    def _parse_labels(self, result: Dict[str, Any], n: int) -> List[str]:
        try:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            lines = [l.strip() for l in text.splitlines() if l.strip()]
        except Exception:
            lines = []

        labels = []
        for i in range(n):
            lbl = lines[i] if i < len(lines) else "Clean"
            labels.append(lbl if lbl in self.labels else "Clean")

        return labels

    def predict(self, tasks: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        texts = [t.get("data", {}).get("text", "") for t in tasks]
        payload = self._build_bulk_payload(texts)

        try:
            r = self.session.post(self.endpoint, json=payload, timeout=30)
            r.raise_for_status()
            result = r.json()
            labels = self._parse_labels(result, len(tasks))
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
                    "model_version": "gemini_2.0_bulk",
                }
            )

        return predictions
