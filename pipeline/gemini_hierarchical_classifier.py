"""
GeminiHierarchicalClassifier – Hierarchical Multi-label Classification (3-tier) via Google AI Studio (Gemini API).

Kiến trúc 3-Tier (giữ nguyên như lmstudio_classifier.py):
  Tier 1 (Spam Check):  "Spam" | "Not Spam"
  Tier 2 (Toxic Check):  "Toxic" | "Clean"   (độc lập với Tier 1)
  Tier 3 (Multi-label):
      - Nếu Tier 2 = "Toxic"  → chọn từ ["Hate Speech", "Harassment", "Obscene"]
      - Nếu Tier 2 = "Clean"  → chọn từ ["Positive", "Negative", "Neutral"]
      (có thể 0, 1 hoặc nhiều nhãn)

Cấu hình:
- GEMINI_API_KEY: bắt buộc (env/.env)
- GEMINI_MODEL: optional (default gemini-2.0-flash)

API endpoint (Google Generative Language API):
  https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=...

Note:
- Prompt yêu cầu trả về JSON object ổn định: {"items": [ ... ]}
- Parser có fallback để chịu được codefence/extra text.
"""

from __future__ import annotations

import json
import os
import re
import time
import random
from typing import Any, Dict, List, Optional

import requests

from lmstudio_classifier import (
    TIER1_LABELS,
    TIER2_LABELS,
    TIER3_TOXIC_LABELS,
    TIER3_CLEAN_LABELS,
)


SYSTEM_PROMPT_JSON_OBJECT = """You are an expert Vietnamese comment classifier using a Hierarchical Multi-label Classification system with 3 Tiers.

For EACH input comment, you must evaluate ALL 3 tiers and return a JSON object.

### Tier 1 – Spam Check
Classify as exactly one of: "Spam" or "Not Spam".
- "Spam": repetitive, promotional, irrelevant, or bot-generated comments.
- "Not Spam": genuine human comments (can still be toxic).

### Tier 2 – Toxic Check (INDEPENDENT from Tier 1)
Classify as exactly one of: "Toxic" or "Clean".
"Toxic": content that includes insults, harassment, threats, profanity, or hate speech targeting a person or group.
"Toxic" requires abusive or aggressive intent.
"Clean": criticism, negative opinions, or complaints without insults or abusive language.
IMPORTANT: A comment can be both "Spam" AND "Toxic", or "Spam" AND "Clean". Tier 1 and Tier 2 are independent.

### Tier 3 – Multi-label (DEPENDS on Tier 2)
Return a list of sub-labels:
- If Tier 2 = "Toxic", choose from: ["Hate Speech", "Harassment", "Obscene"]
  - "Hate Speech": attacks groups based on race, religion, gender, nationality, etc.
  - "Harassment": personal attacks, bullying, threatening individuals.
  - "Obscene": vulgar, sexually explicit, or profane language.
- If Tier 2 = "Clean", choose from: ["Positive", "Negative", "Neutral"]
  - "Positive": positive, supportive, encouraging comments.
  - "Negative": negative, critical, complaining comments (but not toxic).
  - "Neutral": neutral, factual, or informational comments.
Tier 3 can contain 0, 1, or multiple labels simultaneously.

### Output Format
Reply with ONLY a JSON object with this exact shape:
{"items": [
  {"tier1_spam": "Spam"|"Not Spam", "tier2_toxic": "Toxic"|"Clean", "tier3_labels": ["...", ...]},
  ...
]}

General Rules:
- Evaluate tiers independently but logically consistent.
- If Tier 2 = "Toxic", Tier 3 MUST contain at least one toxic label.
- If Tier 2 = "Clean", Tier 3 MUST contain at least one clean label.
- Toxic classification overrides sentiment classification.
- Do not classify something as Spam based only on being short.
- Criticism without insult = Clean + Negative.
- Profanity automatically qualifies as Obscene (Toxic).

- "items" must be an array with one object per input comment, in the same order.
- Do NOT add any explanation, numbering, markdown, or extra text outside the JSON."""


class GeminiHierarchicalClassifier:
    """Gemini-backed classifier that returns the same 3-tier schema as LMStudioClassifier in pipeline."""

    def __init__(
        self,
        model: str = "",
        api_key: str | None = None,
        timeout: int = 60,
        temperature: float = 0.1,
        max_output_tokens_per_item: int = 120,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")

        self.model = (model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")).strip() or "gemini-2.0-flash"
        self.timeout = timeout
        self.temperature = temperature
        self.max_output_tokens_per_item = max_output_tokens_per_item

        self.endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------
    @staticmethod
    def test_connection(model: str = "") -> Dict[str, Any]:
        """Test Gemini API key/model by issuing a tiny generateContent request."""
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return {"ok": False, "models": [], "error": "Missing GEMINI_API_KEY"}

        model_name = (model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")).strip() or "gemini-2.0-flash"
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                SYSTEM_PROMPT_JSON_OBJECT
                                + "\n\nInput comments:\nComment 1: xin chào\n"
                                + "\nReturn JSON now."
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 200,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }

        try:
            r = requests.post(endpoint, json=payload, timeout=10)
            r.raise_for_status()
            data = r.json()
            text = _get_gemini_text(data)
            items = _parse_items(text, expected_n=1)
            if len(items) != 1:
                return {"ok": False, "models": [], "error": "Gemini responded but output could not be parsed."}
            return {"ok": True, "models": [model_name], "error": ""}
        except Exception as e:
            return {"ok": False, "models": [], "error": str(e)}

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    @staticmethod
    def _build_user_message(texts: List[str]) -> str:
        joined = "\n".join(f"Comment {i + 1}: {text}" for i, text in enumerate(texts))
        return "Input comments:\n" + joined

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, tasks: List[Dict[str, Any]], retries: int = 1, **kwargs) -> List[Dict[str, Any]]:
        texts = [str(t.get("data", {}).get("text", "")) for t in tasks]
        user_msg = self._build_user_message(texts)

        max_tokens = max(256, len(texts) * self.max_output_tokens_per_item)

        payload: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {"text": SYSTEM_PROMPT_JSON_OBJECT + "\n\n" + user_msg},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": max_tokens,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }

        def _retry_sleep_seconds(attempt_index: int, response: requests.Response | None) -> float:
            # attempt_index is 0-based (0 = first failure)
            retry_after = None
            if response is not None:
                try:
                    ra = response.headers.get("Retry-After")
                    if ra:
                        retry_after = float(ra)
                except Exception:
                    retry_after = None

            # Exponential backoff with jitter; keep it bounded to avoid long stalls.
            base = min(8.0, 1.0 * (2 ** min(attempt_index, 5)))  # 1,2,4,8,8,...
            jitter = random.uniform(0.0, 0.35 * base)
            delay = base + jitter
            if retry_after is not None:
                delay = max(delay, retry_after)
            return min(delay, 15.0)

        def _is_retryable_http(status_code: int) -> bool:
            return status_code in (408, 429, 500, 502, 503, 504)

        last_error: Exception | None = None
        last_response: requests.Response | None = None
        for attempt in range(1 + max(0, int(retries))):
            try:
                r = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
                last_response = r
                if r.status_code >= 400:
                    # Retryable errors: rate-limit / transient server issues.
                    if _is_retryable_http(int(r.status_code)) and attempt < retries:
                        time.sleep(_retry_sleep_seconds(attempt, r))
                        continue
                    r.raise_for_status()

                data = r.json()
                text = _get_gemini_text(data)
                items = _parse_items(text, expected_n=len(tasks))
                validated = [_validate_tier_result(obj) for obj in items]

                if len(validated) == len(tasks):
                    return validated

                # Parsed but shape mismatch → treat as retryable parsing failure.
                raise ValueError("Gemini output parse mismatch")

            except Exception as e:
                last_error = e
                # If we hit a retryable case (timeouts, connection issues, parse mismatch), backoff.
                if attempt < retries:
                    time.sleep(_retry_sleep_seconds(attempt, last_response))
                    continue
                break

        _ = last_error
        return [_default_result() for _ in range(len(tasks))]


def _get_gemini_text(result: Dict[str, Any]) -> str:
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def _extract_json_by_brackets(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    start = text.find(open_ch)
    end = text.rfind(close_ch)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _parse_items(content: str, expected_n: int) -> List[Dict[str, Any]]:
    """Parse Gemini output into a list of dict items.

    Accepts either:
    - JSON object: {"items": [ ... ]}
    - JSON array: [ ... ]
    - single object: { ... }  (treated as one item)
    """
    if not content:
        return [_default_result() for _ in range(expected_n)]

    content = _strip_code_fences(content)

    # 1) direct json
    items = _try_parse_items_json(content)
    if items is not None:
        return _pad_or_trim(items, expected_n)

    # 2) try extract {...}
    obj_str = _extract_json_by_brackets(content, "{", "}")
    if obj_str:
        items = _try_parse_items_json(obj_str)
        if items is not None:
            return _pad_or_trim(items, expected_n)

    # 3) try extract [...]
    arr_str = _extract_json_by_brackets(content, "[", "]")
    if arr_str:
        items = _try_parse_items_json(arr_str)
        if items is not None:
            return _pad_or_trim(items, expected_n)

    return [_default_result() for _ in range(expected_n)]


def _try_parse_items_json(raw: str) -> Optional[List[Dict[str, Any]]]:
    try:
        data = json.loads(raw)
    except Exception:
        return None

    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]

    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            return [d for d in data["items"] if isinstance(d, dict)]
        return [data]

    return None


def _pad_or_trim(items: List[Dict[str, Any]], expected_n: int) -> List[Dict[str, Any]]:
    if expected_n <= 0:
        return []
    if len(items) >= expected_n:
        return items[:expected_n]
    # pad with defaults
    padded = items[:]
    while len(padded) < expected_n:
        padded.append(_default_result())
    return padded


def _fuzzy_match(raw: str, candidates: List[str], default: Optional[str]) -> Optional[str]:
    raw_lower = raw.lower().strip()
    for c in candidates:
        if c.lower() == raw_lower:
            return c
    for c in candidates:
        if c.lower() in raw_lower or raw_lower in c.lower():
            return c
    return default


def _default_result() -> Dict[str, Any]:
    return {
        "tier1_spam": "Not Spam",
        "tier2_toxic": "Clean",
        "tier3_labels": ["Neutral"],
    }


def _validate_tier_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return _default_result()

    tier1 = str(obj.get("tier1_spam", "Not Spam")).strip()
    if tier1 not in TIER1_LABELS:
        tier1 = _fuzzy_match(tier1, TIER1_LABELS, "Not Spam") or "Not Spam"

    tier2 = str(obj.get("tier2_toxic", "Clean")).strip()
    if tier2 not in TIER2_LABELS:
        tier2 = _fuzzy_match(tier2, TIER2_LABELS, "Clean") or "Clean"

    raw_tier3 = obj.get("tier3_labels", [])
    if isinstance(raw_tier3, str):
        raw_tier3 = [raw_tier3]
    if not isinstance(raw_tier3, list):
        raw_tier3 = []

    valid_pool = TIER3_TOXIC_LABELS if tier2 == "Toxic" else TIER3_CLEAN_LABELS
    tier3: List[str] = []
    for lbl in raw_tier3:
        lbl = str(lbl).strip()
        matched = _fuzzy_match(lbl, valid_pool, None)
        if matched and matched not in tier3:
            tier3.append(matched)

    return {
        "tier1_spam": tier1,
        "tier2_toxic": tier2,
        "tier3_labels": tier3,
    }
