"""
GeminiHierarchicalClassifier – Hierarchical Multi-label Classification (2-tier) via Google AI Studio (Gemini API).

Kiến trúc 2-Tier:
  Tier 1 (Toxic Check):  "Toxic" | "Clean"
  Tier 2 (Multi-label):
      - Nếu Tier 1 = "Toxic"  → chọn từ ["Hate Speech", "Harassment", "Obscene"]
      - Nếu Tier 1 = "Clean"  → chọn từ ["Positive", "Negative", "Neutral"]
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


# =========================================================================== #
#  Custom exceptions
# =========================================================================== #

class GeminiSafetyBlockError(RuntimeError):
    """Raised when the Gemini API returns HTTP 200 but the output is empty
    due to safety filtering (finishReason=SAFETY, blockReason, etc.).
    Retrying the same content will produce the same result, so the caller
    should fall back to defaults or split the batch."""
    pass


# =========================================================================== #
#  Safety settings (shared)
# =========================================================================== #

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
]


# =========================================================================== #
#  Label definitions (2-tier)
# =========================================================================== #

TIER1_LABELS = ["Toxic", "Clean"]
TIER2_TOXIC_LABELS = ["Hate Speech", "Harassment", "Obscene"]
TIER2_CLEAN_LABELS = ["Positive", "Negative", "Neutral"]
TIER2_ALL_LABELS = TIER2_TOXIC_LABELS + TIER2_CLEAN_LABELS


SYSTEM_PROMPT_JSON_OBJECT = """You are an expert Vietnamese social media comment classifier using a Hierarchical Multi-label Classification system with 2 Tiers.

For EACH input comment, you must evaluate ALL 2 tiers and return a JSON object.

---

### Tier 1 – Toxic Check
Classify as exactly one of: "Toxic" or "Clean".

- "Toxic": content that contains insults, harassment, threats, profanity, or hate speech targeting a person or group. Requires abusive or aggressive intent.
- "Clean": criticism, negative opinions, complaints, or sarcasm WITHOUT insults or abusive language targeting a person or group.

Key distinctions:
- Criticism without insult = Clean (e.g., "Hàng xấu, không đáng tiền" = Clean)
- Profanity of any form = Toxic (e.g., "đụ má", "vcl", "cặc", "vl", "đcm" = Toxic)
- Sarcasm that mocks or demeans a specific person = Toxic
- Sarcasm without a target = Clean

---

### Tier 2 – Multi-label Sub-classification (DEPENDS on Tier 1 result)

**If Tier 1 = "Toxic"**, assign ALL applicable labels from: ["Hate Speech", "Harassment", "Obscene"]
- "Hate Speech": attacks or demeans a GROUP based on region, race, religion, gender, nationality, or other identity. The target is a category of people, not a single individual.
  → Example triggers: "Bắc kỳ", "dân 36", "đàn bà", "người Hoa", slurs against ethnic/regional groups.
- "Harassment": personal attacks, bullying, threatening, or mocking a SPECIFIC individual.
  → Example triggers: "mày", "thằng đó", "con này", direct insults aimed at one person.
- "Obscene": vulgar language, sexual content, or profanity regardless of target.
  → Example triggers: "đụ má", "địt mẹ", "cặc", "lồn", "đái", "vcl", "vl", "cc", "đcm", and all variants.

A single comment can and often will have MULTIPLE toxic sub-labels simultaneously.
Example: "Địt mẹ mày, đồ Bắc kỳ" → ["Obscene", "Harassment", "Hate Speech"]
Example: "Con cặc" → ["Obscene"] (no specific target, just profanity)

**If Tier 1 = "Clean"**, assign EXACTLY ONE label from: ["Positive", "Negative", "Neutral"]
- "Positive": supportive, praising, encouraging, or happy comments.
- "Negative": critical, complaining, disappointed, or dissatisfied comments (but NOT toxic).
- "Neutral": factual, informational, or emotionally indifferent comments.

Do NOT combine sentiment labels. Choose the single most dominant sentiment.

---

### Vietnamese Language Rules (CRITICAL)

**Profanity detection — all of the following = Obscene:**
- Full form: "đụ má", "địt mẹ", "cặc", "lồn", "đái", "đéo", "mẹ kiếp", "tiên sư"
- Abbreviated: "đ.m", "đcm", "dm", "vcl", "vl", "cc", "đmm", "đmm"
- Censored/bypass variants: "đ**", "c*c", "d.u ma", "đ()", "c@c", "đ.mạ", "cặk", "lồnn"
- Phonetic substitutions: "dit me", "cak", "lon", "djt me"

**Implicit Hate Speech — no profanity needed, still Hate Speech:**
- Regional targeting: "Bắc kỳ", "dân 36", "Nam kỳ" (used derogatorily)
- Ethnic/religious slurs or negative generalizations about a group
- Statements like "Bọn X toàn lũ Y" (Group X are all Y) = Hate Speech

**Sarcasm and irony:**
- Sarcasm mocking a specific person = Harassment (± Obscene if profanity present)
- Example: "Giỏi thật đấy, thông minh vcl" → Toxic: ["Harassment", "Obscene"]
- Sarcasm with no specific target = Clean + Negative
- Example: "Ồ hay nhỉ, sản phẩm tuyệt vời lắm" (about a product) → Clean: ["Negative"]

---

### Output Format

Reply with ONLY a valid JSON object in this exact shape:
{"items": [
  {"tier1_label": "Toxic"|"Clean", "tier2_labels": ["...", ...]},
  ...
]}

- "items" must be an array with one object per input comment, in the same order as input.
- Do NOT add any explanation, numbering, markdown, backticks, or extra text outside the JSON.

---

### Hard Rules

1. If Tier 1 = "Toxic" → Tier 2 MUST contain at least one of: "Hate Speech", "Harassment", "Obscene".
2. If Tier 1 = "Clean" → Tier 2 MUST contain EXACTLY ONE of: "Positive", "Negative", "Neutral".
3. Toxic classification overrides sentiment classification entirely.
4. Profanity of ANY form (full, abbreviated, censored, phonetic) = Toxic + Obscene.
5. Group-based negative generalization = Toxic + Hate Speech (even without profanity).
6. A comment can be Harassment + Obscene + Hate Speech simultaneously.

---

### Few-shot Examples

Input: "Sản phẩm tốt lắm, giao hàng nhanh, sẽ ủng hộ tiếp"
Output: {"tier1_label": "Clean", "tier2_labels": ["Positive"]}

Input: "Hàng xấu, không đáng tiền, không mua lần 2"
Output: {"tier1_label": "Clean", "tier2_labels": ["Negative"]}

Input: "Chưa dùng nên chưa biết, tạm cho 3 sao"
Output: {"tier1_label": "Clean", "tier2_labels": ["Neutral"]}

Input: "Địt mẹ mày thằng ngu"
Output: {"tier1_label": "Toxic", "tier2_labels": ["Obscene", "Harassment"]}

Input: "Con cặc"
Output: {"tier1_label": "Toxic", "tier2_labels": ["Obscene"]}

Input: "Bọn dân 36 toàn lũ ăn cắp"
Output: {"tier1_label": "Toxic", "tier2_labels": ["Hate Speech"]}

Input: "Bắc kỳ ăn cá rô phi, ăn nhầm lựu đạn chết cha Bắc kỳ"
Output: {"tier1_label": "Toxic", "tier2_labels": ["Hate Speech"]}

Input: "Địt mẹ mày, đồ Bắc kỳ"
Output: {"tier1_label": "Toxic", "tier2_labels": ["Obscene", "Harassment", "Hate Speech"]}

Input: "Mày ngu vcl, làm ăn kiểu gì vậy"
Output: {"tier1_label": "Toxic", "tier2_labels": ["Obscene", "Harassment"]}

Input: "Hàng như cái quần què, mả cha nhà mày lừa đảo à"
Output: {"tier1_label": "Toxic", "tier2_labels": ["Obscene", "Harassment"]}

Input: "Giỏi thật đấy, thông minh vcl, ai cũng phục mày"
Output: {"tier1_label": "Toxic", "tier2_labels": ["Harassment", "Obscene"]}

Input: "Má mày dạy mày tốt nên giờ mày mới khôn như vậy"
Output: {"tier1_label": "Toxic", "tier2_labels": ["Harassment"]}

Input: "Trời nắng quá bạn bị chập mạch phải không"
Output: {"tier1_label": "Toxic", "tier2_labels": ["Harassment"]}

Input: "Sản phẩm bình thường, không có gì nổi bật"
Output: {"tier1_label": "Clean", "tier2_labels": ["Neutral"]}

Input: "Dịch vụ tệ quá, chờ 2 tiếng mà không ai hỗ trợ"
Output: {"tier1_label": "Clean", "tier2_labels": ["Negative"]}
"""
class GeminiHierarchicalClassifier:
    """Gemini-backed classifier that returns 2-tier schema (tier1_label + tier2_labels)."""

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
                "responseMimeType": "application/json",
            },
            "safetySettings": SAFETY_SETTINGS,
        }

        try:
            r = requests.post(endpoint, json=payload, timeout=10)
            # Some deployments reject responseMimeType; retry without it.
            if r.status_code == 400:
                try:
                    body = (r.text or "")
                except Exception:
                    body = ""
                if "responseMimeType" in body or "response_mime_type" in body:
                    payload2 = dict(payload)
                    payload2["generationConfig"] = dict(payload.get("generationConfig", {}))
                    payload2["generationConfig"].pop("responseMimeType", None)
                    r = requests.post(endpoint, json=payload2, timeout=10)
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
        n = len(texts)
        return (
            "Input comments:\n"
            + joined
            + f"\n\nTotal comments: {n}.\n"
            + "Output must contain exactly Total comments items in the same order.\n"
            + "Do NOT merge multiple comments into one item.\n"
            + "If uncertain, still output a best-effort label for each comment."
        )

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self,
        tasks: List[Dict[str, Any]],
        retries: int = 1,
        strict: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        def _should_split_on_error(err: Exception) -> bool:
            msg = str(err).lower()
            return (
                "length mismatch" in msg
                or "parse mismatch" in msg
                or "could not parse json" in msg
                or "empty model output" in msg
                or "non-json" in msg
            )

        def _should_fallback_single(err: Exception) -> bool:
            msg = str(err).lower()
            return "empty model output" in msg or "safety" in msg or isinstance(err, GeminiEmptyOutputError)

        def _predict_strict_with_splitting(batch_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Strict predict that auto-splits batches if Gemini returns wrong item count.

            This avoids failing an entire batch when Gemini only labels the first comment.
            """
            try:
                return _predict_once(batch_tasks)
            except Exception as e:
                if not strict:
                    raise
                if len(batch_tasks) <= 1:
                    if _should_fallback_single(e):
                        # Do not abort the whole run for a single safety-blocked/empty item.
                        if isinstance(e, GeminiEmptyOutputError) and e.response_json is not None:
                            return [_fallback_result_from_gemini_response(e.response_json)]
                        return [_default_result()]
                    raise
                if not _should_split_on_error(e):
                    raise
                mid = len(batch_tasks) // 2
                left = _predict_strict_with_splitting(batch_tasks[:mid])
                right = _predict_strict_with_splitting(batch_tasks[mid:])
                return left + right

        texts = [str(t.get("data", {}).get("text", "")) for t in tasks]
        user_msg = self._build_user_message(texts)

        # Baseline token budget. In practice, JSON can be truncated even when HTTP=200;
        # give some headroom and scale up on retries.
        base_max_tokens = max(512, int(len(texts) * self.max_output_tokens_per_item * 2))

        base_payload: Dict[str, Any] = {
            "contents": [
                {
                    "parts": [
                        {"text": SYSTEM_PROMPT_JSON_OBJECT + "\n\n" + user_msg},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                # maxOutputTokens set per-attempt (scaled)
                "maxOutputTokens": base_max_tokens,
                # Ask Gemini to return JSON directly (reduces chances of markdown/explanations).
                # If unsupported by a specific API version/model, we'll auto-fallback at runtime.
                "responseMimeType": "application/json",
            },
            "safetySettings": SAFETY_SETTINGS,
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
        last_body_snippet: str = ""
        consecutive_empty = 0  # track consecutive empty (non-safety) outputs
        max_attempts = 1 + max(0, int(retries))

        def _predict_once(batch_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            nonlocal last_error, last_response, last_body_snippet, allow_response_mime

            for attempt in range(max_attempts):
                try:
                    # Scale tokens on each retry to avoid truncated JSON.
                    scaled_max_tokens = int(base_max_tokens * (1.8 ** attempt))
                    scaled_max_tokens = max(base_max_tokens, scaled_max_tokens)
                    scaled_max_tokens = min(scaled_max_tokens, 8192)

                    payload = dict(base_payload)
                    payload["generationConfig"] = dict(base_payload.get("generationConfig", {}))
                    payload["generationConfig"]["maxOutputTokens"] = scaled_max_tokens

                    if not allow_response_mime:
                        payload.get("generationConfig", {}).pop("responseMimeType", None)

                    r = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
                    last_response = r
                    if r.status_code >= 400:
                        # Some deployments reject unknown generationConfig keys; fall back if needed.
                        if int(r.status_code) == 400 and allow_response_mime and attempt < (max_attempts - 1):
                            try:
                                body = (r.text or "")[:2000]
                            except Exception:
                                body = ""
                            if "responseMimeType" in body or "response_mime_type" in body:
                                allow_response_mime = False
                                time.sleep(_retry_sleep_seconds(attempt, r))
                                continue
                        # Retryable errors: rate-limit / transient server issues.
                        if _is_retryable_http(int(r.status_code)) and attempt < (max_attempts - 1):
                            time.sleep(_retry_sleep_seconds(attempt, r))
                            continue
                        try:
                            last_body_snippet = (r.text or "")[:1200]
                        except Exception:
                            last_body_snippet = ""
                        r.raise_for_status()

                    try:
                        data = r.json()
                    except Exception as e:
                        last_body_snippet = (r.text or "")[:1200]
                        raise RuntimeError(f"Gemini returned non-JSON response: {last_body_snippet}") from e

                    # Safety-blocked or empty candidate output (HTTP 200 but no usable content).
                    if not (data.get("candidates") or []):
                        raise GeminiEmptyOutputError(
                            "Empty model output (possibly safety-blocked or missing content)", response_json=data
                        )

                    text = _get_gemini_text(data)
                    if not text.strip():
                        raise GeminiEmptyOutputError(
                            "Empty model output (possibly safety-blocked or missing content)", response_json=data
                        )
                    items = _parse_items(text, expected_n=len(batch_tasks), strict=strict)
                    validated = [_validate_tier_result(obj) for obj in items]

                    if len(validated) == len(batch_tasks):
                        return validated

                    # Parsed but shape mismatch → treat as retryable parsing failure.
                    raise ValueError("Gemini output parse mismatch")

                except Exception as e:
                    last_body_snippet = (r.text or "")[:1200]
                    raise RuntimeError(f"Gemini returned non-JSON response: {last_body_snippet}") from e

                # --- Detect safety-blocked responses (HTTP 200 but empty) ---
                text = _get_gemini_text(data)
                if not text:
                    diag = _diagnose_empty_output(data)
                    if diag:
                        # Safety / non-retryable block → do NOT retry the same content.
                        raise GeminiSafetyBlockError(diag)

                    # Transient empty output — but if it keeps happening, the batch
                    # likely contains a problematic comment.  Fall back to single
                    # requests after 2 consecutive empties (for multi-comment batches).
                    consecutive_empty += 1
                    if consecutive_empty >= 2 and len(tasks) > 1:
                        return self._predict_single_fallback(tasks, retries=retries, strict=strict)

                    raise ValueError("Empty model output (possibly safety-blocked or missing content)")

                # Reset counter on successful text extraction
                consecutive_empty = 0

                items = _parse_items(text, expected_n=len(tasks), strict=strict)
                validated = [_validate_tier_result(obj) for obj in items]

            raise RuntimeError("Gemini request failed after retries")

                # Parsed but shape mismatch → treat as retryable parsing failure.
                raise ValueError("Gemini output parse mismatch")

            except GeminiSafetyBlockError:
                # Safety block is deterministic — retrying won't help.
                # Fall back: if batch has >1 item, split into single-comment requests.
                if len(tasks) > 1:
                    return self._predict_single_fallback(tasks, retries=retries, strict=strict)
                # Single comment was blocked → return defaults (or raise in strict mode).
                if strict:
                    raise
                return [_default_result() for _ in range(len(tasks))]

            except Exception as e:
                last_error = e
                # If we hit a retryable case (timeouts, connection issues, parse mismatch), backoff.
                if attempt < (max_attempts - 1):
                    time.sleep(_retry_sleep_seconds(attempt, last_response))
                    continue
                break

        if strict:
            status = None
            if last_response is not None:
                try:
                    status = int(last_response.status_code)
                except Exception:
                    status = None
            msg = str(last_error) if last_error else "Unknown error"
            if status is not None:
                detail = (last_body_snippet or "").strip()
                if detail:
                    raise RuntimeError(f"Gemini request failed (HTTP {status}) after {max_attempts} attempt(s): {msg}\nResponse: {detail}")
                raise RuntimeError(f"Gemini request failed (HTTP {status}) after {max_attempts} attempt(s): {msg}")
            raise RuntimeError(f"Gemini request failed after {max_attempts} attempt(s): {msg}")

        _ = last_error
        return [_default_result() for _ in range(len(tasks))]

    # ------------------------------------------------------------------
    # Single-comment fallback (used when a batch is safety-blocked)
    # ------------------------------------------------------------------
    def _predict_single_fallback(
        self,
        tasks: List[Dict[str, Any]],
        retries: int = 1,
        strict: bool = False,
    ) -> List[Dict[str, Any]]:
        """Predict each comment individually to isolate safety-blocked items.

        Comments that are individually blocked get default labels; the rest
        are classified normally.  This avoids losing an entire batch because
        of one problematic comment.
        """
        results: List[Dict[str, Any]] = []
        for task in tasks:
            try:
                preds = self.predict([task], retries=retries, strict=False)
                results.append(preds[0] if preds else _default_result())
            except GeminiSafetyBlockError:
                # Individual comment blocked → assign defaults silently.
                results.append(_default_result())
            except Exception:
                results.append(_default_result())
        return results


def _get_gemini_text(result: Dict[str, Any]) -> str:
    """Extract the text content from a Gemini API response.

    Returns an empty string if the response has no text (e.g. safety-blocked).
    The caller should use ``_diagnose_empty_output()`` to determine the reason.
    """
    try:
        candidates = result.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if not isinstance(parts, list) or not parts:
            return ""
        chunks: list[str] = []
        for p in parts:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                chunks.append(p.get("text") or "")
        return "".join(chunks)
    except Exception:
        return ""


def _diagnose_empty_output(result: Dict[str, Any]) -> str:
    """Return a human-readable diagnosis when ``_get_gemini_text`` returns empty.

    Inspects ``promptFeedback.blockReason``, ``candidates[0].finishReason``,
    and structural anomalies (missing content/parts) to determine whether the
    Gemini response was safety-blocked or otherwise non-retryable.

    Returns an empty string if no obvious safety block is detected (i.e. the
    empty output might be a transient issue worth retrying).
    """
    parts: List[str] = []

    # Check prompt-level block
    prompt_feedback = result.get("promptFeedback", {})
    block_reason = prompt_feedback.get("blockReason", "")
    if block_reason:
        parts.append(f"promptFeedback.blockReason={block_reason}")

    # Check candidate-level finish reason
    candidates = result.get("candidates", [])
    # Non-retryable finish reasons (deterministic — retrying won't help)
    _NON_RETRYABLE_FINISH = {"SAFETY", "RECITATION", "OTHER", "BLOCKLIST",
                             "PROHIBITED_CONTENT", "SPII"}
    if candidates:
        finish_reason = candidates[0].get("finishReason", "")
        if finish_reason in _NON_RETRYABLE_FINISH:
            parts.append(f"finishReason={finish_reason}")
        elif finish_reason and finish_reason not in ("STOP", "MAX_TOKENS"):
            parts.append(f"finishReason={finish_reason}")

        # Detect missing content or empty parts (candidate exists but no text)
        content = candidates[0].get("content")
        if content is None:
            parts.append("candidate has no content")
        elif not content.get("parts"):
            parts.append("candidate content has no parts")

        # Collect safety ratings that triggered filtering
        safety_ratings = candidates[0].get("safetyRatings", [])
        blocked_cats = [
            sr.get("category", "?")
            for sr in safety_ratings
            if sr.get("blocked") is True
                or sr.get("probability", "").upper() in ("HIGH",)
        ]
        if blocked_cats:
            parts.append(f"blockedCategories={blocked_cats}")
    elif not candidates:
        # No candidates at all
        if block_reason:
            parts.append("no candidates returned (prompt blocked)")
        else:
            parts.append("no candidates returned (unknown reason)")

    if parts:
        return "Safety/Non-retryable: " + "; ".join(parts)
    return ""


def _strip_code_fences(text: str) -> str:
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def _balanced_json_substring(text: str, start_index: int) -> Optional[str]:
    """Return a balanced JSON object/array substring starting at start_index.

    Handles nested {}/[] and ignores braces inside JSON strings.
    Returns None if the substring is not balanced (likely truncated).
    """
    if start_index < 0 or start_index >= len(text):
        return None

    start_ch = text[start_index]
    if start_ch not in "{[":
        return None

    matching = {"{": "}", "[": "]"}
    open_to_close = matching
    close_to_open = {"}": "{", "]": "["}

    stack: list[str] = []
    in_string = False
    escape = False

    for i in range(start_index, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        # not in string
        if ch == '"':
            in_string = True
            continue

        if ch in "{[":
            stack.append(ch)
            continue

        if ch in "}]":
            if not stack:
                return None
            expected_open = close_to_open.get(ch)
            if expected_open != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start_index : i + 1]

    return None


def _iter_json_candidates(text: str) -> List[str]:
    """Generate possible JSON payload substrings from a model output."""
    if not text:
        return []

    # Primary candidate: whole text
    candidates: list[str] = [text.strip()]

    # Balanced substrings starting at every '{' or '['
    seen: set[str] = set(candidates)
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        sub = _balanced_json_substring(text, i)
        if not sub:
            continue
        sub = sub.strip()
        if sub and sub not in seen:
            seen.add(sub)
            candidates.append(sub)

    return candidates


def _extract_json_by_brackets(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    start = text.find(open_ch)
    end = text.rfind(close_ch)
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _parse_items(content: str, expected_n: int, strict: bool = False) -> List[Dict[str, Any]]:
    """Parse Gemini output into a list of dict items.

    Accepts either:
    - JSON object: {"items": [ ... ]}
    - JSON array: [ ... ]
    - single object: { ... }  (treated as one item)

    In strict mode, minor count mismatches (off-by-a-few) are tolerated via
    ``_pad_or_trim`` — Gemini frequently returns N±1 items. Only raises when
    no items could be parsed at all.
    """
    if not content:
        if strict:
            raise ValueError("Empty model output (possibly safety-blocked or missing content)")
        return [_default_result() for _ in range(expected_n)]

    content = _strip_code_fences(content)

    def _resolve(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply pad/trim, raising only when nothing was parsed in strict mode."""
        if expected_n > 0 and len(items) != expected_n:
            if strict and len(items) == 0:
                raise ValueError(
                    f"Gemini parsed 0 items but expected {expected_n}"
                )
            # Any non-zero count (e.g. 11 vs 10, 8 vs 10): silently pad/trim.
            # Gemini frequently returns N±1 items — this is normal LLM behaviour.
        return _pad_or_trim(items, expected_n)

    # 1) direct json
    items = _try_parse_items_json(content)
    if items is not None:
        return _resolve(items)

        if expected_n > 0 and len(items) == expected_n:
            return items if strict else items

        if parsed_any is None:
            parsed_any = items

    if parsed_any is not None:
        if strict and expected_n > 0 and len(parsed_any) != expected_n:
            raise ValueError(f"Gemini parsed items length mismatch: expected {expected_n}, got {len(parsed_any)}")
        return _pad_or_trim(parsed_any, expected_n) if not strict else parsed_any

    # Backward-compatible fallbacks (best-effort extraction by first/last brackets)
    obj_str = _extract_json_by_brackets(content, "{", "}")
    if obj_str:
        items = _try_parse_items_json(obj_str)
        if items is not None:
            return _resolve(items)

    arr_str = _extract_json_by_brackets(content, "[", "]")
    if arr_str:
        items = _try_parse_items_json(arr_str)
        if items is not None:
            return _resolve(items)

    if strict:
        snippet = str(content).strip().replace("\r", "")[:1200]
        raise ValueError(f"Could not parse JSON from Gemini output. Snippet: {snippet}")
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
        "tier1_label": "Clean",
        "tier2_labels": ["Neutral"],
    }


def _validate_tier_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return _default_result()

    # Tier 1
    tier1 = str(obj.get("tier1_label", "Clean")).strip()
    if tier1 not in TIER1_LABELS:
        tier1 = _fuzzy_match(tier1, TIER1_LABELS, "Clean") or "Clean"

    # Tier 2
    raw_tier2 = obj.get("tier2_labels", [])
    if isinstance(raw_tier2, str):
        raw_tier2 = [raw_tier2]
    if not isinstance(raw_tier2, list):
        raw_tier2 = []

    valid_pool = TIER2_TOXIC_LABELS if tier1 == "Toxic" else TIER2_CLEAN_LABELS
    tier2: List[str] = []
    for lbl in raw_tier2:
        lbl = str(lbl).strip()
        matched = _fuzzy_match(lbl, valid_pool, None)
        if matched and matched not in tier2:
            tier2.append(matched)

    return {
        "tier1_label": tier1,
        "tier2_labels": tier2,
    }
