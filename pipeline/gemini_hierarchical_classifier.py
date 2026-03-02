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
Tier 3 must contain at least 1 label.

### Output Format
Reply with ONLY a JSON object with this exact shape:
{"items": [
  {"tier1_spam": "Spam"|"Not Spam", "tier2_toxic": "Toxic"|"Clean", "tier3_labels": ["...", ...]},
  ...
]}

Return ONLY a valid JSON object. Do not include any explanations, markdown formatting, or backticks.
Do NOT repeat, quote, or paraphrase the input comments in any way. Output ONLY the JSON object.

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
                "responseMimeType": "application/json",
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
        last_body_snippet: str = ""
        allow_response_mime = True
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
                    last_error = e
                    # If we hit a retryable case (timeouts, connection issues, parse mismatch), backoff.
                    if attempt < (max_attempts - 1):
                        time.sleep(_retry_sleep_seconds(attempt, last_response))
                        continue
                    raise

            raise RuntimeError("Gemini request failed after retries")

        try:
            # If strict mode is enabled, recover by splitting the batch when Gemini returns wrong count.
            if strict and len(tasks) > 1:
                return _predict_strict_with_splitting(tasks)
            return _predict_once(tasks)
        except Exception as e:
            last_error = e

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


def _get_gemini_text(result: Dict[str, Any]) -> str:
    """Extract text from Gemini `generateContent` response.

    Gemini may split the output across multiple `parts`. If we only read the
    first part, JSON can be truncated even with HTTP 200.
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


class GeminiEmptyOutputError(RuntimeError):
    def __init__(self, message: str, response_json: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.response_json = response_json


def _fallback_result_from_gemini_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort fallback when Gemini returns HTTP 200 but no output.

    If promptFeedback/safetyRatings suggest a category, map to a reasonable toxic label.
    Otherwise, return the global default.
    """
    try:
        fb = resp.get("promptFeedback") or {}
        ratings = fb.get("safetyRatings") or []
        if not isinstance(ratings, list):
            ratings = []

        categories: list[str] = []
        for r in ratings:
            if isinstance(r, dict) and isinstance(r.get("category"), str):
                categories.append(r.get("category") or "")

        cats = " ".join(categories).upper()
        if "HATE" in cats:
            return {"tier1_spam": "Not Spam", "tier2_toxic": "Toxic", "tier3_labels": ["Hate Speech"]}
        if "HARASS" in cats:
            return {"tier1_spam": "Not Spam", "tier2_toxic": "Toxic", "tier3_labels": ["Harassment"]}
        if "SEXU" in cats or "SEXUAL" in cats:
            return {"tier1_spam": "Not Spam", "tier2_toxic": "Toxic", "tier3_labels": ["Obscene"]}
        if "DANGEROUS" in cats:
            return {"tier1_spam": "Not Spam", "tier2_toxic": "Toxic", "tier3_labels": ["Harassment"]}
    except Exception:
        return _default_result()

    return _default_result()


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
    """
    if not content:
        if strict:
            raise ValueError("Empty model output (possibly safety-blocked or missing content)")
        return [_default_result() for _ in range(expected_n)]

    content = _strip_code_fences(content)

    # New robust strategy: try all balanced JSON candidates and prefer a candidate
    # whose parsed length matches expected_n (helps when the model echoes schema/examples).
    parsed_any: Optional[List[Dict[str, Any]]] = None
    for cand in _iter_json_candidates(content):
        items = _try_parse_items_json(cand)
        if items is None:
            continue

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
            if strict and expected_n > 0 and len(items) != expected_n:
                raise ValueError(f"Gemini parsed items length mismatch: expected {expected_n}, got {len(items)}")
            return _pad_or_trim(items, expected_n) if not strict else items

    arr_str = _extract_json_by_brackets(content, "[", "]")
    if arr_str:
        items = _try_parse_items_json(arr_str)
        if items is not None:
            if strict and expected_n > 0 and len(items) != expected_n:
                raise ValueError(f"Gemini parsed items length mismatch: expected {expected_n}, got {len(items)}")
            return _pad_or_trim(items, expected_n) if not strict else items

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
