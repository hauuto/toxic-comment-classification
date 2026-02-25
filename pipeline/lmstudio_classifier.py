"""
lmstudio_classifier.py – Hierarchical Multi-label Classification qua LM Studio.

Kiến trúc 3-Tier:
  Tier 1 (Spam Check):  "Spam" | "Not Spam"
  Tier 2 (Toxic Check):  "Toxic" | "Clean"   (độc lập với Tier 1)
  Tier 3 (Multi-label):
      - Nếu Tier 2 = "Toxic"  → chọn từ ["Hate Speech", "Harassment", "Obscene"]
      - Nếu Tier 2 = "Clean"  → chọn từ ["Positive", "Negative", "Neutral"]
      (có thể 0, 1 hoặc nhiều nhãn)

LM Studio chạy tại http://localhost:1234, endpoint /v1/chat/completions (OpenAI-compatible).
"""

import json
import re
import requests
from typing import List, Dict, Any, Optional


# =========================================================================== #
#  Label definitions
# =========================================================================== #

TIER1_LABELS = ["Spam", "Not Spam"]
TIER2_LABELS = ["Toxic", "Clean"]
TIER3_TOXIC_LABELS = ["Hate Speech", "Harassment", "Obscene"]
TIER3_CLEAN_LABELS = ["Positive", "Negative", "Neutral"]
TIER3_ALL_LABELS = TIER3_TOXIC_LABELS + TIER3_CLEAN_LABELS

# =========================================================================== #
#  System Prompt
# =========================================================================== #

SYSTEM_PROMPT = """You are an expert Vietnamese comment classifier using a Hierarchical Multi-label Classification system with 3 Tiers.

For EACH input comment, you must evaluate ALL 3 tiers and return a JSON object:

### Tier 1 – Spam Check
Classify as exactly one of: "Spam" or "Not Spam".
- "Spam": repetitive, promotional, irrelevant, or bot-generated comments.
- "Not Spam": genuine human comments (can still be toxic).

### Tier 2 – Toxic Check (INDEPENDENT from Tier 1)
Classify as exactly one of: "Toxic" or "Clean".
- "Toxic": contains harmful, offensive, or inappropriate content.
- "Clean": normal, non-harmful content.
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
Reply with ONLY a JSON array (one object per input comment, same order). Each object:
{"tier1_spam": "Spam" or "Not Spam", "tier2_toxic": "Toxic" or "Clean", "tier3_labels": ["label1", ...]}

Example for 2 comments:
[
  {"tier1_spam": "Not Spam", "tier2_toxic": "Toxic", "tier3_labels": ["Obscene", "Harassment"]},
  {"tier1_spam": "Spam", "tier2_toxic": "Clean", "tier3_labels": ["Neutral"]}
]

Do NOT add any explanation, numbering, or extra text outside the JSON array."""


class LMStudioClassifier:
    """Gọi LM Studio local API để phân loại batch bình luận (Hierarchical 3-Tier)."""

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
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------
    @staticmethod
    def test_connection(base_url: str = "http://localhost:1234") -> Dict[str, Any]:
        """Gọi GET /v1/models để kiểm tra LM Studio đang chạy."""
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
            f"Comment {i + 1}: {text}" for i, text in enumerate(texts)
        )
        return joined

    # ------------------------------------------------------------------
    # Parse JSON response
    # ------------------------------------------------------------------
    def _parse_json_response(self, content: str, n: int) -> List[Dict[str, Any]]:
        """Parse model response (expected JSON array) into list of tier dicts.

        Returns list of dicts, each: {tier1_spam, tier2_toxic, tier3_labels}
        """
        # Try to extract JSON array from the response
        parsed = self._extract_json_array(content, n)
        if parsed and len(parsed) == n:
            return [self._validate_tier_result(obj) for obj in parsed]

        # Fallback: try to find individual JSON objects
        parsed = self._extract_json_objects(content, n)
        if parsed and len(parsed) >= n:
            return [self._validate_tier_result(obj) for obj in parsed[:n]]

        # Last resort: return defaults
        return [self._default_result() for _ in range(n)]

    def _extract_json_array(self, content: str, n: int) -> Optional[List[Dict]]:
        """Try to parse a JSON array from the content."""
        # Remove markdown code fences if present
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        content = content.strip()

        # Try direct parse
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass

        # Try to find [...] in the text
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        return None

    def _extract_json_objects(self, content: str, n: int) -> Optional[List[Dict]]:
        """Try to find individual JSON objects {...} in the content."""
        results = []
        for match in re.finditer(r'\{[^{}]+\}', content):
            try:
                obj = json.loads(match.group())
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                continue
        return results if results else None

    def _validate_tier_result(self, obj: Dict) -> Dict[str, Any]:
        """Validate and normalize a single tier result dict."""
        if not isinstance(obj, dict):
            return self._default_result()

        # Tier 1
        tier1 = str(obj.get("tier1_spam", "Not Spam")).strip()
        if tier1 not in TIER1_LABELS:
            tier1 = self._fuzzy_match(tier1, TIER1_LABELS, "Not Spam")

        # Tier 2
        tier2 = str(obj.get("tier2_toxic", "Clean")).strip()
        if tier2 not in TIER2_LABELS:
            tier2 = self._fuzzy_match(tier2, TIER2_LABELS, "Clean")

        # Tier 3
        raw_tier3 = obj.get("tier3_labels", [])
        if isinstance(raw_tier3, str):
            raw_tier3 = [raw_tier3]
        if not isinstance(raw_tier3, list):
            raw_tier3 = []

        # Filter tier3 labels based on tier2
        valid_pool = TIER3_TOXIC_LABELS if tier2 == "Toxic" else TIER3_CLEAN_LABELS
        tier3 = []
        for lbl in raw_tier3:
            lbl = str(lbl).strip()
            matched = self._fuzzy_match(lbl, valid_pool, None)
            if matched and matched not in tier3:
                tier3.append(matched)

        return {
            "tier1_spam": tier1,
            "tier2_toxic": tier2,
            "tier3_labels": tier3,
        }

    @staticmethod
    def _fuzzy_match(raw: str, candidates: List[str], default: Optional[str]) -> Optional[str]:
        """Fuzzy-match a raw string to one of the candidates."""
        raw_lower = raw.lower().strip()
        # Exact match
        for c in candidates:
            if c.lower() == raw_lower:
                return c
        # Partial match
        for c in candidates:
            if c.lower() in raw_lower or raw_lower in c.lower():
                return c
        return default

    @staticmethod
    def _default_result() -> Dict[str, Any]:
        return {
            "tier1_spam": "Not Spam",
            "tier2_toxic": "Clean",
            "tier3_labels": ["Neutral"],
        }

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, tasks: List[Dict[str, Any]], retries: int = 1, **kwargs) -> List[Dict[str, Any]]:
        """Phân loại batch bình luận qua LM Studio API (Hierarchical 3-Tier).

        Parameters
        ----------
        tasks : list[dict]
            Mỗi dict có dạng ``{"data": {"text": "..."}}``.
        retries : int
            Số lần retry nếu JSON parse thất bại (default 1).

        Returns
        -------
        list[dict]
            Mỗi dict có dạng:
            {
                "tier1_spam": "Spam" | "Not Spam",
                "tier2_toxic": "Toxic" | "Clean",
                "tier3_labels": ["label1", ...]
            }
        """
        texts = [t.get("data", {}).get("text", "") for t in tasks]
        user_msg = self._build_user_message(texts)

        payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
            "max_tokens": len(texts) * 80,  # ~80 tokens per JSON object
        }
        if self.model:
            payload["model"] = self.model

        last_error = None
        for attempt in range(1 + retries):
            try:
                r = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
                r.raise_for_status()
                result = r.json()
                content = result["choices"][0]["message"]["content"]
                parsed = self._parse_json_response(content, len(tasks))

                # Check if we got valid results (not all defaults)
                has_real = any(
                    p["tier1_spam"] != "Not Spam" or p["tier2_toxic"] != "Clean" or p["tier3_labels"] != ["Neutral"]
                    for p in parsed
                )
                if has_real or attempt == retries:
                    return parsed

            except Exception as e:
                last_error = e
                if attempt == retries:
                    break

        # All retries failed → return defaults
        return [self._default_result() for _ in range(len(tasks))]
