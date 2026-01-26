from __future__ import annotations

# Consolidated preprocessing pipeline (moved from src/Preprocess2/*)

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import json
import re
import unicodedata

import pandas as pd


@dataclass(frozen=True)
class PreprocessConfig:
    # Filtering
    len_threshold: int = 40
    space_threshold: int = 10
    require_vietnamese: bool = True

    # Text transforms
    enable_teencode: bool = True
    enable_vncorenlp: bool = True
    enable_emoji_decode: bool = True
    emoji_format_style: str = "colon"  # colon|bracket|plain

    # Resource / model paths (optional overrides)
    dictionary_dir: Optional[str] = None
    emoji_dict_filename: str = "emoji_vi.json"
    teencode_dict_filename: str = "teencode.json"

    vncorenlp_model_dir: Optional[str] = None
    java_home: Optional[str] = None

    # Output behavior
    drop_failed_rows: bool = True


@dataclass
class PreprocessResources:
    teencode_map: Dict[str, str]
    emoji_map: Dict[str, str]
    emoji_pattern: Optional[re.Pattern]

    # Optional dependencies
    vncorenlp: Optional[object] = None
    vncorenlp_available: bool = False
    vncorenlp_error: Optional[str] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _dictionary_dir(config: PreprocessConfig) -> Path:
    if config.dictionary_dir:
        return Path(config.dictionary_dir)
    return Path(__file__).resolve().parent / "dictionary"


def normalize_unicode(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def _passes_length_space_filters(text: str, *, len_threshold: int, space_threshold: int) -> bool:
    if not isinstance(text, str):
        return False
    if len(text) < len_threshold:
        return False
    if len(text.split()) < space_threshold:
        return False
    return True


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_teencode_map(dict_path: Path) -> Dict[str, str]:
    if not dict_path.exists():
        return {}
    try:
        data = _load_json(dict_path)
        return {var.lower(): std for std, variants in data.items() for var in variants}
    except Exception:
        return {}


def _match_case(original_text: str, replacement_text: str) -> str:
    if original_text.isupper():
        return replacement_text.upper()
    if original_text.istitle():
        return replacement_text.capitalize()
    return replacement_text


def teencode_replace(text: str, teencode_map: Dict[str, str]) -> str:
    if not text:
        return ""

    tokens = re.findall(r"\w+|[^\w\s]+", text, re.UNICODE)

    result = []
    for t in tokens:
        if t.isalnum():
            replacement = teencode_map.get(t.lower())
            if replacement:
                t = _match_case(t, replacement)
        result.append(t)

    raw_text = " ".join(result)
    return re.sub(r"\s+([,.?!])", r"\1", raw_text)


def _build_emoji_pattern(emoji_map: Dict[str, str]) -> Optional[re.Pattern]:
    if not emoji_map:
        return None
    sorted_emojis = sorted(emoji_map.keys(), key=len, reverse=True)
    escaped_emojis = [re.escape(e) for e in sorted_emojis]
    try:
        return re.compile("|".join(escaped_emojis))
    except re.error:
        return None


def emoji_decode(text: str, *, emoji_map: Dict[str, str], emoji_pattern: Optional[re.Pattern], format_style: str) -> str:
    if not text or not emoji_map or not emoji_pattern:
        return text or ""

    def replace_emoji(match: re.Match) -> str:
        emoji = match.group(0)
        vi_name = emoji_map.get(emoji, emoji)
        if format_style == "colon":
            return f":{vi_name}:"
        if format_style == "bracket":
            return f"[{vi_name}]"
        if format_style == "plain":
            return f" {vi_name} "
        return f":{vi_name}:"

    return emoji_pattern.sub(replace_emoji, text)


def _try_init_vncorenlp(model_dir: Optional[str], java_home: Optional[str]) -> Tuple[Optional[object], bool, Optional[str]]:
    if not model_dir:
        return None, False, "vncorenlp_model_dir not set"

    try:
        # Import lazily to keep this module usable without Java/VnCoreNLP installed.
        import os
        import sys
        import py_vncorenlp

        if java_home:
            os.environ["JAVA_HOME"] = java_home
            if sys.platform == "win32":
                os.environ["PATH"] = f"{java_home}\\bin;" + os.environ.get("PATH", "")

        model_path = Path(model_dir)
        if not model_path.exists():
            return None, False, f"VnCoreNLP model_dir not found: {model_dir}"

        rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg", "pos"],
            save_dir=str(model_path),
            max_heap_size="-Xmx2g",
        )
        return rdrsegmenter, True, None
    except Exception as e:
        return None, False, str(e)


def vncorenlp_segment(text: str, rdrsegmenter: object) -> str:
    """Segment Vietnamese with VnCoreNLP, preserving emojis/symbols via placeholders."""
    if not text:
        return ""

    icons = []

    def replace_func(match: re.Match) -> str:
        icons.append(match.group())
        return f" ICONPLACEHOLDER{len(icons) - 1} "

    pattern = re.compile(r"[^\w\s,.<>?/;:\"'\[\]{}\\\|`~!@#$%^&*()\-=_+]")
    tmp_text = pattern.sub(replace_func, text)

    try:
        segmented_sentences = rdrsegmenter.word_segment(tmp_text)
    except Exception:
        return text

    full_segmented = " ".join(segmented_sentences)

    final_str = full_segmented
    for i, icon in enumerate(icons):
        final_str = final_str.replace(f"ICONPLACEHOLDER{i}", icon)

    final_str = re.sub(r"\s+([,.:?!])", r"\1", final_str)
    final_str = re.sub(r"\s+", " ", final_str).strip()
    return final_str


def _vi_language_info(text: str) -> Tuple[bool, float]:
    """Return (is_vi, vi_probability). If langdetect isn't installed, behaves as (True, 1.0)."""
    try:
        import langdetect
        from langdetect import LangDetectException

        langdetect.DetectorFactory.seed = 0

        text = str(text)
        if not text.strip():
            return False, 0.0

        try:
            langs = langdetect.detect_langs(text)
            vi_prob = 0.0
            for i in langs:
                if i.lang == "vi":
                    vi_prob = i.prob
                    break
            is_vi = langdetect.detect(text) == "vi"
            return is_vi, float(vi_prob)
        except LangDetectException:
            return False, 0.0
    except Exception:
        # If langdetect isn't available, don't block preprocessing.
        return True, 1.0


def build_resources(config: PreprocessConfig = PreprocessConfig()) -> PreprocessResources:
    dict_dir = _dictionary_dir(config)

    teencode_map = {}
    if config.enable_teencode:
        teencode_map = _build_teencode_map(dict_dir / config.teencode_dict_filename)

    emoji_map: Dict[str, str] = {}
    if config.enable_emoji_decode:
        emoji_path = dict_dir / config.emoji_dict_filename
        if emoji_path.exists():
            try:
                emoji_map = _load_json(emoji_path)
            except Exception:
                emoji_map = {}

    emoji_pattern = _build_emoji_pattern(emoji_map)

    vncorenlp = None
    vn_ok = False
    vn_err = None
    if config.enable_vncorenlp:
        vncorenlp, vn_ok, vn_err = _try_init_vncorenlp(config.vncorenlp_model_dir, config.java_home)

    return PreprocessResources(
        teencode_map=teencode_map,
        emoji_map=emoji_map,
        emoji_pattern=emoji_pattern,
        vncorenlp=vncorenlp,
        vncorenlp_available=vn_ok,
        vncorenlp_error=vn_err,
    )


def preprocess_text(text: str, *, config: PreprocessConfig = PreprocessConfig(), resources: Optional[PreprocessResources] = None) -> str:
    resources = resources or build_resources(config)

    x = normalize_unicode(str(text) if text is not None else "")

    if config.enable_teencode and resources.teencode_map:
        x = teencode_replace(x, resources.teencode_map)

    if config.enable_vncorenlp and resources.vncorenlp_available and resources.vncorenlp is not None:
        x = vncorenlp_segment(x, resources.vncorenlp)

    if config.enable_emoji_decode and resources.emoji_map and resources.emoji_pattern is not None:
        x = emoji_decode(x, emoji_map=resources.emoji_map, emoji_pattern=resources.emoji_pattern, format_style=config.emoji_format_style)

    return x


def preprocess_df(
    df: pd.DataFrame,
    text_col: str = "text",
    *,
    config: PreprocessConfig = PreprocessConfig(),
    resources: Optional[PreprocessResources] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    resources = resources or build_resources(config)

    if text_col not in df.columns:
        raise KeyError(f"Missing column '{text_col}'")

    work = df.copy()
    work[text_col] = work[text_col].astype(str).apply(normalize_unicode)

    keep_mask = []
    vi_probs = []
    for raw in work[text_col].values:
        ok = _passes_length_space_filters(raw, len_threshold=config.len_threshold, space_threshold=config.space_threshold)
        if not ok:
            keep_mask.append(False)
            continue

        if config.require_vietnamese:
            is_vi, vi_prob = _vi_language_info(raw)
            if not is_vi:
                keep_mask.append(False)
                continue
            keep_mask.append(True)
            vi_probs.append(vi_prob)
        else:
            keep_mask.append(True)

    if config.drop_failed_rows:
        filtered = work[keep_mask].copy()
    else:
        filtered = work.copy()

    if config.require_vietnamese:
        # If drop_failed_rows=False, align vi_probs with kept rows only.
        if config.drop_failed_rows:
            filtered["vi_chance"] = vi_probs
            filtered["is_vi"] = True
        else:
            filtered["is_vi"] = False
            filtered["vi_chance"] = 0.0

    # Decode stage
    iterator = filtered[text_col]
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Preprocessing")
        except Exception:
            pass

    processed_texts = [
        preprocess_text(t, config=config, resources=resources)
        for t in iterator
    ]

    filtered[text_col] = processed_texts

    # Helpful warning for notebook users
    if config.enable_vncorenlp and not resources.vncorenlp_available and resources.vncorenlp_error:
        filtered.attrs["vncorenlp_warning"] = resources.vncorenlp_error

    return filtered


def save_processed_csv(df: pd.DataFrame, path: str, *, text_col: str = "text", index_label: str = "no") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Keep only text column by default (matches previous behavior)
    df[[text_col]].to_csv(out, index=True, encoding="utf-8-sig", index_label=index_label)

