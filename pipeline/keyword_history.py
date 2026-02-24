"""
keyword_history.py – Track which keywords have been successfully crawled
for each platform (voz, threads).  Persisted in keyword_history.json.
"""
import os
import json
import threading

_lock = threading.Lock()
_HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keyword_history.json")


def _ensure_file():
    if not os.path.isfile(_HISTORY_PATH):
        with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump({"voz": [], "threads": []}, f, ensure_ascii=False, indent=2)


def load_history() -> dict:
    """Return ``{"voz": [...], "threads": [...]}``."""
    _ensure_file()
    with _lock:
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    # guarantee keys
    data.setdefault("voz", [])
    data.setdefault("threads", [])
    return data


def add_keyword(platform: str, keyword: str) -> None:
    """Add *keyword* under *platform* if not already present, then save."""
    platform = platform.lower()
    with _lock:
        _ensure_file()
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault(platform, [])
        if keyword not in data[platform]:
            data[platform].append(keyword)
            with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


def get_keywords(platform: str) -> list:
    """Return list of crawled keywords for *platform*."""
    return load_history().get(platform.lower(), [])
