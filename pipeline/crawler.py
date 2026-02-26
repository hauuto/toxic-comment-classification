"""
crawler.py – All crawlers consolidated into one module:
  • URL-based crawlers (Facebook, YouTube, TikTok, Threads) via Playwright / YouTube API
  • Keyword-based crawlers (VOZ, Threads) via Selenium + DuckDuckGo
  • Keyword history management (keyword_history.json)
  • YouTube API crawler (from .env YOUTUBE_API_KEY)
"""
import os
import re
import csv
import json
import time
import random
import subprocess
import unicodedata
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests as _requests
import cloudscraper

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from ddgs import DDGS
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return False
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from nlp_pipeline import VietnameseCommentPreprocessor
from nlp_pipeline.warehouse import append_to_warehouse

load_dotenv()


# =========================================================================== #
#  Chrome version auto-detection (cross-machine compatibility)
# =========================================================================== #

def _detect_chrome_major_version() -> int | None:
    """Auto-detect the installed Chrome/Chromium major version.

    Tries multiple strategies so it works regardless of which Chrome version
    each team member has installed:
      1. Windows registry (most reliable on Windows)
      2. CLI ``--version`` (Linux/macOS/Windows)
      3. Windows file version via PowerShell
      4. Returns None → lets undetected-chromedriver auto-detect (fallback)

    Returns the major version number (e.g. 133) or None if detection fails.
    """
    import platform

    # --- Strategy 1: Windows registry ---
    if platform.system() == "Windows":
        try:
            import winreg
            for reg_path in [
                r"SOFTWARE\Google\Chrome\BLBeacon",
                r"SOFTWARE\Wow6432Node\Google\Chrome\BLBeacon",
            ]:
                try:
                    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path)
                    version_str, _ = winreg.QueryValueEx(key, "version")
                    winreg.CloseKey(key)
                    major = int(version_str.split(".")[0])
                    if major > 0:
                        return major
                except Exception:
                    continue
        except ImportError:
            pass

    # --- Strategy 2: CLI --version ---
    chrome_binaries: list[str] = []
    if platform.system() == "Windows":
        for p in [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        ]:
            if os.path.isfile(p):
                chrome_binaries.append(p)
    else:
        chrome_binaries = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]

    for binary in chrome_binaries:
        try:
            output = subprocess.check_output(
                [binary, "--version"], stderr=subprocess.DEVNULL, timeout=5,
            ).decode().strip()
            # "Google Chrome 133.0.6943.98" → 133
            match = re.search(r"(\d+)\.", output)
            if match:
                return int(match.group(1))
        except Exception:
            continue

    # --- Strategy 3: Windows file version via PowerShell ---
    if platform.system() == "Windows":
        for p in [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        ]:
            if os.path.isfile(p):
                try:
                    output = subprocess.check_output(
                        ["powershell", "-Command",
                         f"(Get-Item '{p}').VersionInfo.FileVersion"],
                        stderr=subprocess.DEVNULL, timeout=5,
                    ).decode().strip()
                    match = re.search(r"(\d+)\.", output)
                    if match:
                        return int(match.group(1))
                except Exception:
                    continue

    # --- Fallback: let uc auto-detect ---
    return None


# Cache the detected version so we only detect once per process
_CHROME_MAJOR_VERSION: int | None = None
_CHROME_VERSION_DETECTED: bool = False


def _get_chrome_version() -> int | None:
    """Return the cached Chrome major version, detecting on first call."""
    global _CHROME_MAJOR_VERSION, _CHROME_VERSION_DETECTED
    if not _CHROME_VERSION_DETECTED:
        _CHROME_MAJOR_VERSION = _detect_chrome_major_version()
        _CHROME_VERSION_DETECTED = True
    return _CHROME_MAJOR_VERSION


# =========================================================================== #
#  Threads-specific text cleaning helpers
# =========================================================================== #

# Patterns for Threads UI / system noise that should never be scraped
_THREADS_UI_SKIP_PATTERNS = re.compile(
    r"^("
    # Login / signup prompts
    r"hãy đăng nhập.*thread.*|"
    r"đăng nhập hoặc đăng ký.*|"
    r"log in to see more.*|"
    r"tiếp tục bằng instagram|"
    r"continue with instagram|"
    r"tiếp tục với instagram|"
    r"dùng ứng dụng|use app|"
    # Policy / legal footer
    r"chính sách quyền riêng tư.*|"
    r"privacy policy.*|"
    r"điều khoản.*|terms.*|"
    # Reply-to prefixes that got scraped as standalone text
    r"đang trả lời\s*<?.*|"
    r"replying to\s*<?.*|"
    # Pure username lines (no spaces, only alphanumeric + _ + .)
    r"[a-z0-9_.]{3,30}"
    r")$",
    re.IGNORECASE,
)

# Trailing "Translate" / "Dịch" button text stuck to comment
_THREADS_TRAILING_TRANSLATE = re.compile(r"(?:[Tt]ranslate|Dịch)\s*$")

# "gia đình" repeated anomaly from emoji <img alt="gia đình"> leak
_THREADS_GIA_DINH_SPAM = re.compile(r"(?:\s*gia đình\s*){2,}:?")
# Single stray "gia đình" token right before/after emoji token or at boundaries
# Also consume trailing colon that may be part of broken emoji syntax "gia đình:"
_THREADS_GIA_DINH_NEAR_EMOJI = re.compile(r"\s*gia đình\s*:?\s*(?=:|$)")


def _clean_threads_text(text: str) -> str | None:
    """Clean a raw Threads comment text. Returns None if text should be skipped."""
    if not text or not text.strip():
        return None

    text = text.strip()

    # Skip UI / system noise
    if _THREADS_UI_SKIP_PATTERNS.fullmatch(text):
        return None

    # Remove trailing "Translate" / "Dịch" button text
    text = _THREADS_TRAILING_TRANSLATE.sub("", text).strip()

    # Fix "gia đình" spam from emoji alt text leak
    text = _THREADS_GIA_DINH_SPAM.sub(" ", text)
    text = _THREADS_GIA_DINH_NEAR_EMOJI.sub(" ", text)

    # Collapse whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()

    if not text:
        return None

    return text


# =========================================================================== #
#  Keyword History – persisted in keyword_history.json
# =========================================================================== #

_kw_lock = threading.Lock()
_KW_HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keyword_history.json")


def _ensure_kw_file():
    if not os.path.isfile(_KW_HISTORY_PATH):
        with open(_KW_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump({"voz": [], "threads": []}, f, ensure_ascii=False, indent=2)


def load_keyword_history() -> dict:
    """Return ``{"voz": [...], "threads": [...]}``."""
    _ensure_kw_file()
    with _kw_lock:
        with open(_KW_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.setdefault("voz", [])
    data.setdefault("threads", [])
    return data


def add_keyword_to_history(platform: str, keyword: str) -> None:
    """Add *keyword* under *platform* if not already present, then save."""
    platform = platform.lower()
    with _kw_lock:
        _ensure_kw_file()
        with open(_KW_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault(platform, [])
        if keyword not in data[platform]:
            data[platform].append(keyword)
            with open(_KW_HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


def get_history_keywords(platform: str) -> list:
    """Return list of crawled keywords for *platform*."""
    return load_keyword_history().get(platform.lower(), [])


# =========================================================================== #
#  Shared helpers
# =========================================================================== #

def _sanitize_keyword(keyword: str) -> str:
    """Remove diacritics, replace spaces with underscores."""
    text = unicodedata.normalize("NFD", keyword)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return text.replace(" ", "_")


def _append_batch_to_csv(file_path: str, batch_data: List[str]) -> int:
    """Incremental CSV writer (id auto-increment)."""
    if not batch_data:
        return 0
    file_exists = os.path.isfile(file_path)
    start_id = 1
    if file_exists:
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        rid = int(row["id"])
                        if rid >= start_id:
                            start_id = rid + 1
                    except (ValueError, KeyError):
                        pass
        except Exception:
            start_id = 1
    with open(file_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text"])
        if not file_exists:
            writer.writeheader()
        for i, text in enumerate(batch_data):
            writer.writerow({"id": start_id + i, "text": text})
    return len(batch_data)


# =========================================================================== #
#  YouTube API Crawler
# =========================================================================== #

def _extract_video_id(url: str) -> str:
    """Extract the 11-char video ID from various YouTube URL formats."""
    patterns = [r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})"]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url.strip()):
        return url.strip()
    return ""


def _sanitize_error(error_msg) -> str:
    """Remove API keys from error messages to prevent leaking secrets."""
    return re.sub(r'key=[A-Za-z0-9_-]+', 'key=***REDACTED***', str(error_msg))


def extract_youtube_comments(
    url_or_id: str, *, log_callback=None, data_callback=None, stop_event=None,
    preprocessor=None, use_decoder: bool = True, use_filter: bool = True,
    use_normalizer: bool = True, use_segmentor: bool = True,
    current_id: int = 1, seen_texts: set = None, extracted_data: list = None,
) -> int:
    """Fetch all comments for a YouTube video via the Data API v3."""
    def log(msg):
        if log_callback: log_callback(msg)
        else: print(msg)

    if seen_texts is None: seen_texts = set()
    if extracted_data is None: extracted_data = []

    video_id = _extract_video_id(url_or_id)
    if not video_id:
        log(f"Không thể trích xuất Video ID từ: {url_or_id}")
        return current_id

    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        log("Lỗi: Không tìm thấy YOUTUBE_API_KEY trong file .env")
        return current_id

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
    except Exception as e:
        log(f"Lỗi khởi tạo YouTube API client: {_sanitize_error(e)}")
        return current_id

    log(f"[YouTube API] Bắt đầu lấy bình luận cho video: {video_id}")
    next_page_token = None
    total_fetched = 0

    while True:
        if stop_event and stop_event.is_set():
            log("[YouTube API] Đã nhận lệnh DỪNG."); break
        try:
            request = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                pageToken=next_page_token, textFormat="plainText",
            )
            response = request.execute()
        except HttpError as e:
            if e.resp.status in (400, 404):
                log(f"[YouTube API] Đã lấy hết bình luận khả dụng (API trả về {e.resp.status}).")
            elif e.resp.status == 403:
                log(f"[YouTube API] Hết quota hoặc bị từ chối: {_sanitize_error(e)}")
            else:
                log(f"[YouTube API] Lỗi HTTP {e.resp.status}: {_sanitize_error(e)}")
            break
        except Exception as e:
            log(f"[YouTube API] Lỗi gọi API: {_sanitize_error(e)}"); break

        items = response.get("items", [])
        if not items: break

        new_batch = []
        for item in items:
            if stop_event and stop_event.is_set(): break
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            full_text = snippet.get("textDisplay", "").strip()
            if not full_text or (len(full_text) < 2 and full_text not in ["Ok", "Dạ"]):
                continue
            if preprocessor:
                processed = preprocessor.process_comment(full_text, use_decoder=use_decoder, use_filter=use_filter, use_normalizer=use_normalizer, use_segmentor=use_segmentor)
                if not processed["is_valid"]: continue
                clean_text = processed["cleaned_text"]
            else:
                clean_text = full_text
            if clean_text not in seen_texts:
                seen_texts.add(clean_text)
                row = {"id": current_id, "text": clean_text}
                new_batch.append(row); extracted_data.append(row); current_id += 1

        if new_batch:
            total_fetched += len(new_batch)
            if data_callback: data_callback(new_batch)
            log(f"[YouTube API] +{len(new_batch)} bình luận (Tổng: {total_fetched})")

        next_page_token = response.get("nextPageToken")
        if not next_page_token: break

    log(f"[YouTube API] Hoàn tất video {video_id} – tổng {total_fetched} bình luận.")
    return current_id


# =========================================================================== #
#  Playwright URL crawlers (Facebook, TikTok, Threads-URL)
# =========================================================================== #

def _extract_facebook(page, log, current_id, seen_texts, stop_event, preprocessor, use_decoder, use_filter, use_normalizer, use_segmentor, extracted_data, data_callback):
    log("Đợi trang ổn định...")
    time.sleep(5)
    
    # Close dialog
    try:
        close_buttons = page.locator('div[aria-label="Đóng"], div[aria-label="Close"], i[data-visualcompletion="css-img"]')
        if close_buttons.count() > 0:
            log("Đóng popup...")
            close_buttons.first.click(timeout=3000)
            time.sleep(2)
    except Exception:
        pass 
        
    replace_tags_js = r"""
        const links = document.querySelectorAll('div[dir="auto"] a');
        links.forEach(a => {
            const textInner = a.innerText ? a.innerText.trim() : "";
            if (textInner && !textInner.startsWith('@')) {
                const formatted = '@' + textInner.replace(/\s+/g, '_');
                const textNode = document.createTextNode(formatted);
                a.parentNode.replaceChild(textNode, a);
            }
        });
        const images = document.querySelectorAll('div[dir="auto"] img');
        images.forEach(img => {
            if (img.alt) {
                const textNode = document.createTextNode(img.alt);
                img.parentNode.replaceChild(textNode, img);
            }
        });
    """

    log("Bắt đầu quét dữ liệu tuần tự Facebook...")
    
    max_empty_scrolls = 5
    empty_scrolls = 0
    
    while True:
        if stop_event and stop_event.is_set():
            return current_id

        view_more_selectors = ['span:text-is("Xem thêm bình luận")', 'span:text-is("View more comments")', 'div[role="button"]:has-text("Xem thêm bình luận")']
        for selector in view_more_selectors:
            try:
                elements = page.locator(selector)
                for i in range(elements.count()):
                    elements.nth(i).click(timeout=2000)
                    time.sleep(1)
            except Exception:
                pass
        
        reply_selectors = ['span:has-text(" xem phản hồi")', 'span:has-text(" replies")', 'div[role="button"]:has-text(" xem phản hồi")']
        for selector in reply_selectors:
            try:
                elements = page.locator(selector)
                for i in range(elements.count()):
                    elements.nth(i).click(timeout=2000)
                    time.sleep(1)
            except Exception:
                pass
                
        see_more_selectors = ['div[dir="auto"][role="button"]:has-text("Xem thêm")', 'div[dir="auto"][role="button"]:has-text("See more")']
        for selector in see_more_selectors:
            try:
                elements = page.locator(selector)
                for i in range(elements.count()):
                    if elements.nth(i).is_visible():
                        elements.nth(i).click(timeout=1000)
                        time.sleep(0.5)
            except Exception:
                pass
        
        time.sleep(2)
        page.evaluate(replace_tags_js)
        
        comment_articles = page.locator('div[role="article"]')
        count = comment_articles.count()
        
        new_batch = []
        start_idx = 1 if count > 0 else 0
        
        for i in range(start_idx, count):
            try:
                article = comment_articles.nth(i)
                text_blocks = article.locator('div[dir="auto"][style*="text-align: start"]')
                
                comment_lines = []
                for j in range(text_blocks.count()):
                    fragment = text_blocks.nth(j).text_content().strip()
                    if fragment and fragment not in ['Thích', 'Phản hồi', 'Chia sẻ', 'Like', 'Reply', 'Share', 'Báo cáo', 'Report', 'Đã chỉnh sửa', 'Edited']:
                        comment_lines.append(fragment)
                
                full_text = "\n".join(comment_lines).strip()
                
                if not full_text: continue
                if len(full_text) < 2 and full_text not in ['Ok', 'Dạ']: continue
                     
                if preprocessor:
                    processed = preprocessor.process_comment(full_text, use_decoder=use_decoder, use_filter=use_filter, use_normalizer=use_normalizer, use_segmentor=use_segmentor)
                    if not processed["is_valid"]: continue
                    clean_text = processed["cleaned_text"]
                else:
                    clean_text = full_text
                     
                if clean_text not in seen_texts:
                    seen_texts.add(clean_text)
                    item = {'id': current_id, 'text': clean_text}
                    new_batch.append(item)
                    extracted_data.append(item)
                    current_id += 1
            except Exception:
                pass
        
        if new_batch:
            if data_callback: data_callback(new_batch)
            log(f"Đã quét thêm {len(new_batch)} bình luận (Tổng Facebook: {len(seen_texts)})...")
            empty_scrolls = 0
        else:
            empty_scrolls += 1
            
        if empty_scrolls >= max_empty_scrolls:
            break
            
        page.mouse.wheel(0, 3000)
        time.sleep(2)
    return current_id

def _extract_tiktok(page, log, current_id, seen_texts, stop_event, preprocessor, use_decoder, use_filter, use_normalizer, use_segmentor, extracted_data, data_callback):
    log("Đợi trang ổn định TikTok...")
    time.sleep(5)
    
    log("Kiểm tra xác minh bot (Captcha) / Đợi tải trang...")
    for _ in range(60):
        try:
            # Nhảy ra khỏi vòng chờ ngay khi trang đã tải xong (hiện ô bình luận, hoặc hiện nút chuyển Tab Bình luận)
            if page.locator('div[data-e2e="comment-input"], p[data-e2e="comment-level-1"], div[class*="DivCommentItemContainer"], div[role="tab"]:has-text("Bình luận"), div[role="tab"]:has-text("Comments"), span:has-text("Bình luận ("), span:has-text("Comments (")').count() > 0:
                break
            time.sleep(1)
        except Exception:
            pass

    try:
        log("Kiểm tra để chuyển sang tab Bình luận (nếu có)...")
        tab_locators = page.locator('div[role="tab"]:has-text("Bình luận"), div[role="tab"]:has-text("Comments"), span:has-text("Bình luận ("), span:has-text("Comments (")')
        for i in range(tab_locators.count()):
            try:
                if tab_locators.nth(i).is_visible():
                    tab_locators.nth(i).click(timeout=1000)
                    time.sleep(2)
                    break
            except Exception:
                pass
    except Exception:
        pass
        
    # Try to close login popup if exists
    try:
        close_btn = page.locator('div[data-e2e="modal-close-inner-button"], div[class*="login-modal"] button, svg.tiktok-11nmmz')
        if close_btn.count() > 0:
            log("Đóng popup...")
            close_btn.first.click(timeout=2000)
    except Exception:
        pass
        
    log("Bắt đầu quét dữ liệu TikTok...")
    max_empty_scrolls = 5
    empty_scrolls = 0
    
    while True:
        if stop_event and stop_event.is_set():
            return current_id
            
        # Expand replies
        try:
            # TikTok "Xem thêm N câu trả lời" / "View more replies" / data-e2e="comment-more-replies"
            replies_btns = page.locator('div[data-e2e="comment-more-replies"], span:has-text("Xem thêm"), span:has-text("View more")')
            for i in range(replies_btns.count()):
                try:
                    if replies_btns.nth(i).is_visible():
                        replies_btns.nth(i).click(timeout=1000)
                        time.sleep(0.5)
                except Exception:
                    pass
        except Exception:
            pass
            
        time.sleep(2)
        
        # Use JS to grab all comment text directly. TikTok DOM is tricky for standard locators
        extract_comments_js = r"""
            () => {
                const commentNodes = document.querySelectorAll('p[data-e2e="comment-level-1"], p[data-e2e="comment-level-2"], span[data-e2e="comment-level-1"], div[class*="DivCommentItemContainer"] span, div[class*="CommentText"], span[class*="SpanCommentContent"]');
                const results = [];
                commentNodes.forEach(node => {
                    // clone node to avoid mutating the live DOM dangerously if not needed
                    const clone = node.cloneNode(true);
                    const images = clone.querySelectorAll('img');
                    images.forEach(img => {
                        if (img.alt) {
                            const textNode = document.createTextNode(img.alt);
                            img.parentNode.replaceChild(textNode, img);
                        }
                    });
                    const text = clone.innerText || clone.textContent;
                    if (text && text.trim()) {
                        results.push(text.trim());
                    }
                });
                return results;
            }
        """
        
        raw_comments = page.evaluate(extract_comments_js)
        new_batch = []
        
        for full_text in raw_comments:
            try:
                if not full_text: continue
                if len(full_text) < 2 and full_text not in ['Ok', 'Dạ']: continue
                
                if preprocessor:
                    processed = preprocessor.process_comment(full_text, use_decoder=use_decoder, use_filter=use_filter, use_normalizer=use_normalizer, use_segmentor=use_segmentor)
                    if not processed["is_valid"]: continue
                    clean_text = processed["cleaned_text"]
                else:
                    clean_text = full_text
                    
                if clean_text not in seen_texts:
                    seen_texts.add(clean_text)
                    item = {'id': current_id, 'text': clean_text}
                    new_batch.append(item)
                    extracted_data.append(item)
                    current_id += 1
            except Exception:
                pass
                
        if new_batch:
            if data_callback: data_callback(new_batch)
            log(f"Đã quét thêm {len(new_batch)} bình luận (Tổng TikTok: {len(seen_texts)})...")
            empty_scrolls = 0
        else:
            empty_scrolls += 1
            
        if empty_scrolls >= max_empty_scrolls:
            break
            
        page.mouse.wheel(0, 2000)
        time.sleep(2)
    return current_id

def _extract_threads(page, log, current_id, seen_texts, stop_event, preprocessor, use_decoder, use_filter, use_normalizer, use_segmentor, extracted_data, data_callback):
    log("Đợi trang ổn định Threads...")
    time.sleep(5)
    
    log("Bắt đầu quét dữ liệu Threads...")
    max_empty_scrolls = 5
    empty_scrolls = 0

    # JS to clean DOM before text extraction:
    # 1. Remove "Translate"/"Dịch" button elements so they don't leak into text_content()
    # 2. Replace emoji <img alt="..."> with proper :token: placeholders (not raw alt text)
    #    This prevents "gia đình" alt text from leaking as repeated garbage
    _threads_dom_cleanup_js = r"""
        // Remove Translate / Dịch buttons (they sit as <span role="link"> near comments)
        document.querySelectorAll('span[role="link"], div[role="button"]').forEach(el => {
            const t = (el.textContent || '').trim().toLowerCase();
            if (t === 'translate' || t === 'dịch' || t === 'see translation' || t === 'xem bản dịch') {
                el.remove();
            }
        });

        // Replace emoji <img> with empty string (decoder will handle Unicode emoji later)
        // The alt text of Threads emoji images is often misleading ("gia đình", etc.)
        document.querySelectorAll('div[dir="auto"] img, span[dir="auto"] img').forEach(img => {
            if (img.alt && img.width && img.width <= 20) {
                // Small image = emoji icon. Remove entirely; unicode emoji in text is enough.
                img.remove();
            }
        });
    """

    while True:
        if stop_event and stop_event.is_set():
            return current_id
            
        time.sleep(2)

        # --- Click "See more" / "Xem thêm" to expand truncated comments ---
        see_more_selectors = [
            'div[role="button"]:has-text("See more")',
            'div[role="button"]:has-text("Xem thêm")',
            'span[role="link"]:has-text("See more")',
            'span[role="link"]:has-text("Xem thêm")',
        ]
        for sel in see_more_selectors:
            try:
                buttons = page.locator(sel)
                for bi in range(buttons.count()):
                    try:
                        if buttons.nth(bi).is_visible():
                            buttons.nth(bi).click(timeout=1000)
                            time.sleep(0.5)
                    except Exception:
                        pass
            except Exception:
                pass

        # Clean DOM before extraction
        try:
            page.evaluate(_threads_dom_cleanup_js)
        except Exception:
            pass

        # Threads DOM is obfuscated. Use generic span/div with dir="auto"
        comment_blocks = page.locator('div[data-pressable-container="true"]')
        count = comment_blocks.count()
        # Fallback to generic selector if no pressable containers found
        if count == 0:
            comment_blocks = page.locator('span[dir="auto"], div[dir="auto"]')
            count = comment_blocks.count()
        new_batch = []
        
        for i in range(count):
            try:
                full_text = comment_blocks.nth(i).text_content().strip()
                if not full_text:
                    continue

                # Apply Threads-specific text cleaning
                full_text = _clean_threads_text(full_text)
                if not full_text:
                    continue

                if len(full_text) < 2 and full_text not in ['Ok', 'Dạ']:
                    continue

                if preprocessor:
                    processed = preprocessor.process_comment(full_text, use_decoder=use_decoder, use_filter=use_filter, use_normalizer=use_normalizer, use_segmentor=use_segmentor)
                    if not processed["is_valid"]: continue
                    clean_text = processed["cleaned_text"]
                else:
                    clean_text = full_text
                    
                if clean_text not in seen_texts:
                    seen_texts.add(clean_text)
                    item = {'id': current_id, 'text': clean_text}
                    new_batch.append(item)
                    extracted_data.append(item)
                    current_id += 1
            except Exception:
                pass
                
        if new_batch:
            if data_callback: data_callback(new_batch)
            log(f"Đã quét thêm {len(new_batch)} bình luận (Tổng Threads: {len(seen_texts)})...")
            empty_scrolls = 0
        else:
            empty_scrolls += 1
            
        if empty_scrolls >= max_empty_scrolls:
            break
            
        page.mouse.wheel(0, 3000)
        time.sleep(2)
    return current_id

def extract_comments_stream(url_input: str, headless: bool = False, 
                            use_decoder: bool = True,
                            use_filter: bool = True,
                            use_normalizer: bool = True,
                            use_segmentor: bool = True,
                            segmentor_backend: str = "vncorenlp",
                            preprocessor: VietnameseCommentPreprocessor | None = None,
                            log_callback=None, data_callback=None, stop_event=None):
    """
    Crawls comments from URLs (FB, YT, TikTok, Threads). Semicolon-separated.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    urls = [u.strip() for u in url_input.split(';') if u.strip()]
    if not urls:
        log("Không tìm thấy URL hợp lệ.")
        return []

    log(f"Đã nhận {len(urls)} URL để quét.")
    
    extracted_data = [] # Full history (optional, callbacks usually handle the main data)
    seen_texts = set()
    current_id = 1
    
    try:
        if preprocessor is None:
            if use_decoder or use_filter or use_normalizer or use_segmentor:
                log("Đang khởi tạo bộ tiền xử lý NLP... Vui lòng đợi vài giây...")
                preprocessor = VietnameseCommentPreprocessor(segmentor_backend=segmentor_backend)
            else:
                preprocessor = None
            
        with sync_playwright() as p:
            log("Mở trình duyệt...")
            browser = p.chromium.launch(headless=headless, args=['--disable-notifications'])
            context = browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            for url in urls:
                if stop_event and stop_event.is_set():
                    log("Đã nhận lệnh DỪNG từ người dùng!")
                    break
                    
                log(f"--- Bắt đầu xử lý URL: {url} ---")

                url_lower = url.lower()

                # YouTube: use API directly (no Playwright page needed)
                if "youtube.com" in url_lower or "youtu.be" in url_lower:
                    current_id = extract_youtube_comments(
                        url,
                        log_callback=log,
                        data_callback=data_callback,
                        stop_event=stop_event,
                        preprocessor=preprocessor,
                        use_decoder=use_decoder,
                        use_filter=use_filter,
                        use_normalizer=use_normalizer,
                        use_segmentor=use_segmentor,
                        current_id=current_id,
                        seen_texts=seen_texts,
                        extracted_data=extracted_data,
                    )
                    log(f"--- Hoàn tất trang {url} ---")
                    continue

                page = context.new_page()
                
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=60000)
                except Exception as e:
                    log(f"Lỗi tải trang {url}: {e}")
                    page.close()
                    continue
                
                try:
                    # Auto-detect platform
                    if "facebook.com" in url_lower or "fb.com" in url_lower or "fb.watch" in url_lower:
                        current_id = _extract_facebook(page, log, current_id, seen_texts, stop_event, preprocessor, use_decoder, use_filter, use_normalizer, use_segmentor, extracted_data, data_callback)
                    elif "tiktok.com" in url_lower:
                        current_id = _extract_tiktok(page, log, current_id, seen_texts, stop_event, preprocessor, use_decoder, use_filter, use_normalizer, use_segmentor, extracted_data, data_callback)
                    elif "threads.net" in url_lower or "threads.com" in url_lower:
                        current_id = _extract_threads(page, log, current_id, seen_texts, stop_event, preprocessor, use_decoder, use_filter, use_normalizer, use_segmentor, extracted_data, data_callback)
                    else:
                        log(f"Nền tảng không được hỗ trợ (chỉ FB, YT, TikTok, Threads): {url}")
                except Exception as e:
                    log(f"Lỗi khi xử lý dữ liệu từ {url}: {str(e)}")
                
                page.close()
                log(f"--- Hoàn tất trang {url} ---")
                
            browser.close()
            log("--- Kết thúc quá trình Playwright ---")
            
    except Exception as e:
        log(f"Lỗi Crawler: {str(e)}")
        
    return extracted_data


# =========================================================================== #
#  VOZ Keyword Crawler
# =========================================================================== #

class VOZCrawler:
    """VOZ forum crawler – search keyword via DuckDuckGo, scrape comments.

    Performance optimisations
    -------------------------
    • **cloudscraper fast-path** – uses ``cloudscraper`` to bypass Cloudflare
      JS challenges without a browser (10-50× faster than Selenium).  Falls
      back to Selenium only when cloudscraper cannot solve the challenge.
    • **Concurrent thread scraping** – ``num_workers`` Chrome-driver instances
      (default 3) scrape different VOZ threads in parallel via
      ``ThreadPoolExecutor``.  Driver creation is staggered via a semaphore
      to avoid Chrome port conflicts.
    • **Per-domain rate limiter** – max 2 requests/s to ``voz.vn`` so we
      don't get IP-banned when cloudscraper makes requests much faster than
      Selenium did.
    • **Eager page-load strategy** – Selenium no longer waits for images /
      stylesheets / ads to finish loading.
    • **Extended blocked resources** – images, fonts, CSS, video assets are
      blocked via CDP so the browser has less work to do.
    • **Reduced sleep delays** – page-load waits reduced from 2-3 s → 0.8-1.5 s.
    • **URL deduplication** – normalises DuckDuckGo results to avoid scraping
      the same VOZ thread twice.
    • **Retry with backoff** – failed threads are retried up to 2 times before
      giving up, so transient Selenium errors don't lose data.
    """

    # Shared across all workers for the cloudscraper fast-path
    _http_session: cloudscraper.CloudScraper | None = None
    _http_session_lock = threading.Lock()

    # Rate-limiter: track the last request timestamp (per-class)
    _rate_limit_lock = threading.Lock()
    _last_request_time: float = 0.0
    _MIN_REQUEST_INTERVAL: float = 0.5  # max ~2 req/s

    # Semaphore to stagger Selenium driver creation (avoid port collisions)
    _driver_create_semaphore = threading.Semaphore(1)

    _BLOCKED_URLS = [
        "*googleads*", "*doubleclick*", "*googlesyndication*",
        "*adservice*", "*google_vignette*",
        "*.gif", "*.png", "*.jpg", "*.jpeg", "*.webp", "*.svg",
        "*.woff", "*.woff2", "*.ttf", "*.eot",
        "*.mp4", "*.webm",
        "*.css",
    ]

    def __init__(self, keyword=None, max_threads=10, max_pages=50, timeout=20,
                 offset_x=-1000, log_callback=None, stop_event=None, data_callback=None,
                 preprocessor=None, use_decoder=True, use_filter=True,
                 use_normalizer=True, use_segmentor=True, num_workers=3):
        self.keyword = keyword
        self.max_threads = max_threads
        self.max_pages = max_pages
        self.timeout = timeout
        self.offset_x = offset_x
        self.log_callback = log_callback
        self.stop_event = stop_event
        self.data_callback = data_callback
        self.preprocessor = preprocessor
        self.use_decoder = use_decoder
        self.use_filter = use_filter
        self.use_normalizer = use_normalizer
        self.use_segmentor = use_segmentor
        self.num_workers = max(1, num_workers)
        self.driver = None
        # Pool of drivers for concurrent scraping (worker_id -> driver)
        self._driver_pool: dict[int, object] = {}
        self._driver_pool_lock = threading.Lock()

    def _log(self, msg: str):
        if self.log_callback: self.log_callback(msg)
        else: print(msg)

    def _stopped(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def get_output_path(self, keyword=None) -> str:
        keyword = keyword or self.keyword
        clean = _sanitize_keyword(keyword)
        return os.path.join(os.getcwd(), f"{clean}_{self.max_threads}_{self.max_pages}.csv")

    # ── HTTP session (cloudscraper fast-path) ────────────────────────────────
    @classmethod
    def _get_http_session(cls) -> cloudscraper.CloudScraper:
        """Lazily create a shared ``cloudscraper`` session that can bypass Cloudflare."""
        if cls._http_session is None:
            with cls._http_session_lock:
                if cls._http_session is None:
                    s = cloudscraper.create_scraper(
                        browser={
                            "browser": "chrome",
                            "platform": "windows",
                            "desktop": True,
                        },
                    )
                    s.headers.update({
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                    })
                    cls._http_session = s
        return cls._http_session

    @classmethod
    def _rate_limit_wait(cls):
        """Enforce a minimum interval between HTTP requests to avoid IP bans."""
        with cls._rate_limit_lock:
            now = time.monotonic()
            elapsed = now - cls._last_request_time
            if elapsed < cls._MIN_REQUEST_INTERVAL:
                time.sleep(cls._MIN_REQUEST_INTERVAL - elapsed)
            cls._last_request_time = time.monotonic()

    def _fast_fetch_page(self, url: str, timeout: int = 15) -> str | None:
        """Try to fetch a VOZ page via ``cloudscraper``.

        Returns the HTML string on success, or ``None`` if Cloudflare or
        another protection is detected (caller should fall back to Selenium).
        """
        try:
            self._rate_limit_wait()
            session = self._get_http_session()
            resp = session.get(url, timeout=timeout, allow_redirects=True)
            if resp.status_code in (403, 503):
                return None
            resp.raise_for_status()
            text = resp.text
            # Cloudflare challenge detection (cloudscraper handles most, but
            # Turnstile or advanced challenges may still slip through)
            if "Just a moment" in text or "cf-browser-verification" in text:
                return None
            # Sanity check: page should contain VOZ forum content
            if ".message-inner" not in text and "bbWrapper" not in text:
                return None
            return text
        except cloudscraper.exceptions.CloudflareChallengeError:
            return None
        except Exception:
            return None

    # ── Selenium driver management ─────────────────────────────────────────
    def _create_driver(self, worker_id: int = 0):
        """Create a new undetected-chromedriver instance (eager strategy).

        Uses a class-level semaphore to ensure only one driver is created at a
        time, preventing Chrome port collisions when multiple workers start
        simultaneously.
        """
        with self._driver_create_semaphore:
            chrome_ver = _get_chrome_version()
            self._log(f"[VOZ] Khởi tạo Chrome Driver (worker {worker_id}) — Chrome v{chrome_ver or 'auto'}...")
            user_data_dir = os.path.join(os.getcwd(), f"chrome_profile_voz_w{worker_id}")
            opts = uc.ChromeOptions()
            opts.add_argument("--disable-blink-features=AutomationControlled")
            opts.page_load_strategy = "eager"          # ← no longer waits for images/CSS
            driver = uc.Chrome(options=opts, user_data_dir=user_data_dir,
                               version_main=chrome_ver, use_subprocess=True)
            try:
                driver.set_window_position(self.offset_x, 0)
                time.sleep(0.5)
                driver.maximize_window()
            except Exception:
                pass
            driver.set_page_load_timeout(self.timeout)
            try:
                driver.execute_cdp_cmd("Network.enable", {})
                driver.execute_cdp_cmd("Network.setBlockedURLs",
                                       {"urls": self._BLOCKED_URLS})
            except Exception:
                pass
            # Small delay after creation so the next worker doesn't collide
            time.sleep(1.0)
            return driver

    def get_driver(self):
        """Legacy single-driver entry point (used when num_workers == 1)."""
        self.driver = self._create_driver(worker_id=0)
        return self.driver

    def _get_pool_driver(self, worker_id: int):
        """Get or create the Selenium driver for *worker_id*."""
        with self._driver_pool_lock:
            drv = self._driver_pool.get(worker_id)
        if drv is not None:
            return drv
        drv = self._create_driver(worker_id)
        with self._driver_pool_lock:
            self._driver_pool[worker_id] = drv
        return drv

    @staticmethod
    def _kill_driver(driver):
        pid = None
        if driver:
            try: pid = driver.service.process.pid
            except Exception: pass
            try: driver.quit()
            except Exception: pass
        if pid:
            try:
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    def force_kill_driver(self, driver=None):
        driver = driver or self.driver
        self._kill_driver(driver)
        if driver is self.driver:
            self.driver = None

    def close(self):
        """Shut down **all** drivers (single + pool)."""
        self.force_kill_driver()
        with self._driver_pool_lock:
            for drv in self._driver_pool.values():
                self._kill_driver(drv)
            self._driver_pool.clear()

    def restart_driver(self):
        self.force_kill_driver()
        return self.get_driver()

    def _restart_pool_driver(self, worker_id: int):
        with self._driver_pool_lock:
            old = self._driver_pool.pop(worker_id, None)
        if old:
            self._kill_driver(old)
        drv = self._create_driver(worker_id)
        with self._driver_pool_lock:
            self._driver_pool[worker_id] = drv
        return drv

    # ── DuckDuckGo search ──────────────────────────────────────────────────
    @staticmethod
    def _normalize_voz_url(url: str) -> str:
        """Normalise a VOZ thread URL for deduplication.

        Strips query params, fragments, trailing ``/page-*`` suffixes, and
        trailing slashes so that variants of the same thread compare equal.
        """
        # Remove query string and fragment
        url = url.split("?")[0].split("#")[0]
        # Remove /page-N suffix
        url = re.sub(r"/page-\d+$", "", url)
        return url.rstrip("/")

    def search_voz(self, keyword=None, limit=None) -> List[str]:
        keyword = keyword or self.keyword
        limit = limit or self.max_threads
        self._log(f"[VOZ] Tìm link cho: {keyword}")
        results: List[str] = []
        seen_normalised: set[str] = set()
        try:
            with DDGS() as ddgs:
                gen = ddgs.text(f"site:voz.vn {keyword}", max_results=limit + 10)
                for r in gen:
                    href = r.get("href")
                    if href and "/t/" in href:
                        normalised = self._normalize_voz_url(href)
                        if normalised not in seen_normalised:
                            seen_normalised.add(normalised)
                            # Store the normalised (clean) URL
                            results.append(normalised)
                    if len(results) >= limit:
                        break
        except Exception as e:
            self._log(f"[VOZ] Lỗi search: {e}")
        self._log(f"[VOZ] Tìm thấy {len(results)} thread links (sau khi loại trùng).")
        return results

    # ── Page-level HTML → comments parser ──────────────────────────────────
    @staticmethod
    def _parse_comments_from_html(html: str) -> tuple[List[str], bool]:
        """Parse VOZ comments from raw HTML.

        Returns ``(comments, has_next_page)``.
        """
        soup = BeautifulSoup(html, "html.parser")
        posts = soup.select(".message-inner")
        comments: List[str] = []
        for post in posts:
            content_tag = post.select_one(".bbWrapper")
            if content_tag:
                for q in content_tag.find_all("blockquote"):
                    q.decompose()
                for br in content_tag.find_all("br"):
                    br.replace_with("\n")
                text = content_tag.get_text(separator="\n", strip=True)
                text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
                text = re.sub(r"\n+", "\n", text)
                if text:
                    comments.append(text)
        has_next = soup.select_one("a.pageNav-jump--next") is not None
        return comments, has_next

    # ── Scrape a single VOZ thread (all pages) ─────────────────────────────
    def scrape_comments(self, url: str, max_pages=None,
                        worker_id: int = 0) -> List[str]:
        """Scrape all comments from a VOZ thread.

        Tries the ``cloudscraper`` fast-path first; falls back to Selenium on
        Cloudflare Turnstile or other unsolvable challenges.
        """
        max_pages = max_pages or self.max_pages
        current_thread_comments: List[str] = []
        base_url = url.split("/page-")[0].rstrip("/")
        self._log(f"  -> Scraping (w{worker_id}): {base_url}")

        # Track whether the fast-path is viable for this thread
        use_fast = True
        fast_path_logged = False

        for page in range(1, max_pages + 1):
            if self._stopped():
                break
            target_url = base_url if page == 1 else f"{base_url}/page-{page}"

            html: str | None = None

            # ── Fast path: cloudscraper ─────────────────────────────────
            if use_fast:
                html = self._fast_fetch_page(target_url)
                if html is not None:
                    comments, has_next = self._parse_comments_from_html(html)
                    if comments:
                        if not fast_path_logged:
                            self._log(f"    [FAST] cloudscraper OK (w{worker_id})")
                            fast_path_logged = True
                        current_thread_comments.extend(comments)
                        if not has_next:
                            break
                        continue
                    # Empty page – maybe Cloudflare slipped through
                    html = None
                # Cloudflare detected → disable fast path for remaining pages
                use_fast = False
                self._log(f"    [INFO] Fast-path unavailable, switching to Selenium (w{worker_id})")

            # ── Slow path: Selenium ────────────────────────────────────
            try:
                drv = self._get_pool_driver(worker_id)
                try:
                    drv.get(target_url)
                except TimeoutException:
                    try:
                        drv.execute_script("window.stop();")
                    except Exception:
                        pass

                if "Just a moment" in (drv.title or ""):
                    self._log(f"    [ALERT] Cloudflare! Waiting... (w{worker_id})")
                    time.sleep(5)

                try:
                    drv.execute_script(
                        "var v=document.getElementById('google_vignette_modal');"
                        "if(v)v.remove();document.body.style.overflow='auto';"
                    )
                except Exception:
                    pass

                time.sleep(random.uniform(0.8, 1.5))
                comments, has_next = self._parse_comments_from_html(drv.page_source)
                if not comments:
                    # Retry once
                    time.sleep(1)
                    comments, has_next = self._parse_comments_from_html(drv.page_source)
                    if not comments:
                        self._log(f"    [WARN] Trang trống hoặc bị chặn (0 post) (w{worker_id}).")
                        break
                current_thread_comments.extend(comments)
                if not has_next:
                    break
            except WebDriverException:
                raise
            except Exception:
                break

        return current_thread_comments

    # ── Main entry: crawl all threads for a keyword ────────────────────────
    def crawl_keyword(self, keyword=None) -> List[str]:
        keyword = keyword or self.keyword
        if not keyword:
            raise ValueError("keyword is required")

        urls = self.search_voz(keyword, self.max_threads)
        if not urls:
            self._log("[VOZ] Không tìm thấy link nào.")
            return []

        all_texts: List[str] = []
        seen_texts: set = set()
        seen_lock = threading.Lock()
        all_texts_lock = threading.Lock()

        # Track how many threads succeeded via fast-path vs Selenium
        _stats = {"fast": 0, "selenium": 0, "failed": 0}

        def _process_batch(raw_batch: List[str]) -> List[str]:
            """NLP-preprocess a raw batch and deduplicate."""
            processed: List[str] = []
            for text in raw_batch:
                if self.preprocessor:
                    result = self.preprocessor.process_comment(
                        text,
                        use_decoder=self.use_decoder,
                        use_filter=self.use_filter,
                        use_normalizer=self.use_normalizer,
                        use_segmentor=self.use_segmentor,
                    )
                    if not result["is_valid"]:
                        continue
                    clean = result["cleaned_text"]
                else:
                    clean = text
                if clean:
                    with seen_lock:
                        if clean not in seen_texts:
                            seen_texts.add(clean)
                            processed.append(clean)
            return processed

        _MAX_RETRIES = 2

        def _scrape_one_thread(idx_url: tuple[int, str], worker_id: int) -> None:
            """Worker function: scrape one thread with retry-with-backoff."""
            i, url = idx_url
            if self._stopped():
                return
            self._log(f"\n[VOZ] [KEYWORD: {keyword}] [THREAD {i+1}/{len(urls)}] (worker {worker_id})")

            last_error = None
            for attempt in range(_MAX_RETRIES + 1):
                if self._stopped():
                    return
                try:
                    if attempt > 0:
                        backoff = min(2 ** attempt, 8)
                        self._log(f"    [RETRY] Attempt {attempt+1}/{_MAX_RETRIES+1} "
                                  f"after {backoff}s backoff (w{worker_id})")
                        time.sleep(backoff)

                    raw_batch = self.scrape_comments(url, self.max_pages,
                                                     worker_id=worker_id)
                    if not raw_batch:
                        # No data – could be an empty thread or Cloudflare block
                        self._log(f"    [WARN] Không có data (w{worker_id})")
                        if attempt < _MAX_RETRIES:
                            # Try restarting driver before retry
                            try:
                                self._restart_pool_driver(worker_id)
                                self._log(f"    [RECOVERY] Driver mới sẵn sàng (w{worker_id}).")
                            except Exception:
                                pass
                            continue
                        _stats["failed"] += 1
                        return

                    processed_batch = _process_batch(raw_batch)
                    if processed_batch:
                        with all_texts_lock:
                            start_idx = len(all_texts) + 1
                            all_texts.extend(processed_batch)
                        wh_rows = [{"text": t} for t in processed_batch]
                        wh_count = append_to_warehouse(wh_rows)
                        self._log(f"    [WAREHOUSE] +{wh_count} dòng (w{worker_id})")
                        if self.data_callback:
                            gui_batch = [
                                {"id": start_idx + j, "text": t}
                                for j, t in enumerate(processed_batch)
                            ]
                            self.data_callback(gui_batch)
                    # Success – break out of retry loop
                    return

                except WebDriverException as e:
                    last_error = e
                    self._log(f"    [CRITICAL] Lỗi/Treo (w{worker_id}): {e}")
                    if attempt < _MAX_RETRIES:
                        self._log(f"    [RECOVERY] Restarting Driver (w{worker_id})...")
                        try:
                            self._restart_pool_driver(worker_id)
                            self._log(f"    [RECOVERY] Driver mới sẵn sàng (w{worker_id}).")
                        except Exception:
                            self._log(f"    [FATAL] Không thể restart driver (w{worker_id}).")
                            _stats["failed"] += 1
                            return
                    else:
                        self._log(f"    [GIVE UP] Hết retry cho thread này (w{worker_id}).")
                        _stats["failed"] += 1
                except Exception as e:
                    self._log(f"    [ERROR] (w{worker_id}) {e}")
                    _stats["failed"] += 1
                    return

        # ── Dispatch: concurrent or sequential ─────────────────────────
        effective_workers = min(self.num_workers, len(urls))

        if effective_workers <= 1:
            # Sequential – behaves exactly like the old code
            for idx_url in enumerate(urls):
                if self._stopped():
                    self._log("[VOZ] Đã nhận lệnh DỪNG.")
                    break
                _scrape_one_thread(idx_url, worker_id=0)
        else:
            self._log(f"[VOZ] Bắt đầu cào song song với {effective_workers} workers...")
            # Use a work queue approach: each worker pulls the next URL when
            # it finishes, rather than submitting all futures at once.  This
            # avoids the race condition where multiple workers try to create
            # their Selenium drivers simultaneously.
            url_queue: list[tuple[int, str]] = list(enumerate(urls))
            queue_lock = threading.Lock()

            def _worker_loop(worker_id: int) -> None:
                while True:
                    if self._stopped():
                        return
                    with queue_lock:
                        if not url_queue:
                            return
                        idx_url = url_queue.pop(0)
                    _scrape_one_thread(idx_url, worker_id)

            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = []
                for wid in range(effective_workers):
                    futures.append(executor.submit(_worker_loop, wid))
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as e:
                        self._log(f"    [ERROR] Worker exception: {e}")

        if all_texts:
            add_keyword_to_history("voz", keyword)
            self._log(f"[VOZ] Đã thêm '{keyword}' vào lịch sử keyword.")

        self._log(f"[VOZ] Hoàn tất '{keyword}' – tổng {len(all_texts)} bình luận. "
                  f"(thất bại: {_stats['failed']})")
        return all_texts


# =========================================================================== #
#  Threads Keyword Crawler
# =========================================================================== #

class ThreadsCrawler:
    """Threads (Meta) crawler – search keyword via DuckDuckGo, scrape comments."""

    def __init__(self, keyword=None, max_posts=10, max_scroll=30, timeout=20,
                 offset_x=-1000, log_callback=None, stop_event=None, data_callback=None,
                 preprocessor=None, use_decoder=True, use_filter=True,
                 use_normalizer=True, use_segmentor=True):
        self.keyword = keyword
        self.max_posts = max_posts
        self.max_scroll = max_scroll
        self.timeout = timeout
        self.offset_x = offset_x
        self.log_callback = log_callback
        self.stop_event = stop_event
        self.data_callback = data_callback
        self.preprocessor = preprocessor
        self.use_decoder = use_decoder
        self.use_filter = use_filter
        self.use_normalizer = use_normalizer
        self.use_segmentor = use_segmentor
        self.driver = None
        self.comments: List[str] = []

    def _log(self, msg: str):
        if self.log_callback: self.log_callback(msg)
        else: print(msg)

    def _stopped(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def get_output_path(self, keyword=None) -> str:
        keyword = keyword or self.keyword
        clean = _sanitize_keyword(keyword)
        return os.path.join(os.getcwd(), f"threads_{clean}_{self.max_posts}_{self.max_scroll}.csv")

    def get_driver(self):
        chrome_ver = _get_chrome_version()
        self._log(f"[Threads] Khởi tạo Chrome Driver — Chrome v{chrome_ver or 'auto'}...")
        user_data_dir = os.path.join(os.getcwd(), "chrome_profile_threads")
        opts = uc.ChromeOptions()
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--disable-notifications")
        opts.page_load_strategy = "normal"
        driver = uc.Chrome(options=opts, user_data_dir=user_data_dir, version_main=chrome_ver, use_subprocess=True)
        try:
            driver.set_window_position(self.offset_x, 0)
            time.sleep(1)
            driver.maximize_window()
        except Exception: pass
        driver.set_page_load_timeout(self.timeout)
        try:
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd("Network.setBlockedURLs", {
                "urls": ["*googleads*", "*doubleclick*", "*googlesyndication*",
                         "*adservice*", "*google_vignette*"]
            })
        except Exception: pass
        self.driver = driver
        return driver

    def force_kill_driver(self, driver=None):
        driver = driver or self.driver
        pid = None
        if driver:
            try: pid = driver.service.process.pid
            except Exception: pass
            try: driver.quit()
            except Exception: pass
        if pid:
            try: subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception: pass
        if driver is self.driver: self.driver = None

    def close(self): self.force_kill_driver()

    def restart_driver(self):
        self.force_kill_driver()
        return self.get_driver()

    def search_threads(self, keyword=None, limit=None) -> List[str]:
        keyword = keyword or self.keyword
        limit = limit or self.max_posts
        self._log(f"[Threads] Tìm link cho: {keyword}")
        results: List[str] = []
        try:
            with DDGS() as ddgs:
                gen = ddgs.text(f"site:threads.net {keyword}", max_results=limit + 20)
                for r in gen:
                    href = r.get("href", "")
                    if "threads.net" in href and "/post/" in href and href not in results:
                        results.append(href)
                    if len(results) >= limit: break
        except Exception as e:
            self._log(f"[Threads] Lỗi search: {e}")
        self._log(f"[Threads] Tìm thấy {len(results)} post links.")
        return results

    def _dismiss_login_popup(self):
        try:
            ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(0.5)
        except Exception: pass
        for sel in ['[aria-label="Close"]', '[aria-label="Đóng"]']:
            try:
                self.driver.find_element(By.CSS_SELECTOR, sel).click()
                time.sleep(0.5)
                return
            except Exception: continue

    def _scroll_to_load_replies(self, max_scroll=None):
        max_scroll = max_scroll or self.max_scroll
        prev_height = 0
        for i in range(max_scroll):
            if self._stopped(): break
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1.5, 2.5))
            # Click "View more replies" / "Xem thêm" reply expansion buttons
            try:
                btns = self.driver.find_elements(
                    By.XPATH,
                    "//*[contains(text(),'View more replies') or "
                    "contains(text(),'Xem thêm') or "
                    "contains(text(),'View replies') or "
                    "contains(text(),'more repl')]",
                )
                for btn in btns:
                    try:
                        self.driver.execute_script("arguments[0].click();", btn)
                        time.sleep(random.uniform(1.0, 2.0))
                    except Exception: pass
            except Exception: pass
            # Click "See more" / "Xem thêm" to expand truncated comment text
            try:
                see_more_btns = self.driver.find_elements(
                    By.XPATH,
                    "//div[@role='button' and (text()='See more' or text()='Xem thêm')] | "
                    "//span[@role='link' and (text()='See more' or text()='Xem thêm')]",
                )
                for btn in see_more_btns:
                    try:
                        self.driver.execute_script("arguments[0].click();", btn)
                        time.sleep(random.uniform(0.5, 1.0))
                    except Exception: pass
            except Exception: pass
            curr_height = self.driver.execute_script("return document.body.scrollHeight")
            if curr_height == prev_height:
                time.sleep(1.0)
                curr_height = self.driver.execute_script("return document.body.scrollHeight")
                if curr_height == prev_height: break
            prev_height = curr_height
            if (i + 1) % 5 == 0:
                self._log(f"    [SCROLL] Lần {i+1}/{max_scroll}...")

    def _parse_comments(self, page_source: str) -> List[str]:
        soup = BeautifulSoup(page_source, "html.parser")
        results: List[str] = []
        seen: set = set()
        UI_TEXTS = {
            "Reply", "Trả lời", "Like", "Thích", "Share", "Chia sẻ",
            "Repost", "More", "Follow", "Theo dõi", "Log in", "Đăng nhập",
            "Sign up", "Đăng ký", "Search", "Tìm kiếm", "Verified",
            "liked", "likes", "replies", "reply",
            # Threads-specific UI noise
            "Translate", "translate", "Dịch",
            "See translation", "Xem bản dịch",
            "Tiếp tục với Instagram", "Continue with Instagram",
            "Tiếp tục bằng Instagram", "Dùng ứng dụng", "Use app",
        }

        # Pre-process: remove small emoji <img> tags to prevent alt text leak
        # (e.g. <img alt="gia đình" width="16"> → garbage text)
        for img in soup.select("img"):
            width = img.get("width")
            if width and str(width).isdigit() and int(width) <= 20:
                img.decompose()
            elif img.get("alt") and not img.get("src", "").startswith("http"):
                # Inline emoji image with no real src → remove
                img.decompose()

        # Remove "Translate"/"Dịch" button elements from the DOM
        for el in soup.select('span[role="link"], div[role="button"]'):
            t = el.get_text(strip=True).lower()
            if t in {"translate", "dịch", "see translation", "xem bản dịch"}:
                el.decompose()

        blocks = soup.select('div[data-pressable-container="true"]')
        if blocks:
            for block in blocks:
                username = ""
                uname_el = block.select_one('a[role="link"] span')
                if uname_el: username = uname_el.get_text(strip=True)
                spans = block.select('div[dir="auto"]')
                texts = []
                for s in spans:
                    t = s.get_text(strip=True)
                    if t and len(t) > 1 and t != username and t not in UI_TEXTS and not t.startswith("http"):
                        texts.append(t)
                if texts:
                    main_text = max(texts, key=len)
                    # Apply Threads-specific text cleaning
                    main_text = _clean_threads_text(main_text)
                    if main_text and main_text not in seen:
                        seen.add(main_text)
                        results.append(main_text)
        if not results:
            for el in soup.select('div[dir="auto"], span[dir="auto"]'):
                t = el.get_text(strip=True)
                if t and len(t) > 3 and t not in seen and t not in UI_TEXTS and not t.startswith("http"):
                    t = _clean_threads_text(t)
                    if t and t not in seen:
                        seen.add(t)
                        results.append(t)
        return results

    def scrape_comments(self, url: str, max_scroll=None) -> List[str]:
        if self.driver is None: self.get_driver()
        self._log(f"  -> Scraping: {url}")
        try:
            try: self.driver.get(url)
            except TimeoutException:
                try: self.driver.execute_script("window.stop();")
                except Exception: pass
            time.sleep(random.uniform(3.0, 5.0))
            self._dismiss_login_popup()
            self._scroll_to_load_replies(max_scroll)
            post_comments = self._parse_comments(self.driver.page_source)
            self._log(f"    [OK] {len(post_comments)} comments.")
            return post_comments
        except WebDriverException: raise
        except Exception as e:
            self._log(f"    [ERROR] {e}")
            return []

    def crawl_keyword(self, keyword=None) -> List[str]:
        keyword = keyword or self.keyword
        if not keyword: raise ValueError("keyword is required")
        if self.driver is None: self.get_driver()

        urls = self.search_threads(keyword, self.max_posts)
        if not urls:
            self._log("[Threads] Không tìm thấy link nào.")
            return []

        all_texts: List[str] = []
        seen_texts: set = set()

        for i, url in enumerate(urls):
            if self._stopped():
                self._log("[Threads] Đã nhận lệnh DỪNG.")
                break

            self._log(f"\n[Threads] [KEYWORD: {keyword}] [POST {i+1}/{len(urls)}]")
            try:
                raw_batch = self.scrape_comments(url, self.max_scroll)
                if not raw_batch:
                    self._log("    [WARN] Không có data -> có thể bị chặn.")
                    raise WebDriverException("Zero data returned")

                processed_batch: List[str] = []
                for text in raw_batch:
                    if self.preprocessor:
                        result = self.preprocessor.process_comment(
                            text, use_decoder=self.use_decoder, use_filter=self.use_filter,
                            use_normalizer=self.use_normalizer, use_segmentor=self.use_segmentor,
                        )
                        if not result["is_valid"]: continue
                        clean = result["cleaned_text"]
                    else:
                        clean = text
                    if clean and clean not in seen_texts:
                        seen_texts.add(clean)
                        processed_batch.append(clean)

                if processed_batch:
                    all_texts.extend(processed_batch)
                    self.comments.extend(processed_batch)
                    wh_rows = [{"text": t} for t in processed_batch]
                    wh_count = append_to_warehouse(wh_rows)
                    self._log(f"    [WAREHOUSE] +{wh_count} dòng")
                    if self.data_callback:
                        gui_batch = [{"id": idx, "text": t} for idx, t in enumerate(processed_batch, start=len(all_texts) - len(processed_batch) + 1)]
                        self.data_callback(gui_batch)

            except WebDriverException as e:
                self._log(f"    [CRITICAL] Lỗi/Treo: {e}")
                self._log("    [RECOVERY] Restarting Driver...")
                try:
                    self.restart_driver()
                    self._log("    [RECOVERY] Driver mới sẵn sàng.")
                except Exception:
                    self._log("    [FATAL] Không thể restart driver.")
                    break
            except Exception as e:
                self._log(f"    [ERROR] Bỏ qua: {e}")

            time.sleep(random.uniform(2.0, 4.0))

        if all_texts:
            add_keyword_to_history("threads", keyword)
            self._log(f"[Threads] Đã thêm '{keyword}' vào lịch sử keyword.")

        self._log(f"[Threads] Hoàn tất '{keyword}' – tổng {len(all_texts)} bình luận.")
        return all_texts

