"""
keyword_crawler.py – VOZ & Threads keyword-based crawlers.
Ported from src/utils/crawler.py with GUI integration
(log_callback, data_callback, stop_event, NLP preprocessing, warehouse).
"""
import os
import re
import csv
import time
import random
import subprocess
import unicodedata
import threading
from typing import List, Optional

from bs4 import BeautifulSoup
from ddgs import DDGS
import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from warehouse import append_to_warehouse
from keyword_history import add_keyword as _add_kw_to_history


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
#  VOZ Crawler
# =========================================================================== #

class VOZCrawler:
    """VOZ forum crawler – search keyword via DuckDuckGo, scrape comments."""

    def __init__(
        self,
        keyword: Optional[str] = None,
        max_threads: int = 10,
        max_pages: int = 50,
        timeout: int = 20,
        offset_x: int = -1000,
        log_callback=None,
        stop_event: Optional[threading.Event] = None,
        data_callback=None,
        preprocessor=None,
        use_decoder: bool = True,
        use_filter: bool = True,
        use_normalizer: bool = True,
        use_segmentor: bool = True,
    ):
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

        self.driver = None

    # -- logging ----------------------------------------------------------- #
    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def _stopped(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    # -- output path ------------------------------------------------------- #
    def get_output_path(self, keyword: Optional[str] = None) -> str:
        keyword = keyword or self.keyword
        clean = _sanitize_keyword(keyword)
        filename = f"{clean}_{self.max_threads}_{self.max_pages}.csv"
        return os.path.join(os.getcwd(), filename)

    # -- driver management ------------------------------------------------- #
    def get_driver(self):
        self._log("[VOZ] Khởi tạo Chrome Driver...")
        user_data_dir = os.path.join(os.getcwd(), "chrome_profile_fixed")
        opts = uc.ChromeOptions()
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.page_load_strategy = "normal"

        driver = uc.Chrome(
            options=opts,
            user_data_dir=user_data_dir,
            version_main=None,
            use_subprocess=True,
        )
        try:
            driver.set_window_position(self.offset_x, 0)
            time.sleep(1)
            driver.maximize_window()
        except Exception:
            pass
        driver.set_page_load_timeout(self.timeout)
        try:
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd("Network.setBlockedURLs", {
                "urls": ["*googleads*", "*doubleclick*", "*googlesyndication*",
                         "*adservice*", "*google_vignette*", "*.gif"]
            })
        except Exception:
            pass
        self.driver = driver
        return driver

    def force_kill_driver(self, driver=None):
        driver = driver or self.driver
        pid = None
        if driver:
            try:
                pid = driver.service.process.pid
            except Exception:
                pass
            try:
                driver.quit()
            except Exception:
                pass
        if pid:
            try:
                subprocess.call(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        if driver is self.driver:
            self.driver = None

    def close(self):
        self.force_kill_driver()

    def restart_driver(self):
        self.force_kill_driver()
        return self.get_driver()

    # -- search ------------------------------------------------------------ #
    def search_voz(self, keyword: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
        keyword = keyword or self.keyword
        limit = limit or self.max_threads
        self._log(f"[VOZ] Tìm link cho: {keyword}")
        results = []
        try:
            with DDGS() as ddgs:
                gen = ddgs.text(f"site:voz.vn {keyword}", max_results=limit + 10)
                for r in gen:
                    href = r.get("href")
                    if href and "/t/" in href and href not in results:
                        results.append(href)
                    if len(results) >= limit:
                        break
        except Exception as e:
            self._log(f"[VOZ] Lỗi search: {e}")
        self._log(f"[VOZ] Tìm thấy {len(results)} thread links.")
        return results

    # -- scrape one thread ------------------------------------------------- #
    def scrape_comments(self, url: str, max_pages: Optional[int] = None) -> List[str]:
        if self.driver is None:
            self.get_driver()
        max_pages = max_pages or self.max_pages
        current_thread_comments = []
        base_url = url.split("/page-")[0].rstrip("/")
        self._log(f"  -> Scraping: {base_url}")

        for page in range(1, max_pages + 1):
            if self._stopped():
                break
            try:
                target_url = base_url if page == 1 else f"{base_url}/page-{page}"
                try:
                    self.driver.get(target_url)
                except TimeoutException:
                    try:
                        self.driver.execute_script("window.stop();")
                    except Exception:
                        pass

                if "Just a moment" in (self.driver.title or ""):
                    self._log("    [ALERT] Cloudflare! Waiting...")
                    time.sleep(5)

                try:
                    self.driver.execute_script("""
                        var v = document.getElementById('google_vignette_modal');
                        if (v) v.remove();
                        document.body.style.overflow = 'auto';
                    """)
                except Exception:
                    pass

                time.sleep(random.uniform(2, 3))
                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                posts = soup.select(".message-inner")
                if not posts:
                    time.sleep(2)
                    soup = BeautifulSoup(self.driver.page_source, "html.parser")
                    posts = soup.select(".message-inner")
                    if not posts:
                        self._log("    [WARN] Trang trống hoặc bị chặn (0 post).")
                        break

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
                            current_thread_comments.append(text)

                if not soup.select_one("a.pageNav-jump--next"):
                    break
            except WebDriverException:
                raise
            except Exception:
                break

        return current_thread_comments

    # -- main entry: crawl keyword ----------------------------------------- #
    def crawl_keyword(self, keyword: Optional[str] = None, persist: bool = True) -> List[str]:
        keyword = keyword or self.keyword
        if not keyword:
            raise ValueError("keyword is required")

        if self.driver is None:
            self.get_driver()

        output_path = self.get_output_path(keyword)
        urls = self.search_voz(keyword, self.max_threads)
        if not urls:
            self._log("[VOZ] Không tìm thấy link nào.")
            return []

        all_texts: List[str] = []
        seen_texts: set = set()

        for i, url in enumerate(urls):
            if self._stopped():
                self._log("[VOZ] Đã nhận lệnh DỪNG.")
                break

            self._log(f"\n[VOZ] [KEYWORD: {keyword}] [THREAD {i+1}/{len(urls)}]")
            try:
                raw_batch = self.scrape_comments(url, self.max_pages)
                if not raw_batch:
                    self._log("    [WARN] Không có data -> Nghi ngờ treo.")
                    raise WebDriverException("Zero data returned")

                # NLP preprocessing
                processed_batch: List[str] = []
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

                    if clean and clean not in seen_texts:
                        seen_texts.add(clean)
                        processed_batch.append(clean)

                if processed_batch:
                    all_texts.extend(processed_batch)

                    # Save to per-keyword CSV
                    if persist:
                        count = _append_batch_to_csv(output_path, processed_batch)
                        self._log(f"    [CSV] +{count} dòng -> {os.path.basename(output_path)}")

                    # Save to warehouse
                    wh_rows = [{"text": t} for t in processed_batch]
                    wh_count = append_to_warehouse(wh_rows)
                    self._log(f"    [WAREHOUSE] +{wh_count} dòng")

                    # Callback to GUI
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
                self._log(f"    [ERROR] {e}")

        # Keyword history
        if all_texts:
            _add_kw_to_history("voz", keyword)
            self._log(f"[VOZ] Đã thêm '{keyword}' vào lịch sử keyword.")

        self._log(f"[VOZ] Hoàn tất '{keyword}' – tổng {len(all_texts)} bình luận.")
        return all_texts


# =========================================================================== #
#  Threads Crawler
# =========================================================================== #

class ThreadsCrawler:
    """Threads (Meta) crawler – search keyword via DuckDuckGo, scrape comments."""

    def __init__(
        self,
        keyword: Optional[str] = None,
        max_posts: int = 10,
        max_scroll: int = 30,
        timeout: int = 20,
        offset_x: int = -1000,
        log_callback=None,
        stop_event: Optional[threading.Event] = None,
        data_callback=None,
        preprocessor=None,
        use_decoder: bool = True,
        use_filter: bool = True,
        use_normalizer: bool = True,
        use_segmentor: bool = True,
    ):
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

        self.driver: Optional[uc.Chrome] = None
        self.comments: List[str] = []

    # -- logging ----------------------------------------------------------- #
    def _log(self, msg: str):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def _stopped(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    # -- output path ------------------------------------------------------- #
    def get_output_path(self, keyword: Optional[str] = None) -> str:
        keyword = keyword or self.keyword
        clean = _sanitize_keyword(keyword)
        filename = f"threads_{clean}_{self.max_posts}_{self.max_scroll}.csv"
        return os.path.join(os.getcwd(), filename)

    # -- driver management ------------------------------------------------- #
    def get_driver(self) -> uc.Chrome:
        self._log("[Threads] Khởi tạo Chrome Driver...")
        user_data_dir = os.path.join(os.getcwd(), "chrome_profile_threads")
        opts = uc.ChromeOptions()
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--disable-notifications")
        opts.page_load_strategy = "normal"

        driver = uc.Chrome(
            options=opts,
            user_data_dir=user_data_dir,
            version_main=None,
            use_subprocess=True,
        )
        try:
            driver.set_window_position(self.offset_x, 0)
            time.sleep(1)
            driver.maximize_window()
        except Exception:
            pass
        driver.set_page_load_timeout(self.timeout)
        try:
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd("Network.setBlockedURLs", {
                "urls": ["*googleads*", "*doubleclick*", "*googlesyndication*",
                         "*adservice*", "*google_vignette*"]
            })
        except Exception:
            pass
        self.driver = driver
        return driver

    def force_kill_driver(self, driver=None):
        driver = driver or self.driver
        pid = None
        if driver:
            try:
                pid = driver.service.process.pid
            except Exception:
                pass
            try:
                driver.quit()
            except Exception:
                pass
        if pid:
            try:
                subprocess.call(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        if driver is self.driver:
            self.driver = None

    def close(self):
        self.force_kill_driver()

    def restart_driver(self):
        self.force_kill_driver()
        return self.get_driver()

    # -- search ------------------------------------------------------------ #
    def search_threads(self, keyword: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
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
                    if len(results) >= limit:
                        break
        except Exception as e:
            self._log(f"[Threads] Lỗi search: {e}")
        self._log(f"[Threads] Tìm thấy {len(results)} post links.")
        return results

    # -- helpers ----------------------------------------------------------- #
    def _dismiss_login_popup(self):
        try:
            ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(0.5)
        except Exception:
            pass
        for sel in ['[aria-label="Close"]', '[aria-label="Đóng"]']:
            try:
                self.driver.find_element(By.CSS_SELECTOR, sel).click()
                time.sleep(0.5)
                return
            except Exception:
                continue

    def _scroll_to_load_replies(self, max_scroll: Optional[int] = None):
        max_scroll = max_scroll or self.max_scroll
        prev_height = 0
        for i in range(max_scroll):
            if self._stopped():
                break
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1.5, 2.5))
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
                    except Exception:
                        pass
            except Exception:
                pass
            curr_height = self.driver.execute_script("return document.body.scrollHeight")
            if curr_height == prev_height:
                time.sleep(1.0)
                curr_height = self.driver.execute_script("return document.body.scrollHeight")
                if curr_height == prev_height:
                    break
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
        }
        blocks = soup.select('div[data-pressable-container="true"]')
        if blocks:
            for block in blocks:
                username = ""
                uname_el = block.select_one('a[role="link"] span')
                if uname_el:
                    username = uname_el.get_text(strip=True)
                spans = block.select('div[dir="auto"]')
                texts = []
                for s in spans:
                    t = s.get_text(strip=True)
                    if t and len(t) > 1 and t != username and t not in UI_TEXTS and not t.startswith("http"):
                        texts.append(t)
                if texts:
                    main_text = max(texts, key=len)
                    if main_text not in seen:
                        seen.add(main_text)
                        results.append(main_text)
        if not results:
            for el in soup.select('div[dir="auto"], span[dir="auto"]'):
                t = el.get_text(strip=True)
                if t and len(t) > 3 and t not in seen and t not in UI_TEXTS and not t.startswith("http"):
                    seen.add(t)
                    results.append(t)
        return results

    # -- scrape one post --------------------------------------------------- #
    def scrape_comments(self, url: str, max_scroll: Optional[int] = None) -> List[str]:
        if self.driver is None:
            self.get_driver()
        self._log(f"  -> Scraping: {url}")
        try:
            try:
                self.driver.get(url)
            except TimeoutException:
                try:
                    self.driver.execute_script("window.stop();")
                except Exception:
                    pass
            time.sleep(random.uniform(3.0, 5.0))
            self._dismiss_login_popup()
            self._scroll_to_load_replies(max_scroll)
            post_comments = self._parse_comments(self.driver.page_source)
            self._log(f"    [OK] {len(post_comments)} comments.")
            return post_comments
        except WebDriverException:
            raise
        except Exception as e:
            self._log(f"    [ERROR] {e}")
            return []

    # -- main entry: crawl keyword ----------------------------------------- #
    def crawl_keyword(self, keyword: Optional[str] = None, persist: bool = True) -> List[str]:
        keyword = keyword or self.keyword
        if not keyword:
            raise ValueError("keyword is required")

        if self.driver is None:
            self.get_driver()

        output_path = self.get_output_path(keyword)
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

                # NLP preprocessing
                processed_batch: List[str] = []
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

                    if clean and clean not in seen_texts:
                        seen_texts.add(clean)
                        processed_batch.append(clean)

                if processed_batch:
                    all_texts.extend(processed_batch)
                    self.comments.extend(processed_batch)

                    if persist:
                        count = _append_batch_to_csv(output_path, processed_batch)
                        self._log(f"    [CSV] +{count} dòng -> {os.path.basename(output_path)}")

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
            _add_kw_to_history("threads", keyword)
            self._log(f"[Threads] Đã thêm '{keyword}' vào lịch sử keyword.")

        self._log(f"[Threads] Hoàn tất '{keyword}' – tổng {len(all_texts)} bình luận.")
        return all_texts
