#General lib
import os
import pandas as pd
import json
from dotenv import load_dotenv

load_dotenv()


#lib for YouTube crawler
from googleapiclient.discovery import build


class YoutubeCrawler:

    def __init__(self, video_id):
        self.video_id = video_id
        self.comments = []
        self.next_page_token = None
        self.youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

    def get_comments(self):
        if self.check_video_id(video_id=self.video_id):
            while True:
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=self.video_id,
                    maxResults=100, # Tối đa 100 comment mỗi lần gọi
                    pageToken=self.next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()

                for item in response.get("items", []):
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    self.comments.append({
                        "text": comment.get("textDisplay", "")
                    })

                # Kiểm tra nếu còn trang tiếp theo
                self.next_page_token = response.get("nextPageToken")
                if not self.next_page_token:
                    break
        else:
            raise Exception(f"Video id {self.video_id} đã được cào.")

    def check_video_id(self, video_id):
        with open('../reports/crawled_videos.json') as file:
            crawled = json.load(file)
        if video_id in crawled and crawled[video_id]["status"] == "crawled":
            return False
        return True


    def output(self):
        output = pd.DataFrame(self.comments)
        output.to_csv(f"../data/raw/{self.video_id}.csv")

        with open ("../reports/crawled_videos.json", 'r', encoding="utf-8") as f:
            report = json.load(f)
            report[self.video_id] = {"status": "crawled"}

        with open ("../reports/crawled_videos.json", 'w', encoding="utf-8" ) as f:
            json.dump(report, f, ensure_ascii=False, indent=2)



#lib for VOZ Crawler

import time
import warnings
import unicodedata
import re
import random
import subprocess
from typing import Iterable, List, Optional

from bs4 import BeautifulSoup
from ddgs import DDGS
import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException, WebDriverException


class VOZCrawler:
    """Simple VOZ crawler (ported from notebooks/1-Data-Collection.ipynb cell 5)."""

    def __init__(
        self,
        keyword: Optional[str] = None,
        max_threads: int = 10,
        max_pages: int = 50,
        timeout: int = 20,
        offset_x: int = -1000,
        chrome_version_main: int = 143,
        verbose: bool = True,
    ):
        self.keyword = keyword
        self.max_threads = max_threads
        self.max_pages = max_pages
        self.timeout = timeout
        self.offset_x = offset_x
        self.chrome_version_main = chrome_version_main
        self.verbose = verbose

        self.driver = None

    def get_output_path(self, keyword: Optional[str] = None) -> str:
        keyword = keyword or self.keyword
        if not keyword:
            raise ValueError("keyword is required")

        text = unicodedata.normalize('NFD', keyword)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        text = text.replace('đ', 'd').replace('Đ', 'D')
        clean_keyword = text.replace(' ', '_')

        filename = f"{clean_keyword}_{self.max_threads}_{self.max_pages}.csv"
        output_dir = os.path.join(os.pardir, "data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def get_driver(self):
        if self.verbose:
            print("[INIT] Khoi tao Driver moi...")

        user_data_dir = os.path.join(os.getcwd(), "chrome_profile_fixed")

        opts = uc.ChromeOptions()
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.page_load_strategy = 'normal'

        driver = None

        # try fixed version first (same as notebook)
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

        # block ads
        try:
            driver.execute_cdp_cmd('Network.enable', {})
            driver.execute_cdp_cmd('Network.setBlockedURLs', {
                "urls": ["*googleads*", "*doubleclick*", "*googlesyndication*", "*adservice*", "*google_vignette*", "*.gif"]
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
                    ['taskkill', '/F', '/T', '/PID', str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if self.verbose:
                    print(f"    [SYSTEM] Killed PID: {pid}")
            except Exception:
                pass

        if driver is self.driver:
            self.driver = None

    def close(self):
        self.force_kill_driver(self.driver)

    def restart_driver(self):
        self.force_kill_driver(self.driver)
        return self.get_driver()

    @staticmethod
    def append_batch_to_csv(file_path: str, batch_data: List[str]) -> int:
        if not batch_data:
            return 0

        file_exists = os.path.isfile(file_path)
        start_id = 1
        if file_exists:
            try:
                df_check = pd.read_csv(file_path, usecols=['id'])
                if not df_check.empty:
                    start_id = int(df_check['id'].max()) + 1
            except Exception:
                start_id = 1

        df_new = pd.DataFrame(batch_data, columns=['text'])
        df_new.insert(0, 'id', range(start_id, start_id + len(df_new)))
        df_new.to_csv(file_path, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
        return len(df_new)

    def search_voz(self, keyword: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
        keyword = keyword or self.keyword
        if not keyword:
            raise ValueError("keyword is required")
        limit = limit or self.max_threads

        if self.verbose:
            print(f"[SEARCH] Tim link cho: {keyword}")

        results = []
        try:
            with DDGS() as ddgs:
                gen = ddgs.text(f"site:voz.vn {keyword}", max_results=limit + 10)
                for r in gen:
                    href = r.get('href')
                    if href and "/t/" in href and href not in results:
                        results.append(href)
                    if len(results) >= limit:
                        break
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Search loi: {e}")
        return results

    def scrape_comments(self, url: str, max_pages: Optional[int] = None) -> List[str]:
        if self.driver is None:
            self.get_driver()

        max_pages = max_pages or self.max_pages

        current_thread_comments = []
        base_url = url.split('/page-')[0]
        if base_url.endswith('/'):
            base_url = base_url[:-1]

        if self.verbose:
            print(f"  -> Scraping: {base_url}")

        for page in range(1, max_pages + 1):
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
                    if self.verbose:
                        print("    [ALERT] Cloudflare! Waiting...")
                    time.sleep(5)

                try:
                    self.driver.execute_script("""
                        var vignette = document.getElementById('google_vignette_modal');
                        if (vignette) vignette.remove();
                        document.body.style.overflow = 'auto';
                    """)
                except Exception:
                    pass

                time.sleep(random.uniform(2, 3))

                soup = BeautifulSoup(self.driver.page_source, 'html.parser')

                posts = soup.select('.message-inner')
                if not posts:
                    time.sleep(2)
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                    posts = soup.select('.message-inner')

                    if not posts:
                        if self.verbose:
                            print("    [WARN] Trang trong hoac bi chan (0 post).")
                        break

                for post in posts:
                    content_tag = post.select_one('.bbWrapper')
                    if content_tag:
                        for q in content_tag.find_all('blockquote'):
                            q.decompose()
                        for br in content_tag.find_all('br'):
                            br.replace_with('\n')

                        text = content_tag.get_text(separator='\n', strip=True)
                        text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)
                        text = re.sub(r'\n+', '\n', text)

                        if text:
                            current_thread_comments.append(text)

                if not soup.select_one('a.pageNav-jump--next'):
                    break

            except WebDriverException:
                raise
            except Exception:
                break

        return current_thread_comments

    def crawl_keyword(self, keyword: Optional[str] = None, persist: bool = True) -> List[str]:
        keyword = keyword or self.keyword
        if not keyword:
            raise ValueError("keyword is required")

        if self.driver is None:
            self.get_driver()

        output_path = self.get_output_path(keyword)
        urls = self.search_voz(keyword, self.max_threads)
        if not urls:
            return []

        all_texts = []

        for i, url in enumerate(urls):
            if self.verbose:
                print(f"\n[KEYWORD: {keyword}] [THREAD {i+1}/{len(urls)}]")

            try:
                batch_data = self.scrape_comments(url, self.max_pages)

                if batch_data:
                    all_texts.extend(batch_data)
                    if persist:
                        count = self.append_batch_to_csv(output_path, batch_data)
                        if self.verbose:
                            print(f"    [SUCCESS] + {count} dòng.")
                else:
                    if self.verbose:
                        print("    [WARN] Khong co data -> Nghi ngo trinh duyet treo.")
                    raise WebDriverException("Zero data returned (Zombie Browser)")

            except Exception as e:
                if self.verbose:
                    print(f"    [CRITICAL] Phat hien loi/Treo: {e}")
                    print("    [RECOVERY] Restarting Driver...")

                try:
                    self.restart_driver()
                    if self.verbose:
                        print("    [RECOVERY] Driver moi san sang.")
                except Exception:
                    if self.verbose:
                        print("    [FATAL] Khong the restart driver.")
                    break

        return all_texts

    def crawl_keywords(self, keywords: Iterable[str], persist: bool = True) -> dict:
        results = {}
        for kw in keywords:
            results[kw] = self.crawl_keyword(kw, persist=persist)
            time.sleep(2)
        return results


# --------------------------------------------------------------------------- #
#  Threads Crawler (Selenium + DDGS – cào public posts bằng từ khóa)
# --------------------------------------------------------------------------- #

from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


class ThreadsCrawler:
    """Crawler cho Threads (Meta) – cào comment public posts bằng từ khóa.

    Flow giống VOZCrawler:
        1. Tìm ``site:threads.net <keyword>`` qua DuckDuckGo (DDGS)
        2. Mở từng link post bằng undetected-chromedriver
        3. Cuộn trang để lazy-load replies
        4. Parse HTML lấy nội dung comment
        5. Xuất CSV

    Cách dùng::

        crawler = ThreadsCrawler(keyword="chính trị", max_posts=10)
        crawler.crawl_keyword()
        crawler.output()
        crawler.close()
    """

    def __init__(
        self,
        keyword: Optional[str] = None,
        max_posts: int = 10,
        max_scroll: int = 30,
        timeout: int = 20,
        offset_x: int = -1000,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        keyword : str | None
            Từ khóa tìm kiếm mặc định.
        max_posts : int
            Số post tối đa cần cào cho mỗi keyword.
        max_scroll : int
            Số lần cuộn tối đa để load replies trong 1 post.
        timeout : int
            Timeout (giây) cho mỗi lần load trang.
        offset_x : int
            Vị trí x cửa sổ Chrome (đẩy sang màn hình phụ).
        verbose : bool
            In log ra console.
        """
        self.keyword = keyword
        self.max_posts = max_posts
        self.max_scroll = max_scroll
        self.timeout = timeout
        self.offset_x = offset_x
        self.verbose = verbose

        self.driver: Optional[uc.Chrome] = None
        self.comments: List[str] = []

    # ================================================================== #
    #  Driver management
    # ================================================================== #
    def get_driver(self) -> uc.Chrome:
        if self.verbose:
            print("[INIT] Khởi tạo Chrome Driver cho Threads...")

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
                "urls": [
                    "*googleads*", "*doubleclick*", "*googlesyndication*",
                    "*adservice*", "*google_vignette*",
                ]
            })
        except Exception:
            pass

        self.driver = driver
        return driver

    def force_kill_driver(self, driver=None) -> None:
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
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if self.verbose:
                    print(f"    [SYSTEM] Killed PID: {pid}")
            except Exception:
                pass
        if driver is self.driver:
            self.driver = None

    def close(self) -> None:
        self.force_kill_driver()

    def restart_driver(self) -> uc.Chrome:
        self.force_kill_driver()
        return self.get_driver()

    # ================================================================== #
    #  Output path
    # ================================================================== #
    def get_output_path(self, keyword: Optional[str] = None) -> str:
        keyword = keyword or self.keyword
        if not keyword:
            raise ValueError("keyword is required")

        text = unicodedata.normalize("NFD", keyword)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = text.replace("đ", "d").replace("Đ", "D")
        clean = text.replace(" ", "_")

        filename = f"threads_{clean}_{self.max_posts}_{self.max_scroll}.csv"
        output_dir = os.path.join(os.pardir, "data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    # ================================================================== #
    #  Bước 1: Tìm link post bằng DDGS
    # ================================================================== #
    def search_threads(
        self,
        keyword: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """Tìm link bài Threads qua DuckDuckGo.

        Returns
        -------
        list[str]
            Danh sách URL dạng ``https://www.threads.net/@user/post/...``
        """
        keyword = keyword or self.keyword
        if not keyword:
            raise ValueError("keyword is required")
        limit = limit or self.max_posts

        if self.verbose:
            print(f"[SEARCH] Tìm link Threads cho: {keyword}")

        results: List[str] = []
        try:
            with DDGS() as ddgs:
                gen = ddgs.text(
                    f"site:threads.net {keyword}",
                    max_results=limit + 20,
                )
                for r in gen:
                    href = r.get("href", "")
                    if (
                        "threads.net" in href
                        and "/post/" in href
                        and href not in results
                    ):
                        results.append(href)
                    if len(results) >= limit:
                        break
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Search lỗi: {e}")

        if self.verbose:
            print(f"[SEARCH] Tìm thấy {len(results)} post links.")
        return results

    # ================================================================== #
    #  Bước 2: Đóng popup đăng nhập
    # ================================================================== #
    def _dismiss_login_popup(self) -> None:
        """Threads thường hiện popup yêu cầu đăng nhập – đóng nó."""
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

    # ================================================================== #
    #  Bước 3: Cuộn trang load replies
    # ================================================================== #
    def _scroll_to_load_replies(self, max_scroll: Optional[int] = None) -> None:
        max_scroll = max_scroll or self.max_scroll
        prev_height = 0

        for i in range(max_scroll):
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(random.uniform(1.5, 2.5))

            # Click "View more replies" / "Xem thêm"
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
                        self.driver.execute_script(
                            "arguments[0].click();", btn
                        )
                        time.sleep(random.uniform(1.0, 2.0))
                    except Exception:
                        pass
            except Exception:
                pass

            curr_height = self.driver.execute_script(
                "return document.body.scrollHeight"
            )
            if curr_height == prev_height:
                time.sleep(1.0)
                curr_height = self.driver.execute_script(
                    "return document.body.scrollHeight"
                )
                if curr_height == prev_height:
                    break
            prev_height = curr_height

            if self.verbose and (i + 1) % 5 == 0:
                print(f"    [SCROLL] Lần {i + 1}/{max_scroll}...")

    # ================================================================== #
    #  Bước 4: Parse comments từ HTML
    # ================================================================== #
    def _parse_comments(self, page_source: str) -> List[str]:
        """Parse text comments từ HTML Threads post."""
        soup = BeautifulSoup(page_source, "html.parser")
        results: List[str] = []
        seen: set = set()

        UI_TEXTS = {
            "Reply", "Trả lời", "Like", "Thích", "Share", "Chia sẻ",
            "Repost", "More", "Follow", "Theo dõi", "Log in", "Đăng nhập",
            "Sign up", "Đăng ký", "Search", "Tìm kiếm", "Verified",
            "liked", "likes", "replies", "reply",
        }

        # Strategy 1: data-pressable-container (mỗi block = 1 post/reply)
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
                    if (
                        t
                        and len(t) > 1
                        and t != username
                        and t not in UI_TEXTS
                        and not t.startswith("http")
                    ):
                        texts.append(t)

                if texts:
                    main_text = max(texts, key=len)
                    if main_text not in seen:
                        seen.add(main_text)
                        results.append(main_text)

        # Strategy 2 (fallback): mọi div[dir="auto"] có text
        if not results:
            for el in soup.select('div[dir="auto"], span[dir="auto"]'):
                t = el.get_text(strip=True)
                if (
                    t
                    and len(t) > 3
                    and t not in seen
                    and t not in UI_TEXTS
                    and not t.startswith("http")
                ):
                    seen.add(t)
                    results.append(t)

        return results

    # ================================================================== #
    #  Bước 5: Cào 1 post (giống VOZCrawler.scrape_comments)
    # ================================================================== #
    def scrape_comments(
        self,
        url: str,
        max_scroll: Optional[int] = None,
    ) -> List[str]:
        """Cào comments từ 1 URL post Threads."""
        if self.driver is None:
            self.get_driver()

        if self.verbose:
            print(f"  -> Scraping: {url}")

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

            if self.verbose:
                print(f"    [OK] {len(post_comments)} comments.")
            return post_comments

        except WebDriverException:
            raise
        except Exception as e:
            if self.verbose:
                print(f"    [ERROR] {e}")
            return []

    # ================================================================== #
    #  crawl_keyword (giống VOZCrawler.crawl_keyword)
    # ================================================================== #
    def crawl_keyword(
        self,
        keyword: Optional[str] = None,
        persist: bool = True,
    ) -> List[str]:
        """Tìm posts theo keyword → cào comments."""
        keyword = keyword or self.keyword
        if not keyword:
            raise ValueError("keyword is required")

        if self.driver is None:
            self.get_driver()

        output_path = self.get_output_path(keyword)
        urls = self.search_threads(keyword, self.max_posts)
        if not urls:
            return []

        all_texts: List[str] = []

        for i, url in enumerate(urls):
            if self.verbose:
                print(f"\n[KEYWORD: {keyword}] [POST {i + 1}/{len(urls)}]")

            try:
                batch = self.scrape_comments(url, self.max_scroll)

                if batch:
                    all_texts.extend(batch)
                    self.comments.extend(batch)
                    if persist:
                        count = self._append_batch_to_csv(output_path, batch)
                        if self.verbose:
                            print(f"    [SUCCESS] + {count} dòng.")
                else:
                    if self.verbose:
                        print("    [WARN] Không có data -> có thể bị chặn.")
                    raise WebDriverException("Zero data returned (Zombie Browser)")

            except WebDriverException as e:
                if self.verbose:
                    print(f"    [CRITICAL] Lỗi/Treo: {e}")
                    print("    [RECOVERY] Restarting Driver...")
                try:
                    self.restart_driver()
                    if self.verbose:
                        print("    [RECOVERY] Driver mới sẵn sàng.")
                except Exception:
                    if self.verbose:
                        print("    [FATAL] Không thể restart driver.")
                    break

            except Exception as e:
                if self.verbose:
                    print(f"    [ERROR] Bỏ qua: {e}")

            time.sleep(random.uniform(2.0, 4.0))

        if self.verbose:
            print(f"\n[RESULT] Tổng cộng {len(all_texts)} comments cho '{keyword}'.")
        return all_texts

    # ================================================================== #
    #  crawl_keywords (giống VOZCrawler.crawl_keywords)
    # ================================================================== #
    def crawl_keywords(
        self,
        keywords: Iterable[str],
        persist: bool = True,
    ) -> dict:
        """Cào comments cho nhiều từ khóa.

        Returns
        -------
        dict
            ``{keyword: [texts]}``
        """
        results = {}
        for kw in keywords:
            results[kw] = self.crawl_keyword(kw, persist=persist)
            time.sleep(3)
        return results

    # ================================================================== #
    #  crawl_profile
    # ================================================================== #
    def crawl_profile(
        self,
        profile_url: str,
        max_posts: int = 20,
        max_scroll_profile: int = 15,
    ) -> List[str]:
        """Cào comments từ các post trên profile 1 user.

        Parameters
        ----------
        profile_url : str
            URL profile, ví dụ ``https://www.threads.net/@username``
        max_posts : int
            Số post tối đa.
        max_scroll_profile : int
            Số lần cuộn trang profile.
        """
        if self.driver is None:
            self.get_driver()

        if self.verbose:
            print(f"[PROFILE] {profile_url}")

        try:
            self.driver.get(profile_url)
        except TimeoutException:
            try:
                self.driver.execute_script("window.stop();")
            except Exception:
                pass

        time.sleep(random.uniform(3.0, 5.0))
        self._dismiss_login_popup()

        post_urls: List[str] = []
        seen_urls: set = set()
        prev_count = 0

        for scroll_i in range(max_scroll_profile):
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            time.sleep(random.uniform(2.0, 3.0))

            links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/post/"]')
            for link in links:
                href = link.get_attribute("href") or ""
                if "/post/" in href and href not in seen_urls:
                    seen_urls.add(href)
                    post_urls.append(href)

            if self.verbose and (scroll_i + 1) % 3 == 0:
                print(
                    f"    [SCROLL PROFILE] Lần {scroll_i + 1}, "
                    f"tìm thấy {len(post_urls)} posts..."
                )
            if len(post_urls) >= max_posts:
                break
            if len(post_urls) == prev_count:
                break
            prev_count = len(post_urls)

        post_urls = post_urls[:max_posts]
        if self.verbose:
            print(f"[INFO] {len(post_urls)} post links. Bắt đầu cào replies...")

        all_texts: List[str] = []
        for i, url in enumerate(post_urls):
            if self.verbose:
                print(f"\n  [POST {i + 1}/{len(post_urls)}]")
            try:
                batch = self.scrape_comments(url)
                all_texts.extend(batch)
                self.comments.extend(batch)
            except WebDriverException as e:
                if self.verbose:
                    print(f"    [CRITICAL] {e} -> Restart driver...")
                try:
                    self.restart_driver()
                except Exception:
                    break
            except Exception as e:
                if self.verbose:
                    print(f"    [ERROR] Bỏ qua: {e}")
            time.sleep(random.uniform(2.0, 4.0))

        if self.verbose:
            print(f"\n[RESULT] Tổng cộng {len(all_texts)} comments từ profile.")
        return all_texts

    # ================================================================== #
    #  CSV helpers
    # ================================================================== #
    @staticmethod
    def _append_batch_to_csv(file_path: str, batch_data: List[str]) -> int:
        """Ghi incremental vào CSV (giống VOZCrawler.append_batch_to_csv)."""
        if not batch_data:
            return 0
        file_exists = os.path.isfile(file_path)
        start_id = 1
        if file_exists:
            try:
                df_check = pd.read_csv(file_path, usecols=["id"])
                if not df_check.empty:
                    start_id = int(df_check["id"].max()) + 1
            except Exception:
                start_id = 1

        df_new = pd.DataFrame(batch_data, columns=["text"])
        df_new.insert(0, "id", range(start_id, start_id + len(df_new)))
        df_new.to_csv(
            file_path, mode="a", header=not file_exists,
            index=False, encoding="utf-8-sig",
        )
        return len(df_new)

    def output(self, name: Optional[str] = None) -> str:
        """Xuất toàn bộ self.comments ra file CSV."""
        name = name or self.keyword or "threads"

        output_dir = os.path.join(os.pardir, "data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{name}.csv")

        df = pd.DataFrame(self.comments, columns=["text"])
        df.insert(0, "id", range(1, len(df) + 1))
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        if self.verbose:
            print(f"[OUTPUT] Đã lưu {len(df)} dòng → {output_path}")
        return output_path

    # ================================================================== #
    #  Tiện ích
    # ================================================================== #
    def clear(self) -> None:
        """Xóa toàn bộ comments đã thu thập."""
        self.comments.clear()

    def __len__(self) -> int:
        return len(self.comments)

    def __repr__(self) -> str:
        return f"ThreadsCrawler(comments={len(self.comments)}, keyword={self.keyword!r})"

