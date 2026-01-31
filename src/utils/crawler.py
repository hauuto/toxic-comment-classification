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
