import time
from playwright.sync_api import sync_playwright

from nlp_pipeline import VietnameseCommentPreprocessor
from youtube_crawler import extract_youtube_comments

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
        
    # JS to mutate DOM so text_content() grabs @Name_Here and alt text for emojis
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

        # Expand "View more comments"
        view_more_selectors = ['span:text-is("Xem thêm bình luận")', 'span:text-is("View more comments")', 'div[role="button"]:has-text("Xem thêm bình luận")']
        for selector in view_more_selectors:
            try:
                elements = page.locator(selector)
                for i in range(elements.count()):
                    elements.nth(i).click(timeout=2000)
                    time.sleep(1)
            except Exception:
                pass
        
        # Expand "View N replies" 
        reply_selectors = ['span:has-text(" xem phản hồi")', 'span:has-text(" replies")', 'div[role="button"]:has-text(" xem phản hồi")']
        for selector in reply_selectors:
            try:
                elements = page.locator(selector)
                for i in range(elements.count()):
                    elements.nth(i).click(timeout=2000)
                    time.sleep(1)
            except Exception:
                pass
                
        # Expand "See more" (long comments)
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
    
    while True:
        if stop_event and stop_event.is_set():
            return current_id
            
        time.sleep(2)
        
        # Threads DOM is obfuscated. Use generic span with dir auto
        comment_blocks = page.locator('span[dir="auto"], div[dir="auto"]')
        count = comment_blocks.count()
        new_batch = []
        
        # Threads emojis are usually just unicode, no img replace needed mostly, but we'll try just in case 
        replace_emoji_js = r"""
            const images = document.querySelectorAll('div[dir="auto"].xzsf02u img');
            images.forEach(img => {
                if (img.alt) {
                    const textNode = document.createTextNode(img.alt);
                    img.parentNode.replaceChild(textNode, img);
                }
            });
        """
        page.evaluate(replace_emoji_js)
        
        for i in range(count):
            try:
                full_text = comment_blocks.nth(i).text_content().strip()
                if not full_text: continue
                # Skip the exact text of the login wall button block
                if full_text in ['Tiếp tục với Instagram', 'Continue with Instagram', 'Dùng ứng dụng', 'Use app']: continue
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
                            log_callback=None, data_callback=None, stop_event=None):
    """
    Crawls comments iteratively. Supports multiple URLs separated by semicolon.
    Supports Facebook, YouTube, TikTok, Threads. Auto-detects based on URL.
    - log_callback(msg): send status messages to GUI.
    - data_callback(list_of_dicts): send newly extracted comments to GUI/CSV.
    - stop_event (threading.Event): check if user requested to stop.
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
        if use_decoder or use_filter or use_normalizer or use_segmentor:
            log("Đang khởi tạo bộ tiền xử lý NLP (VnCoreNLP)... Vui lòng đợi vài giây...")
            preprocessor = VietnameseCommentPreprocessor()
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
