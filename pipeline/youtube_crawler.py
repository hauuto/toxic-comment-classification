"""
youtube_crawler.py – YouTube comment crawler using the YouTube Data API v3.
Replaces the old Playwright-based YouTube scraper.
"""
import os
import re
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()


def _extract_video_id(url: str) -> str:
    """Extract the 11-char video ID from various YouTube URL formats."""
    patterns = [
        r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    # Maybe the user passed a raw video id
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url.strip()):
        return url.strip()
    return ""


def _sanitize_error(error_msg) -> str:
    """Remove API keys from error messages to prevent leaking secrets in logs."""
    return re.sub(r'key=[A-Za-z0-9_-]+', 'key=***REDACTED***', str(error_msg))


def extract_youtube_comments(
    url_or_id: str,
    *,
    log_callback=None,
    data_callback=None,
    stop_event=None,
    preprocessor=None,
    use_decoder: bool = True,
    use_filter: bool = True,
    use_normalizer: bool = True,
    use_segmentor: bool = True,
    current_id: int = 1,
    seen_texts: set = None,
    extracted_data: list = None,
) -> int:
    """Fetch all comments for a YouTube video via the Data API v3.

    Returns the next ``current_id`` after processing.
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    if seen_texts is None:
        seen_texts = set()
    if extracted_data is None:
        extracted_data = []

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
            log("[YouTube API] Đã nhận lệnh DỪNG.")
            break

        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
            )
            response = request.execute()
        except HttpError as e:
            # YouTube API sometimes returns 404/400 mid-pagination when a
            # pageToken becomes stale.  This is normal – treat as end-of-comments.
            if e.resp.status in (400, 404):
                log(f"[YouTube API] Đã lấy hết bình luận khả dụng (API trả về {e.resp.status} ở trang cuối).")
            elif e.resp.status == 403:
                log(f"[YouTube API] Hết quota hoặc bị từ chối: {_sanitize_error(e)}")
            else:
                log(f"[YouTube API] Lỗi HTTP {e.resp.status}: {_sanitize_error(e)}")
            break
        except Exception as e:
            log(f"[YouTube API] Lỗi gọi API: {_sanitize_error(e)}")
            break

        items = response.get("items", [])
        if not items:
            break

        new_batch = []
        for item in items:
            if stop_event and stop_event.is_set():
                break
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            full_text = snippet.get("textDisplay", "").strip()
            if not full_text:
                continue
            if len(full_text) < 2 and full_text not in ["Ok", "Dạ"]:
                continue

            if preprocessor:
                processed = preprocessor.process_comment(
                    full_text,
                    use_decoder=use_decoder,
                    use_filter=use_filter,
                    use_normalizer=use_normalizer,
                    use_segmentor=use_segmentor,
                )
                if not processed["is_valid"]:
                    continue
                clean_text = processed["cleaned_text"]
            else:
                clean_text = full_text

            if clean_text not in seen_texts:
                seen_texts.add(clean_text)
                row = {"id": current_id, "text": clean_text}
                new_batch.append(row)
                extracted_data.append(row)
                current_id += 1

        if new_batch:
            total_fetched += len(new_batch)
            if data_callback:
                data_callback(new_batch)
            log(f"[YouTube API] +{len(new_batch)} bình luận (Tổng: {total_fetched})")

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    log(f"[YouTube API] Hoàn tất video {video_id} – tổng {total_fetched} bình luận.")
    return current_id
