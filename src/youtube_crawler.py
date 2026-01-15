from googleapiclient.discovery import build
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
# Cấu hình API
load_dotenv()
API_KEY = os.getenv("API_KEY")
VIDEO_ID = "Ym6h7ZLtdhE" # Thay bằng ID video của bạn

def get_youtube_comments(video_id):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None

    while True:
        # Gọi API lấy danh sách comment
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100, # Tối đa 100 comment mỗi lần gọi
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        # Trích xuất dữ liệu từ JSON
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": f"{comment["textDisplay"]}"
            })

        # Kiểm tra nếu còn trang tiếp theo
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# Chạy và in kết quả
all_comments = get_youtube_comments(VIDEO_ID)
print(f"Tổng số comment lấy được: {len(all_comments)}")

for c in all_comments[:5]: # In thử 5 comment đầu tiên
    print(f"{c['text']}")
    
cmt = pd.DataFrame(all_comments)
cmt.to_csv(f"../data/{VIDEO_ID}_raw.csv")


# Labels:
# 