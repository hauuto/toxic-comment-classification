from googleapiclient.discovery import build
import pandas as pd
from dotenv import load_dotenv
import os
import json



def get_youtube_comments():
    video_id = input("Nhập id của video vào đây: ")
    load_dotenv()
    youtube = build("youtube", "v3", developerKey=os.getenv("API_KEY"))
    comments = []
    next_page_token = None

    if check_video_id(video_id):
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

        return output(comments, video_id)
    else:
        print("Video đã được cào")
        return 0


def check_video_id(video_id):
    with open('../reports/crawled_videos.json') as file:
        crawled = json.load(file)

    if video_id in crawled and crawled[video_id]["status"] == "crawled":
        return False
    return True


def output(comments, video_id):


    output = pd.DataFrame(comments)
    output.to_csv(f"../data/raw/{video_id}.csv")




    with open ("../reports/crawled_videos.json", 'r', encoding="utf-8") as f:
        report = json.load(f)
    report[video_id] = {"status": "crawled"}



    with open ("../reports/crawled_videos.json", 'w', encoding="utf-8" ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)




    print(f"Tổng số comment lấy được: {len(comments)}")
    for c in comments[:5]:
        print(f"{c['text']}")
    return comments
