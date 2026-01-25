from googleapiclient.discovery import build
import pandas as pd
from dotenv import load_dotenv
import os
import json

load_dotenv()

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
                        "text": f"{comment["textDisplay"]}"
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





class VOZCrawler:
    def __init__(self, keyword, max_threads, max_pages):
        self.keyword = keyword



