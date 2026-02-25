# toxic-comment-classification

Dự án thuộc môn học phần **Xử lý ngôn ngữ tự nhiên**

## Cách chạy project (Python)

### Cài đặt Poetry (nếu chưa có)
```
pip install poetry
```

### Cài đặt venv và dependency
```
poetry install
```

## Auto labeling bằng Gemini (Google AI Studio)

### 1) Cấu hình API key
- Tạo file `.env` ở root project (hoặc set biến môi trường hệ thống).
- Thêm:
```
GEMINI_API_KEY=YOUR_KEY_HERE
GEMINI_MODEL=gemini-2.0-flash
```

### 2) Chạy GUI (pipeline)
- Tab Labeling sẽ tự dùng Gemini khi có `GEMINI_API_KEY`.
- Nếu không có `GEMINI_API_KEY` thì giữ nguyên fallback sang LM Studio như trước.

### 3) Chạy CLI Gemini 3-tier
```
python pipeline/run_gemini_labeling_3tier.py --resume
```
Output giữ nguyên format 3-tier: `id,text,tier1_spam,tier2_toxic,tier3_labels` (Tier3 join bằng `|`).

## Crawl Google Maps theo khu vực (Places API)

### Cấu hình
- Tạo API key trên Google Maps Platform và bật **Places API**.
- Thêm vào `.env`:
```
GOOGLE_MAPS_API_KEY=YOUR_KEY_HERE
```

### Dùng trong GUI
- Vào tab **Keyword Crawler** → chọn Platform **Google Maps**.
- Nhập `Center Lat/Lng`, `Radius(m)`, `Step(m)` và keyword (ví dụ: "quán cà phê", "nhà hàng").

### Giới hạn
- Places Details API chỉ trả **một số review** cho mỗi địa điểm (không có API chính thức để lấy 100% toàn bộ review).
- Tool sẽ cố gắng lấy tối đa reviews mà API trả về, quét được nhiều địa điểm bằng cách quét theo lưới trong vùng.

### Kiểm tra Poetry
```
poetry env info
poetry show
```

### Thêm dependency
```
poetry add <dependency_name>
```

#### Reference: https://python-poetry.org/docs/basic-usage/
