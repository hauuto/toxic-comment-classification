# Báo cáo phân tích dữ liệu Comment YouTube

## Mô tả
Module này cung cấp báo cáo chi tiết về dữ liệu comment YouTube trước và sau khi preprocessing.

## Tính năng chính

### 📊 Thống kê cơ bản
- So sánh số lượng comment trước và sau xử lý
- Phân tích độ dài và số từ trung bình
- Phát hiện tỷ lệ spam
- Tính toán tỷ lệ dữ liệu được giữ lại

### 🔍 Phân tích chất lượng văn bản
- Phân bố độ dài text theo các khoảng
- Top từ khóa xuất hiện nhiều nhất
- Thống kê chi tiết về độ dài và số từ

### 📹 Phân tích theo video
- Phân bố số lượng comment theo từng video
- So sánh hiệu quả preprocessing cho từng video

### 🔧 Phát hiện thay đổi preprocessing
- Phân tích các thay đổi do quá trình xử lý
- Đếm số records bị loại bỏ
- Phát hiện các pattern xử lý (underscores, emoji, etc.)

### 📈 Biểu đồ trực quan
- Biểu đồ so sánh raw vs processed data
- Histogram phân bố độ dài text
- Pie chart phân bố theo video
- Histogram phân bố số từ

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements_analysis.txt
```

2. Hoặc cài đặt thủ công:
```bash
pip install pandas numpy matplotlib seaborn
```

## Sử dụng

### Cách 1: Chạy script đơn giản
```bash
python run_report.py
```

### Cách 2: Sử dụng class DataAnalysisReport
```python
from data_analysis_report import DataAnalysisReport

# Khởi tạo analyzer
analyzer = DataAnalysisReport()

# Chạy phân tích đầy đủ
analyzer.run_full_analysis()

# Hoặc chạy từng bước
analyzer.load_data()
analyzer.generate_basic_statistics()
analyzer.analyze_text_quality()
analyzer.create_visualizations()
```

### Cách 3: Phân tích tùy chỉnh
```python
from data_analysis_report import DataAnalysisReport

analyzer = DataAnalysisReport()
analyzer.load_data()

# Chỉ chạy thống kê cơ bản
raw_stats, processed_stats = analyzer.generate_basic_statistics()

# Chỉ tạo biểu đồ
analyzer.create_visualizations()
```

## Cấu trúc dữ liệu đầu vào

Script yêu cầu cấu trúc thư mục sau:
```
data/
├── raw/
│   ├── video1.csv
│   ├── video2.csv
│   └── ...
├── video1_processed.csv
├── video2_processed.csv
├── combined_processed.csv
└── ...
```

### Format file CSV:
- **Raw data**: `no,text`
- **Processed data**: `no,text`
- **Combined data**: `no1,no,text,source`

## Kết quả đầu ra

### 📄 Báo cáo text
- Hiển thị trên console với format đẹp
- Xuất file `.txt` vào thư mục `reports/`

### 📊 Biểu đồ
- Lưu file PNG với độ phân giải cao
- Hiển thị trực tiếp trên màn hình

### 📁 Thư mục reports
Tất cả kết quả được lưu trong:
```
reports/
├── data_analysis_charts.png
└── data_analysis_summary_YYYYMMDD_HHMMSS.txt
```

## Ví dụ output

```
📊 THỐNG KÊ CƠ BẢN
==============================================================

📈 TRƯỚC KHI XỬ LÝ (Raw Data):
----------------------------------------------------------------------------------------------------
Video                Số comments  Độ dài TB    Số từ TB     Spam     Tỷ lệ spam  
----------------------------------------------------------------------------------------------------
eLcYOrgGl-A          32          85.3         12.4         2        6.25%
IzSYlr3VI1A         1250        92.1         13.8         45       3.60%
...
TỔNG                 5420        89.7         13.2         125      2.31%

📈 SAU KHI XỬ LÝ (Processed Data):
----------------------------------------------------------------------------------------------------
Video                Số comments  Độ dài TB    Số từ TB     Spam     Tỷ lệ spam  
----------------------------------------------------------------------------------------------------
eLcYOrgGl-A          15          78.9         11.2         1        6.67%
IzSYlr3VI1A         890         84.5         12.1         28       3.15%
...
TỔNG                 3850        82.3         11.8         75       1.95%
```

## Tùy chỉnh

### Thay đổi đường dẫn dữ liệu
```python
analyzer = DataAnalysisReport(base_path="/path/to/your/data")
```

### Thêm video mới
Chỉnh sửa list `video_files` trong class:
```python
self.video_files = [
    "video1",
    "video2", 
    "your_new_video"
]
```

### Tùy chỉnh spam detection
Chỉnh sửa method `detect_spam_keywords()` để thêm pattern mới:
```python
spam_patterns = [
    r'your_pattern',
    r'another_pattern'
]
```

## Lỗi thường gặp

1. **FileNotFoundError**: Kiểm tra đường dẫn dữ liệu
2. **ImportError**: Cài đặt lại các thư viện required
3. **UnicodeDecodeError**: Đảm bảo file CSV được encode UTF-8
4. **Empty data**: Kiểm tra format CSV và tên cột

## Yêu cầu hệ thống
- Python 3.7+
- RAM: 2GB+ (tùy kích thước dữ liệu)
- Dung lượng: 100MB+ cho cache và output

## Liên hệ
Nếu có vấn đề hoặc cần hỗ trợ, vui lòng tạo issue hoặc liên hệ team phát triển.