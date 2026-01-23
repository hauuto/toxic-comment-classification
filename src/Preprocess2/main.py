from demoji import EmojiDecoder
from english_filter import EnglishFilter
from short_text_filter import ShortTextFilter
from teencode_decoder import TeencodeConverter
from vncorenlp_decoder import VnCoreNLPDecoder
import pandas as pd
from tqdm import tqdm # Thư viện tạo thanh tiến trình
import unicodedata
import os
import glob

# --- KHỞI TẠO CÁC BỘ LỌC/XỬ LÝ ---
emojiDecoder = EmojiDecoder("emoji_vi.json")
teencodeDecoder = TeencodeConverter("teencode.json")

englishFilter = EnglishFilter()
shortTextFilter = ShortTextFilter()

JAVA_PATH = r"C:\Program Files\Java\jdk-21"
MODEL_PATH = r"K:\GithubRepo\comment-classification\src\vncorenlp"

coreNlp = VnCoreNLPDecoder(MODEL_PATH, JAVA_PATH)

# --- CÁC HÀM XỬ LÝ ---

def normalize_unicode(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Đưa về chuẩn NFC (chuẩn phổ biến nhất cho tiếng Việt)
    return unicodedata.normalize('NFC', text)

def stageFilt(x: pd.DataFrame):
    print("--- Bắt đầu lọc dữ liệu (Filter) ---")
    x = shortTextFilter.filt(x, 'text', len_threshold=40, space_threshold=10)
    x = englishFilter.filt(x, 'text')
    return x

def decode(x: str):
    x = teencodeDecoder.replace(x)
    x = coreNlp.segment_text(x)
    x = emojiDecoder.decode(x)
    return x

def stageDecode(df: pd.DataFrame):
    print("--- Bắt đầu giải mã (Decode) ---")
    
    df['text'] = df['text'].astype(str).apply(normalize_unicode)
    df['text'] = df['text'].apply(teencodeDecoder.replace)
    
    processed_texts = []
    for text in tqdm(df['text'], desc="Đang xử lý"):
        processed_texts.append(decode(text))
    
    df['text'] = processed_texts
    
    return df

def saveFile(df: pd.DataFrame, path: str):
    print(f"--- Đang lưu file vào: {path} ---")
    df.drop(columns=['no'])
    df[["text"]].to_csv(path, index=True, encoding='utf-8-sig', index_label="no")
    print("Đã lưu xong!")

# --- MAIN RUNNING ---

raw_path = r"K:\GithubRepo\comment-classification\data\raw"
output_path = r"K:\GithubRepo\comment-classification\data"

# Lấy tất cả file CSV trong thư mục raw
csv_files = glob.glob(os.path.join(raw_path, "*.csv"))

print(f"Tìm thấy {len(csv_files)} file CSV trong thư mục raw:")
for file in csv_files:
    print(f"  - {os.path.basename(file)}")

# Xử lý từng file
for input_file_path in csv_files:
    print(f"\n{'='*60}")
    print(f"Đang xử lý file: {os.path.basename(input_file_path)}")
    print(f"{'='*60}")
    
    try:
        # Đọc dữ liệu
        data = pd.read_csv(input_file_path)
        print(f"Đã đọc {len(data)} dòng dữ liệu")
        
        # Áp dụng pipeline
        data = stageFilt(data)
        data = stageDecode(data)
        
        # Tạo tên file output với suffix _processed
        base_name = os.path.splitext(os.path.basename(input_file_path))[0]
        output_file_name = f"{base_name}_processed.csv"
        output_file_path = os.path.join(output_path, output_file_name)
        
        # Lưu file
        saveFile(data, output_file_path)
        
        print(f"✓ Đã hoàn thành xử lý file: {output_file_name}")
        
    except Exception as e:
        print(f"✗ Lỗi khi xử lý file {os.path.basename(input_file_path)}: {str(e)}")
        continue

print(f"\n{'='*60}")
print("Hoàn thành xử lý tất cả file!")
print(f"{'='*60}")

# --- KẾT HỢP TẤT CẢ FILE THÀNH 1 FILE ---
print(f"\n{'='*60}")
print("Bắt đầu kết hợp tất cả file đã xử lý thành 1 file...")
print(f"{'='*60}")

# Tìm tất cả file _processed.csv
processed_files = glob.glob(os.path.join(output_path, "*_processed.csv"))
num_files = len(processed_files)

if num_files == 0:
    print("Không tìm thấy file nào đã được xử lý!")
else:
    print(f"Tìm thấy {num_files} file đã xử lý:")
    
    combined_data = []
    sample_ratio = 1.0 / num_files
    
    for processed_file in processed_files:
        print(f"  - Đang lấy mẫu từ {os.path.basename(processed_file)} với tỷ lệ {sample_ratio:.3f}")
        
        try:
            # Đọc file đã xử lý
            df = pd.read_csv(processed_file)
            
            # Lấy mẫu theo tỷ lệ 1/số_lượng_file
            if len(df) > 0:
                sample_size = max(1, int(len(df) * sample_ratio))  # Đảm bảo ít nhất 1 dòng
                sampled_df = df.sample(n=min(sample_size, len(df)), random_state=42)
                
                # Thêm cột source để biết dữ liệu từ file nào
                sampled_df['source'] = os.path.splitext(os.path.basename(processed_file))[0]
                
                combined_data.append(sampled_df)
                print(f"    ✓ Đã lấy {len(sampled_df)} dòng từ {len(df)} dòng")
            
        except Exception as e:
            print(f"    ✗ Lỗi khi xử lý file {os.path.basename(processed_file)}: {str(e)}")
            continue
    
    # Kết hợp tất cả dữ liệu
    if combined_data:
        final_combined = pd.concat(combined_data, ignore_index=True)
        
        # Xáo trộn dữ liệu
        final_combined = final_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Lưu file kết hợp
        combined_output_path = os.path.join(output_path, "combined_processed.csv")
        final_combined.to_csv(combined_output_path, index=True, encoding='utf-8-sig', index_label="no")
        
        print(f"\n✓ Đã kết hợp thành công!")
        print(f"  - Tổng số dòng: {len(final_combined)}")
        print(f"  - File được lưu tại: {combined_output_path}")
        
        # Hiển thị thống kê theo nguồn
        print(f"\nThống kê theo nguồn:")
        source_counts = final_combined['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  - {source}: {count} dòng")
    else:
        print("Không có dữ liệu nào để kết hợp!")

print(f"\n{'='*60}")
print("Hoàn tất toàn bộ quá trình xử lý!")
print(f"{'='*60}")