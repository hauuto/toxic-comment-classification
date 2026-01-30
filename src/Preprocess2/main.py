from Decoders.demoji import EmojiDecoder
from Filters.english_filter import EnglishFilter
from Filters.short_text_filter import ShortTextFilter
from Decoders.teencode_decoder import TeencodeConverter
from Decoders.repetition_decoder import RepetitionDecoder
from src.Preprocess2.Decoders.link_filter import LinkFilter
from Decoders.vncorenlp_decoder import VnCoreNLPDecoder
from Normalizers.vietnamese_typing_normalizer import VietnameseTypingNormalizer
import pandas as pd
from tqdm import tqdm # Thư viện tạo thanh tiến trình
import unicodedata
import os
import glob
import argparse

# --- KHỞI TẠO CÁC BỘ LỌC/XỬ LÝ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(BASE_DIR, "Json")

emojiDecoder = EmojiDecoder(os.path.join(JSON_DIR, "emoji_vi.json"))
teencodeDecoder = TeencodeConverter(os.path.join(JSON_DIR, "teencode.json"))
repetitionDecoder = RepetitionDecoder()
viTypingNormalizer = VietnameseTypingNormalizer()

englishFilter = EnglishFilter()
shortTextFilter = ShortTextFilter()
linkFilter = LinkFilter()

JAVA_PATH = r"C:\Program Files\Java\jdk-21"
MODEL_PATH = r"K:\GithubRepo\comment-classification\src\vncorenlp"

coreNlp = VnCoreNLPDecoder(MODEL_PATH, JAVA_PATH)

# --- CÁC HÀM XỬ LÝ ---

def normalize_unicode(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Đưa về chuẩn NFC (chuẩn phổ biến nhất cho tiếng Việt)
    return unicodedata.normalize('NFC', text)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột của DataFrame.
    - Cột đầu tiên -> 'id'
    - Cột thứ hai -> 'text'
    Hoặc tự động detect cột text nếu có tên như 'text', 'comment', 'content', etc.
    """
    columns = list(df.columns)
    
    # Các tên cột có thể là text
    text_column_names = ['text', 'comment', 'content', 'message', 'body', 'description']
    # Các tên cột có thể là id  
    id_column_names = ['id', 'no', 'index', 'idx', 'stt']
    
    # Tìm cột text
    text_col = None
    for name in text_column_names:
        matching = [c for c in columns if name.lower() in c.lower()]
        if matching:
            text_col = matching[0]
            break
    
    # Nếu không tìm thấy, lấy cột thứ 2 (hoặc cột cuối nếu chỉ có 1 cột)
    if text_col is None:
        text_col = columns[1] if len(columns) > 1 else columns[0]
    
    # Tìm cột id
    id_col = None
    for name in id_column_names:
        matching = [c for c in columns if name.lower() == c.lower()]
        if matching:
            id_col = matching[0]
            break
    
    # Nếu không tìm thấy, lấy cột đầu tiên (nếu khác text_col)
    if id_col is None:
        id_col = columns[0] if columns[0] != text_col else None
    
    # Rename columns
    rename_map = {}
    if text_col and text_col != 'text':
        rename_map[text_col] = 'text'
    if id_col and id_col != 'id':
        rename_map[id_col] = 'id'
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  📋 Đã chuẩn hóa cột: {rename_map}")
    
    # Đảm bảo có cột text
    if 'text' not in df.columns:
        raise ValueError(f"Không tìm thấy cột text trong file! Các cột hiện có: {list(df.columns)}")
    
    return df

def stageFilt(x: pd.DataFrame):
    print("--- Bắt đầu lọc dữ liệu (Filter) ---")
    x = linkFilter.filt(x, 'text')
    x = shortTextFilter.filt(x, 'text', len_threshold=20, space_threshold=5)
    x = englishFilter.filt(x, 'text')
    return x

def _preserve_emoticons_clean(text: str) -> str:
    import re
    emoticon_patterns = [r"[:=]-?\)+", r":>", r":d", r":3"]
    placeholders = []
    def stash(m):
        placeholders.append(m.group(0))
        return f" EMO_PLACE_{len(placeholders)-1} "

    for pat in emoticon_patterns:
        text = re.sub(pat, stash, text, flags=re.IGNORECASE)

    text = re.sub(r"[@#%*]", "", text)
    tokens = text.split()
    cleaned_tokens = [t for t in tokens if not re.fullmatch(r"[.,;:!?]+", t)]
    text = " ".join(cleaned_tokens)

    for i, emo in enumerate(placeholders):
        text = text.replace(f"EMO_PLACE_{i}", emo)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def decode(x: str):
    x = normalize_unicode(x)
    x = viTypingNormalizer.replace(x)
    x = teencodeDecoder.replace(x)
    x = repetitionDecoder.normalize(x)
    x = coreNlp.segment_text(x)
    x = emojiDecoder.decode(x)
    x = x.lower()
    x = _preserve_emoticons_clean(x)
    return x

def stageDecode(df: pd.DataFrame):
    print("--- Bắt đầu giải mã (Decode) ---")
    df['text'] = df['text'].astype(str)
    processed_texts = []
    for text in tqdm(df['text'], desc="Đang xử lý"):
        processed_texts.append(decode(text))
    df['text'] = processed_texts
    return df

def saveFile(df: pd.DataFrame, path: str):
    print(f"--- Đang lưu file vào: {path} ---")
    # Chỉ giữ lại cột text, dùng index làm id
    output_df = df[["text"]].copy()
    output_df.to_csv(path, index=True, encoding='utf-8-sig', index_label="no")
    print("Đã lưu xong!")

# --- ARGUMENT PARSER ---
parser = argparse.ArgumentParser(description='Xử lý dữ liệu comment từ file CSV')
parser.add_argument('files', nargs='*', default=None, 
                    help='Tên các file cần xử lý (không cần đuôi .csv). Mặc định: xử lý tất cả file trong thư mục raw')
parser.add_argument('--no-combine', action='store_true',
                    help='Không kết hợp các file sau khi xử lý')
args = parser.parse_args()

# --- MAIN RUNNING ---

raw_path = r"K:\GithubRepo\comment-classification\data\raw"
output_path = r"K:\GithubRepo\comment-classification\data"

# Lấy danh sách file cần xử lý
if args.files:
    # Nếu có chỉ định file cụ thể
    csv_files = []
    for filename in args.files:
        # Thêm đuôi .csv nếu chưa có
        if not filename.endswith('.csv'):
            filename = filename + '.csv'
        filepath = os.path.join(raw_path, filename)
        if os.path.exists(filepath):
            csv_files.append(filepath)
        else:
            print(f"⚠️ Không tìm thấy file: {filename}")
    
    if not csv_files:
        print("❌ Không có file hợp lệ nào để xử lý!")
        exit(1)
else:
    # Mặc định: lấy tất cả file CSV trong thư mục raw
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
        print(f"  📋 Các cột: {list(data.columns)}")
        
        # Chuẩn hóa tên cột
        data = normalize_columns(data)
        
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
if not args.no_combine:
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
        
        for processed_file in processed_files:
            # Lấy tên file để in log
            file_name = os.path.basename(processed_file)
            print(f"  - Đang đọc toàn bộ dữ liệu từ {file_name}")
            
            try:
                # Đọc file đã xử lý
                df = pd.read_csv(processed_file)
                
                if len(df) > 0:
                    # Thêm cột source để biết dữ liệu từ file nào
                    df['source'] = os.path.splitext(file_name)[0]
                    
                    combined_data.append(df)
                    print(f"    ✓ Đã lấy toàn bộ {len(df)} dòng")
                else:
                    print(f"    ! File trống, bỏ qua.")
                
            except Exception as e:
                print(f"    ✗ Lỗi khi xử lý file {file_name}: {str(e)}")
                continue
        
        # Kết hợp tất cả dữ liệu
        if combined_data:
            # pd.concat sẽ nối toàn bộ danh sách DataFrame lại với nhau
            final_combined = pd.concat(combined_data, ignore_index=True)
            
            # Xáo trộn dữ liệu (Shuffle) - Giữ nguyên nếu bạn muốn dữ liệu trộn đều các nguồn
            final_combined = final_combined.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Lưu file kết hợp đầy đủ
            combined_output_path = os.path.join(output_path, "combined_processed.csv")
            final_combined.to_csv(combined_output_path, index=True, encoding='utf-8-sig', index_label="no")

            # Tạo thêm file combine.csv (chỉ gồm no, text)
            combine_simple_path = os.path.join(output_path, "combine.csv")
            if "text" in final_combined.columns:
                simple_df = final_combined[["text"]].copy()
                simple_df.to_csv(combine_simple_path, index=True, encoding='utf-8-sig', index_label="no")
            
            print(f"\n✓ Đã kết hợp thành công!")
            print(f"  - Tổng số dòng: {len(final_combined)}")
            print(f"  - File được lưu tại: {combined_output_path}")
            print(f"  - File đơn giản (no,text): {combine_simple_path}")
            
            # Hiển thị thống kê theo nguồn
            print(f"\nThống kê số dòng theo nguồn:")
            source_counts = final_combined['source'].value_counts()
            for source, count in source_counts.items():
                print(f"  - {source}: {count} dòng")
        else:
            print("Không có dữ liệu nào để kết hợp!")

print(f"\n{'='*60}")
print("Hoàn tất toàn bộ quá trình xử lý!")
print(f"{'='*60}")