import re
import unicodedata
import emoji
import sys
import os
import pandas as pd
from teencode_converter import TeencodeConverter

# --- SETUP MÔI TRƯỜNG ---
sys.stdout.reconfigure(encoding='utf-8')
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-21"
from py_vncorenlp import VnCoreNLP

BASE_DIR = r"K:\GithubRepo\comment-classification\src\vncorenlp"
if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(f"Không tìm thấy thư mục VnCoreNLP tại: {BASE_DIR}")

vncorenlp = VnCoreNLP(annotators=["wseg", "pos"], save_dir=BASE_DIR)


class TextPreprocessor:
    def __init__(self, vncorenlp_instance=None, teencode_path="teencode.json"):
        self.vncorenlp = vncorenlp_instance
        self.teencode_converter = TeencodeConverter(teencode_path)

    def normalize_unicode(self, text):
        if not isinstance(text, str): return str(text)
        return unicodedata.normalize('NFC', text)

    def to_lower(self, text):
        return text.lower()

    def remove_urls(self, text):
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def standardize_punctuation(self, text):
        text = re.sub(r'!+', ' ! ', text)
        text = re.sub(r'\?+', ' ? ', text)
        text = re.sub(r'\.+', ' . ', text)
        text = re.sub(r',+', ' , ', text)
        return text

    def remove_duplicate_characters(self, text):
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def normalize_teencode(self, text):
        return self.teencode_converter.replace(text)

    # --- SỬA ĐỔI QUAN TRỌNG: MASKING KHÔNG DÙNG GẠCH DƯỚI ---
    def mask_emojis(self, text):
        found_emojis = []
        
        def replace_callback(char, data_dict):
            demojized = emoji.demojize(char, delimiters=(" :", ": "))
            found_emojis.append(demojized)
            # Dùng mã EMOJITOKEN liền mạch (coi như 1 từ tiếng Anh)
            # VnCoreNLP sẽ không cắt chữ này ra.
            return f" EMOJITOKEN{len(found_emojis)-1} "

        masked_text = emoji.replace_emoji(text, replace=replace_callback)
        return masked_text, found_emojis

    # --- SỬA ĐỔI QUAN TRỌNG: RESTORE ĐƠN GIẢN HƠN ---
    def restore_emojis(self, text, found_emojis):
        """
        Khôi phục emoji từ mã EMOJITOKEN
        """
        # Đề phòng trường hợp VnCoreNLP tách số ra khỏi chữ (VD: EMOJITOKEN 0)
        # Ta dùng regex để tìm: EMOJITOKEN + (khoảng trắng tùy ý) + Số
        def restore_callback(match):
            idx = int(match.group(1))
            if 0 <= idx < len(found_emojis):
                return " " + found_emojis[idx] + " "
            return match.group(0) # Nếu lỗi index thì giữ nguyên

        # Tìm tất cả pattern EMOJITOKEN + số
        text = re.sub(r'EMOJITOKEN\s*(\d+)', restore_callback, text)
        return text

    def segment_text(self, text):
        if self.vncorenlp:
            try:
                sentences = self.vncorenlp.word_segment(text)
                return ' '.join(sentences)
            except Exception as e:
                print(f"Lỗi Segment: {e}")
                return text
        return text

    def process(self, text):
        if not isinstance(text, str): return ""
        
        # 1. Pipeline làm sạch
        text = self.normalize_unicode(text)
        text = self.to_lower(text)
        text = self.remove_urls(text)
        text = self.remove_duplicate_characters(text)
        text = self.standardize_punctuation(text)
        text = self.normalize_teencode(text) 
        
        # 2. Ẩn Emoji (Masking) bằng từ khóa an toàn EMOJITOKEN
        text, emoji_storage = self.mask_emojis(text)
        
        # 3. Tách từ
        # VnCoreNLP thấy "EMOJITOKEN0" sẽ coi là tên riêng (Np) hoặc từ lạ -> Giữ nguyên
        text = self.segment_text(text)
        
        # 4. Trả lại Emoji
        text = self.restore_emojis(text, emoji_storage)
        
        # 5. Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

# --- MAIN ---
if __name__ == "__main__":    
    print("--- Bắt đầu xử lý ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    teencode_path_absolute = os.path.join(current_dir, "teencode.json")
    
    processor = TextPreprocessor(
        vncorenlp_instance=vncorenlp, 
        teencode_path=teencode_path_absolute
    )

    # Test với chuỗi cứng trước để đảm bảo logic đúng
    test_str = "ai từ nhóm VSTAR qua đây ko, đoàn mình ấy, xin lỗi anh Qúy 🤣🙃"
    print(f"\nTEST NHANH:\nInput: {test_str}")
    print(f"Output: {processor.process(test_str)}\n")

    # Xử lý file CSV
    inputfile = r"K:\GithubRepo\comment-classification\data\IzSYlr3VI1A_raw.csv"
    
    data = None
    for enc in ['utf-8-sig', 'utf-8', 'utf-16']:
        try:
            data = pd.read_csv(inputfile, encoding=enc)
            # Check nhanh
            str(data['text'].iloc[0]) 
            break
        except Exception:
            continue
            
    if data is not None:
        data = data.dropna(subset=['text']) 
        lim = 6
        cur = 0
        print("-" * 40)
        for input_text in data["text"]:
            input_str = str(input_text)
            # Bỏ qua nếu dòng quá ngắn hoặc rỗng
            if not input_str.strip(): continue

            output_text = processor.process(input_str)
            print(f"Input:  {input_str}")
            print(f"Output: {output_text}")
            print("-" * 40)
            cur += 1
            if cur >= lim: break
    else:
        print("Không đọc được file CSV.")