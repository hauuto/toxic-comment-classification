import sys
import os
import re
import json
import logging
import unicodedata
import pandas as pd
import emoji
from tqdm import tqdm
from py_vncorenlp import VnCoreNLP
from typing import List, Dict, Tuple

# --- 1. CONFIGURATION (CẤU HÌNH) ---
class Config:
    # Cấu hình Java cho VnCoreNLP
    JAVA_HOME = r"C:\Program Files\Java\jdk-21"
    ENCODING = 'utf-8'
    
    # Các đường dẫn
    # Lấy thư mục chứa file script hiện tại làm gốc
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = r"K:\GithubRepo\comment-classification\data"
    VNCORENLP_DIR = r"K:\GithubRepo\comment-classification\src\vncorenlp"
    
    # Tên file Input/Output
    INPUT_FILE = os.path.join(DATA_DIR, "IzSYlr3VI1A_raw.csv")
    OUTPUT_FILE = os.path.join(BASE_DIR, "IzSYlr3VI1A_preprocess.csv")
    
    # File tài nguyên
    TEENCODE_FILE = os.path.join(BASE_DIR, "teencode.json")
    SPAM_KEYWORDS_FILE = os.path.join(BASE_DIR, "spamkeyword.json")

    # Ngưỡng lọc rác
    MIN_TEXT_LENGTH = 2      # Bỏ comment quá ngắn
    MAX_WORD_LENGTH = 15     # Bỏ từ quá dài (spam ký tự)
    SPAM_THRESHOLD = 1       # Số từ khóa spam tối thiểu để bị loại

# Setup môi trường
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sys.stdout.reconfigure(encoding=Config.ENCODING)
os.environ["JAVA_HOME"] = Config.JAVA_HOME


# --- 2. HELPER CLASSES ---

class TeencodeConverter:
    def __init__(self, json_path: str):
        self.teencode_dict = self._load_dict(json_path)

    def _load_dict(self, json_path: str) -> Dict[str, str]:
        if not os.path.exists(json_path):
            return {}
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Chuyển về dạng: key (lower) -> value (standard)
            return {var.lower(): std for std, variants in data.items() for var in variants}
        except Exception as e:
            logging.error(f"Lỗi đọc teencode: {e}")
            return {}

    def replace(self, text: str) -> str:
        if not text: return ""
        # Tách từ và dấu câu để thay thế chính xác (vd: "hnay," -> "hnay" + ",")
        tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        result = []
        for token in tokens:
            if token.isalnum(): # Nếu là chữ/số thì tra từ điển
                result.append(self.teencode_dict.get(token.lower(), token))
            else: # Nếu là dấu câu thì giữ nguyên
                result.append(token)
        return ' '.join(result)


class SpamChecker:
    def __init__(self, json_path: str):
        self.patterns = self._load_patterns(json_path)

    def _load_patterns(self, json_path: str) -> Dict[str, re.Pattern]:
        if not os.path.exists(json_path):
            return {}
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            patterns = {}
            for category, keywords in data.items():
                if keywords:
                    escaped = [re.escape(k.lower()) for k in keywords]
                    patterns[category] = re.compile('|'.join(escaped), re.IGNORECASE)
            return patterns
        except Exception:
            return {}

    def is_spam(self, text: str, threshold: int = 1) -> bool:
        if not text: return False
        count = 0
        text_lower = text.lower()
        for pattern in self.patterns.values():
            if pattern.search(text_lower):
                count += 1
        return count >= threshold


# --- 3. TEXT PREPROCESSOR (CORE LOGIC) ---

class TextPreprocessor:
    def __init__(self, vncorenlp_dir: str, teencode_path: str):
        if not os.path.exists(vncorenlp_dir):
            raise FileNotFoundError(f"Không tìm thấy VnCoreNLP tại: {vncorenlp_dir}")
        
        # Khởi tạo VnCoreNLP
        self.vncorenlp = VnCoreNLP(annotators=["wseg", "pos"], save_dir=vncorenlp_dir)
        self.teencode_converter = TeencodeConverter(teencode_path)

    def _mask_emojis(self, text: str) -> Tuple[str, List[str]]:
        found_emojis = []
        def replace_cb(char, _):
            demojized = emoji.demojize(char)
            found_emojis.append(demojized)
            return f" EMOJITOKEN{len(found_emojis)-1} "
        return emoji.replace_emoji(text, replace=replace_cb), found_emojis

    def _restore_emojis(self, text: str, found_emojis: List[str]) -> str:
        def restore_cb(match):
            idx = int(match.group(1))
            return f" {found_emojis[idx]} " if 0 <= idx < len(found_emojis) else match.group(0)
        return re.sub(r'EMOJITOKEN\s*(\d+)', restore_cb, text)

    def process(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip(): return ""
        
        # 1. Clean cơ bản
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'http\S+|www\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # 2. Teencode
        text = self.teencode_converter.replace(text)
        
        # 3. Dấu câu
        text = re.sub(r'([!?.,])\1+', r' \1 ', text)

        # 4. Mask Emoji -> Annotate (JSON) -> Smart Lowercase
        text, emojis = self._mask_emojis(text)
        
        try:
            # --- KHẮC PHỤC LỖI TẠI ĐÂY ---
            # Sử dụng .annotate() để lấy JSON Object thay vì String
            output = self.vncorenlp.annotate_text(text)
            
            # Cấu trúc JSON trả về: {'sentences': [[{'index': 1, 'form': 'Từ', 'posTag': 'Nhãn'}, ...]]}
            processed_tokens = []
            
            if isinstance(output, dict) and 'sentences' in output:
                for sentence in output['sentences']: # Duyệt qua các câu
                    for token in sentence:           # Duyệt qua các từ trong câu
                        # Lấy chính xác dữ liệu từ key, không cắt chuỗi thủ công nữa
                        word = token.get('form', '')
                        tag = token.get('posTag', '')
                        
                        # Logic Smart Lowercase
                        if tag in ['Np', 'Ny', 'M', 'Ab'] or "EMOJITOKEN" in word:
                            processed_tokens.append(word)
                        else:
                            processed_tokens.append(word.lower())
                
                # Ghép lại
                if processed_tokens:
                    text = ' '.join(processed_tokens)

                text = self.vncorenlp.word_segment(text)
            else:
                # Nếu output rỗng hoặc sai format, dùng text gốc (đã clean sơ)
                pass

        except Exception as e:
            # Fallback an toàn
            logging.error(f"Lỗi xử lý NLP: {e}")
            pass 
        
        # 5. Restore Emoji & Final Clean
        text = self._restore_emojis(text, emojis)
        return re.sub(r'\s+', ' ', text).strip()


# --- 4. DATA PIPELINE ---

class DataPipeline:
    def __init__(self, config):
        self.cfg = config
        logging.info("Đang khởi tạo các models...")
        self.preprocessor = TextPreprocessor(self.cfg.VNCORENLP_DIR, self.cfg.TEENCODE_FILE)
        self.spam_checker = SpamChecker(self.cfg.SPAM_KEYWORDS_FILE)

    def load_data(self, filepath: str) -> pd.DataFrame:
        logging.info(f"Đọc file: {filepath}")
        for enc in ['utf-8-sig', 'utf-8', 'utf-16']:
            try:
                df = pd.read_csv(filepath, encoding=enc)
                logging.info(f"Encoding '{enc}': OK. Số dòng: {len(df)}")
                return df
            except Exception:
                continue
        raise ValueError("Không đọc được file CSV (thử kiểm tra encoding hoặc đường dẫn).")

    def filter_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        
        # Lọc dòng rỗng/trùng
        df = df.dropna(subset=['text']).drop_duplicates(subset=['text'], keep='first')
        
        # Hàm kiểm tra hợp lệ
        def is_valid_content(text):
            text = str(text).strip()
            # Quá ngắn
            if len(text) < self.cfg.MIN_TEXT_LENGTH: return False
            # Từ dài vô lý (aaaaaaaa...)
            longest = max(text.split(), key=len, default="")
            if len(longest) > self.cfg.MAX_WORD_LENGTH: return False
            # Dính spam keywords
            if self.spam_checker.is_spam(text, self.cfg.SPAM_THRESHOLD): return False
            return True

        valid_mask = df['text'].apply(is_valid_content)
        df_clean = df[valid_mask].copy()
        
        dropped = initial_count - len(df_clean)
        logging.info(f"Đã lọc bỏ {dropped} dòng rác/spam. Còn lại: {len(df_clean)}")
        return df_clean

    def run(self):
        # 1. Load Data
        try:
            df = self.load_data(self.cfg.INPUT_FILE)
        except Exception as e:
            logging.error(str(e))
            return

        # 2. Filter
        df = self.filter_noise(df)

        logging.info("Bắt đầu xử lý (Preprocessing)...")
        tqdm.pandas(desc="Tiến độ")
        df['processed_text'] = df['text'].progress_apply(self.preprocessor.process)

        # Lọc bỏ dòng rỗng sau xử lý
        df_final = df[df['processed_text'].str.strip().astype(bool)]
        
        # Lưu file: THÊM index=False
        output_path = self.cfg.OUTPUT_FILE
        df_final[['processed_text']].to_csv(output_path, index=True, encoding='utf-8-sig')
        logging.info(f"XONG! Kết quả lưu tại: {output_path}")

# --- 5. MAIN ENTRY POINT ---
if __name__ == "__main__":
    pipeline = DataPipeline(Config)
    pipeline.run()