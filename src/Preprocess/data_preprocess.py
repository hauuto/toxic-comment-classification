import re
import unicodedata
import emoji
import sys
import os

# Dictionary teencode thủ công (Cập nhật dựa trên quan sát dữ liệu thực tế)
# Đây là các từ phổ biến trên Youtube/Tiktok/Facebook Việt Nam
TEENCODE_DICT = {
    "o": "không", "ko": "không", "k": "không", "kh": "không", "khong": "không",
    "dc": "được", "đc": "được", "dk": "được",
    "t": "tôi", "tui": "tôi", "tao": "tôi",
    "m": "mày", "mk": "mình", "mik": "mình",
    "c": "các", "cc": "các",
    "ng": "người", "n": "người",
    "pt": "phát triển",
    "bh": "bây giờ",
    "tr": "trời", "trùi": "trời",
    "clgt": "cái lề gì thốn",
    "dell": "đéo", "đell": "đéo", "dek": "đéo", "đếch": "đéo",
    "vc": "vãi chưởng", "vcl": "vãi cả lúa", "vkl": "vãi cả lúa",
    "h": "giờ",
    "g": "gì",
    "z": "vậy", "zậy": "vậy",
    "thik": "thích", "thix": "thích",
    "iu": "yêu",
    "add": "đồng ý",
    "nt": "nhắn tin",
    "r": "rồi",
    "fb": "facebook",
    "face": "facebook",
    "hn": "hà nội",
    "sg": "sài gòn",
    "uk": "ừ", "uh": "ừ", "uhm": "ừ",
    "v": "vậy",
    "wa": "quá",
    "wá": "quá",
    "j": "gì",
    "vs": "với",
    "vn": "việt nam",
    "hiu": "hiểu",
    "bun": "buồn",
    "thui": "thôi",
    "nha": "nhé",
    "hem": "không",
    "bit": "biết",
    "thg": "thằng",
    "b": "bạn",
    "ch": "chưa",
    "cx": "cũng"
}

class TextPreprocessor:
    def __init__(self, vncorenlp_instance=None):
        """
        Khởi tạo bộ xử lý.
        :param vncorenlp_instance: Instance của VnCoreNLP đã được load (để tránh load lại nhiều lần gây chậm)
        """
        self.vncorenlp = vncorenlp_instance

    def normalize_unicode(self, text):
        """
        Chuẩn hóa Unicode về dạng NFC (Dựng sẵn).
        Tiếng Việt tổ hợp (NFD) sẽ gây lỗi khi so sánh chuỗi.
        """
        if not isinstance(text, str):
            return str(text)
        return unicodedata.normalize('NFC', text)

    def to_lower(self, text):
        return text.lower()

    def remove_urls(self, text):
        """Loại bỏ các đường link"""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    def standardize_punctuation(self, text):
        """
        Xử lý dấu câu lặp lại và dấu câu dính liền.
        Ví dụ: "hay quá!!!!!!" -> "hay quá !"
        """
        # Thay thế các dấu câu lặp lại bằng 1 dấu (vd: ... -> .)
        # Giữ lại chấm than và hỏi chấm để giữ cảm xúc
        text = re.sub(r'!+', ' ! ', text)
        text = re.sub(r'\?+', ' ? ', text)
        text = re.sub(r'\.+', ' . ', text)
        text = re.sub(r',+', ' , ', text)
        
        # Xóa các ký tự đặc biệt không cần thiết (giữ lại chữ, số, và emoji dạng text)
        # Lưu ý: Regex này cần cẩn thận để không xóa mất các token emoji sau này
        return text

    def process_emojis(self, text):
        return emoji.demojize(text, delimiters=(" :", ": "))

    def normalize_teencode(self, text):
        """
        Thay thế các từ teencode bằng từ chuẩn dựa trên từ điển.
        """
        words = text.split()
        # List comprehension để thay thế nhanh
        normalized_words = [TEENCODE_DICT.get(word, word) for word in words]
        return ' '.join(normalized_words)

    def remove_duplicate_characters(self, text):
        """
        Xử lý các từ kéo dài.
        Ví dụ: "nguuuuuuuu" -> "ngu", "hayyyyy" -> "hay"
        """
        # Thay thế ký tự lặp lại 3 lần trở lên bằng 1 ký tự
        # \1 là backreference tới group ký tự tìm thấy
        return re.sub(r'(.)\1{2,}', r'\1', text)

    def segment_text(self, text):
        """
        Tách từ sử dụng VnCoreNLP.
        Đầu vào: "học sinh học sinh học"
        Đầu ra: "học_sinh học sinh_học" (tùy ngữ cảnh)
        """
        if self.vncorenlp:
            try:
                # word_segment trả về list các câu, mỗi câu là list các từ
                sentences = self.vncorenlp.word_segment(text)
                # Nối lại thành chuỗi
                return ' '.join(sentences)
            except Exception as e:
                print(f"Lỗi Segment: {e}")
                return text
        return text

    def process(self, text):
        """
        Hàm wrapper chạy toàn bộ pipeline
        """
        if not isinstance(text, str):
            return ""
        
        text = self.normalize_unicode(text)
        text = self.to_lower(text)
        text = self.remove_urls(text)
        text = self.process_emojis(text) # Chuyển emoji thành text trước
        text = self.remove_duplicate_characters(text) # nguuu -> ngu
        text = self.standardize_punctuation(text)
        text = self.normalize_teencode(text) # ko -> không
        
        # Segment cuối cùng để đảm bảo các từ ghép như "đất_nước" được bắt đúng
        # sau khi đã clean sạch sẽ
        text = self.segment_text(text)
        
        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text