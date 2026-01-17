import json
import os

class TeencodeConverter:
    def __init__(self, json_path="teencode.json"):
        """
        Khởi tạo converter.
        :param json_path: Đường dẫn đến file teencode.json
        """
        self.teencode_dict = self._load_and_flip_dictionary(json_path)

    def _load_and_flip_dictionary(self, json_path):
        """
        Đọc file JSON (dạng Standard -> List[Variants]) 
        và đảo ngược thành (Variant -> Standard) để tiện tra cứu.
        """
        if not os.path.exists(json_path):
            print(f"Cảnh báo: Không tìm thấy file {json_path}.")
            return {}

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            flipped_dict = {}
            for standard_word, variants in raw_data.items():
                for variant in variants:
                    # Chuyển về chữ thường để đảm bảo đồng nhất
                    flipped_dict[variant.lower()] = standard_word.lower()
            
            return flipped_dict
        except Exception as e:
            print(f"Lỗi khi đọc file teencode: {e}")
            return {}

    def replace(self, text):
        """
        Thay thế các từ teencode trong văn bản bằng từ chuẩn.
        """
        if not text:
            return ""
        
        words = text.split()
        # Tra cứu trong dictionary, nếu không có thì giữ nguyên từ gốc
        normalized_words = [self.teencode_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)

# Ví dụ cách dùng (nếu chạy file này trực tiếp)
if __name__ == "__main__":
    converter = TeencodeConverter()
    sample = "hnay t ko đi học dc"
    print(converter.replace(sample))
    # Output: hôm nay tôi không đi học được (giả sử json có 'hnay')