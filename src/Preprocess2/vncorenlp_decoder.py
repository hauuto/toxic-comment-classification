import os
import sys
from typing import List, Dict, Union
import py_vncorenlp
import re

class VnCoreNLPDecoder:
    def __init__(self, model_dir: str, java_home: str = None):
        """
        Khởi tạo VnCoreNLP.
        :param model_dir: Đường dẫn đến thư mục chứa VnCoreNLP (file .jar và models).
        :param java_home: Đường dẫn đến thư mục JDK (nếu chưa set trong biến môi trường).
        """
        # 1. Setup JAVA_HOME nếu được cung cấp
        if java_home:
            os.environ["JAVA_HOME"] = java_home
            # Thêm vào path để đảm bảo hệ thống tìm thấy java.exe
            if sys.platform == "win32":
                os.environ["PATH"] = f"{java_home}\\bin;" + os.environ["PATH"]

        # 2. Kiểm tra thư mục model
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục VnCoreNLP tại: {model_dir}")
        try:
            self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg", "pos"], 
                save_dir=model_dir,
                max_heap_size="-Xmx2g"
            )
        except Exception as e:
            raise RuntimeError(f"Lỗi khởi tạo VnCoreNLP: {e}. Hãy kiểm tra JAVA_HOME.")

    # def segment_text(self, text: str) -> List[str]:
    #     if not text: return []
    #     return self.rdrsegmenter.word_segment(text)
    
    def segment_text(self, text: str) -> str:
        if not text:
            return ""

        # 1. Trích xuất icon và thay thế bằng Placeholder bằng regex callback
        icons = []
        def replace_func(match):
            icons.append(match.group())
            return f" ICONPLACEHOLDER{len(icons)-1} "

        # Regex tìm các ký tự đặc biệt/emoji
        pattern = re.compile(r'[^\w\s,.<>?/;:"\'\[\]{}\\\|`~!@#$%^&*()\-=_+]')
        tmp_text = pattern.sub(replace_func, text)

        # 2. Gọi VnCoreNLP (Bọc trong try-except để bắt lỗi bộ nhớ nếu có)
        try:
            segmented_sentences = self.rdrsegmenter.word_segment(tmp_text)
        except Exception as e:
            print(f"Lỗi khi segment: {e}")
            return text # Trả về text gốc nếu lỗi

        full_segmented = " ".join(segmented_sentences)

        # 3. Chèn lại icon gốc
        final_str = full_segmented
        for i, icon in enumerate(icons):
            final_str = final_str.replace(f"ICONPLACEHOLDER{i}", icon)

        # 4. Dọn dẹp dấu câu và khoảng trắng
        # Sửa dấu câu sát từ đứng trước
        final_str = re.sub(r'\s+([,.:?!])', r'\1', final_str)
        # Xử lý khoảng trắng thừa
        final_str = re.sub(r'\s+', ' ', final_str).strip()

        return final_str

    def annotate_text(self, text: str) -> Dict:
        if not text: return {}
        return self.rdrsegmenter.annotate_text(text)
    