from typing import List, Dict, Optional
import re
import json
import os

class TeencodeConverter:
    def __init__(self, json_path: str):
        self.teencode_dict = self._load_dict(json_path)

    def _load_dict(self, json_path: str) -> Dict[str, str]:
        if not os.path.exists(json_path):
            return {}
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Flatten dict: key là teencode (viết thường), value là từ chuẩn
            return {var.lower(): std for std, variants in data.items() for var in variants}
        except Exception:
            return {}

    def _match_case(self, original_text: str, replacement_text: str) -> str:
        """
        Hàm này giúp giữ nguyên kiểu viết hoa của từ gốc.
        Ví dụ: Tui -> Tôi, TUI -> TÔI, tui -> tôi
        """
        if original_text.isupper():  # Nếu gốc là TUI -> TÔI
            return replacement_text.upper()
        elif original_text.istitle(): # Nếu gốc là Tui -> Tôi
            return replacement_text.capitalize()
        else:
            return replacement_text

    def replace(self, text: str) -> str:
        if not text: return ""
        
        tokens = re.findall(r"\w+|[^\w\s]+", text, re.UNICODE)
        
        result = []
        for t in tokens:
            if t.isalnum(): 
                replacement = self.teencode_dict.get(t.lower())
                if replacement:
                    t = self._match_case(t, replacement)
            result.append(t)
            
        raw_text = ' '.join(result)
        
    
        final_text = re.sub(r'\s+([,.?!])', r'\1', raw_text)
        
        return final_text