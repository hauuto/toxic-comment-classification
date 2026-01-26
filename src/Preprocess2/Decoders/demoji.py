import json
import re
import os
from typing import Optional


class EmojiDecoder:    
    def __init__(self, emoji_dict_path: Optional[str] = None):
        if emoji_dict_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            emoji_dict_path = os.path.join(current_dir, "..", "Json", "emoji_vi.json")
        
        self.emoji_dict = self._load_emoji_dict(os.path.abspath(emoji_dict_path))
        sorted_emojis = sorted(self.emoji_dict.keys(), key=len, reverse=True)
        escaped_emojis = [re.escape(emoji) for emoji in sorted_emojis]
        self.emoji_pattern = re.compile('|'.join(escaped_emojis))
    
    def _load_emoji_dict(self, path: str) -> dict:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file emoji dictionary: {path}")
        except json.JSONDecodeError:
            raise ValueError(f"File {path} không phải JSON hợp lệ")
    
    def decode(self, text: str, format_style: str = "colon") -> str:
        def replace_emoji(match):
            emoji = match.group(0)
            vi_name = self.emoji_dict.get(emoji, emoji)
            
            if format_style == "colon":
                return f":{vi_name}:"
            elif format_style == "bracket":
                return f"[{vi_name}]"
            elif format_style == "plain":
                return f" {vi_name} "
            else:
                return f":{vi_name}:"
        
        return self.emoji_pattern.sub(replace_emoji, text)
    
    def remove_emoji(self, text: str) -> str:
        return self.emoji_pattern.sub('', text)
    
    def extract_emojis(self, text: str) -> list:
        return self.emoji_pattern.findall(text)
    
    def get_emoji_info(self, emoji: str) -> Optional[str]:
        return self.emoji_dict.get(emoji)
    
    def count_emojis(self, text: str) -> int:
        return len(self.emoji_pattern.findall(text))
