import json
import os
import re
from typing import Dict, List, Tuple, Optional

class SpamChecker:
    """
    Module kiểm tra spam dựa trên từ khóa.
    Hỗ trợ phân loại theo nhiều danh mục spam khác nhau.
    """
    
    def __init__(self, json_path: str = "spamkeyword.json"):
        """
        Khởi tạo SpamChecker.
        
        Args:
            json_path: Đường dẫn đến file JSON chứa từ khóa spam
        """
        self.json_path = json_path
        self.spam_keywords = self._load_keywords()
        self._build_patterns()
    
    def _load_keywords(self) -> Dict[str, List[str]]:
        """Load từ khóa spam từ file JSON"""
        if not os.path.exists(self.json_path):
            print(f"Cảnh báo: Không tìm thấy file {self.json_path}")
            return {}
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Lỗi khi đọc file spam keywords: {e}")
            return {}
    
    def _build_patterns(self):
        """Xây dựng regex patterns cho từng category"""
        self.patterns = {}
        for category, keywords in self.spam_keywords.items():
            if keywords:
                # Escape special regex characters và tạo pattern
                escaped_keywords = [re.escape(kw.lower()) for kw in keywords]
                # Sử dụng word boundary cho các từ có ký tự chữ/số
                pattern = '|'.join(escaped_keywords)
                self.patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def reload_keywords(self):
        """Reload từ khóa từ file (khi file được cập nhật)"""
        self.spam_keywords = self._load_keywords()
        self._build_patterns()
    
    def check_text(self, text: str) -> Dict[str, List[str]]:
        """
        Kiểm tra văn bản và trả về các từ khóa spam tìm thấy theo category.
        
        Args:
            text: Văn bản cần kiểm tra
            
        Returns:
            Dict với key là category, value là list các từ khóa spam tìm thấy
        """
        if not text or not isinstance(text, str):
            return {}
        
        text_lower = text.lower()
        found_spam = {}
        
        for category, keywords in self.spam_keywords.items():
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                found_spam[category] = matched_keywords
        
        return found_spam
    
    def is_spam(self, text: str, threshold: int = 1) -> bool:
        """
        Kiểm tra nhanh xem văn bản có phải spam không.
        
        Args:
            text: Văn bản cần kiểm tra
            threshold: Số từ khóa spam tối thiểu để coi là spam
            
        Returns:
            True nếu là spam, False nếu không
        """
        found = self.check_text(text)
        total_matches = sum(len(keywords) for keywords in found.values())
        return total_matches >= threshold
    
    def get_spam_score(self, text: str) -> Tuple[float, Dict[str, int]]:
        """
        Tính điểm spam cho văn bản.
        
        Args:
            text: Văn bản cần kiểm tra
            
        Returns:
            Tuple gồm (điểm spam tổng, dict số lượng spam theo category)
        """
        found = self.check_text(text)
        
        category_counts = {cat: len(keywords) for cat, keywords in found.items()}
        total_score = sum(category_counts.values())
        
        return total_score, category_counts
    
    def get_spam_categories(self, text: str) -> List[str]:
        """
        Lấy danh sách các category spam trong văn bản.
        
        Args:
            text: Văn bản cần kiểm tra
            
        Returns:
            List các category spam tìm thấy
        """
        found = self.check_text(text)
        return list(found.keys())
    
    def get_detailed_report(self, text: str) -> Dict:
        """
        Tạo báo cáo chi tiết về spam trong văn bản.
        
        Args:
            text: Văn bản cần kiểm tra
            
        Returns:
            Dict chứa thông tin chi tiết về spam
        """
        found = self.check_text(text)
        score, category_counts = self.get_spam_score(text)
        
        return {
            "is_spam": score > 0,
            "spam_score": score,
            "categories": list(found.keys()),
            "category_counts": category_counts,
            "matched_keywords": found,
            "text_length": len(text),
            "spam_density": score / max(len(text.split()), 1)  # spam per word
        }
    
    def filter_spam_comments(self, comments: List[str], threshold: int = 1) -> Tuple[List[str], List[str]]:
        """
        Lọc danh sách comments thành spam và non-spam.
        
        Args:
            comments: Danh sách các comment
            threshold: Ngưỡng spam
            
        Returns:
            Tuple gồm (list non-spam, list spam)
        """
        non_spam = []
        spam = []
        
        for comment in comments:
            if self.is_spam(comment, threshold):
                spam.append(comment)
            else:
                non_spam.append(comment)
        
        return non_spam, spam
    
    def get_all_keywords(self) -> Dict[str, List[str]]:
        """Lấy tất cả từ khóa spam theo category"""
        return self.spam_keywords.copy()
    
    def get_categories(self) -> List[str]:
        """Lấy danh sách các category"""
        return list(self.spam_keywords.keys())
    
    def count_keywords(self) -> Dict[str, int]:
        """Đếm số từ khóa trong mỗi category"""
        return {cat: len(keywords) for cat, keywords in self.spam_keywords.items()}


# Ví dụ cách dùng
if __name__ == "__main__":
    checker = SpamChecker()
    
    # Test với một số comment mẫu
    test_comments = [
        "Video hay quá, cảm ơn bạn!",
        "Mua ngay giá rẻ khuyến mãi hôm nay!",
        "Check ib mình nhé, inbox để biết thêm chi tiết",
        "Bạn trúng thưởng 100 triệu, chuyển tiền ngay!",
        "Nội dung rất bổ ích, mình học được nhiều điều"
    ]
    
    print("=" * 60)
    print("SPAM CHECKER TEST")
    print("=" * 60)
    
    for comment in test_comments:
        report = checker.get_detailed_report(comment)
        print(f"\n📝 Comment: {comment}")
        print(f"   🚨 Is Spam: {report['is_spam']}")
        print(f"   📊 Score: {report['spam_score']}")
        if report['matched_keywords']:
            print(f"   🏷️ Categories: {', '.join(report['categories'])}")
            for cat, keywords in report['matched_keywords'].items():
                print(f"      - {cat}: {keywords}")
    
    print("\n" + "=" * 60)
    print("FILTER TEST")
    print("=" * 60)
    
    non_spam, spam = checker.filter_spam_comments(test_comments)
    print(f"\n✅ Non-spam ({len(non_spam)}):")
    for c in non_spam:
        print(f"   - {c[:50]}...")
    print(f"\n🚫 Spam ({len(spam)}):")
    for c in spam:
        print(f"   - {c[:50]}...")
