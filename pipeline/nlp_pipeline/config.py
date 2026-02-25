"""
Configuration and constants for the Vietnamese comment preprocessing pipeline.
"""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPPINGS_DIR = os.path.join(BASE_DIR, "mappings")
DATA_DIR = os.path.join(BASE_DIR, "data")

ABBREVIATIONS_PATH = os.path.join(MAPPINGS_DIR, "abbreviations.json")
EMOJI_MAPPING_PATH = os.path.join(MAPPINGS_DIR, "emoji_vi.json")
PROFANITY_LIST_PATH = os.path.join(MAPPINGS_DIR, "profanity_list.json")

# --- Filter thresholds ---
MIN_CHAR_LENGTH = 20          # Minimum characters to keep a comment
MIN_VIETNAMESE_RATIO = 0.3    # Minimum ratio of Vietnamese chars in text
MAX_SPAM_REPEAT = 6          # Max consecutive identical chars before flagging as spam

# --- Normalization ---
MAX_REPEAT_CHARS = 3          # Max allowed consecutive repeated chars in words
MAX_REPEAT_PUNCTUATION = 3    # Max allowed consecutive repeated punctuation (!, ?, .)
MAX_REPEAT_ICON_CHARS = 3     # Max allowed repeated chars in icons (:))) -> :)))

# --- Vietnamese character detection ---
VIETNAMESE_CHARS = set(
    "àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
    "ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự"
    "ỳýỷỹỵđ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆ"
    "ÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰ"
    "ỲÝỶỸỴĐ"
)

# --- Via signature patterns to remove ---
VIA_SIGNATURES = [
    r"via\s+theNEXTvoz\s+for\s+iPhone",
    r"Gửi (?:bằng|từ)\s+.*?(?:vozFApp|VOZVNApp|IrisOS)",
    r"Gửi từ\s+\w[\w\s]*?bằng\s+(?:vozFApp|VOZVNApp)",
]

# --- VOZ admin / metadata junk patterns (from 2-Preprocessing.ipynb) ---
# Comments matching ANY of these are dropped entirely (mod actions,
# ban notices, system messages, "sent from" footers – not user content).
_INFRACTION_KEYWORDS = [
    r"kích động", r"gây war", r"war", r"gây gổ", r"chửi", r"xúc phạm", r"thô tục",
    r"vô văn hóa", r"thiếu văn hóa", r"cãi nhau", r"thái độ", r"công kích",
    r"phân biệt", r"vùng miền", r"pbvm", r"kỳ thị", r"phản động", r"tổ lái", r"lái",
    r"chính trị", r"tôn giáo", r"sex", r"18\+", r"đồi trụy", r"nhạy cảm",
    r"bàn luận", r"lạc đề", r"spam", r"quảng cáo", r"seeding", r"nâng bi",
    r"dìm hàng", r"báo cũ", r"nguồn cấm", r"f33", r"f17", r"f\d+",
    r"thread", r"thớt", r"post", r"bài viết", r"comment", r"đào mộ", r"up bài",
    r"tiêu đề", r"tít", r"title", r"caps", r"viết hoa", r"không dấu",
    r"giá", r"sđt", r"điện thoại", r"địa chỉ", r"liên hệ", r"thông tin",
    r"lập lờ", r"gom", r"chung",
    r"vi phạm", r"rule", r"nội quy", r"quy định", r"k phù hợp", r"không phù hợp", r"ban",
    r"banned", r"xử lý", r"nhắc nhở", r"warning", r"clone",
]
_INFRACTION_REGEX = "|".join(_INFRACTION_KEYWORDS)

ADMIN_JUNK_PATTERNS = [
    r"(?:URL\s+)?nick bị (?:xử lý|ban|khóa|ra đảo):",
    r"URL thread/post bị (?:xử lý|ban|khóa|xóa):",
    r"Nick bị band (?:)",
    r"Mod\s+(?:xóa|ban|xử lý|nhắc nhở|warn|gộp|edit|chuyển)\s*:",
    rf"Lý do:.*(?:{_INFRACTION_REGEX})",
    r"Lý do bị band",
    r"Thời hạn:.*(?:vĩnh viễn|đến cuối năm|forever|\d+|[\d/\.-]+)",
    r"Thắc mắc:.*(?:tại sao|ban|nick|xóa|mod|admin|lý do)",
    r"voz không khuyến khích",
    r"chức năng report",
    r"vui lòng đọc kỹ nội quy",
    r"góp ý về việc",
    r"kiện cáo",
    r"https://voz.vn",
    r"^\s*@[a-z0-9]+\s*$",
    r"\s*via\s+thenextvoz[\s\S]*",
    r"(?:sent from|gửi từ).*(?:iphone|ipad|samsung|android|bphone|xiaomi|redmi|vsmart|pixel|blackberry|nokia|sony)[\s\S]*",
    r"(?:sent from|gửi từ).+using\s+(?:vozFApp|tapatalk|nextvoz)[\s\S]*",
    r"sent from my phone[\s\S]*",
    r"gửi từ điện thoại[\s\S]*",
    r"More options.*",
    # --- Threads-specific noise patterns ---
    r"^hãy đăng nhập.*thread.*$",
    r"^đăng nhập hoặc đăng ký.*$",
    r"^tiếp tục (?:bằng|với) instagram$",
    r"^continue with instagram$",
    r"^chính sách quyền riêng tư.*$",
    r"^privacy policy.*$",
    r"^(?:đang trả lời|replying to)\s*<?.*$",
    # "gia đình" repeated anomaly (emoji alt text leak from Threads)
    r"(?:gia đình\s*){2,}",
]
