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
