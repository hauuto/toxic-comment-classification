"""Quick validation script for the Threads scraper fixes."""
import sys
import re
import os

# Force UTF-8 output to avoid cp1252 encoding errors on Windows
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Test 1: Import all modified modules
try:
    from nlp_pipeline.decoder import Decoder, _TRAILING_TRANSLATE, _GIA_DINH_SPAM, _GIA_DINH_NEAR_EMOJI
    from nlp_pipeline.filter import Filter
    from nlp_pipeline.config import ADMIN_JUNK_PATTERNS
    from crawler import _clean_threads_text, _THREADS_UI_SKIP_PATTERNS, _THREADS_TRAILING_TRANSLATE, _THREADS_GIA_DINH_SPAM
    print("[PASS] All imports successful")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: _clean_threads_text filters
tests_skip = [
    "hãy đăng nhập để xem thêm thread trả lời nhé.",
    "đăng nhập hoặc đăng ký threads",
    "tiếp tục bằng instagram",
    "chính sách quyền riêng tư",
    "heavy_metal_philosophy",
    "staffordshirepremiertravel",
    "replying to<mention>",
    "đang trả lời<mention>",
]
for t in tests_skip:
    result = _clean_threads_text(t)
    assert result is None, f"[FAIL] Should skip: '{t}' but got: '{result}'"
print(f"[PASS] _clean_threads_text correctly skips {len(tests_skip)} UI noise strings")

# Test 3: Trailing "translate" removal
tests_translate = [
    ("đừng quen em :))translate", "đừng quen em :))"),
    ("truyện thôi bàtranslate", "truyện thôi bà"),
    ("hello worldTranslate", "hello world"),
    ("bình thường Dịch", "bình thường"),
]
for raw, expected in tests_translate:
    result = _clean_threads_text(raw)
    assert result == expected, f"[FAIL] translate strip: '{raw}' -> '{result}', expected '{expected}'"
print("[PASS] _clean_threads_text correctly strips trailing Translate/Dich")

# Test 4: "gia đình" spam removal
tests_gia_dinh = [
    ("best player gia đình:cười_toát_mồ_hôi:gia đìnhgia đình:", "best player :cười_toát_mồ_hôi:"),
    ("nightmare gia đình:mặt_điên:gia đình", "nightmare :mặt_điên:"),
    ("gia đìnhgia đình:", ""),  # pure noise -> None
]
pass_count = 0
for raw, expected in tests_gia_dinh:
    result = _clean_threads_text(raw)
    if expected == "":
        assert result is None, f"[FAIL] gia dinh: '{raw}' -> '{result}', expected None"
    else:
        # Normalize spaces for comparison
        result_clean = re.sub(r'\s+', ' ', result).strip() if result else result
        expected_clean = re.sub(r'\s+', ' ', expected).strip()
        assert result_clean == expected_clean, f"[FAIL] gia dinh: '{raw}' -> '{result_clean}', expected '{expected_clean}'"
    pass_count += 1
print(f"[PASS] _clean_threads_text correctly removes 'gia dinh' spam ({pass_count} cases)")

# Test 5: Filter.is_threads_ui_noise
f = Filter()
assert f.is_threads_ui_noise("heavy_metal_philosophy") == True, "[FAIL] username detection"
assert f.is_threads_ui_noise("đăng nhập hoặc đăng ký threads") == True, "[FAIL] login prompt"
assert f.is_threads_ui_noise("Tôi nghĩ bạn nói đúng") == False, "[FAIL] normal comment falsely flagged"
print("[PASS] Filter.is_threads_ui_noise works correctly")

# Test 6: Decoder.clean_threads_artifacts
decoder = Decoder.__new__(Decoder)  # Skip __init__ (needs emoji file)
assert decoder.clean_threads_artifacts("hello worldtranslate") == "hello world", "[FAIL] decoder translate strip"
assert decoder.clean_threads_artifacts("text gia đìnhgia đình:") == "text", "[FAIL] decoder gia dinh strip"
print("[PASS] Decoder.clean_threads_artifacts works correctly")

# Test 7: ADMIN_JUNK_PATTERNS includes Threads patterns
admin_re = re.compile("|".join(ADMIN_JUNK_PATTERNS), re.IGNORECASE | re.MULTILINE)
assert admin_re.search("hãy đăng nhập để xem thêm thread"), "[FAIL] admin_junk missing threads login"
assert admin_re.search("gia đìnhgia đình"), "[FAIL] admin_junk missing gia dinh spam"
print(f"[PASS] ADMIN_JUNK_PATTERNS has {len(ADMIN_JUNK_PATTERNS)} patterns including Threads-specific ones")

print("\n=== ALL TESTS PASSED ===")
