"""Unit tests cho VietnameseNormalizer – kiểu mới (ngữ âm học)."""

from src.Preprocess2.Normalizers.vietnamese_typing_normalizer import (
    VietnameseNormalizer,
)

normalizer = VietnameseNormalizer()


def n(text: str) -> str:
    """Shortcut."""
    return normalizer.normalize(text)


# ================================================================== #
#  1. Nguyên âm đơn – dấu đặt ngay trên nguyên âm chính
# ================================================================== #
class TestSingleVowel:
    def test_a_acute(self):
        assert n("á") == "á"

    def test_ta_tilde(self):
        assert n("tã") == "tã"

    def test_nha_grave(self):
        assert n("nhà") == "nhà"

    def test_nhan_tilde(self):
        assert n("nhãn") == "nhãn"

    def test_ganh_acute(self):
        assert n("gánh") == "gánh"

    def test_nggang_acute(self):
        assert n("ngáng") == "ngáng"


# ================================================================== #
#  2. Âm đệm tròn môi + nguyên âm đơn → dấu lên nguyên âm chính
# ================================================================== #
class TestRoundedGlide:
    """hoà, hoè, quỳ, quà, quờ, thuỷ, nguỵ, hoàn, quét, quát, quỵt, suýt"""

    def test_hoa_grave(self):
        # hoà – dấu trên à (không phải trên o)
        assert n("hoà") == "hoà"

    def test_hoe_grave(self):
        assert n("hoè") == "hoè"

    def test_quy_grave(self):
        # quỳ – dấu trên ỳ
        assert n("quỳ") == "quỳ"

    def test_qua_grave(self):
        assert n("quà") == "quà"

    def test_quo_huyen(self):
        assert n("quờ") == "quờ"

    def test_thuy_hoi(self):
        # thuỷ – dấu trên ỷ (kiểu mới)
        assert n("thuỷ") == "thuỷ"

    def test_nguy_nang(self):
        assert n("nguỵ") == "nguỵ"

    def test_hoan_grave(self):
        assert n("hoàn") == "hoàn"

    def test_quet_sac(self):
        assert n("quét") == "quét"

    def test_quat_sac(self):
        assert n("quát") == "quát"

    def test_quyt_nang(self):
        assert n("quỵt") == "quỵt"

    def test_suyt_sac(self):
        assert n("suýt") == "suýt"


# ================================================================== #
#  3a. Nguyên âm đôi dạng đóng (iê, yê, uô, ươ + coda)
#      → dấu lên chữ cái thứ 2
# ================================================================== #
class TestClosedDiphthong:
    """yếu, uốn, ườn, tiến, chuyến, muốn, mượn, thiện, thuộm, người,
    viếng, muống, cường"""

    def test_yeu_sac(self):
        assert n("yếu") == "yếu"

    def test_uon_sac(self):
        assert n("uốn") == "uốn"

    def test_uon_huyen(self):
        assert n("ườn") == "ườn"

    def test_tien_sac(self):
        assert n("tiến") == "tiến"

    def test_chuyen_sac(self):
        assert n("chuyến") == "chuyến"

    def test_muon_sac(self):
        assert n("muốn") == "muốn"

    def test_muon_nang(self):
        assert n("mượn") == "mượn"

    def test_thien_nang(self):
        assert n("thiện") == "thiện"

    def test_thuom_nang(self):
        assert n("thuộm") == "thuộm"

    def test_nguoi_huyen(self):
        assert n("người") == "người"

    def test_vieng_sac(self):
        assert n("viếng") == "viếng"

    def test_muong_sac(self):
        assert n("muống") == "muống"

    def test_cuong_huyen(self):
        assert n("cường") == "cường"


# ================================================================== #
#  3b. Nguyên âm đôi dạng mở (ia, ya, ua, ưa – không coda)
#      → dấu lên chữ cái thứ 1
# ================================================================== #
class TestOpenDiphthong:
    """nghĩa, tủa, cứa, thùa, khứa"""

    def test_nghia_hoi(self):
        assert n("nghĩa") == "nghĩa"

    def test_tua_hoi(self):
        assert n("tủa") == "tủa"

    def test_cua_sac(self):
        assert n("cứa") == "cứa"

    def test_thua_huyen(self):
        assert n("thùa") == "thùa"

    def test_khua_sac(self):
        assert n("khứa") == "khứa"


# ================================================================== #
#  4. Trường hợp đặc biệt: "gi" và "qu"
# ================================================================== #
class TestGiQu:
    def test_gi_alone_grave(self):
        # "gì" – chữ "gi" đứng một mình → dấu trên i
        assert n("gì") == "gì"

    def test_gi_alone_sac(self):
        assert n("gí") == "gí"

    def test_gia_hoi(self):
        # "giả" – gi + a → dấu trên a
        assert n("giả") == "giả"

    def test_gian_sac(self):
        assert n("giấn") == "giấn"

    def test_gieng_sac(self):
        # giếng – gi + ê + ng → dấu trên ê
        assert n("giếng") == "giếng"

    def test_qua_nang(self):
        assert n("quạ") == "quạ"

    def test_quyen_huyen(self):
        assert n("quyền") == "quyền"


# ================================================================== #
#  5. Đầu vào đặc biệt
# ================================================================== #
class TestEdgeCases:
    def test_empty_string(self):
        assert n("") == ""

    def test_non_string(self):
        assert n(None) == ""
        assert n(123) == ""

    def test_no_tone(self):
        assert n("abc") == "abc"
        assert n("hoa") == "hoa"

    def test_uppercase_preserved(self):
        result = n("HOÀ")
        assert result == "HOÀ"

    def test_mixed_sentence(self):
        result = n("Anh ấy hoà nhã")
        # Mỗi từ được chuẩn hoá riêng
        assert "hoà" in result or "HOÀ" in result.upper()

    def test_multiple_words(self):
        result = n("thuỷ quỳ hoà")
        assert "thuỷ" in result
        assert "quỳ" in result
        assert "hoà" in result


# ================================================================== #
#  6. Kiểm tra chuyển đổi dấu sai vị trí → đúng vị trí kiểu mới
# ================================================================== #
class TestOldToNew:
    """Chuyển dấu từ kiểu cũ (sai) sang kiểu mới (đúng)."""

    def test_hoa_old_to_new(self):
        # "hóa" (kiểu cũ, dấu trên o) → "hoá" (kiểu mới, dấu trên a)
        assert n("hóa") == "hoá"

    def test_thuy_old_to_new(self):
        # "thủy" (kiểu cũ, dấu trên u) → "thuỷ" (kiểu mới, dấu trên y)
        assert n("thủy") == "thuỷ"

    def test_nguy_old_to_new(self):
        assert n("ngụy") == "nguỵ"

    def test_quyt_old_to_new(self):
        assert n("quỵt") == "quỵt"

    def test_chua_old_to_new(self):
        # "chúa" (kiểu cũ) → "chúa" (kiểu mới, dấu trên u – vì ua mở)
        assert n("chúa") == "chúa"

    def test_nghia_old_to_new(self):
        # "nghĩa" đã đúng kiểu mới
        assert n("nghĩa") == "nghĩa"

    def test_duong_old_to_new(self):
        # "đường" (kiểu cũ dấu trên ư) → "đường" (kiểu mới dấu trên ơ)
        assert n("đường") == "đường"

    def test_luong_old_to_new(self):
        assert n("lương") == "lương"

    def test_suyt_old_to_new(self):
        # "suýt" – dấu trên y (kiểu mới)
        assert n("suýt") == "suýt"
