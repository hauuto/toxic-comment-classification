"""
Unit tests for VietnameseNormalizer – kiểu mới (ngữ âm học) tone placement.
"""
import unittest
from pipeline.nlp_pipeline.vietnamese_typing_normalizer import VietnameseNormalizer


class TestVietnameseNormalizer(unittest.TestCase):
    def setUp(self):
        self.norm = VietnameseNormalizer()

    # ------------------------------------------------------------------ #
    #  Rule 1: Âm tiết [-tròn môi], âm chính là nguyên âm đơn
    #  → dấu thanh trên nguyên âm chính
    # ------------------------------------------------------------------ #
    def test_single_vowel_simple(self):
        """á, tã, nhà, nhãn, gánh, ngáng"""
        self.assertEqual(self.norm.normalize("á"), "á")
        self.assertEqual(self.norm.normalize("tã"), "tã")
        self.assertEqual(self.norm.normalize("nhà"), "nhà")
        self.assertEqual(self.norm.normalize("nhãn"), "nhãn")
        self.assertEqual(self.norm.normalize("gánh"), "gánh")
        self.assertEqual(self.norm.normalize("ngáng"), "ngáng")

    # ------------------------------------------------------------------ #
    #  Rule 2: Âm tiết [+tròn môi] (âm đệm o/u), âm chính nguyên âm đơn
    #  → dấu thanh trên nguyên âm chính (sau âm đệm)
    #  Ví dụ: hoà, hoè, quỳ, quà, quờ, thuỷ, nguỵ, hoàn, quét, quát,
    #          quỵt, suýt
    # ------------------------------------------------------------------ #
    def test_rounded_glide_o(self):
        """hoà, hoè, hoàn"""
        self.assertEqual(self.norm.normalize("hòa"), "hoà")
        self.assertEqual(self.norm.normalize("hóa"), "hoá")
        self.assertEqual(self.norm.normalize("hoà"), "hoà")
        self.assertEqual(self.norm.normalize("hòe"), "hoè")
        self.assertEqual(self.norm.normalize("hoè"), "hoè")
        self.assertEqual(self.norm.normalize("hoàn"), "hoàn")

    def test_rounded_glide_qu(self):
        """quỳ, quà, quờ, quét, quát, quỵt"""
        self.assertEqual(self.norm.normalize("quỳ"), "quỳ")
        self.assertEqual(self.norm.normalize("quà"), "quà")
        self.assertEqual(self.norm.normalize("quờ"), "quờ")
        self.assertEqual(self.norm.normalize("quét"), "quét")
        self.assertEqual(self.norm.normalize("quát"), "quát")
        self.assertEqual(self.norm.normalize("quỵt"), "quỵt")

    def test_rounded_glide_u(self):
        """thuỷ, nguỵ, suýt"""
        self.assertEqual(self.norm.normalize("thủy"), "thuỷ")
        self.assertEqual(self.norm.normalize("thuỷ"), "thuỷ")
        self.assertEqual(self.norm.normalize("ngụy"), "nguỵ")
        self.assertEqual(self.norm.normalize("nguỵ"), "nguỵ")
        self.assertEqual(self.norm.normalize("suýt"), "suýt")

    # ------------------------------------------------------------------ #
    #  Rule 3a: Nguyên âm đôi dạng đóng [-khép] (iê, yê, uô, ươ)
    #  có âm cuối (p,t,c,ch,m,n,ng,nh,o,u,i) → dấu lên chữ thứ 2
    #  Ví dụ: yếu, uốn, ườn, tiến, chuyến, muốn, mượn, thiện, thuộm,
    #          người, viếng, muống, cường
    # ------------------------------------------------------------------ #
    def test_closed_diphthong(self):
        self.assertEqual(self.norm.normalize("yếu"), "yếu")
        self.assertEqual(self.norm.normalize("uốn"), "uốn")
        self.assertEqual(self.norm.normalize("tiến"), "tiến")
        self.assertEqual(self.norm.normalize("chuyến"), "chuyến")
        self.assertEqual(self.norm.normalize("muốn"), "muốn")
        self.assertEqual(self.norm.normalize("mượn"), "mượn")
        self.assertEqual(self.norm.normalize("thiện"), "thiện")
        self.assertEqual(self.norm.normalize("người"), "người")
        self.assertEqual(self.norm.normalize("viếng"), "viếng")
        self.assertEqual(self.norm.normalize("muống"), "muống")
        self.assertEqual(self.norm.normalize("cường"), "cường")

    # ------------------------------------------------------------------ #
    #  Rule 3b: Nguyên âm đôi dạng mở [+khép] (ia, ya, ua, ưa)
    #  → nhất loạt dấu lên chữ thứ 1
    #  Ví dụ: nghĩa, tủa, cứa, thùa, khứa
    # ------------------------------------------------------------------ #
    def test_open_diphthong(self):
        self.assertEqual(self.norm.normalize("nghĩa"), "nghĩa")
        # Test that tone stays on first vowel of open diphthong even if misplaced
        self.assertEqual(self.norm.normalize("nghiã"), "nghĩa")  # wrong position → fix
        self.assertEqual(self.norm.normalize("tủa"), "tủa")
        self.assertEqual(self.norm.normalize("cứa"), "cứa")
        self.assertEqual(self.norm.normalize("thùa"), "thùa")
        self.assertEqual(self.norm.normalize("khứa"), "khứa")
        self.assertEqual(self.norm.normalize("mía"), "mía")
        self.assertEqual(self.norm.normalize("lụa"), "lụa")
        self.assertEqual(self.norm.normalize("mùa"), "mùa")
        self.assertEqual(self.norm.normalize("chừa"), "chừa")

    # ------------------------------------------------------------------ #
    #  Rule 4: Trường hợp đặc biệt "gi" – chữ "gi" đơn lẻ
    #  (gì, gí, gỉ, gĩ, gị) → dấu trên "i"
    # ------------------------------------------------------------------ #
    def test_gi_standalone(self):
        """gì, gí – 'gi' là toàn bộ âm tiết, dấu trên 'i'."""
        self.assertEqual(self.norm.normalize("gì"), "gì")
        self.assertEqual(self.norm.normalize("gí"), "gí")

    def test_gi_onset(self):
        """giá, giấy, giếng – 'gi' là phụ âm đầu."""
        self.assertEqual(self.norm.normalize("giá"), "giá")
        self.assertEqual(self.norm.normalize("giấy"), "giấy")
        self.assertEqual(self.norm.normalize("giếng"), "giếng")

    # ------------------------------------------------------------------ #
    #  Trường hợp "ui" – u là nguyên âm chính, i là bán âm cuối
    #  Ví dụ: túi, đuổi, cúi, ngủi, xùi
    # ------------------------------------------------------------------ #
    def test_ui_nucleus_coda(self):
        """'ui' → u là âm chính, i là bán âm cuối → dấu trên u."""
        self.assertEqual(self.norm.normalize("túi"), "túi")
        self.assertEqual(self.norm.normalize("cúi"), "cúi")
        self.assertEqual(self.norm.normalize("xùi"), "xùi")
        self.assertEqual(self.norm.normalize("đuổi"), "đuổi")

    # ------------------------------------------------------------------ #
    #  Trường hợp "oi" – o là nguyên âm chính, i là bán âm cuối
    # ------------------------------------------------------------------ #
    def test_oi_nucleus_coda(self):
        """'oi' → o là âm chính, i là bán âm cuối → dấu trên o."""
        self.assertEqual(self.norm.normalize("tói"), "tói")
        self.assertEqual(self.norm.normalize("nói"), "nói")
        self.assertEqual(self.norm.normalize("mòi"), "mòi")
        self.assertEqual(self.norm.normalize("hỏi"), "hỏi")

    # ------------------------------------------------------------------ #
    #  Trường hợp "ai", "ao", "au", "ay", "ây" ...
    # ------------------------------------------------------------------ #
    def test_common_vowel_pairs(self):
        self.assertEqual(self.norm.normalize("tài"), "tài")
        self.assertEqual(self.norm.normalize("bào"), "bào")
        self.assertEqual(self.norm.normalize("sáu"), "sáu")
        self.assertEqual(self.norm.normalize("tây"), "tây")
        self.assertEqual(self.norm.normalize("máy"), "máy")

    # ------------------------------------------------------------------ #
    #  Trường hợp "ươi", "uôi" – tam nguyên âm
    # ------------------------------------------------------------------ #
    def test_triphthongs(self):
        self.assertEqual(self.norm.normalize("người"), "người")
        self.assertEqual(self.norm.normalize("cười"), "cười")
        self.assertEqual(self.norm.normalize("buồi"), "buồi")

    # ------------------------------------------------------------------ #
    #  Edge cases: mixed text, sentences
    # ------------------------------------------------------------------ #
    def test_sentence(self):
        result = self.norm.normalize("Hoà bình và nghĩa lý")
        self.assertEqual(result, "Hoà bình và nghĩa lý")

    def test_no_tone(self):
        """No-tone text should pass through unchanged."""
        self.assertEqual(self.norm.normalize("hello world"), "hello world")
        self.assertEqual(self.norm.normalize("xin chao"), "xin chao")

    def test_empty_and_none(self):
        self.assertEqual(self.norm.normalize(""), "")
        self.assertEqual(self.norm.normalize(None), "")

    # ------------------------------------------------------------------ #
    #  Ường – standalone
    # ------------------------------------------------------------------ #
    def test_uon_standalone(self):
        self.assertEqual(self.norm.normalize("ườn"), "ườn")

    # ------------------------------------------------------------------ #
    #  thuộm
    # ------------------------------------------------------------------ #
    def test_thuom(self):
        self.assertEqual(self.norm.normalize("thuộm"), "thuộm")

    # ------------------------------------------------------------------ #
    #  Rule 5: Xử lý chữ Hoa / chữ Thường (Capitalization)
    #  → Phải bảo toàn đúng case của chữ cái mang dấu
    # ------------------------------------------------------------------ #
    def test_capitalization(self):
        """Bảo toàn in hoa toàn bộ, viết hoa chữ cái đầu hoặc xen kẽ."""
        # Viết hoa chữ cái đầu
        self.assertEqual(self.norm.normalize("Hòa"), "Hoà")
        self.assertEqual(self.norm.normalize("Thủy"), "Thuỷ")
        self.assertEqual(self.norm.normalize("Khỏe"), "Khoẻ")

        # In hoa toàn bộ
        self.assertEqual(self.norm.normalize("HÒA"), "HOÀ")
        self.assertEqual(self.norm.normalize("THỦY"), "THUỶ")
        self.assertEqual(self.norm.normalize("QUÝ"), "QUÝ")
        self.assertEqual(self.norm.normalize("HOÀN"), "HOÀN")

        # Mixed case (nếu dữ liệu bị dơ) — tone chuyển sang nguyên âm chính,
        # chữ cái cũ mất dấu thanh nhưng giữ nguyên case gốc.
        self.assertEqual(self.norm.normalize("hÒa"), "hOà")
        self.assertEqual(self.norm.normalize("tHủy"), "tHuỷ")

    # ------------------------------------------------------------------ #
    #  Rule 6: Chuyển đổi mạnh từ Chuẩn Cũ (Old Style) -> Chuẩn Mới
    #  → Các từ mở oa, oe, uy phải đẩy dấu về âm thứ 2
    # ------------------------------------------------------------------ #
    def test_old_style_to_new_style(self):
        """Ép chuẩn từ kiểu gõ cũ sang kiểu mới."""
        self.assertEqual(self.norm.normalize("hòa"), "hoà")
        self.assertEqual(self.norm.normalize("thủy"), "thuỷ")
        self.assertEqual(self.norm.normalize("khỏe"), "khoẻ")
        self.assertEqual(self.norm.normalize("tùy"), "tuỳ")
        self.assertEqual(self.norm.normalize("hủy"), "huỷ")
        self.assertEqual(self.norm.normalize("chóe"), "choé")
        self.assertEqual(self.norm.normalize("xóa"), "xoá")
        self.assertEqual(self.norm.normalize("lõa"), "loã")

    # ------------------------------------------------------------------ #
    #  Rule 7: Sửa lỗi gõ sai vị trí dấu (Typos / Misplacements)
    #  → User gõ sai dấu ở từ đóng (phụ âm cuối) hoặc sai do bộ gõ
    # ------------------------------------------------------------------ #
    def test_fix_typos_closed_syllables(self):
        """Sửa lỗi gõ sai vị trí dấu ở các từ có phụ âm cuối."""
        self.assertEqual(self.norm.normalize("hòan"), "hoàn")
        self.assertEqual(self.norm.normalize("tóan"), "toán")
        self.assertEqual(self.norm.normalize("lụât"), "luật")
        self.assertEqual(self.norm.normalize("thụân"), "thuận")
        self.assertEqual(self.norm.normalize("giường"), "giường")  # already correct
        self.assertEqual(self.norm.normalize("chuỵện"), "chuyện")

    def test_fix_typo_qu(self):
        """Sửa lỗi gõ sai chữ Qúy (do máy tính tự sửa hoặc user gõ sai)."""
        self.assertEqual(self.norm.normalize("qúy"), "quý")
        self.assertEqual(self.norm.normalize("qủa"), "quả")
        self.assertEqual(self.norm.normalize("qùa"), "quà")
        self.assertEqual(self.norm.normalize("Qúy"), "Quý")

    # ------------------------------------------------------------------ #
    #  Rule 8: Ưu tiên tuyệt đối các chữ có mũ/râu (â, ă, ê, ô, ơ, ư)
    #  → Bất kể đứng đâu, có mũ/râu là phải gánh dấu thanh
    # ------------------------------------------------------------------ #
    def test_priority_vowels_with_hat_or_breve(self):
        """Dấu thanh phải bám vào â, ă, ê, ô, ơ, ư."""
        self.assertEqual(self.norm.normalize("thuở"), "thuở")
        self.assertEqual(self.norm.normalize("hoẵng"), "hoẵng")
        self.assertEqual(self.norm.normalize("quẫy"), "quẫy")
        self.assertEqual(self.norm.normalize("hoặc"), "hoặc")
        self.assertEqual(self.norm.normalize("huệ"), "huệ")
        self.assertEqual(self.norm.normalize("quỳnh"), "quỳnh")  # Nhóm y

    # ------------------------------------------------------------------ #
    #  Rule 9: Các cụm nguyên âm 3 chữ phức tạp (uyê, uya, oay, oai)
    # ------------------------------------------------------------------ #
    def test_complex_vowel_groups(self):
        self.assertEqual(self.norm.normalize("tuyết"), "tuyết")
        self.assertEqual(self.norm.normalize("khuya"), "khuya")
        self.assertEqual(self.norm.normalize("tuýp"), "tuýp")  # Vay mượn (tuýp kem)
        self.assertEqual(self.norm.normalize("loay hoay"), "loay hoay")
        self.assertEqual(self.norm.normalize("ngoái"), "ngoái")
        self.assertEqual(self.norm.normalize("khuỷu"), "khuỷu")  # u-y-u (mở)

    # ------------------------------------------------------------------ #
    #  Rule 10: Ngoại lệ phụ âm "gi" kết hợp với các nguyên âm phức tạp
    # ------------------------------------------------------------------ #
    def test_gi_complex_combinations(self):
        self.assertEqual(self.norm.normalize("giặc"), "giặc")
        self.assertEqual(self.norm.normalize("giường"), "giường")
        self.assertEqual(self.norm.normalize("giục"), "giục")
        self.assertEqual(self.norm.normalize("giữa"), "giữa")

if __name__ == "__main__":
    unittest.main()
