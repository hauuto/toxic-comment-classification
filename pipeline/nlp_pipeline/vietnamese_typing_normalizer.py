"""
Vietnamese Typing Normalizer – Chuẩn hóa vị trí dấu thanh điệu tiếng Việt.
Ported from src/Preprocess2/Normalizers/vietnamese_typing_normalizer.py
"""
import unicodedata


class VietnameseNormalizer:
    """Chuẩn hóa vị trí dấu thanh điệu tiếng Việt theo kiểu mới (ngữ âm học)."""

    # Combining marks for the 5 Vietnamese tones (NFD)
    TONES = frozenset("\u0300\u0301\u0309\u0303\u0323")

    # Base vowel letters (lowercase, NFD – i.e. without combining marks)
    VOWELS = frozenset("aeouiy")

    # Nguyên âm đôi dạng đóng (closed diphthongs) – khi có âm cuối
    CLOSED_DIPHTHONGS = {"iê", "yê", "uô", "ươ"}

    # Nguyên âm đôi dạng mở (open diphthongs) – không có âm cuối
    OPEN_DIPHTHONGS = {"ia", "ya", "ua", "ưa"}

    # Các phụ âm cuối hợp lệ trong âm tiết tiếng Việt
    CODA_CONSONANTS = frozenset("ptcmnhkg")

    # Combining marks that modify a vowel shape (NOT tones)
    # Circumflex \u0302, breve \u0306, horn \u031B
    VOWEL_MODIFIERS = frozenset("\u0302\u0306\u031B")

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _is_vowel(self, ch: str) -> bool:
        return ch.lower() in self.VOWELS

    def _is_tone(self, ch: str) -> bool:
        return ch in self.TONES

    @staticmethod
    def _is_combining_mark(ch: str) -> bool:
        """Bất kỳ combining mark nào (Mn category): tone, circumflex, breve, horn, …"""
        return len(ch) == 1 and unicodedata.category(ch) == "Mn"

    def _nfc_vowel_at(self, chars: list[str], idx: int, n: int) -> str:
        """Trả về nguyên âm NFC (lowercase) tại vị trí *idx* trong mảng NFD.

        Ghép base char + vowel modifiers (circumflex, breve, horn), bỏ qua tones,
        rồi NFC hóa. Ví dụ: e + \\u0302 → ê, o + \\u031B → ơ, u + \\u031B → ư.
        """
        result = chars[idx].lower()
        j = idx + 1
        while j < n:
            ch = chars[j]
            if ch == "":
                j += 1
                continue
            if not self._is_combining_mark(ch):
                break
            if ch in self.VOWEL_MODIFIERS:
                result += ch
            j += 1
        return unicodedata.normalize("NFC", result)

    def _vowel_string(self, chars: list[str], indices: list[int], n: int) -> str:
        """Trả về chuỗi nguyên âm NFC (lowercase) cho dãy vowel indices."""
        return "".join(self._nfc_vowel_at(chars, i, n) for i in indices)

    def _has_coda(self, chars: list[str], after: int, n: int) -> bool:
        """Kiểm tra sau vị trí *after* có phụ âm cuối (coda) hay không.

        Coda bao gồm: p, t, c, ch, m, n, ng, nh  và các bán nguyên âm cuối
        o, u, i (chúng đóng vai trò âm cuối, không phải âm chính).
        """
        j = after
        # Bỏ qua combining marks còn sót (tone, circumflex, breve, horn…)
        while j < n and self._is_combining_mark(chars[j]):
            j += 1
        if j >= n:
            return False
        ch = chars[j].lower()
        # Phụ âm cuối hoặc bán nguyên âm cuối (o, u, i khi đứng sau nguyên âm đôi)
        return ch in self.CODA_CONSONANTS or ch in ("o", "u", "i")

    # ------------------------------------------------------------------ #
    #  Xác định vị trí đặt dấu thanh – Kiểu mới
    # ------------------------------------------------------------------ #
    def _choose_tone_position(
        self,
        chars: list[str],
        vowel_indices: list[int],
        n: int,
        is_after_qu: bool = False,
        is_after_gi: bool = False,
    ) -> int:
        """Trả về index (trong *chars*) nơi đặt dấu thanh điệu."""

        k = len(vowel_indices)

        # --- 1 nguyên âm duy nhất → dấu ngay trên nó -----------------
        if k == 1:
            return vowel_indices[0]

        # Chuỗi nguyên âm (NFC lowercase) để nhận diện kiểu âm chính
        vowel_str = self._vowel_string(chars, vowel_indices, n)

        # --- Xử lý âm đệm tròn môi (o/u đứng trước nguyên âm chính) ---
        # "o" hoặc "u" có thể là âm đệm (glide /w/) hoặc là nguyên âm chính.
        # Phải phân biệt: "hoà" (o = glide) vs "tói" (o = nucleus, i = coda)
        #                  "thuỷ" (u = glide) vs "túi" (u = nucleus, i = coda)
        if k >= 2 and vowel_str[0] in ("o", "u") and not is_after_qu:
            first_nfc = self._nfc_vowel_at(chars, vowel_indices[0], n)
            second_nfc = self._nfc_vowel_at(chars, vowel_indices[1], n)
            pair_check = first_nfc + second_nfc
            is_glide = False

            # Nếu cặp nguyên âm là diphthong chuẩn (uô, ua, ươ, ưa, iê, ...)
            # → KHÔNG phải glide, để xử lý ở phần diphthong bên dưới
            if pair_check in self.CLOSED_DIPHTHONGS or pair_check in self.OPEN_DIPHTHONGS:
                is_glide = False
            elif first_nfc == "o":
                # "o" là glide khi đứng trước nguyên âm chính thực sự
                # Nhưng "oi", "oy" → o là nucleus, i/y là bán âm cuối
                if k == 2 and second_nfc in ("i", "y"):
                    is_glide = False  # oi, oy: o là âm chính
                else:
                    is_glide = True
            elif first_nfc == "u":
                if k >= 3:
                    is_glide = True
                elif k == 2:
                    # "u" + nguyên âm: u là glide nếu nguyên âm sau là âm chính
                    # thực sự (a, e, ê, ơ, â, ...), KHÔNG phải bán âm cuối (i, y)
                    # "ui", "uy" khi KHÔNG có phụ âm đầu tròn môi → u là nucleus
                    if second_nfc == "i":
                        # "ui": u là âm chính, i là bán âm cuối (túi, cúi, xùi)
                        is_glide = False
                    elif second_nfc == "y":
                        # "uy": u là âm đệm, y là âm chính (thuỷ, nguỵ, huỷ)
                        is_glide = True
                    else:
                        is_glide = True

            if is_glide:
                main_indices = vowel_indices[1:]
                if len(main_indices) == 1:
                    return main_indices[0]
                return self._diphthong_position(chars, main_indices, n)

        # --- Nguyên âm đôi ------------------------------------------
        if k >= 2:
            return self._diphthong_position(chars, vowel_indices, n)

        # fallback (shouldn't reach)
        return vowel_indices[0]

    def _diphthong_position(
        self, chars: list[str], indices: list[int], n: int
    ) -> int:
        """Xác định vị trí dấu cho tổ hợp nguyên âm đôi / ba."""

        vowel_str = self._vowel_string(chars, indices, n)
        k = len(indices)

        # Lấy 2 ký tự đầu để kiểm tra diphthong
        pair = vowel_str[:2] if k >= 2 else ""

        # Nguyên âm đôi dạng mở (ia, ya, ua, ưa) – nhất loạt dấu lên chữ thứ 1
        # Theo quy tắc kiểu mới: âm tiết [+khép] → luôn đặt dấu vào chữ cái
        # thứ nhất trong tổ hợp 2 chữ cái biểu diễn cho âm chính.
        if pair in self.OPEN_DIPHTHONGS:
            return indices[0]

        # Nguyên âm đôi dạng đóng (iê, yê, uô, ươ) – dấu lên chữ thứ 2
        if pair in self.CLOSED_DIPHTHONGS:
            return indices[1]

        # Tổ hợp 3 nguyên âm (uyê, iêu, ươi, uôi …) → dấu lên chữ giữa
        if k >= 3:
            return indices[1]

        # 2 nguyên âm không phải diphthong chuẩn
        first_lower = self._nfc_vowel_at(chars, indices[0], n)
        second_lower = self._nfc_vowel_at(chars, indices[1], n)

        # Nếu nguyên âm thứ 2 là bán âm cuối (i, y, o, u) → nguyên âm thứ 1
        # là âm chính → dấu trên nguyên âm thứ 1.
        # Ví dụ: ai, ay, ao, au, oi, oy, ui, uy, ưi, ...
        if second_lower in ("i", "y", "o", "u"):
            return indices[0]

        # Nếu có phụ âm cuối sau nguyên âm thứ 2 → nguyên âm thứ 2 là âm chính
        # (nguyên âm thứ 1 đóng vai trò âm đệm đã lọt qua bộ lọc glide)
        # Ví dụ: oăn, oắt, ...
        last_vowel_pos = indices[-1]
        after = last_vowel_pos + 1
        while after < n and self._is_combining_mark(chars[after]):
            after += 1
        has_coda = after < n and chars[after].lower() in self.CODA_CONSONANTS
        if has_coda:
            return indices[1]

        # Mặc định: dấu lên nguyên âm đầu (nguyên âm chính + bán âm cuối)
        return indices[0]

    # ------------------------------------------------------------------ #
    #  Normalize toàn bộ văn bản
    # ------------------------------------------------------------------ #
    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        nfd = unicodedata.normalize("NFD", text)
        chars = list(nfd)
        n = len(chars)
        i = 0

        while i < n:
            # Bỏ qua ký tự không phải nguyên âm
            if not self._is_vowel(chars[i]):
                i += 1
                continue

            # ----------------------------------------------------------
            # Xử lý "gi" và "qu" là phụ âm đầu
            # ----------------------------------------------------------
            is_after_gi = False
            is_after_qu = False
            onset_tone = None  # Tone tìm thấy trên nguyên âm onset (u/i)

            if i > 0:
                prev = chars[i - 1].lower()

                # --- "qu" → u là phần của phụ âm, bỏ qua ---------------
                if chars[i].lower() == "u" and prev == "q":
                    is_after_qu = True
                    # Thu thập dấu thanh trên "u" nếu có (ví dụ: qúy → quý)
                    i += 1
                    while i < n and self._is_combining_mark(chars[i]):
                        if self._is_tone(chars[i]):
                            if onset_tone is None:
                                onset_tone = chars[i]
                            chars[i] = ""
                        i += 1
                    if i >= n or not self._is_vowel(chars[i]):
                        # Không còn nguyên âm → gắn tone lại vào "u"
                        if onset_tone is not None:
                            ins = i
                            chars.insert(ins, onset_tone)
                            n += 1
                            i += 1
                            onset_tone = None
                        continue

                # --- "gi" → i là phần của phụ âm … TRỪ KHI chữ chỉ là
                #     "gi" (ví dụ: gì, gí, gỉ, gĩ, gị) ----------------
                elif chars[i].lower() == "i" and prev == "g":
                    j = i + 1
                    while j < n and self._is_combining_mark(chars[j]):
                        j += 1
                    if j < n and self._is_vowel(chars[j]):
                        is_after_gi = True
                        # Thu thập dấu thanh trên "i" nếu có
                        for mi in range(i + 1, j):
                            if self._is_tone(chars[mi]):
                                if onset_tone is None:
                                    onset_tone = chars[mi]
                                chars[mi] = ""
                        i = j
                        if i >= n or not self._is_vowel(chars[i]):
                            continue

            # ----------------------------------------------------------
            # Thu thập cụm nguyên âm
            # ----------------------------------------------------------
            vowel_indices: list[int] = []
            while i < n:
                if self._is_vowel(chars[i]):
                    vowel_indices.append(i)
                    i += 1
                    while i < n and self._is_combining_mark(chars[i]):
                        i += 1
                else:
                    break

            if not vowel_indices:
                continue

            # ----------------------------------------------------------
            # Tách dấu thanh ra khỏi mọi nguyên âm trong cụm
            # ----------------------------------------------------------
            tone = None
            for idx in vowel_indices:
                pos = idx + 1
                while pos < n and (chars[pos] == "" or (self._is_combining_mark(chars[pos]) and not self._is_tone(chars[pos]))):
                    pos += 1
                if pos < n and self._is_tone(chars[pos]):
                    if tone is None:
                        tone = chars[pos]
                    chars[pos] = ""

            # Nếu không tìm thấy tone trên cụm nguyên âm, dùng tone
            # đã thu thập từ onset (qu/gi) nếu có
            if tone is None and onset_tone is not None:
                tone = onset_tone

            if tone is None:
                continue

            # ----------------------------------------------------------
            # Xác định vị trí đúng và gắn dấu
            # ----------------------------------------------------------
            tone_pos = self._choose_tone_position(
                chars, vowel_indices, n, is_after_qu, is_after_gi
            )
            insert_at = tone_pos + 1
            while (
                insert_at < n
                and chars[insert_at] != ""
                and self._is_combining_mark(chars[insert_at])
                and not self._is_tone(chars[insert_at])
            ):
                insert_at += 1
            chars.insert(insert_at, tone)
            n += 1
            i += 1  # adjust for inserted char

        return unicodedata.normalize("NFC", "".join(chars))
