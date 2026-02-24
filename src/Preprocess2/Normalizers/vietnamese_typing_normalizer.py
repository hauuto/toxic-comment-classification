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
        # "qu" đã bị tách ở ngoài nên ở đây chỉ xét "o" và "u" làm âm đệm
        # Ví dụ: hoà, hoè  →  âm đệm "o" + nguyên âm chính
        #        chuyến     →  âm đệm "u" + nguyên âm đôi yê + coda n
        if k >= 2 and vowel_str[0] in ("o", "u") and not is_after_qu:
            # Kiểm tra xem ký tự đầu có phải âm đệm hay không:
            # - "o" luôn là âm đệm khi đứng trước nguyên âm khác (hoà, hoè)
            # - "u" là âm đệm khi phần còn lại ≥ 2 nguyên âm tạo thành
            #   diphthong (uyê, uyê, uya…) – ví dụ: chuyến, xuyên, khuya
            # - "u" KHÔNG là âm đệm khi chính nó là phần đầu của diphthong
            #   uô/ươ hoặc ua/ưa
            first_nfc = self._nfc_vowel_at(chars, vowel_indices[0], n)
            is_glide = False
            if first_nfc == "o":
                is_glide = True
            elif first_nfc == "u" and k >= 3:
                # u + 2 nguyên âm phía sau → u là âm đệm
                is_glide = True
            elif first_nfc == "u" and k == 2:
                # u + 1 nguyên âm: kiểm tra có phải diphthong uô/ươ/ua/ưa
                second_nfc = self._nfc_vowel_at(chars, vowel_indices[1], n)
                pair_check = first_nfc + second_nfc
                if pair_check not in self.CLOSED_DIPHTHONGS and pair_check not in self.OPEN_DIPHTHONGS:
                    # Không phải diphthong → u là âm đệm (ví dụ: uy, uê)
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

        # Nguyên âm đôi dạng mở (ia, ya, ua, ưa) – dấu lên chữ thứ 1
        if pair in self.OPEN_DIPHTHONGS:
            last_vowel_pos = indices[-1]
            if not self._has_coda(chars, last_vowel_pos + 1, n):
                return indices[0]
            # Nếu có coda → thực ra đây là dạng đóng viết sai → dấu lên chữ thứ 2
            return indices[1]

        # Nguyên âm đôi dạng đóng (iê, yê, uô, ươ) – dấu lên chữ thứ 2
        if pair in self.CLOSED_DIPHTHONGS:
            return indices[1]

        # Tổ hợp 3 nguyên âm (uyê, iêu, ươi, uôi …) → dấu lên chữ giữa
        if k >= 3:
            return indices[1]

        # 2 nguyên âm không phải diphthong chuẩn:
        #   - âm đệm u/o + nguyên âm đơn (ví dụ: "uy", "oa") → dấu lên
        #     nguyên âm cuối (nguyên âm chính)
        #   - nguyên âm chính + bán nguyên âm cuối (ai, ao, au, oi, …)
        #     → dấu lên nguyên âm đầu (nguyên âm chính)
        last_vowel_pos = indices[-1]
        after = last_vowel_pos + 1
        while after < n and self._is_combining_mark(chars[after]):
            after += 1
        has_coda = after < n and chars[after].lower() in self.CODA_CONSONANTS
        if has_coda:
            return indices[1]

        first_lower = self._nfc_vowel_at(chars, indices[0], n)
        second_lower = self._nfc_vowel_at(chars, indices[1], n)

        # Nếu nguyên âm thứ nhất là u/o và nguyên âm thứ hai KHÔNG phải
        # u/o → khả năng cao là âm đệm + nguyên âm chính → dấu lên chữ 2
        if first_lower in ("u", "o") and second_lower not in ("u", "o"):
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

            if i > 0:
                prev = chars[i - 1].lower()

                # --- "qu" → u là phần của phụ âm, bỏ qua ---------------
                if chars[i].lower() == "u" and prev == "q":
                    is_after_qu = True
                    # Bỏ qua "u" (và mọi combining mark nếu có)
                    i += 1
                    while i < n and self._is_combining_mark(chars[i]):
                        i += 1
                    # Tiếp tục thu thập nguyên âm phía sau
                    if i >= n or not self._is_vowel(chars[i]):
                        continue

                # --- "gi" → i là phần của phụ âm … TRỪ KHI chữ chỉ là
                #     "gi" (ví dụ: gì, gí, gỉ, gĩ, gị) ----------------
                elif chars[i].lower() == "i" and prev == "g":
                    # Kiểm tra sau "i" (và combining marks) có nguyên âm khác không
                    j = i + 1
                    while j < n and self._is_combining_mark(chars[j]):
                        j += 1
                    if j < n and self._is_vowel(chars[j]):
                        # "gi" + nguyên_âm → "i" thuộc phụ âm
                        is_after_gi = True
                        i = j  # jump to the next vowel
                        if i >= n or not self._is_vowel(chars[i]):
                            continue
                    # else: chỉ có "gi" → "i" chính là nguyên âm

            # ----------------------------------------------------------
            # Thu thập cụm nguyên âm
            # ----------------------------------------------------------
            vowel_indices: list[int] = []
            while i < n:
                if self._is_vowel(chars[i]):
                    vowel_indices.append(i)
                    i += 1
                    # Bỏ qua MỌI combining marks ngay sau nguyên âm
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

            if tone is None:
                continue

            # ----------------------------------------------------------
            # Xác định vị trí đúng và gắn dấu
            # ----------------------------------------------------------
            tone_pos = self._choose_tone_position(
                chars, vowel_indices, n, is_after_qu, is_after_gi
            )
            # Chèn dấu ngay sau base character + vowel modifiers tại tone_pos
            insert_at = tone_pos + 1
            # Nếu đã có combining mark khác (ví dụ breve, circumflex, horn)
            # thì chèn sau chúng
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




