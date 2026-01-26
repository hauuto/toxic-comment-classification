import re


class VietnameseTypingNormalizer:
    def __init__(self):
        self.rules = self._build_rules()

    def _build_rules(self):
        mapping = {
            "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
            "Oà": "Òa", "Oá": "Óa", "Oả": "Ỏa", "Oã": "Õa", "Oạ": "Ọa",
            "uý": "úy", "uỳ": "ùy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
            "Uý": "Úy", "Uỳ": "Ùy", "Uỷ": "Ủy", "Uỹ": "Ũy", "Uỵ": "Ụy",
            "uà": "ùa", "uá": "úa", "uả": "ủa", "uã": "ũa", "uạ": "ụa",
            "Uà": "Ùa", "Uá": "Úa", "Uả": "Ủa", "Uã": "Ũa", "Uạ": "Ụa",
            "ià": "ìa", "iá": "ía", "iả": "ỉa", "iã": "ĩa", "iạ": "ịa",
            "Ià": "Ìa", "Iá": "Ía", "Iả": "Ỉa", "Iã": "Ĩa", "Iạ": "Ịa",
            "ưà": "ừa", "ưá": "ứa", "ưả": "ửa", "ưã": "ữa", "ưạ": "ựa",
            "Ưà": "Ừa", "Ưá": "Ứa", "Ưả": "Ửa", "Ưã": "Ữa", "Ưạ": "Ựa",
            "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
        }
        pattern = re.compile("|".join(map(re.escape, mapping.keys())))
        return pattern, mapping

    def replace(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        pattern, mapping = self.rules
        return pattern.sub(lambda m: mapping[m.group(0)], text)
