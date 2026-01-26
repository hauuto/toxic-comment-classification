import os
import re
import json


class IconNormalizer:
    def __init__(self, json_path: str = r"K:\GithubRepo\comment-classification\src\Preprocess2\Json\icons.json"):
        self.rules = self._load_rules(json_path)

    def _load_rules(self, json_path: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Không tìm thấy icon json: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rules = []
        for cfg in data.values():
            rules.append({
                "base": cfg["base"],
                "char": cfg["char"],
                "patterns": [re.compile(p) for p in cfg["patterns"]],
                "max_repeat": cfg["max_repeat"],
            })
        return rules

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        for rule in self.rules:
            base = rule["base"]
            char = rule["char"]
            max_r = rule["max_repeat"]

            for pat in rule["patterns"]:
                def _repl(m):
                    cnt = min(m.group().count(char), max_r)
                    return base[:-1] + char * cnt

                text = pat.sub(_repl, text)

        return text
