import os
# ⚠️ PHẢI ĐẶT TRƯỚC py_vncorenlp
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-21"

import re
from py_vncorenlp import VnCoreNLP

BASE_DIR = r"K:\GithubRepo\comment-classification\src\vncorenlp"

print(BASE_DIR)

vncorenlp = VnCoreNLP(
    annotators=["wseg", "pos"],
    save_dir=BASE_DIR
)

text = "lấp lánh kim sa hạt lựu bánh xe bánh ngọt bánh"
output = vncorenlp.word_segment(text)
print(output)
# ['Ông Nguyễn_Khắc_Chúc đang làm_việc tại Đại_học Quốc_gia Hà_Nội .', 'Bà Lan , vợ ông Chúc , cũng làm_việc tại đây .']