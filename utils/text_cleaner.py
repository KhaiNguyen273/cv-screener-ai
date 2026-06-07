"""
utils/text_cleaner.py
---------------------
Bước 2 (tiếp theo): Tiền xử lý và làm sạch văn bản thô từ CV / JD.
Loại bỏ nhiễu, chuẩn hóa khoảng trắng, ký tự đặc biệt.
"""

import re
import unicodedata

# Chuyển sang NFC đảm bảo đúng ký tự nếu không sẽ hiểu theo NFD vd: u + `` =ù chứ không phải 1 chữ "ù"
def normalize_unicode(text: str) -> str:
    """Chuẩn hóa các ký tự Unicode về dạng NFC để đảm bảo nhất quán."""
    return unicodedata.normalize("NFC", text)


def remove_noise_characters(text: str) -> str:
    """
    Loại bỏ các ký tự không mong muốn:
    - Ký tự điều khiển (control characters)
    - Ký tự bullet, dấu đặc biệt không cần thiết
    - Dấu gạch ngang thừa
    """
    # Loại bỏ ký tự điều khiển (trừ newline và tab)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
    # Thay thế các loại bullet bằng dấu gạch ngang
    text = re.sub(r"[•·▪▸►◆●■□▶➤➢➣]", "-", text)
    # Loại bỏ ký tự đặc biệt không cần thiết nhưng giữ dấu câu cơ bản vd các icon 🚀, 💻, ⭐
    text = re.sub(r"[^\w\s\-.,;:()/+@#&%'\"\n]", " ", text, flags=re.UNICODE)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Chuẩn hóa khoảng trắng:
    - Thu gọn nhiều khoảng trắng thành một
    - Xóa khoảng trắng đầu/cuối mỗi dòng
    - Thu gọn nhiều dòng trống liên tiếp thành tối đa 2 dòng
    """
    # 2 dấu cách trở lên hoặc dấu tab đi liền nhau thành 1 dấu cách
    text = re.sub(r"[ \t]+", " ", text)
    # Xóa khoảng trắng đầu/cuối mỗi dòng
    lines = [line.strip() for line in text.split("\n")]
    # Loại bỏ các dòng hoàn toàn trống ở đầu/cuối khối
    text = "\n".join(lines)
    # Thu gọn nhiều dòng trống thành tối đa 2 dòng
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# 4. Làm sạch cv jd
def clean_text(raw_text: str) -> str:
    """
    Nguyễn Văn A  
    • Kỹ năng: Python  ,,  C++ 🚀 

    • Kinh nghiệm: 2 năm

    ========>  biến thành

    Nguyễn Văn A
    - Kỹ năng: Python , C++

    - Kinh nghiệm: 2 năm
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = normalize_unicode(raw_text)
    text = remove_noise_characters(text)
    text = normalize_whitespace(text)
    return text
