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
    # Loại bỏ ký tự điều khiển (trừ newline, carriage return và tab)
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
    # None hoặc chuỗi toàn khoảng trắng → trả ""
    if not raw_text or not raw_text.strip():
        return ""

    # Tiếng Việt có thể được lưu theo 2 cách khác nhau NFD (chữ ù = ký tự u + dấu huyền `) và NFC (chữ ù = 1 ký tự duy nhất)
    text = normalize_unicode(raw_text)

    # Xóa các ký tự vô hình không in được như \x00 (null), \x01... 
    # Thay bullet → dấu gạch ngang
    # Xóa ký tự đặc biệt / icon
    text = remove_noise_characters(text)

    # Thu gọn spaces/tabs nhiều dấu thành 1 dấu
    # Thu gọn các dòng trống tối đa 2
    text = normalize_whitespace(text)
    return text
