"""
core/pdf_parser.py
------------------
Bước 2: Bóc tách văn bản từ file CV (PDF hoặc DOCX).
Sử dụng pdfplumber cho PDF và python-docx cho DOCX.
"""

import os
import pdfplumber
from docx import Document


def extract_text_from_pdf(file_path: str) -> str:
    """
    Đọc file PDF và trả về toàn bộ nội dung dưới dạng chuỗi văn bản thô.

    Args:
        file_path: Đường dẫn tới file PDF.

    Returns:
        Chuỗi văn bản thô được ghép từ tất cả các trang.
    """
    raw_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text.append(text)
    return "\n".join(raw_text)


def extract_text_from_docx(file_path: str) -> str:
    """
    Đọc file DOCX và trả về toàn bộ nội dung dưới dạng chuỗi văn bản thô.

    Args:
        file_path: Đường dẫn tới file DOCX.

    Returns:
        Chuỗi văn bản thô được ghép từ tất cả các đoạn.
    """
    doc = Document(file_path)
    raw_text = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(raw_text)


def parse_cv(file_path: str) -> str:
    """
    Tự động phát hiện định dạng file và gọi hàm parser tương ứng.

    Args:
        file_path: Đường dẫn tới file CV (PDF hoặc DOCX).

    Returns:
        Chuỗi văn bản thô từ CV.

    Raises:
        ValueError: Nếu định dạng file không được hỗ trợ.
        FileNotFoundError: Nếu file không tồn tại.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Định dạng file '{ext}' không được hỗ trợ. Chỉ chấp nhận PDF hoặc DOCX.")
