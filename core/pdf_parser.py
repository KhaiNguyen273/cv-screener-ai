import os
import numpy as np
import pdfplumber
from docx import Document

POPPLER_PATH = os.getenv("POPPLER_PATH", None)

# Load EasyOCR reader để tránh load model mỗi lần gọi
_ocr_reader = None

def _get_ocr_reader():
    """Singleton: chỉ khởi tạo EasyOCR reader một lần."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # Load model tiếng Việt + Anh
        _ocr_reader = easyocr.Reader(["vi", "en"], gpu=False)
    return _ocr_reader


def extract_text_from_pdf(file_path: str) -> str:
    """
    Đọc file PDF và trả về toàn bộ nội dung.
    Nếu pdfplumber trả về rỗng (PDF scan) → fallback sang EasyOCR.
    """

    # list rỗng để chứa text từng trang.
    raw_text = []

    # Thử pdfplumber trước (PDF có text layer)
    with pdfplumber.open(file_path) as pdf:
        # loop từng trang
        for page in pdf.pages:
            # trích text từ trang hiện tại (có text layer → trả về chuỗi string, (chỉ có ảnh) → trả về None hoặc "")
            text = page.extract_text()
            if text:
                # append khi có nội dung
                raw_text.append(text)

    # Nếu không lấy được text → PDF dạng ảnh scan → dùng OCR
    if not raw_text:
        print(f"  [OCR] pdfplumber rỗng, dùng EasyOCR: {os.path.basename(file_path)}")
        raw_text = _ocr_fallback_pdf(file_path)

    return "\n".join(raw_text)


def _ocr_fallback_pdf(file_path: str) -> list[str]:
    """
    Chuyển từng trang PDF thành ảnh rồi chạy EasyOCR.
    Cần cài thêm: pip install pdf2image
    Và poppler: https://github.com/oschwartz10612/poppler-windows (Windows)
                 sudo apt install poppler-utils (Linux)
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("  [OCR] Thiếu pdf2image. Chạy: pip install pdf2image")
        return []

    reader = _get_ocr_reader()
    results = []

    #  poppler binary render từng trang PDF thành ảnh PIL nhờ pdf2image (Python không tự render được pdf)
    pages = convert_from_path(
        file_path,
        dpi=200,
        poppler_path=POPPLER_PATH
    )

    for i, page_img in enumerate(pages):
        print(f"  [OCR] Đang xử lý trang {i + 1}/{len(pages)}...")
        # Chuyển PIL Image → numpy array (EasyOCR không nhận PIL trực tiếp) 
        # OCR nhận numpy array(chứa các giá trị pixel của ảnh) hoặc đường dẫn file
        page_np = np.array(page_img)
        # EasyOCR nhận dạng chữ từ ảnh
        ocr_result = reader.readtext(page_np, detail=0, paragraph=True)
        # Ghép các đoạn text trong trang thành 1 chuỗi
        page_text = "\n".join(ocr_result)
        if page_text.strip():
            results.append(page_text)

    return results


def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    # Duyệt toàn bộ paragraphs trong document.
    # para.text — lấy text thuần của paragraph, bỏ toàn bộ formatting (bold, italic, màu sắc, font).
    raw_text = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(raw_text)


def parse_cv(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower() # → ".pdf" hoặc ".docx"

    if ext == ".pdf":
        return extract_text_from_pdf(file_path) # pdfplumber mở file
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_path) # python-docx mở file
    else:
        raise ValueError(f"Định dạng file '{ext}' không được hỗ trợ.")