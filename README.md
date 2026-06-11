# CV Screener AI

Ứng dụng phân tích và đánh giá nhiều CV ứng viên so với Job Description bằng AI — tự động chấm điểm, xếp hạng, và gợi ý câu hỏi phỏng vấn.

## Cấu trúc dự án

cv_screener/
├── app.py # Giao diện Streamlit — input, pipeline, dashboard
├── core/
│ ├── pdf_parser.py # Bóc tách văn bản từ PDF/DOCX
│ ├── nlp_engine.py # Nhận diện thực thể NER — kỹ năng, học vấn, kinh nghiệm
│ ├── scoring_algo.py # Vector hóa & tính điểm cosine similarity
│ └── genai_helper.py # Sinh báo cáo JSON bằng AI (Groq → Gemini fallback)
├── utils/
│ └── text_cleaner.py # Làm sạch và chuẩn hóa văn bản
├── topcv/
│ ├── top_cv_api.py # Scraper TopCV bằng Selenium
│ └── top_cv_type.py # Dataclass Job
├── requirements.txt
├── .env.example
└── README.md

## Pipeline xử lý

[JD Text] + [CV Files (PDF/DOCX)]
↓ Bước 1 : Streamlit Input
↓ Bước 2 : Parse + Clean → pdf_parser + text_cleaner
↓ Bước 3 : NER → skills, education, experience
↓ Bước 4 : Scoring → cosine similarity + skill overlap
↓ Bước 5 : AI API → JSON report có cấu trúc
↓ Bước 6 : Dashboard → bảng ranking + gauge charts + interview Q&A

## Cài đặt

### Bước 1 — Cài thư viện

pip install -r requirements.txt

### Bước 2 — Tạo file .env

GROQ_API_KEY=your_api_key_here # Ưu tiên 1 — free 1000 req/ngày
GEMINI_API_KEY=your_api_key_here # Fallback — dùng khi Groq hết quota
POPPLER_PATH = your_path

Lấy Groq API Key miễn phí : https://console.groq.com/keys
Lấy Gemini API Key miễn phí : https://aistudio.google.com/app/apikey
Tải Poppler từ https://github.com/oschwartz10612/poppler-windows/releases - giải nén và lấy path tới Library\bin của thư mục đó

### Bước 3 — Chạy ứng dụng

python -m streamlit run app.py

## Công việc cần làm

### 1. Skill database theo ngành

- File `core/nlp_engine.py` hiện có `TECH_SKILLS_KEYWORDS` chỉ chứa skill của CNTT
- Cần tách thành `SKILLS_BY_CATEGORY` với skill riêng cho từng ngành (Marketing, Kế toán, Thiết kế,...)

### 2. Mở rộng tìm việc theo ngành trên TopCV

- File `topcv/top_cv_api.py` hiện chỉ hardcode link IT:
  `https://www.topcv.vn/tim-viec-lam-cong-nghe-thong-tin-cr257`
- Cần thêm link các ngành khác và cho user chọn ngành trước khi tải job

### 3. Upload file JD trong chế độ Tuyển Dụng

- Hiện tại chỉ nhập JD bằng text area
- Cần thêm tùy chọn upload file PDF/DOCX như bên chế độ Ứng Viên

### 4. Database lưu trữ (làm sau)

- Nhà tuyển dụng lưu lại thông tin và điểm của ứng viên
- Ứng viên lưu lại các công việc quan tâm

### 5. OCR cho CV dạng ảnh / scan (làm sau)

- File `core/pdf_parser.py` hiện chỉ đọc được PDF có text layer (PDF tạo từ Word,...)
- CV scan hoặc chụp ảnh thì `pdfplumber` trả về chuỗi rỗng
- Cần tích hợp OCR (Tesseract hoặc Google Vision) để đọc được CV dạng ảnh
