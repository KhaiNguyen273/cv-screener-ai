# THESIS REPORT — CV Screener AI

## CHƯƠNG 1: GIỚI THIỆU

### 1.1 Lý do chọn đề tài

Trong bối cảnh tuyển dụng hiện nay, doanh nghiệp đối mặt với lượng lớn hồ sơ ứng viên mỗi ngày. Nhiều công ty, đặc biệt là trong lĩnh vực CNTT, nhận được hàng trăm đến hàng nghìn CV cho một vị trí tuyển dụng. Điều này khiến bộ phận HR phải tốn nhiều thời gian để sàng lọc, phân loại và đánh giá từng hồ sơ.

Với số lượng CV lớn, việc đánh giá thủ công dễ dẫn đến sai sót, bỏ sót ứng viên phù hợp, và khó đảm bảo tính đồng nhất khi so sánh. Nhiều CV có cấu trúc khác nhau, định dạng PDF/DOCX hoặc scan ảnh, làm tăng độ phức tạp cho quá trình đọc và trích xuất thông tin.

Xu hướng ứng dụng AI trong HR đang tăng mạnh. AI được sử dụng để tự động hóa các bước sơ tuyển, trích xuất kỹ năng, đối chiếu yêu cầu công việc (JD) và sinh báo cáo đánh giá. Hệ thống CV Screening giúp HR giảm tải công việc thủ công, tăng tốc quy trình tuyển dụng và nâng cao độ chính xác trong đánh giá.

Lợi ích của hệ thống CV Screening:
- Tự động đọc và chuyển đổi CV PDF/DOCX sang văn bản.
- Trích xuất kỹ năng, học vấn và kinh nghiệm từ nội dung CV.
- So khớp CV với JD dựa trên kỹ năng và ngữ nghĩa.
- Chấm điểm, xếp hạng ứng viên và sinh báo cáo phân tích AI.
- Giảm thời gian sàng lọc, cải thiện trải nghiệm HR và ứng viên.

### 1.2 Mục tiêu đề tài

#### Mục tiêu tổng quát

Xây dựng hệ thống hỗ trợ doanh nghiệp tự động đánh giá và xếp hạng CV dựa trên Job Description, sử dụng AI để tạo báo cáo phân tích và đưa ra đề xuất tuyển dụng.

#### Mục tiêu cụ thể

- Đọc file PDF.
- Đọc file DOCX.
- Trích xuất kỹ năng từ CV và JD.
- So khớp JD với CV bằng kỹ thuật NLP và embedding.
- Chấm điểm ứng viên dựa trên kỹ năng, ngữ nghĩa và kinh nghiệm.
- Sinh báo cáo AI phân tích ứng viên, điểm mạnh, điểm yếu và đề xuất câu hỏi phỏng vấn.

### 1.3 Phạm vi đề tài

| Có thực hiện | Không thực hiện |
|---|---|
| PDF | Video interview |
| DOCX | Phân tích giọng nói |
| Trích xuất kỹ năng | ATS Enterprise |
| So khớp JD |  |
| Chấm điểm |  |
| Sinh báo cáo AI |  |

### 1.4 Ý nghĩa thực tiễn

- Đối với doanh nghiệp: Giảm chi phí tuyển dụng, rút ngắn thời gian sàng lọc, lựa chọn được hồ sơ phù hợp nhanh hơn.
- Đối với HR: Giảm công việc thủ công, có công cụ đánh giá đồng nhất, chuẩn hóa quy trình tuyển dụng.
- Đối với sinh viên/ứng viên: Nhận phản hồi nhanh hơn, cải thiện cơ hội khi hồ sơ được đánh giá khách quan.
- Đối với nghiên cứu AI: Áp dụng NLP, embedding và mô hình ngôn ngữ trong bài toán tuyển dụng thực tế.

## CHƯƠNG 2: CƠ SỞ LÝ THUYẾT

### 2.1 Python

#### Lịch sử phát triển Python

- Python 1.0: Được phát hành năm 1994 bởi Guido van Rossum. Phiên bản đầu tiên hỗ trợ các cấu trúc dữ liệu cơ bản và lập trình thủ tục.
- Python 2.x: Trải qua nhiều cải tiến, hỗ trợ Unicode, thu thập rác tốt hơn và mở rộng thư viện chuẩn. Python 2 phổ biến trong thập niên 2000.
- Python 3.x: Ra mắt năm 2008, thay đổi một số cú pháp không tương thích ngược. Python 3 là phiên bản hiện đại, được cập nhật liên tục và khuyến khích sử dụng.

#### Đặc điểm

- Interpreted: Python được thực thi mà không cần biên dịch trước, giúp phát triển nhanh.
- OOP: Hỗ trợ lập trình hướng đối tượng với lớp, kế thừa và đóng gói.
- Dynamic Typing: Kiểu dữ liệu được xác định tại thời điểm chạy, giúp viết mã linh hoạt.

#### Ứng dụng

- AI: Xây dựng mô hình học máy, xử lý ngôn ngữ tự nhiên, deep learning.
- Data Science: Phân tích, trực quan hóa, xử lý dữ liệu.
- Web: Phát triển ứng dụng web bằng Flask, Django, FastAPI.
- Automation: Tự động hóa xử lý tệp, kiểm thử, scraping.

#### Ưu nhược điểm

- Ưu điểm:
  - Dễ học và đọc.
  - Hệ sinh thái thư viện phong phú.
  - Phù hợp rapid prototyping.
- Nhược điểm:
  - Hiệu năng chậm hơn ngôn ngữ biên dịch.
  - Quản lý đa luồng có giới hạn do GIL.
  - Cần chú ý khi triển khai production quy mô lớn.

### 2.2 Trí tuệ nhân tạo AI

#### Khái niệm AI

AI là lĩnh vực nghiên cứu máy móc thực hiện nhiệm vụ thông minh như con người, bao gồm học máy, xử lý ngôn ngữ, nhận diện hình ảnh và ra quyết định.

#### AI trong tuyển dụng

AI có thể tự động phân loại CV, trích xuất thông tin, đánh giá sự phù hợp giữa ứng viên và JD, và sinh câu hỏi phỏng vấn.

#### AI Generative

AI Generative là mô hình tạo nội dung mới dựa trên dữ liệu đầu vào. Trong hệ thống này, Gemini và Groq được dùng để sinh bảng phân tích JSON từ CV và JD.

### 2.3 NLP

#### Các bước NLP

- Tokenization: Phân tách văn bản thành từ hoặc cụm từ.
- Stopword Removal: Loại bỏ từ dừng không mang ý nghĩa.
- Lemmatization: Chuẩn hóa từ về thể gốc.
- Named Entity Recognition (NER): Nhận diện thực thể như kỹ năng, học vấn, kinh nghiệm.

#### NLP trong CV Screening

Ví dụ:

CV:
```
PythonJavaSQL
```
JD:
```
PythonMySQLDocker
```

Hệ thống NLP sẽ nhận diện kỹ năng từ CV và JD, chuẩn hóa "MySQL"/"SQL" thành các kỹ năng tương ứng, sau đó so sánh để tìm ra kỹ năng khớp và thiếu.

### 2.4 Machine Learning

#### Sentence Transformer

Sentence Transformer là mô hình embedding chuyển văn bản thành vector số. Từ đó, hệ thống tính được mức độ tương đồng ngữ nghĩa giữa CV và JD.

#### Embedding

Embedding là quá trình vector hóa văn bản, đưa ngữ nghĩa câu/chữ vào không gian số để so sánh bằng phép toán đại số.

#### Cosine Similarity

Công thức:

\[ \mathrm{Cos}(x, y) = \frac{x \cdot y}{\|x\| \cdot \|y\|} \]

Cosine similarity đo góc giữa hai vector. Giá trị gần 1 nghĩa là hai văn bản có ý nghĩa tương đồng.

### 2.5 Gemini AI và Groq AI

- Gemini là nền tảng generative AI của Google. Trong dự án, Gemini được dùng để sinh báo cáo JSON khi Groq không khả dụng.
- Groq là dịch vụ AI chuyên thực thi mô hình LLM nhanh và có quota miễn phí. Hệ thống ưu tiên dùng Groq trước, sau đó fallback sang Gemini.
- Vai trò: tạo phân tích chuyên sâu, tóm tắt điểm mạnh/điểm yếu, sinh câu hỏi phỏng vấn và đề xuất tuyển dụng.

## CHƯƠNG 3: PHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG

### 3.1 Phân tích yêu cầu

#### Yêu cầu chức năng

- Upload CV.
- Upload JD.
- Xem kết quả đánh giá và báo cáo.

#### Yêu cầu phi chức năng

- Nhanh: Xử lý CV và JD trong thời gian chấp nhận được.
- Chính xác: Kết quả có khả năng phản ánh đúng kỹ năng và kinh nghiệm.
- Bảo mật: Không lưu trữ dữ liệu nhạy cảm ngoài phạm vi hệ thống.

### 3.2 Use Case Diagram

Actor:
- HR
- Hệ thống

Use Case:
- Upload CV
- Upload JD
- Xem kết quả
- Sinh báo cáo

### 3.3 Activity Diagram

Luồng xử lý tổng quan:

Start ↓ Upload CV/JD ↓ Parse Text ↓ Preprocessing NLP ↓ Matching ↓ Scoring ↓ AI Report ↓ Result

### 3.4 Kiến trúc hệ thống

Kiến trúc chính:

User ↓ Streamlit UI (app.py) ↓ Python Backend ↓ NLP Engine (core/nlp_engine.py) ↓ Gemini/Groq (core/genai_helper.py)

- Streamlit cung cấp giao diện upload và hiển thị kết quả.
- Backend xử lý parsing, làm sạch và gọi pipeline.
- NLP Engine trích xuất kỹ năng, học vấn, kinh nghiệm.
- AI Engine tạo báo cáo JSON và đề xuất phỏng vấn.

### 3.5 Thiết kế dữ liệu

Nếu mở rộng thành MySQL:

Bảng CV:

| Field | Type |
|---|---|
| id | int |
| name | varchar |
| skill | text |
| raw_text | text |
| score | float |

Bảng Job:

| Field | Type |
|---|---|
| id | int |
| title | varchar |
| jd_text | text |
| required_skills | text |

## CHƯƠNG 4: XÂY DỰNG HỆ THỐNG

### 4.1 Cấu trúc thư mục project

Cấu trúc chính:

```
app.py
core/
  pdf_parser.py
  nlp_engine.py
  scoring_algo.py
  genai_helper.py
topcv/
  categories.py
  top_cv_api.py
utils/
  text_cleaner.py
```

Giải thích:
- `app.py`: Giao diện Streamlit, điều phối pipeline.
- `core/pdf_parser.py`: Đọc PDF/DOCX, fallback OCR với EasyOCR.
- `core/nlp_engine.py`: Trích xuất kỹ năng, học vấn, kinh nghiệm bằng spaCy và PhraseMatcher.
- `core/scoring_algo.py`: Vector hóa text bằng SentenceTransformer và tính cosine similarity.
- `core/genai_helper.py`: Gọi Groq/Gemini để sinh báo cáo JSON.
- `topcv/`: Xử lý thu thập job từ TopCV.
- `utils/text_cleaner.py`: Làm sạch văn bản trước khi phân tích.

### 4.2 Module đọc CV

Hệ thống hỗ trợ:
- `pdf_parser.py` dùng `pdfplumber` để đọc PDF có text layer.
- `python-docx` để đọc file DOCX.
- OCR bằng `easyocr` và `pdf2image` nếu PDF là ảnh scan.

### 4.3 Tiền xử lý dữ liệu

Các bước tiền xử lý:
- Làm sạch ký tự thừa, khoảng trắng.
- Chuẩn hóa định dạng văn bản.
- Tách câu và loại bỏ lỗi định dạng do PDF.

Ví dụ trước xử lý: văn bản chứa nhiều dòng trống, ký tự đặc biệt.
Ví dụ sau xử lý: chuỗi clean, dễ cho NLP và embedding.

### 4.4 Trích xuất kỹ năng

Hệ thống dùng:
- `PhraseMatcher` của spaCy để nhận diện kỹ năng nhanh và chính xác.
- Dictionary kỹ năng `SKILLS_BY_CATEGORY`, `SKILL_ALIASES` để chuẩn hóa biến thể.

Ví dụ: `python`, `python3`, `py` có thể được chuẩn hóa về cùng một skill.

### 4.5 Chấm điểm CV

Cơ chế chấm điểm:
- Embedding văn bản CV và JD bằng `SentenceTransformer("all-MiniLM-L6-v2")`.
- Tính `Cosine Similarity` giữa hai vector để đo điểm ngữ nghĩa.
- Tính `skill_score` dựa trên overlap giữa kỹ năng CV và JD.
- Tính `experience_score` dựa trên số năm kinh nghiệm trích xuất được.
- Tổng hợp thành `final_score` với trọng số semantic 50%, skill 35%, experience 10%, GPA 5%.

### 4.6 AI Analysis

`core/genai_helper.py` xây dựng prompt chi tiết và gọi:
- Groq trước tiên.
- Gemini nếu Groq thất bại hoặc hết quota.

AI trả về báo cáo JSON gồm:
- verdict
- overall_summary
- strengths
- weaknesses
- interview_questions
- development_suggestions
- hiring_recommendation

### 4.7 Giao diện

Giao diện Streamlit hiển thị:
- Điểm tổng quan, kỹ năng, kinh nghiệm, tiềm năng.
- Danh sách kỹ năng khớp và còn thiếu.
- Nhận xét tổng quan và gợi ý phát triển.
- Bảng leaderboard xếp hạng nhiều CV.

## CHƯƠNG 5: KẾT QUẢ THỰC NGHIỆM

### 5.1 Môi trường thử nghiệm

- CPU: Intel / AMD tiêu chuẩn.
- RAM: 16GB.
- Python: 3.12.
- Thư viện: Streamlit, sentence-transformers, spaCy, easyocr, pdfplumber.

### 5.2 Bộ dữ liệu

Ví dụ thử nghiệm:
- 50 CV ngành IT.
- 10 JD ví dụ.

### 5.3 Kết quả

Bảng mẫu:

| CV | Điểm |
|---|---|
| CV190 | 90 |
| CV285 | 82 |

### 5.4 Đánh giá

Ưu điểm:
- Nhanh, tự động hóa sàng lọc CV.
- Kết hợp kỹ năng và ngữ nghĩa.

Nhược điểm:
- Phụ thuộc vào chất lượng CV.
- Chưa hiểu ngữ cảnh phức tạp trong mô tả công việc.

## CHƯƠNG 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### Kết luận

Hệ thống đã xây dựng được pipeline đọc CV/DOCX, trích xuất kỹ năng, chấm điểm CV so với JD và sinh báo cáo AI. Mục tiêu tự động hóa đánh giá CV cơ bản đã đạt được.

### Hướng phát triển

- Tích hợp ATS để lưu trữ hồ sơ và quản lý tuyển dụng.
- Mở rộng thành phỏng vấn AI / video interview.
- Phân tích dữ liệu LinkedIn và hồ sơ mạng xã hội.
- Học từ dữ liệu tuyển dụng thực tế để cải thiện mô hình đánh giá.
