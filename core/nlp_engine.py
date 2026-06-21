import re
from typing import Dict, List, Set, Optional
from datetime import datetime
from utils.database.skill_repo import (
    load_skills,
    load_aliases
)

# ── Load spaCy model ──────────────────────────────────────────
nlp_model: Optional[object] = None
PhraseMatcher = None
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    try:
        nlp_model = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model 'en_core_web_sm' chưa cài. Chạy: python -m spacy download en_core_web_sm")
        nlp_model = None
except ImportError:
    print("spaCy chưa cài. Chạy: pip install spacy")
    nlp_model = None


# Đây là một Dictionary có các key là tên ngành nghề, còn value là một Set chứa các kỹ năng
SKILLS_BY_CATEGORY: Dict[str, Set[str]] = load_skills()
# vd: {
#     "IT / Software": {"python", "javascript", "node.js"},
#     "Marketing": {"seo"}
# }

# Đây là một Dictionary ánh xạ từ tên kỹ năng gốc
SKILL_ALIASES: Dict[str, List[str]] = load_aliases()
# vd: {
#     "javascript": ["javascript", "js", "es6"],
#     "node.js": ["node.js", "nodejs"],
#     "python": ["python"]
# }

# Tổng hợp các kỹ năng của các ngành với nhau để dùng 1 PhraseMatcher duy nhất tìm kiếm
ALL_SKILLS: Set[str] = set()
for category_skills in SKILLS_BY_CATEGORY.values():
    ALL_SKILLS.update(category_skills)
# vd: {"python", "javascript", "node.js", "seo"}



# Đây là một Dictionary dùng để "tra cứu ngược". Đảm bảo luôn tìm được
SKILL_LOOKUP: Dict[str, str] = {}
# vd:{
#     "python": "python",
#     "javascript": "javascript",
#     "js": "javascript",
#     "es6": "javascript",
#     "node.js": "node.js",
#     "nodejs": "node.js",
#     "seo": "seo"
# }

# là một Instance (đối tượng) của lớp PhraseMatcher thuộc thư viện spaCy.
SKILL_MATCHER: Optional[object] = None

if nlp_model is not None:
    try:
        SKILL_MATCHER = PhraseMatcher(nlp_model.vocab, attr="LOWER")
        patterns = []
        
        # Thêm skill gốc
        for skill in ALL_SKILLS:
            patterns.append(nlp_model.make_doc(skill))
            # Ghi vào sổ tra cứu để truy cứu tên khác của kỹ năng như js vẫn hiểu là javascript
            SKILL_LOOKUP[skill.lower()] = skill
        
        # duyệt qua từng tên và bí danh của skills
        for canonical, aliases in SKILL_ALIASES.items():
            for alias in aliases:
                # Nạp toàn bộ vào patterns để ghi vào SKILL_MATCHER
                patterns.append(nlp_model.make_doc(alias))
                # Ghi vào sổ tra cứu để truy cứu tên khác của kỹ năng như js vẫn hiểu là javascript
                SKILL_LOOKUP[alias.lower()] = canonical
        
        # Thêm vào SKILL_MATCHER để sau này nó tìm
        SKILL_MATCHER.add("SKILL", patterns)
    except Exception as e:
        print(f"Warning: PhraseMatcher initialization failed: {e}")
        SKILL_MATCHER = None

# ── Lookup table: skill -> category (để lấy category nhanh hơn) ──
SKILL_TO_CATEGORY: Dict[str, str] = {}
for category, skills in SKILLS_BY_CATEGORY.items():
    for skill in skills:
        SKILL_TO_CATEGORY[skill] = category

# ── Từ khóa học vấn ───────────────────────────────────────────────────────────
EDUCATION_KEYWORDS: List[str] = [
    "bachelor", "master", "phd", "doctorate", "associate", "diploma",
    "bsc", "msc", "mba", "b.e", "m.e", "b.tech", "m.tech",
    "cử nhân", "thạc sĩ", "tiến sĩ", "đại học", "cao đẳng",
    "university", "college", "institute", "school of",
    "graduated", "tốt nghiệp",
]

# ── Pattern kinh nghiệm (cải tiến để match nhiều dạng) ──────────────────────────
EXPERIENCE_PATTERNS = [
    # Pattern 0: Bắt kiểu số đứng trước chữ "năm" (Ví dụ: "3 năm kinh nghiệm", "2.5 năm làm việc")
    # \d+(?:\.\d+)? bắt số nguyên hoặc số thập phân (3 hoặc 2.5)
    # (?:\+)? bắt dấu cộng nếu có (ví dụ: "5+ năm")
    r"(\d+(?:\.\d+)?)(?:\+)?\s*năm",

    # ── Pattern 1: "5 years", "3.5 years", "5+ years"
    # \d+(?:\.\d+)?  → số nguyên hoặc số thập phân (3, 3.5)
    # \+?            → hỗ trợ "5+ years"
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b",

    # ── Pattern 2: "experience of 3 years"
    # bắt kiểu viết ngược "experience of X years"
    r"experience\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b",

    # ── Pattern 3: "3 years experience"
    # bắt kiểu phổ biến trong CV
    r"(\d+(?:\.\d+)?)\s*years?'?\s*(?:of\s+)?experience\b",

    # ── Pattern 4: "over 3 years", "more than 3 years"
    # bắt các từ định lượng kinh nghiệm
    r"(?:over|more\s+than|at\s+least)\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b",

    # ── Pattern 5: date range "2020 - 2023"
    # dùng để tính tổng năm kinh nghiệm từ timeline làm việc
    r"(\d{4})\s*[-–—]\s*(\d{4}|present|current|ongoing|nay|hiện tại)",
]

ZERO_EXP_KEYWORDS = {
    "no experience", "không yêu cầu kinh nghiệm", "entry level", 
    "fresher", "thực tập sinh", "intern", "internship", 
    "mới tốt nghiệp", "new graduate"
}

def _normalize_skill_text(text: str) -> str:
    """
    Normalize skill text để tăng khả năng match.
    Ví dụ: "C++" -> "c++", "Node.js" -> "node.js".
    """
    return text.lower().strip()

# Tìm kỹ năng (dùng PhraseMatcher thay vì regex loop)
def _extract_skills_fallback(
    text: str,
    use_spacy: bool = False
) -> List[str]:
    """
    Fallback skill extractor khi spaCy PhraseMatcher không khả dụng.

    - Keyword matching với word boundary.
    - Hỗ trợ aliases.
    - Trả về canonical skills.
    """

    text_lower = text.lower()
    found: Set[str] = set()

    # Match skill gốc
    for skill in ALL_SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        if re.search(pattern, text_lower):
            found.add(skill)

    # Match aliases -> canonical skill
    for canonical, aliases in SKILL_ALIASES.items():
        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(alias.lower()) + r"(?!\w)"

            if re.search(pattern, text_lower):
                found.add(canonical)

    # Bổ sung spaCy nếu được bật
    if use_spacy and nlp_model is not None:
        try:
            doc = nlp_model(text)

            # noun chunks
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()

                if chunk_text in ALL_SKILLS:
                    found.add(chunk_text)

                for canonical, aliases in SKILL_ALIASES.items():
                    if chunk_text in {a.lower() for a in aliases}:
                        found.add(canonical)

        except Exception as e:
            print(f"spaCy fallback error: {e}")

    return sorted(found)


def _extract_skills(text: str) -> List[str]:
    """
    Mục đích
    Nhận vào text → trả về list canonical skills tìm được:

    "I have experience with Python, JS and node.js" → ["javascript", "node.js", "python"]
    """
    if not nlp_model or not SKILL_MATCHER:
        # chạy nếu spaCy hay PhraseMatcher chưa có
        return _extract_skills_fallback(text)
    
    try:
        # spaCy tokenize text thành Doc object
        # vd: Vị trí 0: "I"
        # Vị trí 1: "know"
        # Vị trí 2: "JS"
        doc = nlp_model(text)
        # dùng set bỏ các kỹ năng trùng, tạo biến tên found
        found: Set[str] = set()
        
        # SKILL_MATCHER(doc) quét qua danh sách các Token trên.
        # Nhờ cái đã được nạp từ trước, nó phát hiện ra vị trí khớp lệnh
        for match_id, start, end in SKILL_MATCHER(doc):
            span_text = doc[start:end].text.lower()
            # Lấy từ đi tra để ra đúng skill ví dụ js -> tra được javascript
            canonical = SKILL_LOOKUP.get(span_text, span_text)
            if canonical in ALL_SKILLS:
                found.add(canonical)
            
        return sorted(found)
    except Exception as e:
        print(f"Error in _extract_skills: {e}")
        return _extract_skills_fallback(text)

# Lấy danh sách skill theo category (nhanh hơn dùng lookup table)
def _get_skills_by_category(text: str) -> Dict[str, List[str]]:
    """
    Trích xuất kỹ năng từ văn bản với thông tin category.
    Dùng SKILL_TO_CATEGORY lookup thay vì loop từng category.
    """
    skills = _extract_skills(text)
    result: Dict[str, List[str]] = {}
    
    for skill in skills:
        category = SKILL_TO_CATEGORY.get(skill)
        if category:
            result.setdefault(category, []).append(skill)
    
    return result

# Tìm học vấn
def _extract_education(text: str) -> List[str]:
    """Trích xuất thông tin học vấn từ văn bản."""
    # văn bản thường
    text_lower = text.lower()
    # danh sách lưu
    found: List[str] = []
    # tìm các từ trong EDUCATION_KEYWORDS như đại học,cao đẳng...
    for keyword in EDUCATION_KEYWORDS:
        if keyword.lower() in text_lower:
            # Lấy dòng chứa keyword để trả về ngữ cảnh
            for line in text.split("\n"):
                if keyword.lower() in line.lower() and line.strip():
                    found.append(line.strip())
    # Khử trùng, giữ tối đa 5 dòng nếu bị trùng vd như Đại học cũng khớp chữ học
    # dùng set để bỏ trùng
    seen: Set[str] = set()
    # danh sách lưu cái đã có bỏ trùng 
    unique: List[str] = []
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:5]

# Tìm kinh nghiệm
def _calculate_years_from_periods(periods: List[str]) -> Optional[float]:
    """
    Tính tổng số năm từ các khoảng thời gian, xử lý overlap.
    Ví dụ: ["2020 - 2023", "2023 - 2026"] -> 6 năm (không phải 7).
    Ví dụ overlap: ["2020 - 2023", "2022 - 2025"] -> 5 năm (2020-2025).
    """
    current_year = datetime.now().year
    
    # Parse tất cả periods
    parsed_periods = []
    # Duyệt qua từng chuỗi 2020 - 2023,...
    for period in periods:
        match = re.search(r"(\d{4})\s*[-–—]\s*(\d{4}|present|nay|hiện tại|ongoing|current)", period, re.IGNORECASE)
        if match:
            try:   
                # Lấy ra năm đầu
                start_year = int(match.group(1))
                # Lấy ra năm cuối
                end_text = match.group(2)
                
                # Nếu end là "present", "nay", "hiện tại", "ongoing", "current"
                if end_text.lower() in ["present", "nay", "hiện tại", "ongoing", "current"]:
                    end_year = current_year
                else:
                    end_year = int(end_text)
                
                if start_year <= end_year:
                    # nếu năm bắt đầu <= năm kết thúc nạp vào parsed_periods dạng tuple (2020, 2023)
                    parsed_periods.append((start_year, end_year))
            except (ValueError, AttributeError):
                pass
    
    if not parsed_periods:
        return None
    
    # Sắp xếp theo start_year
    parsed_periods.sort()
    
    # Gộp khoảng thời gian trùng nhau
    merged = []
    for start, end in parsed_periods:
        # kiểm tra merged đã có và năm bắt đầu của cái đang xét <= năm kết thúc của cái đã có trong merged
        if merged and start <= merged[-1][1]:
            # sửa lại năm cuối trong merged thành năm lớn hơn
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            # không có thì thêm mới
            merged.append((start, end))

    # Sau vòng lặp dạng merged = [(2020, 2025)]
    # Tính tổng năm từ merged
    total_years = sum(end - start for start, end in merged)
    
    return total_years if total_years > 0 else None

def _extract_experience(text: str) -> Dict:
    """
    Trích xuất thông tin kinh nghiệm:
    - số năm (5 years, 3.5 years, 5+ years, over 4 years)
    - khoảng thời gian (2020 - 2023)
    - xử lý entry level
    """

    years_mentioned: List[float] = []
    periods: List[str] = []

    text_lower = text.lower()

    # Kiểm tra xem có từ khóa trong bộ không có kinh nghiệm làm việc hay không 
    entry_level = any(
        keyword.lower() in text_lower
        for keyword in ZERO_EXP_KEYWORDS
    )

    # Lấy 4 ký tự trong EXPERIENCE_PATTERNS đã định nghĩa sau đó tìm các dòng trong chuỗi experience truyền vào từ hàm _detect_cv_sections 
    for pattern in EXPERIENCE_PATTERNS[:4]:
        # Cái khớp thì thêm vào matches
        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:

            # re.findall có thể trả:
            # - string "2.5"
            # - tuple ("3.5", "years")
            if isinstance(match, tuple):
                match = match[0]

            try:
                year_val = float(match)
                # Thêm các năm đã có đề cập vào
                years_mentioned.append(year_val)

            except (ValueError, TypeError):
                # tránh crash silent
                continue

    # Tính số năm nếu ghi theo khoảng 2020 - 2023
    period_pattern = EXPERIENCE_PATTERNS[5]
    period_matches = re.findall(period_pattern, text, re.IGNORECASE)

    for start, end in period_matches:
        periods.append(f"{start} - {end}")

    # Tính năm có nhiều khoảng
    calculated_years = _calculate_years_from_periods(periods) if periods else None

    # Quyết định kết quả ưu tiên mốc năm làm thực tế hơn 
    if calculated_years is not None:
        final_years = calculated_years

    elif years_mentioned:
        # nên MAX, không ép int ngay (mất 3.5 -> 3)
        final_years = max(years_mentioned)

    else:
        final_years = None

    # Cho kết quả số năm kinh nghiệm và lấy khoảng 5 mốc thời gian tránh nhiều 
    result = {
        "years": final_years,
        "periods": list(set(periods))[:5],
    }

    if entry_level:
        result["entry_level"] = True

    return result

# Tìm liên hệ
def _extract_contact_info(text: str) -> Dict:
    """Trích xuất email và số điện thoại từ CV. Fixed regex patterns."""
    # Email pattern: tất cả ký tự trước @, domain, và TLD. Dấu . PHẢI escape thành \\.
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    
    # Phone pattern: hỗ trợ +, (), -, spaces.
    phone_pattern = r"(?:\+?\d[\d\s\-().]{7,}\d)"

    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)

    return {
        "email": emails[0] if emails else None,
        "phone": phones[0].strip() if phones else None,
    }


def _detect_cv_sections(text: str) -> Dict[str, str]:
    """
    Nhận diện section CV (robust cho PDF / LinkedIn / ATS text).
    """

    sections = {}

    # Danh sách keyword để nhận diện các section phổ biến trong CV
    section_keywords = {
        "experience": ["experience", "work experience", "professional experience", "employment", "kinh nghiệm"],
        "education": ["education", "educational background", "qualifications", "học vấn"],
        "skills": ["skills", "technical skills", "competencies", "kỹ năng"],
        "projects": ["projects", "portfolio", "dự án"],
        "certifications": ["certifications", "certificates", "chứng chỉ"],
        "languages": ["languages", "language skills", "ngôn ngữ"],
    }

    # Tách text thành từng dòng để scan
    lines = text.split("\n")

    # section hiện tại đang parse
    current_section = None

    # buffer chứa nội dung của section hiện tại vd: buffer = ["Python", "React", "Node.js", "Docker"]
    buffer = []

    # flush buffer vào sections dict
    def flush():
        nonlocal current_section, buffer # truy cập biến của hàm cha _detect_cv_sections()

        # chỉ lưu nếu có section + có nội dung
        if current_section and buffer:
            sections[current_section] = "\n".join(buffer).strip()

        # reset buffer sau khi flush
        buffer = []

    # chuẩn hóa text để so sánh keyword
    def normalize(line: str) -> str:
        return line.lower().strip().replace(":", "").replace("-", "").strip()

    # kiểm tra xem dòng có phải header không
    def is_header(line: str) -> bool:
        norm = normalize(line)

        # nếu quá dài thì gần như không phải header
        # norm.split() tách thành list các từ vd: "python react nodejs" → ["python", "react", "nodejs"]
        if len(norm.split()) > 8: 
            # Check with word boundaries for long lines
            has_keyword = any(
                re.search(rf"\b{re.escape(kw)}\b", norm)
                # \b = word boundary: tránh "nosql" match "sql"
                for kws in section_keywords.values() # duyệt qua từng list keyword
                for kw in kws # duyệt qua từng keyword trong list
            )
            if not has_keyword:
                return False

        # nếu có keyword của bất kỳ section nào → coi là header
        for kws in section_keywords.values():
            for kw in kws:
                if re.search(rf"\b{re.escape(kw)}\b", norm):
                    return True

        return False

    # ─────────────────────────────
    # MAIN LOOP: duyệt từng dòng CV
    # ─────────────────────────────
    for line in lines:
        # bỏ khoảng trắng đầu cuối 
        raw = line.strip()

        # bỏ dòng rỗng không xử lý
        if not raw:
            continue

        norm = normalize(raw)

        matched_section = None

        # tìm xem dòng này thuộc section nào không
        for section, keywords in section_keywords.items():
            for kw in keywords:

                # dùng word boundary để tránh match sai (vd: sql vs nosql)
                if re.search(rf"\b{re.escape(kw.lower())}\b", norm):
                    matched_section = section
                    break # thoát vòng keyword

            if matched_section:
                break # thoát vòng section

        # ── CASE 1: gặp đúng tên section (match keyword) ──
        if matched_section and matched_section != current_section:
            flush()  # lưu section cũ
            current_section = matched_section  # chuyển sang section mới
            continue

        # ── CASE 2: gặp header nhưng không match keyword ──
        if is_header(raw):
            flush()  # kết thúc section cũ
            current_section = None  # tạm reset context
            continue

        # ── CASE 3: nội dung thuộc section hiện tại ──
        if current_section:
            buffer.append(raw)

    # flush lần cuối khi kết thúc loop
    flush()

    return sections

def _extract_gpa(text: str) -> Dict:
    """
    Trích xuất GPA từ text.
    Hỗ trợ: GPA: 3.45/4.0, Điểm: 9.2/10, 3.45 out of 4.0, v.v.
    """
    text_lower = text.lower()
    
    patterns = [
        # Từ khóa + số + denominator (CÓ capture group 2)
        r"(?:gpa|điểm|score|tích\s*lũy|điểm\s*trung\s*bình|cumulative|overall|final)\s*[:\s]+(\d+(?:\.\d+)?)\s*/\s*(4\.?0?|10)(?!\d)",

        # Số / Denominator (không cần từ khóa)  
        r"(\d+(?:\.\d+)?)\s*/\s*(4\.?0?|10)(?!\d)",

        # Từ khóa + số (không dấu /)
        r"(?:gpa|điểm|score|tích\s*lũy|điểm\s*trung\s*bình|cumulative|overall|final)\s*[:\s]*(\d+(?:\.\d+)?)(?!\d)",

        # Số trong ngoặc
        r"(\d+(?:\.\d+)?)\s*\(\s*(?:trên|on)?\s*(4\.?0?|10|scale)\s*\)",

        # out of
        r"(\d+(?:\.\d+)?)\s+out\s+of\s+(4\.?0?|10)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                gpa_value = float(match.group(1))
                # Lấy group 2 nếu có (denominator)
                try:
                    scale_text = match.group(2)
                except IndexError:
                    scale_text = None
                
                # Tự suy ra scale nếu không có group 2
                if scale_text is None:
                    scale = 10.0 if gpa_value > 4.0 else None
                else:
                    scale = 10.0 if "10" in scale_text else 4.0
                
                if 0 <= gpa_value <= 10:
                    return {"gpa": round(gpa_value, 2), "gpa_scale": scale}
            except (ValueError, IndexError):
                continue

    return {"gpa": None, "gpa_scale": None}


def extract_entities(text: str, source: str = "cv") -> Dict:
    """
    Pipeline NER đầy đủ: trích xuất kỹ năng, học vấn, kinh nghiệm.

    Args:
        text: Văn bản đã làm sạch.
        source: "cv" hoặc "jd" — ảnh hưởng đến trường nào được trả về.

    Returns:
        Dict chứa các thực thể được nhận diện.
    """
    # Nhận diện các sections trước
    sections = _detect_cv_sections(text) if source == "cv" else {}
    
    # Tìm kỹ năng (ưu tiên từ Skills section)
    skills_text = sections.get("skills") or text
    skills = _extract_skills(skills_text)
    
    # Tìm học vấn (từ Education section)
    if source == "cv":
        education_text = sections.get("education") or text
        education = _extract_education(education_text)
    else:
        education = []
    
    # Tìm kinh nghiệm (từ Experience section)
    experience_text = sections.get("experience") or text
    experience = _extract_experience(experience_text)
    
    # Tìm liên hệ
    contact = _extract_contact_info(text) if source == "cv" else {}

    # Tìm GPA
    gpa_info = _extract_gpa(education_text) if source == "cv" else {}

    return {
        "skills": skills,
        "education": education,
        "experience": experience,
        "contact": contact,
        "gpa": gpa_info.get("gpa"), 
        "gpa_scale": gpa_info.get("gpa_scale"),
        "raw_skill_count": len(skills),
        "sections_detected": list(sections.keys()) if sections else [],
    }
