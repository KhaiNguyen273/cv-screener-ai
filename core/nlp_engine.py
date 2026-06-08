"""
core/nlp_engine.py
------------------
Bước 3: Nhận diện thực thể (NER) từ văn bản CV và JD.
Trích xuất kỹ năng, học vấn, kinh nghiệm bằng spaCy và regex rules.
"""

import re
from typing import Dict, List, Set, Optional
from datetime import datetime

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

# ── Danh sách kỹ năng phân theo ngành (SKILLS_BY_CATEGORY) ─────────────────────
SKILLS_BY_CATEGORY: Dict[str, Set[str]] = {
    "IT / Công nghệ": {
        # Languages
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "kotlin", "swift", "php", "ruby", "scala", "r", "matlab", "bash",
        # Web / Frontend
        "react", "vue", "angular", "html", "css", "nextjs", "nuxtjs", "svelte",
        "tailwind", "bootstrap", "jquery", "webpack", "vite",
        # Backend / Frameworks
        "django", "flask", "fastapi", "spring", "node.js", "express", "laravel",
        "rails", "asp.net", "dotnet",
        # Data / AI / ML
        "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "spark", "hadoop", "kafka", "airflow", "mlflow", "hugging face",
        "langchain", "llm", "generative ai", "rag",
        # Cloud / DevOps
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "github actions", "ci/cd", "linux", "git",
        # Database
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "oracle", "sqlite", "dynamodb", "firebase",
        # Other
        "rest api", "graphql", "microservices", "agile", "scrum", "jira",
    },
    "Marketing / Truyền thông": {
        "seo", "sem", "digital marketing", "content marketing", "social media",
        "email marketing", "marketing automation", "analytics", "ga4", "google analytics",
        "facebook ads", "google ads", "instagram", "linkedin", "tiktok",
        "copywriting", "branding", "brand strategy", "market research", "campaign management",
        "crm", "salesforce", "hubspot", "mailchimp", "hootsuite",
        "content management", "wordpress", "shopify", "conversion rate", "ab testing",
    },
    "Thiết kế / UX": {
        "figma", "adobe xd", "sketch", "photoshop", "illustrator", "indesign",
        "ui/ux design", "ux research", "wireframing", "prototyping", "user testing",
        "design thinking", "user experience", "user interface", "graphic design",
        "web design", "mobile design", "interaction design", "animation",
        "css", "sass", "responsive design", "accessibility", "color theory",
        "typography", "layout design", "design system", "component design",
    },
    "Kế toán / Tài chính": {
        "accounting", "bookkeeping", "tax accounting", "financial reporting",
        "auditing", "forensic accounting", "quickbooks", "sage", "xero",
        "excel", "advanced excel", "pivot table", "vlookup", "financial analysis",
        "budget planning", "forecasting", "cost analysis", "cash flow", "general ledger",
        "accounts payable", "accounts receivable", "reconciliation", "gaap", "ifrs",
        "tax compliance", "payroll", "financial statements", "investment analysis",
    },
    "Kinh doanh / Quản lý": {
        "project management", "project planning", "risk management", "stakeholder management",
        "agile", "scrum", "kanban", "jira", "asana", "monday.com", "smartsheet",
        "business analysis", "requirements gathering", "process improvement",
        "supply chain", "operations", "logistics", "vendor management",
        "strategic planning", "business development", "sales management",
        "performance management", "change management", "organizational development",
    },
    "Nhân sự (HR)": {
        "recruitment", "talent acquisition", "employer branding", "onboarding",
        "employee relations", "performance management", "compensation benefits",
        "training development", "learning management", "hris", "workday", "successfactors",
        "hr analytics", "recruitment analytics", "succession planning", "career development",
        "employee engagement", "organizational culture", "conflict resolution",
        "compliance", "labor law", "employee satisfaction",
    },
    "Bán hàng (Sales)": {
        "sales", "business development", "account management", "account executive",
        "inside sales", "outside sales", "territory management", "lead generation",
        "sales pipeline", "customer acquisition", "customer retention", "upselling", "cross-selling",
        "negotiation", "closing skills", "sales forecasting", "quota management",
        "crm", "salesforce", "hubspot", "pipedrive", "zoho crm",
        "sales strategy", "sales metrics", "roi calculation",
    },
    "Hỗ trợ khách hàng (Customer Service)": {
        "customer service", "customer support", "technical support", "help desk",
        "ticketing system", "zendesk", "intercom", "freshdesk", "jira service desk",
        "customer success", "account management", "complaint handling", "troubleshooting",
        "documentation", "knowledge base", "customer training", "onboarding",
        "customer retention", "customer satisfaction", "nps", "customer feedback",
    },
}

# ── Danh sách từ khóa kỹ năng kỹ thuật phổ biến (để backward compatibility) ────────
TECH_SKILLS_KEYWORDS: Set[str] = SKILLS_BY_CATEGORY.get("IT / Công nghệ", set())
# ── Build ALL_SKILLS at module level (optimization) ────────────────────────
ALL_SKILLS: Set[str] = set()
for category_skills in SKILLS_BY_CATEGORY.values():
    ALL_SKILLS.update(category_skills)

# ── Skill synonyms/variations mapping: canonical -> [variations] ──
SKILL_ALIASES: Dict[str, List[str]] = {
    "react": ["reactjs", "react.js"],
    "vue": ["vuejs", "vue.js"],
    "angular": ["angularjs", "angular.js"],
    "nextjs": ["next.js"],
    "nuxtjs": ["nuxt.js"],
    "node.js": ["nodejs", "node js"],
    "postgresql": ["postgres", "postgre"],
    "mysql": ["mysql5"],
    "mongodb": ["mongo"],
    "javascript": ["js", "java script"],
    "typescript": ["ts", "type script"],
    "aws": ["amazon"],
    "gcp": ["google cloud"],
    "azure": ["microsoft azure"],
    "kubernetes": ["k8s", "k8"],
    "python": ["py"],
}

# ── Build PhraseMatcher cho skills (hiệu năng tốt hơn regex loop) ──
SKILL_LOOKUP: Dict[str, str] = {}  # skill_text.lower() -> canonical_skill
SKILL_MATCHER: Optional[object] = None  # PhraseMatcher instance

if nlp_model is not None:
    try:
        SKILL_MATCHER = PhraseMatcher(nlp_model.vocab, attr="LOWER")
        patterns = []
        
        # Thêm skill gốc
        for skill in ALL_SKILLS:
            patterns.append(nlp_model.make_doc(skill))
            SKILL_LOOKUP[skill.lower()] = skill
        
        # Thêm aliases -> map về canonical form
        for canonical, aliases in SKILL_ALIASES.items():
            for alias in aliases:
                patterns.append(nlp_model.make_doc(alias))
                SKILL_LOOKUP[alias.lower()] = canonical
        
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
    "sinh viên mới tốt nghiệp", "new graduate"
}

def _normalize_skill_text(text: str) -> str:
    """
    Normalize skill text để tăng khả năng match.
    Ví dụ: "C++" -> "c++", "Node.js" -> "node.js".
    """
    return text.lower().strip()

# 5.1 Tìm kỹ năng (dùng PhraseMatcher thay vì regex loop)

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
    Trích xuất kỹ năng từ văn bản bằng spaCy PhraseMatcher.
    Hỗ trợ tất cả skills trong SKILLS_BY_CATEGORY + aliases.
    Hiệu năng: O(text_length) thay vì O(num_skills × text_length).
    
    Args:
        text: Văn bản cần tìm kỹ năng.
    
    Returns:
        Danh sách kỹ năng được tìm thấy (canonical form), sắp xếp alphabetically.
    """
    if not nlp_model or not SKILL_MATCHER:
        # fall back to regex if PhraseMatcher not available
        return _extract_skills_fallback(text)
    
    try:
        doc = nlp_model(text)
        found: Set[str] = set()
        
        for match_id, start, end in SKILL_MATCHER(doc):
            span_text = doc[start:end].text.lower()
            # Lấy canonical form từ lookup
            canonical = SKILL_LOOKUP.get(span_text, span_text)
            if canonical in ALL_SKILLS:
                found.add(canonical)
        
        return sorted(found)
    except Exception as e:
        print(f"Error in _extract_skills: {e}")
        return _extract_skills_fallback(text)

# 5.1a Lấy danh sách skill theo category (nhanh hơn dùng lookup table)
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

# 5.2 Tìm học vấn
def _extract_education(text: str) -> List[str]:
    """Trích xuất thông tin học vấn từ văn bản."""
    text_lower = text.lower()
    found: List[str] = []
    for keyword in EDUCATION_KEYWORDS:
        if keyword.lower() in text_lower:
            # Lấy dòng chứa keyword để trả về ngữ cảnh
            for line in text.split("\n"):
                if keyword.lower() in line.lower() and line.strip():
                    found.append(line.strip())
    # Khử trùng, giữ tối đa 5 dòng nếu bị trùng
    seen: Set[str] = set()
    unique: List[str] = []
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:5]

# 5.3 Tìm kinh nghiệm
def _calculate_years_from_periods(periods: List[str]) -> Optional[float]:
    """
    Tính tổng số năm từ các khoảng thời gian, xử lý overlap.
    Ví dụ: ["2020 - 2023", "2023 - 2026"] -> 6 năm (không phải 7).
    Ví dụ overlap: ["2020 - 2023", "2022 - 2025"] -> 5 năm (2020-2025).
    """
    current_year = datetime.now().year
    
    # Parse tất cả periods
    parsed_periods = []
    for period in periods:
        match = re.search(r"(\d{4})\s*[-–—]\s*(\d{4}|present|nay|hiện tại|ongoing|current)", period, re.IGNORECASE)
        if match:
            try:
                start_year = int(match.group(1))
                end_text = match.group(2)
                
                # Nếu end là "present", "nay", "hiện tại", "ongoing", "current"
                if end_text.lower() in ["present", "nay", "hiện tại", "ongoing", "current"]:
                    end_year = current_year
                else:
                    end_year = int(end_text)
                
                if start_year <= end_year:
                    parsed_periods.append((start_year, end_year))
            except (ValueError, AttributeError):
                pass
    
    if not parsed_periods:
        return None
    
    # Sắp xếp theo start_year
    parsed_periods.sort()
    
    # Merge overlapping periods
    merged = []
    for start, end in parsed_periods:
        if merged and start <= merged[-1][1]:
            # Overlap: mở rộng end date của period trước
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            # Không overlap: thêm period mới
            merged.append((start, end))
    
    # Tính tổng năm từ merged periods
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

    # ── 1. Check entry level ─
    entry_level = any(
        keyword.lower() in text_lower
        for keyword in ZERO_EXP_KEYWORDS
    )

    # ── 2. Extract "X years" patterns ─────────────────────
    for pattern in EXPERIENCE_PATTERNS[:4]:
        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:

            # re.findall có thể trả:
            # - string
            # - tuple (nếu regex có nhiều group)
            if isinstance(match, tuple):
                match = match[0]

            try:
                year_val = float(match)
                years_mentioned.append(year_val)

            except (ValueError, TypeError):
                # tránh crash silent
                continue

    # ── 3. Extract date ranges 
    period_pattern = EXPERIENCE_PATTERNS[4]
    period_matches = re.findall(period_pattern, text, re.IGNORECASE)

    for start, end in period_matches:
        periods.append(f"{start} - {end}")

    # ── 4. Calculate years from periods 
    calculated_years = _calculate_years_from_periods(periods) if periods else None

    # ── 5. Decide final result 
    if calculated_years is not None:
        final_years = calculated_years

    elif years_mentioned:
        # nên MAX, không ép int ngay (mất 3.5 -> 3)
        final_years = max(years_mentioned)

    else:
        final_years = None

    result = {
        "years": final_years,
        "periods": list(set(periods))[:5],
    }

    if entry_level:
        result["entry_level"] = True

    return result

# 5.4 Tìm liên hệ
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

    # buffer chứa nội dung của section hiện tại
    buffer = []

    # flush buffer vào sections dict
    def flush():
        nonlocal current_section, buffer

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
        if len(norm.split()) > 8:
            # Check with word boundaries for long lines
            has_keyword = any(
                re.search(rf"\b{re.escape(kw)}\b", norm)
                for kws in section_keywords.values()
                for kw in kws
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
        raw = line.strip()

        # bỏ dòng rỗng
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
                    break

            if matched_section:
                break

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


def extract_entities(text: str, source: str = "cv") -> Dict:
    """
    Pipeline NER đầy đủ: trích xuất kỹ năng, học vấn, kinh nghiệm.

    Args:
        text: Văn bản đã làm sạch.
        source: "cv" hoặc "jd" — ảnh hưởng đến trường nào được trả về.

    Returns:
        Dict chứa các thực thể được nhận diện.
    """
    # Nhận diện sections trước
    sections = _detect_cv_sections(text) if source == "cv" else {}
    
    # 5.1 Tìm kỹ năng (ưu tiên từ Skills section)
    skills_text = sections.get("skills") or text
    skills = _extract_skills(skills_text)
    
    # 5.2 Tìm học vấn (từ Education section)
    if source == "cv":
        education_text = sections.get("education") or text
        education = _extract_education(education_text)
    else:
        education = []
    
    # 5.3 Tìm kinh nghiệm (từ Experience section)
    experience_text = sections.get("experience") or text
    experience = _extract_experience(experience_text)
    
    # 5.4 Tìm liên hệ
    contact = _extract_contact_info(text) if source == "cv" else {}

    return {
        "skills": skills,
        "education": education,
        "experience": experience,
        "contact": contact,
        "raw_skill_count": len(skills),
        "sections_detected": list(sections.keys()) if sections else [],
    }
