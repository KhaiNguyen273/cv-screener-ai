"""
core/nlp_engine.py
------------------
Bước 3: Nhận diện thực thể (NER) từ văn bản CV và JD.
Trích xuất kỹ năng, học vấn, kinh nghiệm bằng spaCy và regex rules.
"""

import re
from typing import Dict, List, Set

# ── Danh sách từ khóa kỹ năng kỹ thuật phổ biến ──────────────────────────────
TECH_SKILLS_KEYWORDS: Set[str] = {
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "php", "ruby", "scala", "r", "matlab", "bash",
    # Web / Frontend
    "react", "vue", "angular", "html", "css", "nextjs", "nuxtjs", "svelte",
    "tailwind", "bootstrap", "jquery", "webpack", "vite",
    # Backend / Frameworks
    "django", "flask", "fastapi", "spring", "node.js", "express", "laravel",
    "rails", "asp.net", ".net",
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
    "figma", "tableau", "power bi", "excel", "powerpoint",
}

# ── Từ khóa học vấn ───────────────────────────────────────────────────────────
EDUCATION_KEYWORDS: List[str] = [
    "bachelor", "master", "phd", "doctorate", "associate", "diploma",
    "bsc", "msc", "mba", "b.e", "m.e", "b.tech", "m.tech",
    "cử nhân", "thạc sĩ", "tiến sĩ", "đại học", "cao đẳng",
    "university", "college", "institute", "school of",
    "graduated", "tốt nghiệp",
]

# ── Pattern kinh nghiệm ───────────────────────────────────────────────────────
EXPERIENCE_PATTERNS: List[str] = [
    r"(\d+)\+?\s*(?:year|yr)s?\s+(?:of\s+)?experience",
    r"experience\s+(?:of\s+)?(\d+)\+?\s*(?:year|yr)s?",
    r"(\d+)\+?\s*(?:năm|year|yr)s?\s*(?:kinh nghiệm|experience)",
    r"(\d{4})\s*[-–—]\s*(\d{4}|present|nay|hiện tại)",
]

ZERO_EXP_KEYWORDS = {
    "no experience", "không yêu cầu kinh nghiệm", "entry level", 
    "fresher", "thực tập sinh", "intern", "internship", 
    "sinh viên mới tốt nghiệp", "new graduate"
}

# 5.1 Tìm kỹ năng
def _extract_skills(text: str) -> List[str]:
    """Trích xuất kỹ năng từ văn bản bằng keyword matching (case-insensitive)."""
    text_lower = text.lower()
    found: List[str] = []
    for skill in TECH_SKILLS_KEYWORDS:
        # Dùng word boundary để tránh match nhầm (vd: "r" trong "framework" tránh nhằm nếu search ngôn ngữ R chỉ khớp R 1 mình)
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            # tìm kỹ năng nếu có thêm vào found
            found.append(skill)
    return sorted(found)

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
def _extract_experience(text: str) -> Dict:
    """
    Trích xuất thông tin kinh nghiệm: số năm và các khoảng thời gian.
    """
    # Số năm
    years_mentioned: List[int] = []

    # Mốc thời gian ví dụ 2020 - 2023
    periods: List[str] = []

    # Từ kiểu như "5 years experience", "3+ years of experience", "4 năm kinh nghiệm".
    # và dạng "2020 - 2023" hoặc "2021 - nay"
    for pattern in EXPERIENCE_PATTERNS[:3]:
        # Tìm tất cả chuỗi con khớp mẫu trả về list
        matches = re.findall(pattern, text, re.IGNORECASE)
        # Đổi phần số (ví dụ "5" từ "5 years") thành số
        for match in matches:
            try:
                years_mentioned.append(int(match))
            except (ValueError, TypeError):
                pass

    # Tìm khoảng thời gian dạng "2020 - 2023"
    period_pattern = EXPERIENCE_PATTERNS[3]
    # Tìm tất cả chuỗi con khớp mẫu trả về list
    period_matches = re.findall(period_pattern, text, re.IGNORECASE)
    # Cặp (tuple) chứa (năm bắt đầu, năm kết thúc)
    for match in period_matches:
        start, end = match
        periods.append(f"{start} - {end}")

    return {
        "years": max(years_mentioned) if years_mentioned else None,
        "periods": list(set(periods))[:5],
    }

# 5.4 Tìm liên hệ
def _extract_contact_info(text: str) -> Dict:
    """Trích xuất email và số điện thoại từ CV."""
    email_pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"(?:\+?\d[\d\s\-().]{7,}\d)"

    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)

    return {
        "email": emails[0] if emails else None,
        "phone": phones[0].strip() if phones else None,
    }


def extract_entities(text: str, source: str = "cv") -> Dict:
    """
    Pipeline NER đầy đủ: trích xuất kỹ năng, học vấn, kinh nghiệm.

    Args:
        text: Văn bản đã làm sạch.
        source: "cv" hoặc "jd" — ảnh hưởng đến trường nào được trả về.

    Returns:
        Dict chứa các thực thể được nhận diện.
    """
    # 5.1 Tìm kỹ năng
    skills = _extract_skills(text)
    # 5.2 Tìm học vấn
    education = _extract_education(text) if source == "cv" else []
    # 5.3 Tìm kinh nghiệm
    experience = _extract_experience(text)
    # 5.4 Tìm liên hệ
    contact = _extract_contact_info(text) if source == "cv" else {}

    return {
        "skills": skills,
        "education": education,
        "experience": experience,
        "contact": contact,
        "raw_skill_count": len(skills),
    }
