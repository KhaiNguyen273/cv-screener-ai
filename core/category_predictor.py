from typing import Dict

from core.nlp_engine import SKILLS_BY_CATEGORY


def predict_category(cv_text: str) -> str:
    """Dự đoán ngành nghề chính của CV dựa trên kỹ năng đã trích xuất.

    Hiện tại dùng heuristic đơn giản: đếm kỹ năng theo category.
    """
    from core.nlp_engine import _extract_skills

    skills = _extract_skills(cv_text)
    counts: Dict[str, int] = {}
    for skill in skills:
        category = SKILLS_BY_CATEGORY.get(skill)
        if category:
            counts[category] = counts.get(category, 0) + 1

    if not counts:
        return "IT / Công nghệ"

    return max(counts, key=counts.get)
