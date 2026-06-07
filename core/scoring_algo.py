"""
core/scoring_algo.py
--------------------
Bước 4: Vector hóa và tính điểm tương đồng giữa CV và JD.
Sử dụng Sentence-Transformers + Cosine Similarity.
"""

from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Lazy-load model để tránh load lại mỗi lần gọi
_model: SentenceTransformer = None


def _get_model() -> SentenceTransformer:
    """Singleton: chỉ load model một lần duy nhất."""
    global _model
    if _model is None:
        # all-MiniLM-L6-v2: nhỏ gọn, nhanh, phù hợp production
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

# Biến đoạn text thành vector - *đang lưu tạm thời trong ram
def _embed(texts: List[str]) -> np.ndarray:
    """Chuyển danh sách chuỗi văn bản thành ma trận embedding."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Tính Cosine Similarity giữa hai vector.

    Returns:
        Điểm tương đồng từ 0.0 đến 1.0.
    """
    return float(np.dot(vec_a, vec_b))

# 6.2 Skill Overlap Score
# So sánh 2 kỹ năng cv và jd
def compute_skill_overlap(
    cv_skills: List[str], jd_skills: List[str]
) -> Dict:
    """
    Tính phần trăm kỹ năng JD yêu cầu mà CV đáp ứng được.

    Returns:
        Dict gồm matched_skills, missing_skills, overlap_score.
    """
    cv_set = {s.lower() for s in cv_skills}
    jd_set = {s.lower() for s in jd_skills}

    # jd không có yêu cầu kỹ năng trả về lỗi
    if not jd_set:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "overlap_score": 0.0,
        }
    # Phép giao 2 tập hợp
    matched = sorted(cv_set & jd_set)
    # Phép hiệu 2 tập hợp
    missing = sorted(jd_set - cv_set)
    # Tính tỷ lệ khớp
    overlap_score = len(matched) / len(jd_set)

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "overlap_score": round(overlap_score, 4),
    }

# 6. Chấm điểm
# Dict là cặp key-value
def score_cv_vs_jd(
    cv_text: str,
    jd_text: str,
    cv_entities: Dict,
    jd_entities: Dict,
) -> Dict:
    """
    Tính điểm toàn diện giữa CV và JD.

    Cấu trúc điểm:
    - semantic_score (50%): Độ tương đồng ngữ nghĩa toàn văn bản
    - skill_score (40%): Kỹ năng khớp với JD
    - experience_score (10%): Điểm sơ bộ từ số năm kinh nghiệm

    Args:
        cv_text: Văn bản CV đã làm sạch.
        jd_text: Văn bản JD đã làm sạch.
        cv_entities: Kết quả NER từ CV.
        jd_entities: Kết quả NER từ JD.

    Returns:
        Dict chứa các điểm số chi tiết và thông tin kỹ năng.
    """
    # 6.1 Semantic Score (Cosine Similarity toàn văn)
    # Chuyển CV và JD thành 2 vector
    embeddings = _embed([cv_text, jd_text])
    # Tính gốc giữa 2 vector càng bé là tương đồng
    semantic_score = compute_cosine_similarity(embeddings[0], embeddings[1])
    # Đảm bảo kết quả tính gốc giữa 2 vector không bao giờ nhỏ hơn 0.0 hoặc lớn hơn 1.0
    semantic_score = float(np.clip(semantic_score, 0.0, 1.0))

    # 6.2 Skill Overlap Score
    skill_result = compute_skill_overlap(
        cv_entities.get("skills", []),
        jd_entities.get("skills", []),
    )
    skill_score = skill_result["overlap_score"]

   # 6.3 Experience Score
    cv_years = cv_entities.get("experience", {}).get("years")
    jd_years = jd_entities.get("experience", {}).get("years")

    if jd_years is None:
        # JD không đề cập kinh nghiệm → không yêu cầu → full điểm
        experience_score = 1.0
    elif jd_years == 0:
        # JD ghi rõ không yêu cầu (fresher, intern) → full điểm
        experience_score = 1.0
    elif cv_years is None:
        # JD có yêu cầu nhưng CV không rõ → trung bình
        experience_score = 0.5
    else:
        # Cả hai đều có số năm → tính tỷ lệ
        experience_score = min(cv_years / jd_years, 1.0)

    # 6.4 Weighted Final Score
    final_score = (
        semantic_score * 0.50
        + skill_score * 0.40
        + experience_score * 0.10
    )

    return {
        "final_score": round(final_score * 100, 2),         # Phần trăm
        "semantic_score": round(semantic_score * 100, 2),
        "skill_score": round(skill_score * 100, 2),
        "experience_score": round(experience_score * 100, 2),
        "matched_skills": skill_result["matched_skills"],
        "missing_skills": skill_result["missing_skills"],
        "cv_skill_count": len(cv_entities.get("skills", [])),
        "jd_skill_count": len(jd_entities.get("skills", [])),
    }
