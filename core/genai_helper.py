
import json
import os
import re
import time
from typing import Dict

import google.generativeai as genai
from groq import Groq, RateLimitError as GroqRateLimitError
from dotenv import load_dotenv

load_dotenv()

# ── Cấu hình model ────────────────────────────────────────────────────────────
_GROQ_MODEL   = "llama-3.3-70b-versatile"  # Ưu tiên 1 — free 1000 req/ngày
_GEMINI_MODEL = "gemini-2.5-flash"          # Ưu tiên 2 — fallback khi Groq hết quota


# ── Khởi tạo clients ──────────────────────────────────────────────────────────

def _get_groq_client() -> Groq:
    """Khởi tạo Groq client từ API key trong .env."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Thiếu GROQ_API_KEY. Vui lòng thêm vào file .env")
    return Groq(api_key=api_key)


def _get_gemini_client():
    """Khởi tạo Gemini client từ API key trong .env."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Thiếu GEMINI_API_KEY. Vui lòng thêm vào file .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(_GEMINI_MODEL)


# ── Gọi AI với fallback chain ─────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    """Gọi Groq API, retry tối đa 2 lần nếu bị rate limit tạm thời."""
    client = _get_groq_client()
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except GroqRateLimitError as e:
            err_msg = str(e).lower()
            # Hết quota ngày → không retry, ném lên để fallback sang Gemini
            if "day" in err_msg or "daily" in err_msg:
                raise
            # Rate limit tạm thời (per-minute) → chờ rồi thử lại
            wait = 15 * (attempt + 1)
            print(f"  [Groq] Rate limit tạm thời, chờ {wait}s... (lần {attempt + 1}/3)")
            time.sleep(wait)
    # Hết lần retry
    raise GroqRateLimitError("Groq: vượt quá số lần retry")


def _call_gemini(prompt: str) -> str:
    """Gọi Gemini API, retry tối đa 2 lần nếu bị rate limit tạm thời."""
    model = _get_gemini_client()
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=1,
                    max_output_tokens=8192,
                ),
            )
            # candidates[0] là phương án tốt nhất, ghép các parts thành text
            return "".join(
                part.text
                for part in response.candidates[0].content.parts
                if hasattr(part, "text") and part.text
            )
        except Exception as e:
            err_msg = str(e).lower()
            # Hết quota ngày → không retry
            if "quota" in err_msg and "day" in err_msg:
                raise
            # Lỗi tạm thời → chờ rồi thử lại
            wait = 30 * (attempt + 1)
            print(f"  [Gemini] Lỗi tạm thời, chờ {wait}s... (lần {attempt + 1}/3)")
            time.sleep(wait)
    raise RuntimeError("Gemini: vượt quá số lần retry")


def _call_llm_with_fallback(prompt: str) -> tuple[str, str]:
    """
    Gọi LLM theo thứ tự ưu tiên: Groq → Gemini.
    Trả về (raw_text, model_name_used).
    """
    # ── Ưu tiên 1: Groq ───────────────────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            print(f"  [LLM] Thử Groq ({_GROQ_MODEL})...")
            text = _call_groq(prompt)
            print(f"  [LLM] Groq OK")
            return text, _GROQ_MODEL
        except Exception as e:
            print(f"  [LLM] Groq thất bại ({e}), chuyển sang Gemini...")

    # ── Ưu tiên 2: Gemini ─────────────────────────────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            print(f"  [LLM] Thử Gemini ({_GEMINI_MODEL})...")
            text = _call_gemini(prompt)
            print(f"  [LLM] Gemini OK")
            return text, _GEMINI_MODEL
        except Exception as e:
            print(f"  [LLM] Gemini thất bại ({e})")
            raise

    raise EnvironmentError(
        "Không tìm thấy API key nào (GROQ_API_KEY hoặc GEMINI_API_KEY). "
        "Vui lòng thêm vào file .env"
    )


# 7.2 Tạo prompt
def _build_prompt(
    cv_text: str,
    jd_text: str,
    scores: Dict,
    cv_entities: Dict,
    jd_entities: Dict,
) -> str:
    """
    Xây dựng prompt yêu cầu AI trả về JSON có cấu trúc.
    Không sinh text tự do — output là JSON thuần túy.
    """
    matched_skills = ", ".join(scores.get("matched_skills", [])) or "Không có"
    missing_skills = ", ".join(scores.get("missing_skills", [])) or "Không có"
    cv_education   = "; ".join(cv_entities.get("education", [])) or "Không rõ"
    cv_exp_years   = cv_entities.get("experience", {}).get("years", "Không rõ")

    prompt = f"""
Bạn là chuyên gia tuyển dụng cấp cao. Hãy phân tích CV ứng viên so với Job Description và trả về KẾT QUẢ ĐÚNG CỨU JSON (không có markdown, không có text ngoài JSON).

=== DỮ LIỆU ĐẦU VÀO ===

JOB DESCRIPTION:
{jd_text[:2000]}

CV ỨNG VIÊN:
{cv_text[:2000]}

=== ĐIỂM SỐ ĐÃ TÍNH ===
- Điểm tổng: {scores.get('final_score', 0)}%
- Điểm ngữ nghĩa (Semantic): {scores.get('semantic_score', 0)}%
- Điểm kỹ năng: {scores.get('skill_score', 0)}%
- Điểm kinh nghiệm: {scores.get('experience_score', 0)}%
- Kỹ năng khớp: {matched_skills}
- Kỹ năng còn thiếu: {missing_skills}
- Học vấn: {cv_education}
- Số năm kinh nghiệm: {cv_exp_years}

=== YÊU CẦU OUTPUT ===
Trả về ĐÚNG ĐỊNH DẠNG JSON sau (không thêm bất kỳ text nào bên ngoài JSON):

{{
  "verdict": "Phù hợp cao | Phù hợp trung bình | Cần cân nhắc | Không phù hợp",
  "verdict_reason": "1-2 câu tóm tắt lý do quyết định",
  "overall_summary": "Đoạn tóm tắt tổng quát về ứng viên trong 3-4 câu, phong cách chuyên nghiệp",
  "strengths": [
    "Điểm mạnh 1 (cụ thể, kèm dẫn chứng từ CV)",
    "Điểm mạnh 2",
    "Điểm mạnh 3"
  ],
  "weaknesses": [
    "Điểm yếu / khoảng cách 1 (cụ thể)",
    "Điểm yếu / khoảng cách 2"
  ],
  "interview_questions": [
    {{
      "category": "Kỹ thuật",
      "question": "Câu hỏi phỏng vấn kỹ thuật cụ thể liên quan đến JD"
    }},
    {{
      "category": "Kinh nghiệm",
      "question": "Câu hỏi về kinh nghiệm thực tế"
    }},
    {{
      "category": "Hành vi",
      "question": "Câu hỏi tình huống/hành vi (STAR method)"
    }},
    {{
      "category": "Văn hóa",
      "question": "Câu hỏi đánh giá sự phù hợp văn hóa"
    }},
    {{
      "category": "Kỹ năng còn thiếu",
      "question": "Câu hỏi thăm dò kỹ năng ứng viên còn thiếu"
    }}
  ],
  "development_suggestions": [
    "Gợi ý phát triển 1 cho ứng viên",
    "Gợi ý phát triển 2"
  ],
  "hiring_recommendation": "Mời phỏng vấn ngay | Đưa vào danh sách dự phòng | Từ chối lịch sự"
}}
"""
    return prompt.strip()


def _parse_json_response(raw_response: str) -> Dict:
    """
    Parse JSON từ response của AI.
    Xử lý trường hợp model vẫn wrap trong markdown code block.
    """
    # Xóa markdown code fences nếu có
    text = raw_response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Fallback: trả về dict lỗi có cấu trúc thay vì crash
        return {
            "verdict": "Lỗi phân tích",
            "verdict_reason": f"Không thể parse JSON từ AI: {str(e)}",
            "overall_summary": raw_response[:500],
            "strengths": [],
            "weaknesses": [],
            "interview_questions": [],
            "development_suggestions": [],
            "hiring_recommendation": "Cần kiểm tra thủ công",
            "_parse_error": True,
        }


# Phân tích
def generate_analysis_report(
    cv_text: str,
    jd_text: str,
    scores: Dict,
    cv_entities: Dict,
    jd_entities: Dict,
) -> Dict:
    """
    Gọi AI API để sinh báo cáo phân tích CV vs JD.
    Tự động fallback: Groq → Gemini nếu hết quota.
    Luôn trả về Dict JSON có cấu trúc — không bao giờ trả về text thô.

    Args:
        cv_text:     Văn bản CV đã làm sạch.
        jd_text:     Văn bản JD đã làm sạch.
        scores:      Kết quả tính điểm từ scoring_algo.
        cv_entities: Thực thể NER từ CV.
        jd_entities: Thực thể NER từ JD.

    Returns:
        Dict JSON có cấu trúc chứa toàn bộ báo cáo.
    """
    # Tạo prompt
    prompt = _build_prompt(cv_text, jd_text, scores, cv_entities, jd_entities)

    # Gọi LLM với fallback chain (Groq → Gemini)
    raw_text, model_used = _call_llm_with_fallback(prompt)

    # Xóa markdown, parse JSON câu trả lời
    report = _parse_json_response(raw_text)

    # Thêm metadata vào report
    report["_model_used"] = model_used

    return report