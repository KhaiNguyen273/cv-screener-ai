"""
app.py — CV Screener AI
Chế độ Tuyển Dụng : so sánh nhiều CV vs 1 JD
Chế độ Ứng Viên   : upload 1 CV, cào job TopCV, chọn tối đa 5 job, chấm điểm & xếp hạng
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from core.genai_helper import generate_analysis_report
from core.nlp_engine import extract_entities
from core.pdf_parser import parse_cv
from core.scoring_algo import score_cv_vs_jd
from utils.text_cleaner import clean_text
from topcv import get_jobs_list
from topcv.categories import CATEGORY_CONFIGS

st.set_page_config(page_title="CV Screener AI", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f7f8fa;
}

/* ── Reset ── */
.main { background-color: #f7f8fa; }
p, span, div, label { color: #1a1d29; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.app-header p {
    color: #6b7280;
    font-size: 0.95rem;
    margin: 0;
}

/* ── Mode switcher ── */
.mode-switcher {
    display: flex;
    justify-content: center;
    gap: 0;
    margin: 0 auto 2rem auto;
    width: fit-content;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    overflow: hidden;
}
.mode-btn {
    padding: 9px 28px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    border: none;
    background: #ffffff;
    color: #4b5563;
    transition: all .15s;
}
.mode-btn.active {
    background: #1e3a8a;
    color: #ffffff;
    font-weight: 600;
}

/* ── Section labels ── */
.field-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

/* ── Score cards (number style) ── */
.score-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 1.5rem 0;
}
.score-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.1rem 1rem;
    text-align: left;
}
.score-card .sc-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7280;
    margin-bottom: 6px;
}
.score-card .sc-value {
    font-size: 2rem;
    font-weight: 700;
    color: #111827;
    line-height: 1;
    margin-bottom: 8px;
}
.score-card .sc-bar-bg {
    background: #f3f4f6;
    border-radius: 4px;
    height: 4px;
    width: 100%;
}
.score-card .sc-bar-fill {
    height: 4px;
    border-radius: 4px;
    background: #1e3a8a;
}

/* ── Verdict badge ── */
.verdict-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.85rem;
    border: 1px solid;
}
.verdict-high   { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
.verdict-medium { background: #fffbeb; color: #92400e; border-color: #fde68a; }
.verdict-low    { background: #fef2f2; color: #991b1b; border-color: #fecaca; }

/* ── Summary card ── */
.summary-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 14px;
}
.summary-card h3 {
    font-size: 1.15rem;
    font-weight: 700;
    color: #111827;
    margin: 0 0 10px 0;
}
.summary-card p {
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.65;
    margin: 0;
}
.sub-section-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7280;
    margin: 12px 0 6px 0;
}
.strength-item {
    font-size: 0.88rem;
    color: #1a1d29;
    padding: 3px 0 3px 12px;
    border-left: 2px solid #10b981;
    margin-bottom: 5px;
}
.weakness-item {
    font-size: 0.88rem;
    color: #1a1d29;
    padding: 3px 0 3px 12px;
    border-left: 2px solid #ef4444;
    margin-bottom: 5px;
}

/* ── Skill analysis card ── */
.skill-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
}
.skill-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #9ca3af;
    margin: 14px 0 6px 0;
}
.skill-section-label:first-child { margin-top: 0; }
.skill-tag {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 5px;
    font-size: 0.77rem;
    font-weight: 500;
    margin: 3px 3px 3px 0;
    border: 1px solid;
}
.skill-matched { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
.skill-missing { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
.edu-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-top: 8px;
}
.edu-name { font-size: 0.88rem; font-weight: 600; color: #111827; }
.gpa-value { font-size: 0.88rem; font-weight: 700; color: #1e3a8a; }

/* ── Question cards ── */
.question-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 3px solid #1e3a8a;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.question-category {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #1e3a8a;
    margin-bottom: 4px;
}
.question-text {
    font-size: 0.88rem;
    color: #111827;
    line-height: 1.5;
}

/* ── Dev suggestions ── */
.dev-card {
    background: #1e3a8a;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
}
.dev-card .dev-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #93c5fd;
    margin-bottom: 10px;
}
.dev-card .dev-item {
    font-size: 0.85rem;
    color: #e0eaff;
    padding: 3px 0 3px 14px;
    position: relative;
    margin-bottom: 5px;
}
.dev-card .dev-item::before {
    content: "–";
    position: absolute;
    left: 0;
    color: #93c5fd;
}

/* ── Hiring recommendation ── */
.rec-bar {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem 1.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 16px;
}
.rec-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #9ca3af; }

/* ── Candidate leaderboard ── */
/* ── Candidate leaderboard ── */
.leaderboard-header {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #9ca3af;
    padding: 8px 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 4px;
}
.lb-row {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr 1.4fr 1.6fr;
    align-items: center;
    padding: 10px 8px;
    border-radius: 6px;
    border-bottom: 1px solid #f3f4f6;
    font-size: 0.88rem;
    color: #374151;
    transition: background .1s;
}
.lb-row:hover { background: #f9fafb; }

/* Thay đổi ở đây: Bỏ background màu xanh nhạt cũ, đưa về trong suốt hoặc trắng */
.lb-row.lb-top { 
    background: transparent; 
}

.lb-name { font-weight: 600; color: #1e3a8a; font-size: 0.9rem; }
.lb-score-high { font-weight: 700; color: #1e3a8a; }
.lb-verdict-high { color: #065f46; font-weight: 500; }
.lb-verdict-med  { color: #92400e; font-weight: 500; }
.lb-verdict-low  { color: #991b1b; font-weight: 500; }

/* ── Assessment section ── */
.assessment-header {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9ca3af;
    margin: 2rem 0 6px 0;
}
.assessment-name {
    font-size: 1.6rem;
    font-weight: 700;
    color: #111827;
    margin: 0 0 6px 0;
}
.assessment-summary {
    font-size: 0.9rem;
    color: #6b7280;
    line-height: 1.6;
    margin: 0 0 1.5rem 0;
    max-width: 640px;
}

/* ── Job card ── */
.job-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 10px;
}
.job-card.selected { border-color: #1e3a8a; background: #f0f4ff; }
.job-badge {
    display: inline-block;
    background: #f3f4f6;
    color: #374151;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
}
.job-title-text {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1e3a8a;
    margin: 0 0 2px 0;
}
.job-company { font-size: 0.85rem; color: #6b7280; margin: 0 0 6px 0; }
.job-meta-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 4px;
    font-size: 0.78rem;
    color: #9ca3af;
    align-items: start; /* Ép tất cả các cột xuất phát từ cùng một đỉnh dòng */
}
.job-meta-row span { 
    display: flex; 
    flex-direction: column; 
    justify-content: flex-start; /* Căn nội dung bên trong span sát lên trên */
}
.job-meta-row .meta-val { color: #374151; font-weight: 500; font-size: 0.82rem; }

/* ── Upload area ── */
.upload-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 12px;
}
.upload-filename {
    font-size: 0.95rem;
    font-weight: 600;
    color: #111827;
}
.upload-change {
    font-size: 0.82rem;
    color: #2563eb;
    cursor: pointer;
    margin-left: 10px;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }

/* ── Streamlit button overrides ── */
div[data-testid="stButton"] > button {
    border-radius: 7px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.55rem 1.4rem;
    transition: all .15s;
}

/* 1. KHI NÚT ĐANG ĐƯỢC CHỌN (PRIMARY) */
div[data-testid="stButton"] > button[kind="primary"] {
    background: #1e3a8a !important;
    border: 1px solid #1e3a8a !important;
}
/* Ép toàn bộ chữ hoặc thẻ con (span, p) bên trong nút primary thành màu trắng tuyệt đối */
div[data-testid="stButton"] > button[kind="primary"] *,
div[data-testid="stButton"] > button[kind="primary"] p,
div[data-testid="stButton"] > button[kind="primary"] span {
    color: #ffffff !important;
}

/* Hiệu ứng hover khi nút đang được chọn */
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1e40af !important;
    border-color: #1e40af !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover *,
div[data-testid="stButton"] > button[kind="primary"]:hover p,
div[data-testid="stButton"] > button[kind="primary"]:hover span {
    color: #ffffff !important;
}

/* 2. KHI NÚT KHÔNG ĐƯỢC CHỌN (SECONDARY) */
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
}
/* Đảm bảo nút không chọn luôn hiển thị chữ màu xám tối */
div[data-testid="stButton"] > button[kind="secondary"] *,
div[data-testid="stButton"] > button[kind="secondary"] p,
div[data-testid="stButton"] > button[kind="secondary"] span {
    color: #374151 !important;
}

/* Hiệu ứng hover khi nút KHÔNG được chọn */
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: #f9fafb !important;
    border-color: #9ca3af !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover *,
div[data-testid="stButton"] > button[kind="secondary"]:hover p,
div[data-testid="stButton"] > button[kind="secondary"]:hover span {
    color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _verdict_class(verdict: str) -> str:
    v = verdict.lower()
    if "cao" in v or "ngay" in v:
        return "verdict-high"
    if "trung bình" in v or "dự phòng" in v:
        return "verdict-medium"
    return "verdict-low"


def _score_bar(value: float) -> str:
    pct = max(0, min(100, value))
    return f"""
    <div class="sc-bar-bg">
        <div class="sc-bar-fill" style="width:{pct}%;"></div>
    </div>"""


def run_pipeline(jd_text: str, cv_file) -> dict:
    # Tương tự jd lưu đường dẫn file tạm để dùng cho parse cv lấy ra raw text của cv
    suffix = Path(cv_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(cv_file.read())
        tmp_path = tmp.name
    try:
        raw_cv = parse_cv(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Làm sạch raw text khoảng trắng và ký tự không được phép
    clean_cv = clean_text(raw_cv)
    clean_jd = clean_text(jd_text)

    # Tìm thực thể
    cv_ent   = extract_entities(clean_cv, source="cv")
    jd_ent   = extract_entities(clean_jd, source="jd")

    scores   = score_cv_vs_jd(clean_cv, clean_jd, cv_ent, jd_ent)
    report   = generate_analysis_report(cv_text=clean_cv, jd_text=clean_jd,
                                        scores=scores, cv_entities=cv_ent,
                                        jd_entities=jd_ent)
    return {"scores": scores, "cv_entities": cv_ent,
            "jd_entities": jd_ent, "report": report}


def run_pipeline_from_text(cv_text: str, jd_text: str) -> dict:
    clean_cv = clean_text(cv_text)
    clean_jd = clean_text(jd_text)
    cv_ent   = extract_entities(clean_cv, source="cv")
    jd_ent   = extract_entities(clean_jd, source="jd")
    scores   = score_cv_vs_jd(clean_cv, clean_jd, cv_ent, jd_ent)
    report   = generate_analysis_report(cv_text=clean_cv, jd_text=clean_jd,
                                        scores=scores, cv_entities=cv_ent,
                                        jd_entities=jd_ent)
    return {"scores": scores, "cv_entities": cv_ent,
            "jd_entities": jd_ent, "report": report}


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE DETAIL DASHBOARD (dùng chung 2 chế độ)
# ══════════════════════════════════════════════════════════════════════════════

def _render_candidate_detail(result: dict, candidate_key: str, candidate_name: str = "", mode: str = "recruiter"):
    scores  = result["scores"]
    cv_ent  = result["cv_entities"]
    report  = result["report"]
    verdict = report.get("verdict", "N/A")

    # ── Assessment title ──
    if candidate_name:
        st.markdown(f"""
        <p class="assessment-header">ACTIVE ASSESSMENT</p>
        <p class="assessment-name">{candidate_name}</p>
        <p class="assessment-summary">{report.get('overall_summary', '')}</p>
        """, unsafe_allow_html=True)

    contact = cv_ent.get("contact", {})
    email = contact.get("email")
    phone = contact.get("phone")
    if email or phone:
        contact_html = " &nbsp;·&nbsp; ".join(
            filter(None, [email, phone])
        )
        st.markdown(
            f'<p style="color:#6b7280;font-size:.85rem;margin:0 0 12px 0;">{contact_html}</p>',
            unsafe_allow_html=True,
        )

    # ── Score cards (4 boxes) ──
    final   = scores.get("final_score", 0)
    skill   = scores.get("skill_score", 0)
    exp     = scores.get("experience_score", 0)
    # Tiềm năng = trung bình semantic + skill
    potential = round((scores.get("semantic_score", 0) + skill) / 2, 1)

    st.markdown(f"""
    <div class="score-grid">
        <div class="score-card">
            <div class="sc-label">Điểm tổng quan</div>
            <div class="sc-value">{final:.0f}%</div>
            {_score_bar(final)}
        </div>
        <div class="score-card">
            <div class="sc-label">Phù hợp kỹ năng</div>
            <div class="sc-value">{skill:.0f}%</div>
            {_score_bar(skill)}
        </div>
        <div class="score-card">
            <div class="sc-label">Kinh nghiệm</div>
            <div class="sc-value">{exp:.0f}%</div>
            {_score_bar(exp)}
        </div>
        <div class="score-card">
            <div class="sc-label">Tiềm năng</div>
            <div class="sc-value">{potential:.0f}%</div>
            {_score_bar(potential)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main 2-column layout ──
    left, right = st.columns([3, 2], gap="large")

    with left:
        # Nhận xét tổng quan + điểm mạnh/yếu
        strengths = report.get("strengths", [])
        weaknesses = report.get("weaknesses", [])
        dev_suggestions = report.get("development_suggestions", [])

        strengths_html = "".join(
            f'<div class="strength-item">— {s}</div>' for s in strengths
        )
        weaknesses_html = "".join(
            f'<div class="weakness-item">— {w}</div>' for w in weaknesses
        )

        st.markdown(f"""
        <div class="summary-card">
            <h3>Nhận Xét Tổng Quan</h3>
            <p>{report.get('overall_summary', '')}</p>
            <div class="sub-section-label">Điểm Mạnh</div>
            {strengths_html}
            <div class="sub-section-label">Điểm Yếu</div>
            {weaknesses_html}
        </div>
        """, unsafe_allow_html=True)

        # Câu hỏi phỏng vấn
        questions = report.get("interview_questions", [])
        if questions:
            st.markdown('<div style="font-size:1rem;font-weight:600;color:#111827;margin:1rem 0 10px 0;">Câu Hỏi Phỏng Vấn Gợi Ý</div>', unsafe_allow_html=True)
            q_cols = st.columns(2)
            for i, q in enumerate(questions):
                with q_cols[i % 2]:
                    st.markdown(f"""
                    <div class="question-card">
                        <div class="question-category">{q.get('category','Chung')}</div>
                        <div class="question-text">{q.get('question','')}</div>
                    </div>""", unsafe_allow_html=True)

    with right:
        # Skill analysis card
        matched = scores.get("matched_skills", [])
        missing = scores.get("missing_skills", [])
        edu_list = cv_ent.get("education", [])
        gpa = cv_ent.get("gpa")

        matched_html = "".join(
            f'<span class="skill-tag skill-matched">{s}</span>' for s in matched
        ) or "<span style='color:#9ca3af;font-size:.83rem;'>Không có</span>"

        missing_html = "".join(
            f'<span class="skill-tag skill-missing">{s}</span>' for s in missing
        ) or "<span style='color:#9ca3af;font-size:.83rem;'>Đáp ứng đầy đủ</span>"

        edu_name = edu_list[0] if edu_list else "Không rõ"
        gpa_scale = cv_ent.get("gpa_scale")
        if gpa:
            if gpa_scale:
                scale_label = f"/{int(gpa_scale)}"
            else:
                # Không có dấu "/" trong CV → suy ra từ giá trị: >4 là thang 10, còn lại là thang 4
                scale_label = "/10" if gpa > 4.0 else "/4.0"
            gpa_str = f"{gpa}{scale_label}"
        else:
            gpa_str = "—"

        st.markdown(f"""
        <div class="skill-card">
            <div class="skill-section-label">Kỹ Năng Khớp</div>
            <div>{matched_html}</div>
            <div class="skill-section-label">Kỹ Năng Còn Thiếu</div>
            <div>{missing_html}</div>
            <div class="skill-section-label">Học Vấn</div>
            <div class="edu-row">
                <span class="edu-name">{edu_name}</span>
                <span class="gpa-value">{gpa_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gợi ý phát triển
        if dev_suggestions:
            dev_items_html = "".join(
                f'<div class="dev-item">{s}</div>' for s in dev_suggestions
            )
            st.markdown(f"""
            <div class="dev-card" style="margin-top:12px;">
                <div class="dev-title">Gợi Ý Phát Triển</div>
                {dev_items_html}
            </div>
            """, unsafe_allow_html=True)

    # ── Hiring / Candidate recommendation ──
    if mode == "recruiter":
        rec_text = report.get("hiring_recommendation", "")
        if "ngay" in rec_text.lower(): rec_color = "#065f46"
        elif "dự phòng" in rec_text.lower(): rec_color = "#92400e"
        else: rec_color = "#991b1b"
    else:
        rec_text = report.get("candidate_recommendation", "")
        if "ngay" in rec_text.lower(): rec_color = "#065f46"
        elif "cân nhắc" in rec_text.lower(): rec_color = "#92400e"
        else: rec_color = "#991b1b"

    if rec_text:
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            st.markdown(f"""
            <div style="text-align:center;margin-top:1.5rem;">
                <button style="
                    background:#1e3a8a;color:#fff;border:none;
                    border-radius:8px;padding:10px 32px;
                    font-size:0.9rem;font-weight:600;cursor:default;
                    width:100%;
                ">{rec_text}</button>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LEADERBOARD + RANKING
# ══════════════════════════════════════════════════════════════════════════════

def _render_ranking(all_results: dict, label_col: str = "CV", mode: str = "recruiter"):
    recommendation = "hiring_recommendation" if mode == "recruiter" else "candidate_recommendation"
    # Xây dữ liệu bảng
    rows = []
    for name, res in all_results.items():
        if "error" in res:
            rows.append({
                "name": name, "final": None, "skill": None,
                "exp": None, "verdict": "Lỗi", "rec": res["error"],
                "error": True,
            })
        else:
            s, r = res["scores"], res["report"]
            rows.append({
                "name": name,
                "final": s["final_score"],
                "skill": s["skill_score"],
                "exp":   s["experience_score"],
                "verdict": r.get("verdict", "N/A"),
                "rec":     r.get(recommendation, "N/A"),
                "error": False,
            })

    # Sắp xếp
    rows_valid  = sorted([r for r in rows if not r["error"]], key=lambda x: x["final"], reverse=True)
    rows_error  = [r for r in rows if r["error"]]
    rows_sorted = rows_valid + rows_error

    # ── Leaderboard table ──
    st.markdown('<p style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#9ca3af;margin-bottom:6px;">CANDIDATE LEADERBOARD</p>', unsafe_allow_html=True)

    header_html = f"""
    <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1.4fr 1.6fr;
                padding:8px 8px;font-size:0.72rem;font-weight:700;
                text-transform:uppercase;letter-spacing:.06em;color:#9ca3af;
                border-bottom:2px solid #e5e7eb;">
        <span>{label_col}</span>
        <span>Điểm Tổng</span>
        <span>Kỹ Năng</span>
        <span>Kinh Nghiệm</span>
        <span>Đánh Giá</span>
        <span>Lời Khuyên</span>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    for i, row in enumerate(rows_sorted):
        is_top = i == 0 and not row["error"]
        bg = "#ffffff"
        border = "border-bottom:1px solid #f3f4f6;"

        if row["error"]:
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1.4fr 1.6fr;
                        padding:10px 8px;background:{bg};{border}
                        font-size:0.86rem;align-items:center;border-radius:4px;">
                <span style="font-weight:600;color:#374151;">{row['name']}</span>
                <span style="color:#9ca3af;">—</span>
                <span style="color:#9ca3af;">—</span>
                <span style="color:#9ca3af;">—</span>
                <span style="color:#991b1b;">Lỗi</span>
                <span style="color:#9ca3af;font-size:.8rem;">{row['rec'][:40]}...</span>
            </div>""", unsafe_allow_html=True)
        else:
            v = row["verdict"].lower()
            if "cao" in v: vc = "#065f46"
            elif "trung bình" in v or "dự phòng" in v: vc = "#92400e"
            else: vc = "#991b1b"
            name_style = "font-weight:600;color:#374151;"
            score_style = "color:#374151;"

            st.markdown(f"""
            <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1.4fr 1.6fr;
                        padding:10px 8px;background:{bg};{border}
                        font-size:0.86rem;align-items:center;border-radius:4px;">
                <span style="{name_style}">{row['name']}</span>
                <span style="{score_style}">{row['final']:.0f}%</span>
                <span style="color:#374151;">{row['skill']:.0f}%</span>
                <span style="color:#374151;">{row['exp']:.0f}%</span>
                <span style="color:{vc};font-weight:500;">{row['verdict']}</span>
                <span style="color:#2563eb;font-size:.83rem;">{row['rec']}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Detail tabs ──
    if rows_sorted:
        top_name = rows_sorted[0]["name"] if not rows_sorted[0]["error"] else None
        tabs = st.tabs([r["name"] for r in rows_sorted])
        for tab, row in zip(tabs, rows_sorted):
            with tab:
                res = all_results[row["name"]]
                if "error" in res:
                    st.error(f"Lỗi: {res['error']}")
                else:
                    safe_key = str(list(all_results.keys()).index(row["name"]))
                    _render_candidate_detail(
                        res,
                        candidate_key=safe_key,
                        candidate_name=row["name"],
                        mode=mode,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ TUYỂN DỤNG
# ══════════════════════════════════════════════════════════════════════════════

def render_recruiter_mode():
    col_jd, col_cv = st.columns([1, 1], gap="large")

    with col_jd:
        st.markdown('<p class="field-label">Job Description</p>', unsafe_allow_html=True)

        has_jd_text = bool(st.session_state.get("recruiter_jd", "").strip())
        has_jd_file = st.session_state.get("recruiter_jd_file") is not None

        jd_text = st.text_area(
            label="JD", height=220,
            placeholder="Dán nội dung Job Description tại đây...",
            label_visibility="collapsed",
            key="recruiter_jd",
            disabled=has_jd_file,
        )

        jd_file = st.file_uploader(
            label="Tải lên tệp JD", type=["pdf", "docx"],
            accept_multiple_files=False,
            key="recruiter_jd_file",
            disabled=has_jd_text,
        )
        if has_jd_file and not has_jd_text:
            st.caption("Xóa file JD để nhập text thủ công.")
        if has_jd_text and not has_jd_file:
            st.caption("Xóa text JD để upload file.")

        if jd_file:
            st.success(f"Đã tải lên JD: **{jd_file.name}**")

    with col_cv:
        st.markdown('<p class="field-label">CV Ứng Viên (nhiều file)</p>', unsafe_allow_html=True)
        cv_files = st.file_uploader(
            label="CV", type=["pdf", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="recruiter_cv",
        )
        if cv_files:
            st.success(f"Đã tải lên {len(cv_files)} file: {', '.join(f.name for f in cv_files)}")

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        analyze_btn = st.button("PHÂN TÍCH NGAY", width="stretch",
                                key="recruiter_analyze", type="primary")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if analyze_btn:
        if not jd_text.strip() and not jd_file:
            st.error("Vui lòng nhập nội dung Job Description hoặc tải lên file JD.")
            return
        if not cv_files:
            st.error("Vui lòng tải lên ít nhất 1 file CV.")
            return

        if jd_file:
            # Lấy tên file và đuôi đảm bảo parse_cv hoạt động đúng nhận biết pdf hay doc ,...
            suffix = Path(jd_file.name).suffix.lower()
            # Tạo file tạm trong disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                # Ghi nội dung vào
                tmp.write(jd_file.read())
                tmp_path = tmp.name
            try:
                # Chạy hàm parse_cv với đường dẫn file đã ghi, lấy ra jd
                jd_text = parse_cv(tmp_path)
            finally:
                # Cuối cùng luôn xóa đi
                os.unlink(tmp_path)

        # Tạo dict rỗng để lưu kết quả, key là tên file CV, value là kết quả pipeline.
        all_results = {}
        progress = st.progress(0, text="Bắt đầu phân tích...")
        # Duyệt từng CV để chấm điểm với jd
        for i, cv_file in enumerate(cv_files):
            progress.progress(i / len(cv_files), text=f"Đang xử lý: {cv_file.name}")
            try:
                # Chạy pipeline cho từng CV, lưu kết quả vào dict với key là tên file.
                all_results[cv_file.name] = run_pipeline(jd_text, cv_file)
            except Exception as e:
                # Nếu 1 CV lỗi thì không crash toàn bộ, ghi lỗi rồi chạy tiếp CV tiếp theo.
                all_results[cv_file.name] = {"error": str(e)}
        progress.progress(1.0, text=f"Hoàn tất {len(cv_files)} CV.")

        _render_ranking(all_results, label_col="CV", mode="recruiter")


# ══════════════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ ỨNG VIÊN
# ══════════════════════════════════════════════════════════════════════════════

MAX_SELECT = 5


def _render_job_card(job, idx: int, is_selected: bool, is_full: bool):
    card_class = "job-card selected" if is_selected else "job-card"
    employment_type = (job.tags[0] if job.tags else "Full-time")

    st.markdown(f"""
    <div class="{card_class}">
        <span class="job-badge">{employment_type}</span>
        <p class="job-title-text">{job.title or 'N/A'}</p>
        <p class="job-company">{job.company or 'N/A'}</p>
        <div class="job-meta-row">
            <span>
                <span style="font-size:.7rem;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;">Địa điểm</span>
                <span class="meta-val">{job.location or '—'}</span>
            </span>
            <span>
                <span style="font-size:.7rem;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;">Mức lương</span>
                <span class="meta-val">{job.salary or 'Thỏa thuận'}</span>
            </span>
            <span>
                <span style="font-size:.7rem;font-weight:600;color:#9ca3af;text-transform:uppercase;letter-spacing:.05em;">Đăng lúc</span>
                <span class="meta-val">{job.time_posted or '—'}</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    btn_col, detail_col = st.columns([1, 1])
    with btn_col:
        if is_selected:
            if st.button("Bỏ chọn", key=f"sel_{idx}", width="stretch", type="secondary"):
                st.session_state.selected_jobs.discard(idx)
                st.rerun()
        else:
            disabled = is_full and not is_selected
            label = "Chọn" if not is_full else f"Đã đủ {MAX_SELECT} job"
            if st.button(label, key=f"sel_{idx}", width="stretch",
                         type="primary", disabled=disabled):
                st.session_state.selected_jobs.add(idx)
                st.rerun()

    with detail_col:
        if st.button("Xem chi tiết", key=f"det_{idx}", width="stretch", type="secondary"):
            current = st.session_state.get("expanded_job")
            st.session_state.expanded_job = idx if current != idx else None
            st.rerun()

    if st.session_state.get("expanded_job") == idx:
        with st.expander("", expanded=True):
            if job.detail:
                st.markdown(job.detail)
            else:
                st.warning("Không lấy được mô tả chi tiết.")


def render_candidate_mode():
    # ── Upload CV ──
    cv_name = st.session_state.get("candidate_cv_name", "")
    if cv_name:
        st.markdown(f"""
        <div class="upload-box">
            <span style="font-size:.72rem;font-weight:700;text-transform:uppercase;
                         letter-spacing:.06em;color:#9ca3af;display:block;margin-bottom:4px;">
                Upload CV của bạn
            </span>
            <span class="upload-filename">{cv_name}</span>
        </div>
        """, unsafe_allow_html=True)

    cv_file = st.file_uploader(
        label="Upload CV của bạn" if not cv_name else "Thay đổi CV",
        type=["pdf", "docx"],
        label_visibility="visible" if not cv_name else "visible",
        key="candidate_cv",
        accept_multiple_files=False,
    )

    if cv_file and st.session_state.get("candidate_cv_name") != cv_file.name:
        suffix = Path(cv_file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(cv_file.read())
            tmp_path = tmp.name
        try:
            raw = parse_cv(tmp_path)
        finally:
            os.unlink(tmp_path)
        st.session_state.candidate_cv_text = clean_text(raw)
        st.session_state.candidate_cv_name = cv_file.name
        st.session_state.pop("jobs_list", None)
        st.session_state.selected_jobs = set()
        st.session_state.job_results   = {}
        st.rerun()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if not st.session_state.get("candidate_cv_text"):
        st.info("Hãy upload CV để tiếp tục.")
        return

    # ── Chọn ngành & tải job ──
    col_fetch, col_page, col_btn = st.columns([2, 1, 2])
    with col_fetch:
        selected_industry = st.selectbox(
            "Chọn ngành",
            list(CATEGORY_CONFIGS.keys()),
            key="topcv_industry",
        )
    with col_page:
        page_num = st.number_input("Trang", min_value=1, max_value=20,
                                   value=1, step=1, key="topcv_page")
    with col_btn:
        st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
        fetch_btn = st.button("Tải danh sách Job", key="fetch_jobs",
                              type="primary", width="stretch")

    if fetch_btn:
        with st.spinner(f"Đang tải trang {page_num}..."):
            try:
                from topcv.categories import get_category_url
                jobs = asyncio.run(get_jobs_list(category=selected_industry, page=page_num))
                st.session_state.jobs_list     = jobs
                st.session_state.selected_jobs = set()
                st.session_state.job_results   = {}
            except Exception as e:
                st.error(f"Lỗi khi cào TopCV: {e}")
                return

    all_jobs: list = st.session_state.get("jobs_list", [])
    if not all_jobs:
        return

    jobs_map = {
        i: j for i, j in enumerate(all_jobs)
        if j.detail and len(j.detail.strip()) > 50
    }
    total_all      = len(all_jobs)
    total_with_jd  = len(jobs_map)

    if not jobs_map:
        st.warning("Không có job nào lấy được mô tả chi tiết.")
        return

    st.markdown(f"""
    <p style="font-size:1rem;font-weight:700;color:#111827;margin:1rem 0 .5rem 0;">
        Danh sách Job ({total_with_jd} việc làm)
    </p>
    """, unsafe_allow_html=True)

    keyword = st.text_input("Lọc theo từ khóa (tiêu đề, công ty, tag)",
                            key="job_filter")
    filtered_jobs = [
        (i, j) for i, j in jobs_map.items()
        if not keyword or keyword.lower() in (
            (j.title or "") + (j.company or "") + " ".join(j.tags or [])
        ).lower()
    ]

    if not filtered_jobs:
        st.warning("Không có job nào khớp với từ khóa.")
        return

    selected: set = st.session_state.get("selected_jobs", set())
    is_full = len(selected) >= MAX_SELECT

    for i, job in filtered_jobs:
        _render_job_card(job, i, i in selected, is_full)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if not selected:
        st.info(f"Chọn từ 1 đến {MAX_SELECT} job để chấm điểm CV của bạn.")
        return

    selected_jobs_list = [(i, all_jobs[i]) for i in sorted(selected) if i < len(all_jobs)]
    st.markdown(f"**Đã chọn {len(selected_jobs_list)} job để phân tích:**")
    for _, j in selected_jobs_list:
        st.markdown(f"- **{j.title}** — {j.company}")

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        score_btn = st.button("Chấm Điểm CV vs Các Job Đã Chọn",
                              width="stretch", key="candidate_score", type="primary")

    if score_btn:
        cv_text = st.session_state.candidate_cv_text
        all_results = {}
        progress = st.progress(0, text="Bắt đầu chấm điểm...")
        for idx_in_list, (job_idx, job) in enumerate(selected_jobs_list):
            jd_text = job.toMarkdown()
            label   = f"{job.title} @ {job.company}"
            progress.progress(idx_in_list / len(selected_jobs_list),
                              text=f"Đang phân tích: {label}")
            try:
                all_results[label] = run_pipeline_from_text(cv_text, jd_text)
            except Exception as e:
                all_results[label] = {"error": str(e)}
        progress.progress(1.0, text="Hoàn tất.")
        st.session_state.job_results = all_results

    if st.session_state.get("job_results"):
        _render_ranking(st.session_state.job_results, label_col="Job", mode="candidate")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Khởi tạo các session_state
    if "mode" not in st.session_state:
        st.session_state.mode = "recruiter" # mặc định ban đầu là chế độ tuyển dụng
    if "selected_jobs" not in st.session_state:
        st.session_state.selected_jobs = set()
    if "job_results" not in st.session_state:
        st.session_state.job_results = {}

    # Header
    st.markdown("""
    <div class="app-header">
        <h1>CV Screener AI</h1>
        <p>Phân tích &amp; đánh giá CV so với Job Description bằng AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Mode switcher
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "Chế độ Tuyển dụng", width="stretch",
                key="btn_recruiter",
                type="primary" if st.session_state.mode == "recruiter" else "secondary"
            ):
                st.session_state.mode = "recruiter"
                st.rerun()
        with c2:
            if st.button(
                "Chế độ Ứng viên", width="stretch",
                key="btn_candidate",
                type="primary" if st.session_state.mode == "candidate" else "secondary"
            ):
                st.session_state.mode = "candidate"
                st.rerun()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Render chế độ tương ứng
    if st.session_state.mode == "recruiter":
        render_recruiter_mode()
    else:
        render_candidate_mode()


if __name__ == "__main__":
    main()