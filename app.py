"""
app.py — CV Screener AI
Chế độ Tuyển Dụng : so sánh nhiều CV vs 1 JD (giữ nguyên)
Chế độ Ứng Viên   : upload 1 CV, cào job TopCV, chọn tối đa 5 job, chấm điểm & xếp hạng
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.genai_helper import generate_analysis_report
from core.nlp_engine import extract_entities
from core.pdf_parser import parse_cv
from core.scoring_algo import score_cv_vs_jd
from utils.text_cleaner import clean_text
from topcv import get_jobs_list  # scraper TopCV
from topcv.categories import CATEGORY_CONFIGS

# Cấu hình trang
st.set_page_config(page_title="CV Screener AI", page_icon="🎯", layout="wide")

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }

    /* Mode switcher */
    .mode-bar {
        display: flex; gap: 12px; justify-content: center;
        margin-bottom: 2rem;
    }
    .mode-btn {
        padding: 10px 32px; border-radius: 24px; font-weight: 600;
        font-size: 0.95rem; cursor: pointer; border: 2px solid #2d3250;
        background: #1e2130; color: #8b92a5; transition: all .2s;
    }
    .mode-btn.active {
        background: linear-gradient(135deg,#667eea,#764ba2);
        border-color: transparent; color: #fff;
    }

    /* Score cards */
    .score-card { background: linear-gradient(135deg,#1e2130,#252a3d); border:1px solid #2d3250; border-radius:16px; padding:1.5rem; text-align:center; margin-bottom:1rem; }
    .score-card .score-value { font-size:2.8rem; font-weight:700; background:linear-gradient(135deg,#667eea,#764ba2); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .score-card .score-label { color:#8b92a5; font-size:.85rem; margin-top:4px; }

    /* Verdict badges */
    .verdict-badge { display:inline-block; padding:6px 18px; border-radius:20px; font-weight:600; font-size:.9rem; margin-bottom:8px; }
    .verdict-high   { background:#1a3a2a; color:#4ade80; border:1px solid #166534; }
    .verdict-medium { background:#2d2a1a; color:#fbbf24; border:1px solid #92400e; }
    .verdict-low    { background:#2d1a1a; color:#f87171; border:1px solid #991b1b; }

    /* Skill tags */
    .skill-tag { display:inline-block; padding:4px 12px; border-radius:12px; font-size:.78rem; font-weight:500; margin:3px; }
    .skill-matched { background:#1a3a2a; color:#4ade80; border:1px solid #166534; }
    .skill-missing { background:#2d1a1a; color:#f87171; border:1px solid #991b1b; }

    /* Question cards */
    .question-card { background:#1a1d2e; border-left:3px solid #667eea; border-radius:0 8px 8px 0; padding:10px 16px; margin-bottom:8px; }
    .question-category { font-size:.72rem; color:#667eea; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
    .question-text { color:#c9d1d9; font-size:.9rem; margin-top:4px; }

    /* Section headers */
    .section-header { font-size:1rem; font-weight:600; color:#c9d1d9; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid #2d3250; }

    /* Job card */
    .job-card {
        background:#1a1d2e; border:1px solid #2d3250; border-radius:12px;
        padding:1rem 1.2rem; margin-bottom:10px; transition: border-color .2s;
    }
    .job-card:hover { border-color:#667eea; }
    .job-card.selected { border-color:#4ade80; background:#1a2a1e; }
    .job-title { color:#c9d1d9; font-weight:600; font-size:.95rem; margin:0 0 4px 0; }
    .job-meta  { color:#8b92a5; font-size:.8rem; }
    .job-tag   { display:inline-block; background:#252a3d; color:#8b92a5; padding:2px 8px; border-radius:8px; font-size:.72rem; margin:2px; }

    /* Buttons */
    div[data-testid="stButton"] > button {
        background:linear-gradient(135deg,#667eea,#764ba2); color:#fff;
        border:none; border-radius:8px; padding:.6rem 2rem;
        font-weight:600; font-size:1rem; width:100%; transition:opacity .2s;
    }
    div[data-testid="stButton"] > button:hover { opacity:.85; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _gauge_chart(value: float, title: str) -> go.Figure:
    color = "#4ade80" if value >= 70 else "#fbbf24" if value >= 45 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": "%", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#555"},
            "bar": {"color": color}, "bgcolor": "#1e2130", "bordercolor": "#2d3250",
            "steps": [
                {"range": [0,  45], "color": "#1a1a2e"},
                {"range": [45, 70], "color": "#1a1a2e"},
                {"range": [70,100], "color": "#1a1a2e"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": .75, "value": value},
        },
        title={"text": title, "font": {"size": 13, "color": "#8b92a5"}},
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#c9d1d9")
    return fig


def _verdict_class(verdict: str) -> str:
    v = verdict.lower()
    if "cao" in v or "ngay" in v:   return "verdict-high"
    if "trung bình" in v or "dự phòng" in v: return "verdict-medium"
    return "verdict-low"


def run_pipeline(jd_text: str, cv_file) -> dict:
    """Chạy toàn bộ pipeline CV vs JD. cv_file là UploadedFile của Streamlit."""
    suffix = Path(cv_file.name).suffix.lower()
    #Tạo ra một file tạm thời trên hệ thống lưu trữ của máy tính
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        #Đọc và ghi vào file tạm
        tmp.write(cv_file.read())
        #Lưu đường dẫn vào biến tạm
        tmp_path = tmp.name
    try:
        #Lưu nội dung vào biến
        raw_cv = parse_cv(tmp_path)
    finally:
        #Xóa file tạm thời khỏi hệ thống lưu trữ
        os.unlink(tmp_path)

    #Làm sạch cv jd
    clean_cv  = clean_text(raw_cv)
    clean_jd  = clean_text(jd_text)
    #Trích xuất thực thể
    cv_ent    = extract_entities(clean_cv, source="cv")
    jd_ent    = extract_entities(clean_jd, source="jd")
    #Chấm điểm
    scores    = score_cv_vs_jd(clean_cv, clean_jd, cv_ent, jd_ent)
    #Phân tích
    report    = generate_analysis_report(cv_text=clean_cv, jd_text=clean_jd,
                                         scores=scores, cv_entities=cv_ent,
                                         jd_entities=jd_ent)
    return {"scores": scores, "cv_entities": cv_ent,
            "jd_entities": jd_ent, "report": report}


def run_pipeline_from_text(cv_text: str, jd_text: str) -> dict:
    """Giống run_pipeline nhưng nhận text thô thay vì file upload (dùng cho chế độ ứng viên)."""
    #Làm sạch cv jd
    clean_cv  = clean_text(cv_text)
    clean_jd  = clean_text(jd_text)
    #Trích xuất thực thể
    cv_ent    = extract_entities(clean_cv, source="cv")
    jd_ent    = extract_entities(clean_jd, source="jd")
    #Chấm điểm
    scores    = score_cv_vs_jd(clean_cv, clean_jd, cv_ent, jd_ent)
    #Phân tích
    report    = generate_analysis_report(cv_text=clean_cv, jd_text=clean_jd,
                                         scores=scores, cv_entities=cv_ent,
                                         jd_entities=jd_ent)
    return {"scores": scores, "cv_entities": cv_ent,
            "jd_entities": jd_ent, "report": report}


def _render_candidate_detail(result: dict, candidate_key: str):
    """Dashboard chi tiết cho 1 ứng viên (dùng trong cả 2 chế độ)."""
    scores   = result["scores"]
    cv_ent   = result["cv_entities"]
    report   = result["report"]

    verdict = report.get("verdict", "N/A")
    st.markdown(
        f"""<div style="text-align:center;margin-bottom:1.5rem;">
            <div class="verdict-badge {_verdict_class(verdict)}">{verdict}</div>
            <p style="color:#8b92a5;font-size:.9rem;margin:4px 0 0 0;">
                {report.get('verdict_reason','')}
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    g1,g2,g3,g4 = st.columns(4)
    # Mỗi tuple: (cột, giá trị điểm, tiêu đề, key unique tránh lỗi DuplicateElementId)
    charts = [
        (g1, scores["final_score"],    "Điểm Tổng",  f"gauge_total_{candidate_key}"),
        (g2, scores["semantic_score"], "Ngữ Nghĩa",  f"gauge_sem_{candidate_key}"),
        (g3, scores["skill_score"],    "Kỹ Năng",    f"gauge_skill_{candidate_key}"),
        (g4, scores["experience_score"],"Kinh Nghiệm",f"gauge_exp_{candidate_key}"),
    ]
    for col, val, title, key in charts:
        with col:
            st.plotly_chart(_gauge_chart(val, title),
                            width='stretch', key=key)

    left, right = st.columns([3,2], gap="large")
    with left:
        st.markdown('<div class="section-header">📝 Nhận Xét Tổng Quan</div>', unsafe_allow_html=True)
        st.markdown(f"<p style='color:#c9d1d9;line-height:1.7;font-size:.95rem;'>{report.get('overall_summary','')}</p>", unsafe_allow_html=True)
        for icon, color, key_name, label in [
            ("💪","#4ade80","strengths","Điểm Mạnh"),
            ("⚠️","#f87171","weaknesses","Điểm Yếu / Khoảng Cách"),
            ("🚀","#fbbf24","development_suggestions","Gợi Ý Phát Triển"),
        ]:
            items = report.get(key_name, [])
            if items:
                st.markdown(f'<div class="section-header">{icon} {label}</div>', unsafe_allow_html=True)
                for item in items:
                    st.markdown(f"<p style='color:{color};font-size:.9rem;margin:4px 0;'>→ {item}</p>", unsafe_allow_html=True)

    with right:
        matched = scores.get("matched_skills", [])
        missing = scores.get("missing_skills", [])
        st.markdown(f'<div class="section-header">✅ Kỹ Năng Khớp ({len(matched)})</div>', unsafe_allow_html=True)
        if matched:
            st.markdown("".join(f'<span class="skill-tag skill-matched">{s}</span>' for s in matched), unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#8b92a5;font-size:.85rem;'>Không tìm thấy kỹ năng khớp</p>", unsafe_allow_html=True)

        st.markdown(f'<div class="section-header" style="margin-top:1rem;">❌ Kỹ Năng Còn Thiếu ({len(missing)})</div>', unsafe_allow_html=True)
        if missing:
            st.markdown("".join(f'<span class="skill-tag skill-missing">{s}</span>' for s in missing), unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#8b92a5;font-size:.85rem;'>Ứng viên đáp ứng đầy đủ kỹ năng</p>", unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1rem;">🎓 Học Vấn</div>', unsafe_allow_html=True)
        for edu in cv_ent.get("education", [])[:3]:
            st.markdown(f"<p style='color:#c9d1d9;font-size:.85rem;margin:2px 0;'>• {edu}</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🎤 Câu Hỏi Phỏng Vấn Gợi Ý</div>', unsafe_allow_html=True)
    questions = report.get("interview_questions", [])
    if questions:
        q_cols = st.columns(2)
        for i, q in enumerate(questions):
            with q_cols[i % 2]:
                st.markdown(
                    f"""<div class="question-card">
                        <div class="question-category">{q.get('category','Chung')}</div>
                        <div class="question-text">{q.get('question','')}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    hiring_rec = report.get("hiring_recommendation","")
    if hiring_rec:
        rec_color = "#4ade80" if "ngay" in hiring_rec.lower() else "#fbbf24" if "dự phòng" in hiring_rec.lower() else "#f87171"
        st.markdown(
            f"""<div style="background:#1e2130;border:1px solid #2d3250;border-radius:12px;
                        padding:1.2rem;text-align:center;margin-top:1.5rem;">
                <p style="color:#8b92a5;font-size:.85rem;margin:0 0 6px 0;">KHUYẾN NGHỊ</p>
                <p style="color:{rec_color};font-size:1.2rem;font-weight:700;margin:0;">{hiring_rec}</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # with st.expander("🔍 Xem dữ liệu JSON thô (Debug)"):
    #     st.json(result)


def _render_ranking(all_results: dict, label_col: str = "CV"):
    """Vẽ bảng xếp hạng + tabs chi tiết. Dùng chung cho cả 2 chế độ."""
    st.markdown("### 🏆 Bảng Xếp Hạng")
    rows = []
    for name, res in all_results.items():
        if "error" in res:
            rows.append({label_col: name, "Điểm Tổng":"❌","Ngữ Nghĩa":"-","Kỹ Năng":"-","Kinh Nghiệm":"-","Kết Luận":res["error"],"Khuyến Nghị":"-"})
        else:
            s, r = res["scores"], res["report"]
            rows.append({
                label_col: name,
                "Điểm Tổng": f"{s['final_score']:.0f}%",
                "Ngữ Nghĩa": f"{s['semantic_score']:.0f}%",
                "Kỹ Năng":   f"{s['skill_score']:.0f}%",
                "Kinh Nghiệm": f"{s['experience_score']:.0f}%",
                "Kết Luận":  r.get("verdict","N/A"),
                "Khuyến Nghị": r.get("hiring_recommendation","N/A"),
            })
    df = pd.DataFrame(rows)
    # Sắp xếp: có điểm số đứng trên, lỗi xuống cuối
    valid   = df[df["Điểm Tổng"] != "❌"].sort_values("Điểm Tổng", ascending=False)
    invalid = df[df["Điểm Tổng"] == "❌"]
    st.dataframe(pd.concat([valid, invalid]), width='stretch', hide_index=True)

    # Tab chi tiết — 1 tab / ứng viên hoặc 1 tab / job
    st.markdown("### 📋 Chi Tiết")
    tabs = st.tabs(list(all_results.keys()))
    for tab, (name, res) in zip(tabs, all_results.items()):
        with tab:
            if "error" in res:
                st.error(f"❌ Lỗi: {res['error']}")
            else:
                # Dùng index số làm key để tránh lỗi DuplicateElementId khi tên file có ký tự đặc biệt
                safe_key = str(list(all_results.keys()).index(name))
                _render_candidate_detail(res, candidate_key=safe_key)


# ══════════════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ TUYỂN DỤNG
# ══════════════════════════════════════════════════════════════════════════════

def render_recruiter_mode():
    col_jd, col_cv = st.columns([1,1], gap="large")
    with col_jd:
        st.markdown("#### 📋 Job Description")
        jd_text = st.text_area(
            label="JD", height=300,
            placeholder="Dán toàn bộ nội dung Job Description vào đây...",
            label_visibility="collapsed",
            key="recruiter_jd",
        )
    with col_cv:
        st.markdown("#### 📄 CV Ứng Viên (nhiều file)")
        cv_files = st.file_uploader(
            label="CV", type=["pdf","docx"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="recruiter_cv",
        )
        if cv_files:
            st.success(f"✅ Đã tải lên **{len(cv_files)}** file: {', '.join(f.name for f in cv_files)}")

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1,2,1])
    with mid:
        analyze_btn = st.button("🚀 Phân Tích Ngay", width='stretch', key="recruiter_analyze")

    st.markdown("<hr style='border-color:#2d3250;margin:1.5rem 0;'>", unsafe_allow_html=True)

    if analyze_btn:
        if not jd_text.strip():
            st.error("⚠️ Vui lòng nhập nội dung Job Description.")
            return
        if not cv_files:
            st.error("⚠️ Vui lòng tải lên ít nhất 1 file CV.")
            return

        #Xử lý từng CV với progress bar
        all_results = {}
        progress = st.progress(0, text="Bắt đầu phân tích...")
        for i, cv_file in enumerate(cv_files):
            progress.progress(i / len(cv_files), text=f"⚙️ Đang xử lý: **{cv_file.name}**")
            try:
                #hạy pipeline, lưu kết quả vào all_results với key là tên file
                all_results[cv_file.name] = run_pipeline(jd_text, cv_file)
            except Exception as e:
                all_results[cv_file.name] = {"error": str(e)}
        progress.progress(1.0, text=f"✅ Hoàn tất {len(cv_files)} CV!")

        #Xây bảng ranking + tab chi tiết
        _render_ranking(all_results, label_col="CV")


# ══════════════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ ỨNG VIÊN
# ══════════════════════════════════════════════════════════════════════════════

MAX_SELECT = 5  # tối đa chọn bao nhiêu job để chấm


def _render_job_card(job, idx: int, is_selected: bool, is_full: bool):
    """Render 1 job card với nút chọn / bỏ chọn và nút xem chi tiết."""
    card_class = "job-card selected" if is_selected else "job-card"
    tags_html  = "".join(f'<span class="job-tag">{t}</span>' for t in (job.tags or [])[:6])

    st.markdown(
        f"""<div class="{card_class}">
            <p class="job-title">💼 {job.title or 'N/A'}</p>
            <p class="job-meta">
                🏢 {job.company or 'N/A'} &nbsp;|&nbsp;
                📍 {job.location or 'N/A'} &nbsp;|&nbsp;
                💰 {job.salary or 'Thỏa thuận'} &nbsp;|&nbsp;
                🕐 {job.time_posted or ''}
            </p>
            <div style="margin-top:6px;">{tags_html}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    btn_col, detail_col = st.columns([1,1])
    with btn_col:
        if is_selected:
            # Nút bỏ chọn — xóa index khỏi set
            if st.button("✅ Bỏ chọn", key=f"sel_{idx}", width='stretch'):
                st.session_state.selected_jobs.discard(idx)
                st.rerun()
        else:
            # Disable nút nếu đã đủ MAX_SELECT job
            disabled = is_full and not is_selected
            label    = f"☑ Chọn ({len(st.session_state.selected_jobs)}/{MAX_SELECT})" if not is_full else "🔒 Đã đủ 5 job"
            if st.button(label, key=f"sel_{idx}", width='stretch', disabled=disabled):
                st.session_state.selected_jobs.add(idx)
                st.rerun()

    with detail_col:
        # Toggle mở/đóng chi tiết — nhấn lần 2 sẽ đóng lại
        if st.button("🔍 Xem chi tiết", key=f"det_{idx}", width='stretch'):
            st.session_state.expanded_job = idx if st.session_state.get("expanded_job") != idx else None
            st.rerun()

    # Chi tiết mở rộng
    if st.session_state.get("expanded_job") == idx:
        with st.expander("", expanded=True):
            if job.detail:
                st.markdown(job.detail[:3000] + ("..." if len(job.detail) > 3000 else ""))
            else:
                st.warning("⚠️ Không lấy được mô tả chi tiết cho job này.")

def render_candidate_mode():
    #upload CV
    st.markdown("#### 📄 Upload CV của bạn")
    cv_file = st.file_uploader(
        label="CV", type=["pdf","docx"],
        label_visibility="collapsed",
        key="candidate_cv",
        accept_multiple_files=False,
    )
    if cv_file:
        st.success(f"✅ Đã tải lên: **{cv_file.name}**")

    #Parse CV text ngay khi upload (cache bằng session_state để không parse lại khi rerun)
    if cv_file and st.session_state.get("candidate_cv_name") != cv_file.name:
        suffix = Path(cv_file.name).suffix.lower()
        # Tạo ra một file tạm thời trên hệ thống lưu trữ của máy tính
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # Đọc và ghi vào file tạm
            tmp.write(cv_file.read())
            # Lưu đường dẫn vào biến tạm
            tmp_path = tmp.name
        try:
            # Lưu nội dung vào biến
            raw = parse_cv(tmp_path)
        finally:
            # Xóa file tạm thời khỏi hệ thống lưu trữ
            os.unlink(tmp_path)

        #buộc dùng session_state vì là luồng nhiều bước bấm một nút, tải lên một file, gõ chữ thì app.py sẽ chạy lại 
        st.session_state.candidate_cv_text = clean_text(raw)
        st.session_state.candidate_cv_name = cv_file.name
        st.session_state.pop("jobs_list", None)
        st.session_state.selected_jobs = set()
        st.session_state.job_results   = {}

    st.markdown("<hr style='border-color:#2d3250;margin:1.5rem 0;'>", unsafe_allow_html=True)

    if not st.session_state.get("candidate_cv_text"):
        st.info("👆 Hãy upload CV để tiếp tục.")
        return

    #Cào danh sách job
    col_fetch, col_page = st.columns([3,1])
    with col_fetch:
        selected_industry = st.selectbox(
            "Chọn ngành TopCV",
            list(CATEGORY_CONFIGS.keys()),
            key="topcv_industry",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button(
            f"🔄 Tải danh sách Job {selected_industry} từ TopCV",
            width='stretch', key="fetch_jobs",
        )
    with col_page:
        page_num = st.number_input("Trang TopCV", min_value=1, max_value=20,
                                   value=1, step=1, key="topcv_page")

    if fetch_btn:
        with st.spinner(f"⚙️ Đang cào trang {page_num}..."):
            try:
                jobs = asyncio.run(get_jobs_list(category=selected_industry, page=page_num))
                st.session_state.jobs_list     = jobs
                st.session_state.selected_jobs = set()
                st.session_state.job_results   = {}
            except Exception as e:
                st.error(f"❌ Lỗi khi cào TopCV: {e}")
                return

    all_jobs: list = st.session_state.get("jobs_list", [])
    if not all_jobs:
        return

    # Chỉ giữ job lấy được JD
    # Dùng dict {index_gốc: job} để index trong selected_jobs không bị lệch
    jobs_map = {
        i: j for i, j in enumerate(all_jobs)
        if j.detail and len(j.detail.strip()) > 50
    }

    total_all    = len(all_jobs)
    total_with_jd = len(jobs_map)

    if not jobs_map:
        st.warning("⚠️ Không có job nào lấy được mô tả chi tiết. Thử tải lại hoặc chọn trang khác.")
        return

    st.markdown(
        f"<p style='color:#8b92a5;font-size:.85rem;margin-bottom:1rem;'>"
        f"Tìm thấy <b style='color:#c9d1d9;'>{total_with_jd}</b> job có JD "
        f"(bỏ qua {total_all - total_with_jd} job không lấy được mô tả)</p>",
        unsafe_allow_html=True,
    )

    # Danh sách job + chọn
    selected: set = st.session_state.get("selected_jobs", set())
    is_full = len(selected) >= MAX_SELECT

    st.markdown(f"#### 📋 Danh sách Job ({total_with_jd} việc làm) — Chọn tối đa {MAX_SELECT}")

    keyword = st.text_input("🔎 Lọc theo từ khóa (tiêu đề, công ty, tag)",
                             key="job_filter", placeholder="python, react, hà nội...")
    filtered_jobs = [
        (i, j) for i, j in jobs_map.items()
        if not keyword or keyword.lower() in (
            (j.title or "") + (j.company or "") + " ".join(j.tags or [])
        ).lower()
    ]

    if not filtered_jobs:
        st.warning("Không có job nào khớp với từ khóa.")
        return

    for i, job in filtered_jobs:
        _render_job_card(job, i, i in selected, is_full)

    st.markdown("<hr style='border-color:#2d3250;margin:1.5rem 0;'>", unsafe_allow_html=True)

    # Chấm điểm
    if not selected:
        st.info(f"👆 Chọn từ 1 đến {MAX_SELECT} job để chấm điểm CV của bạn.")
        return

    selected_jobs_list = [(i, all_jobs[i]) for i in sorted(selected) if i < len(all_jobs)]
    st.markdown(f"#### ✅ Đã chọn {len(selected_jobs_list)} job để phân tích")
    for _, j in selected_jobs_list:
        st.markdown(f"- **{j.title}** — {j.company}")

    _, mid, _ = st.columns([1,2,1])
    with mid:
        score_btn = st.button("🚀 Chấm Điểm CV vs Các Job Đã Chọn",
                              width='stretch', key="candidate_score")

    if score_btn:
        # Lấy text CV đã parse từ session_state — tránh parse lại file mỗi lần rerun
        cv_text = st.session_state.candidate_cv_text
        all_results = {}
        progress = st.progress(0, text="Bắt đầu chấm điểm...")

        for idx_in_list, (job_idx, job) in enumerate(selected_jobs_list):
            # Chuyển Job object thành chuỗi Markdown để đưa vào pipeline như JD text
            jd_text = job.toMarkdown()
            label   = f"{job.title} @ {job.company}"
            # Cập nhật progress bar — idx_in_list/total cho ra % từ 0.0 đến <1.0
            progress.progress(
                idx_in_list / len(selected_jobs_list),
                text=f"⚙️ Đang phân tích: **{label}**",
            )
            try:
                # Chạy toàn bộ pipeline, lưu kết quả với key là "Tên job @ Công ty"
                all_results[label] = run_pipeline_from_text(cv_text, jd_text)
            except Exception as e:
                # Không để 1 job lỗi làm crash toàn bộ — ghi lỗi rồi chạy tiếp
                all_results[label] = {"error": str(e)}

        # Kéo progress bar lên 100% sau khi xong tất cả
        progress.progress(1.0, text="✅ Hoàn tất!")
        # Lưu vào session_state để kết quả không mất khi Streamlit rerun
        st.session_state.job_results = all_results

    # Kết quả xếp hạng
    # Dùng .get() thay vì truy cập thẳng — tránh KeyError nếu chưa chấm lần nào
    if st.session_state.get("job_results"):
        _render_ranking(st.session_state.job_results, label_col="Job")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Khởi tạo session state — chỉ chạy lần đầu, các lần rerun sau giữ nguyên giá trị
    if "mode" not in st.session_state:
        st.session_state.mode = "recruiter"       # chế độ mặc định
    if "selected_jobs" not in st.session_state:
        st.session_state.selected_jobs = set()    # set index các job đã chọn
    if "job_results" not in st.session_state:
        st.session_state.job_results = {}         # kết quả chấm điểm job

    # Header
    st.markdown("""
        <div style="text-align:center;padding:2rem 0 1rem 0;">
            <h1 style="font-size:2.4rem;font-weight:700;color:#c9d1d9;margin:0;">
                🎯 CV Screener
                <span style="background:linear-gradient(135deg,#667eea,#764ba2);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI</span>
            </h1>
            <p style="color:#8b92a5;margin-top:8px;font-size:1rem;">
                Phân tích & đánh giá CV so với Job Description bằng AI
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Mode switcher
    # type="primary" làm nút sáng lên để biết đang ở chế độ nào
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🏢 Chế Độ Tuyển Dụng", width='stretch',
                         key="btn_recruiter",
                         type="primary" if st.session_state.mode == "recruiter" else "secondary"):
                st.session_state.mode = "recruiter"
                st.rerun()
        with c2:
            if st.button("👤 Chế Độ Ứng Viên", width='stretch',
                         key="btn_candidate",
                         type="primary" if st.session_state.mode == "candidate" else "secondary"):
                st.session_state.mode = "candidate"
                st.rerun()

    # Mô tả chế độ hiện tại
    if st.session_state.mode == "recruiter":
        st.markdown(
            "<p style='text-align:center;color:#8b92a5;font-size:.88rem;margin-bottom:1.5rem;'>"
            "📌 Upload nhiều CV — nhập 1 JD — xếp hạng ứng viên phù hợp nhất</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='text-align:center;color:#8b92a5;font-size:.88rem;margin-bottom:1.5rem;'>"
            "📌 Upload CV của bạn — duyệt job từ TopCV — chọn job phù hợp — xem điểm khớp</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#2d3250;margin:.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)

    # Render chế độ tương ứng
    if st.session_state.mode == "recruiter":
        render_recruiter_mode()   # nhiều CV vs 1 JD
    else:
        render_candidate_mode()   # 1 CV vs nhiều job từ TopCV


if __name__ == "__main__":
    main()
