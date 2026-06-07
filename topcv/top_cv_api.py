import asyncio
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from .top_cv_type import Job

CATEGORY_FAMILY = "r257~b258"
MAX_DETAIL_WORKERS = 5

# Các pattern cần xóa — thêm tùy ý
_APPLY_PATTERNS = [
    r"(?i)#{1,3}\s*cách thức ứng tuyển.*",          # Tiêu đề markdown trở đi
    r"(?i)\*\*cách thức ứng tuyển\*\*.*",
    r"(?i)ứng viên nộp hồ sơ trực tuyến.*?bấm\s+ứng tuyển ngay[^\n]*\n?",
    r"(?i)hạn nộp hồ sơ\s*:.*",                     # tuỳ chọn
]

JD_SELECTORS = [
    ".job-description",
    "#job-description",
    "[class*='job-description']",
    ".job-detail__information-detail--content",
    ".detail-row--content",
    "[class*='description-content']",
    ".content-text",
    "section.job-detail",
]


def _strip_apply_section(text: str) -> str:
    """Xóa đoạn 'Cách thức ứng tuyển' và các dòng boilerplate."""
    if not text:
        return text
    for pattern in _APPLY_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    # Xóa dòng trắng thừa
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_driver():
    # Cấu hình Chrome chạy ẩn (không hiện cửa sổ trình duyệt)
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    # Tránh lỗi bộ nhớ dùng chung khi chạy trong Docker / Linux server
    options.add_argument("--disable-dev-shm-usage")
    # Ẩn dấu hiệu tự động hóa để tránh bị TopCV chặn
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    # Giả lập trình duyệt thật để tránh bị block
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    # Tự động tải đúng phiên bản ChromeDriver khớp với Chrome đang cài
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    # Xóa thuộc tính navigator.webdriver — trang web dùng thuộc tính này để phát hiện bot
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def convert_html_2_listObj(html: str) -> list[Job]:
    # Đưa HTML thô vào BeautifulSoup để dễ tìm kiếm phần tử theo CSS selector
    soup = BeautifulSoup(html, "html.parser")
    jobs: list[Job] = []

    # Mỗi div .job-item-search-result là 1 job card trên trang danh sách
    for job_div in soup.select(".job-item-search-result"):
        job = {}
        job["id"] = job_div.get("data-job-id")

        title_tag = job_div.select_one("h3.title a span")
        link_tag  = job_div.select_one("h3.title a")
        job["title"] = title_tag.get_text(strip=True) if title_tag else None
        raw_url = link_tag["href"] if link_tag else None
        # Xóa query params (?utm_source=...) khỏi URL để giữ URL sạch khi scrape detail
        job["url"] = raw_url.split("?")[0] if raw_url else None

        company_tag  = job_div.select_one(".company-name")
        company_link = job_div.select_one("a.company")
        job["company"]     = company_tag.get_text(strip=True) if company_tag else None
        job["company_url"] = company_link["href"] if company_link else None

        salary_tag    = job_div.select_one(".title-salary")
        job["salary"] = salary_tag.get_text(strip=True) if salary_tag else None

        city_tag        = job_div.select_one(".city-text")
        job["location"] = city_tag.get_text(strip=True) if city_tag else None

        exp_tag           = job_div.select_one(".exp span")
        job["experience"] = exp_tag.get_text(strip=True) if exp_tag else None

        job["tags"] = [
            t.get_text(strip=True) for t in job_div.select(".tag a.item-tag")
        ]

        time_tag           = job_div.select_one(".label-update")
        job["time_posted"] = time_tag.get_text(strip=True) if time_tag else None

        # data-src thay vì src vì TopCV dùng lazy-load ảnh
        img_tag          = job_div.select_one(".avatar img")
        job["image_url"] = img_tag.get("data-src") if img_tag else None

        jobs.append(Job(**job))

    return jobs


def _scrape_one_job_detail(job: Job) -> str:
    if not job.url:
        return ""
    # Mỗi thread tạo driver riêng — Chrome driver không thể dùng chung giữa các thread
    driver = _make_driver()
    try:
        driver.get(job.url)
        try:
            # TopCV render bằng JS — nếu lấy page_source ngay thì HTML chưa có nội dung
            # chờ tối đa 8s đến khi thẻ <body> xuất hiện thì chạy tiếp
            WebDriverWait(driver, 8).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except Exception:
            pass

        # Chờ thêm 0.5s để đảm bảo JS render xong nội dung JD bên trong <body>
        time.sleep(0.5)
        # Đưa HTML thô vào BeautifulSoup để tìm kiếm phần tử chứa JD
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Thử lần lượt từng selector — TopCV không nhất quán CSS class giữa các loại job
        for selector in JD_SELECTORS:
            el = soup.select_one(selector)
            # Kiểm tra độ dài > 100 ký tự để loại bỏ các element rỗng hoặc quá ngắn
            if el and len(el.get_text(strip=True)) > 100:
                print(f"  [OK] {job.id} dùng selector '{selector}'")
                # Chuyển HTML → Markdown bằng md() để dễ đưa vào LLM hơn HTML thô
                return _strip_apply_section(md(str(el)))

        print(f"  [MISS] {job.id} — không tìm được selector JD")
        return ""

    except Exception as e:
        print(f"  [ERR] {job.id}: {e}")
        return ""
    finally:
        # Đảm bảo driver luôn được đóng dù có lỗi hay không — tránh rò rỉ tiến trình Chrome
        driver.quit()


# 3.1 lấy danh sách các công việc
def _fetch_jobs_sync(page: int) -> list[Job]:
    driver = _make_driver()
    try:
        url = (
            f"https://www.topcv.vn/tim-viec-lam-cong-nghe-thong-tin-cr257"
            f"?page={page}&category_family={CATEGORY_FAMILY}"
        )
        print(f"[DEBUG] Mở trang danh sách: {url}")
        driver.get(url)

        try:
            # TopCV render bằng JS — chờ tối đa 15s đến khi thấy ít nhất 1 job card
            # dùng CSS selector thay vì TAG_NAME vì cần đảm bảo job đã được render ra DOM
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".job-item-search-result"))
            )
        except Exception:
            print("[DEBUG] TIMEOUT: không thấy job list")
            return []

        # Lúc này HTML đã có đầy đủ job → parse được
        jobs = convert_html_2_listObj(driver.page_source)
        print(f"[DEBUG] Parse được {len(jobs)} jobs")
        return jobs

    finally:
        # Đảm bảo driver luôn được đóng dù có lỗi hay không
        driver.quit()


# 3.2 lấy chi tiết công việc
def _fetch_details_parallel(jobs: list[Job]) -> list[Job]:
    print(f"[DEBUG] Lấy detail song song ({MAX_DETAIL_WORKERS} workers)...")
    # Tạo pool 5 thread chạy song song — mỗi thread scrape 1 job detail cùng lúc
    with ThreadPoolExecutor(max_workers=MAX_DETAIL_WORKERS) as pool:
        # Dict {future: job} — để khi future hoàn thành biết job nào vừa xong
        futures = {pool.submit(_scrape_one_job_detail, job): job for job in jobs}
        # as_completed: future nào xong trước xử lý trước — không chờ theo thứ tự submit
        for future in as_completed(futures):
            job = futures[future]
            try:
                job.detail = future.result()
            except Exception as e:
                print(f"  [ERR] future {job.id}: {e}")
                job.detail = ""
    return jobs


# 3. Cào danh sách job
async def get_jobs_list(page: int = 1) -> list[Job]:
    loop = asyncio.get_event_loop()

    # 3.1 lấy danh sách các công việc
    # run_in_executor: đẩy hàm blocking (Selenium) vào thread riêng
    # để event loop của Streamlit không bị chặn trong lúc chờ
    # max_workers=1 vì chỉ cần 1 driver để tải trang danh sách
    with ThreadPoolExecutor(max_workers=1) as pool:
        jobs = await loop.run_in_executor(pool, _fetch_jobs_sync, page)

    if not jobs:
        return []

    # 3.2 lấy chi tiết công việc
    # max_workers=1 ở đây vì _fetch_details_parallel đã tự tạo pool 5 thread bên trong
    # không cần thêm lớp song song ngoài
    with ThreadPoolExecutor(max_workers=1) as pool:
        jobs = await loop.run_in_executor(pool, _fetch_details_parallel, jobs)

    return jobs