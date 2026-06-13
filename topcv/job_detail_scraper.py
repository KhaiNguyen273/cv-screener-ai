import re
import time
from typing import Optional

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

_APPLY_PATTERNS = [
    r"(?i)#{1,3}\s*cách thức ứng tuyển.*",
    r"(?i)\*\*cách thức ứng tuyển\*\*.*",
    r"(?i)ứng viên nộp hồ sơ trực tuyến.*?bấm\s+ứng tuyển ngay[^\n]*\n?",
    r"(?i)hạn nộp hồ sơ\s*:.*",
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
    if not text:
        return text
    for pattern in _APPLY_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def scrape_job_detail(job: Job, wait_seconds: float = 0.5) -> str:
    if not job.url:
        return ""

    driver = _make_driver()
    try:
        driver.get(job.url)
        try:
            WebDriverWait(driver, 8).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except Exception:
            pass

        time.sleep(wait_seconds)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        for selector in JD_SELECTORS:
            element = soup.select_one(selector)
            if element and len(element.get_text(strip=True)) > 100:
                print(f"  [OK] {job.id} dùng selector '{selector}'")
                return _strip_apply_section(md(str(element)))

        print(f"  [MISS] {job.id} — không tìm được selector JD")
        return ""
    except Exception as e:
        print(f"  [ERR] {job.id}: {e}")
        return ""
    finally:
        driver.quit()
