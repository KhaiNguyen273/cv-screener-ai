import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .categories import get_category_url
from .job_detail_scraper import _make_driver, scrape_job_detail
from .job_parser import convert_html_2_listObj
from .top_cv_type import Job
from .url_builder import build_search_url

MAX_DETAIL_WORKERS = 5


def _fetch_jobs_sync(page: int, category_url: str) -> List[Job]:
    driver = _make_driver()
    try:
        url = build_search_url(base_url=category_url, page=page)
        print(f"[DEBUG] Mở trang danh sách: {url}")
        driver.get(url)

        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".job-item-search-result"))
            )
        except Exception:
            print("[DEBUG] TIMEOUT: không thấy job list")
            return []

        jobs = convert_html_2_listObj(driver.page_source)
        print(f"[DEBUG] Parse được {len(jobs)} jobs")
        return jobs

    finally:
        driver.quit()


def _fetch_details_parallel(jobs: List[Job]) -> List[Job]:
    print(f"[DEBUG] Lấy detail song song ({MAX_DETAIL_WORKERS} workers)...")
    with ThreadPoolExecutor(max_workers=MAX_DETAIL_WORKERS) as pool:
        futures = {pool.submit(scrape_job_detail, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                job.detail = future.result()
            except Exception as e:
                print(f"  [ERR] future {job.id}: {e}")
                job.detail = ""
    return jobs


async def get_jobs_list(category: str = "IT / Công nghệ", page: int = 1) -> List[Job]:
    loop = asyncio.get_event_loop()
    category_url = get_category_url(category)

    with ThreadPoolExecutor(max_workers=1) as pool:
        jobs = await loop.run_in_executor(pool, _fetch_jobs_sync, page, category_url)

    if not jobs:
        return []

    with ThreadPoolExecutor(max_workers=1) as pool:
        jobs = await loop.run_in_executor(pool, _fetch_details_parallel, jobs)

    return jobs
