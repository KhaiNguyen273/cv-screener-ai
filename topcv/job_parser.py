from bs4 import BeautifulSoup

from .top_cv_type import Job


def convert_html_2_listObj(html: str) -> list[Job]:
    """Chuyển HTML trang danh sách job TopCV thành list Job."""
    soup = BeautifulSoup(html, "html.parser")
    jobs: list[Job] = []

    for job_div in soup.select(".job-item-search-result"):
        job = {}
        job["id"] = job_div.get("data-job-id")

        title_tag = job_div.select_one("h3.title a span")
        link_tag = job_div.select_one("h3.title a")
        job["title"] = title_tag.get_text(strip=True) if title_tag else None
        raw_url = link_tag["href"] if link_tag else None
        job["url"] = raw_url.split("?")[0] if raw_url else None

        company_tag = job_div.select_one(".company-name")
        company_link = job_div.select_one("a.company")
        job["company"] = company_tag.get_text(strip=True) if company_tag else None
        job["company_url"] = company_link["href"] if company_link else None

        salary_tag = job_div.select_one(".title-salary")
        job["salary"] = salary_tag.get_text(strip=True) if salary_tag else None

        city_tag = job_div.select_one(".city-text")
        job["location"] = city_tag.get_text(strip=True) if city_tag else None

        exp_tag = job_div.select_one(".exp span")
        job["experience"] = exp_tag.get_text(strip=True) if exp_tag else None

        job["tags"] = [
            t.get_text(strip=True) for t in job_div.select(".tag a.item-tag")
        ]

        time_tag = job_div.select_one(".label-update")
        job["time_posted"] = time_tag.get_text(strip=True) if time_tag else None

        img_tag = job_div.select_one(".avatar img")
        job["image_url"] = img_tag.get("data-src") if img_tag else None

        jobs.append(Job(**job))

    return jobs
