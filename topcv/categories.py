from typing import Dict, List

CATEGORY_CONFIGS: Dict[str, Dict[str, object]] = {
    "IT / Công nghệ": {
        "url": "https://www.topcv.vn/tim-viec-lam-cong-nghe-thong-tin-cr257",
        "keywords": [
            "python",
            "java",
            "c++",
            "frontend",
            "backend",
            "devops",
            "data",
            "mobile",
        ],
    },
    "Marketing / Truyền thông": {
        "url": "https://www.topcv.vn/tim-viec-lam-truyen-thong-marketing",
        "keywords": [
            "seo",
            "facebook ads",
            "content marketing",
            "digital marketing",
            "pr",
            "brand",
        ],
    },
    "Thiết kế / UX": {
        "url": "https://www.topcv.vn/tim-viec-lam-ux-designer",
        "keywords": [
            "ux",
            "ui",
            "product design",
            "graphic design",
            "prototype",
        ],
    },
    "Kinh doanh / Quản lý": {
        "url": "https://www.topcv.vn/tim-viec-lam-quan-ly-kinh-doanh",
        "keywords": [
            "sales",
            "business development",
            "account manager",
            "project manager",
            "operation",
        ],
    },
    "Kế toán / Tài chính": {
        "url": "https://www.topcv.vn/tim-viec-lam-ke-toan-tai-chinh",
        "keywords": [
            "accounting",
            "finance",
            "audit",
            "tax",
            "controller",
        ],
    },
    "Nhân sự (HR)": {
        "url": "https://www.topcv.vn/tim-viec-lam-hr",
        "keywords": [
            "recruitment",
            "talent acquisition",
            "training",
            "people operations",
        ],
    },
    "Bán hàng (Sales)": {
        "url": "https://www.topcv.vn/tim-viec-lam-sales",
        "keywords": [
            "sales",
            "account executive",
            "business development",
            "customer acquisition",
        ],
    },
    "Hỗ trợ khách hàng (Customer Service)": {
        "url": "https://www.topcv.vn/tim-viec-lam-customer-service",
        "keywords": [
            "customer support",
            "call center",
            "help desk",
            "service advisor",
        ],
    },
}

DEFAULT_CATEGORY_URL = CATEGORY_CONFIGS["IT / Công nghệ"]["url"]


def get_category_url(category_name: str) -> str:
    """Trả về URL ngành tương ứng hoặc dùng mặc định nếu không tìm thấy."""
    config = CATEGORY_CONFIGS.get(category_name)
    return config["url"] if config else DEFAULT_CATEGORY_URL


def get_category_keywords(category_name: str) -> List[str]:
    """Trả về danh sách keywords cho ngành, dùng rỗng nếu không tìm thấy."""
    config = CATEGORY_CONFIGS.get(category_name)
    return config.get("keywords", []) if config else []
