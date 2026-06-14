def build_search_url(base_url: str, page: int = 1) -> str:
    """Xây dựng URL tìm kiếm TopCV dựa trên base_url ngành và page."""
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}page={page}"
