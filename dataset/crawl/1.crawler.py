# region Crawler
import asyncio
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from dotenv import load_dotenv

# Setup workspace paths
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env")

CONFIG_PATH = CURRENT_DIR / "crawler_config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)

crawler_cfg = config.get("crawler", {})
OUTPUT_FILE = CURRENT_DIR / crawler_cfg.get("output_path", "1.crawler.json")


def normalize_url(url: str) -> str:
    """Strip hash fragments and whitespace from URL."""
    return url.split("#")[0].strip()


def get_section_prefix(url: str) -> str:
    """Extract documentation section base prefix from URL."""
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 3:
        return f"https://{parsed.netloc}/{'/'.join(parts[:3])}/"
    return url.rsplit("/", 1)[0] + "/"


def get_page_title(markdown: str, fallback_url: str) -> str:
    """Extract clean H1 title or derive from URL slug."""
    for line in markdown.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("#") and not trimmed.startswith("##"):
            clean_title = re.sub(r"\[.*?\]", "", trimmed.lstrip("#")).strip()
            if clean_title:
                return clean_title
    slug = fallback_url.rstrip("/").split("/")[-1].replace(".html", "").replace("-", " ")
    return slug.title() or "Documentation Page"


async def crawl_site(
    root_urls: list[str],
    max_depth: int = 3,
    max_concurrency: int = 30,
    css_selector: str = "article",
) -> list[dict]:
    """Recursively crawls documentation roots and extracts hierarchical document trees."""
    semaphore = asyncio.Semaphore(max_concurrency)
    visited: set[str] = set()
    prefixes = [get_section_prefix(u) for u in root_urls]
    run_config = CrawlerRunConfig(css_selector=css_selector)

    async def _crawl_node(crawler: AsyncWebCrawler, url: str, depth: int = 1) -> dict | None:
        url = normalize_url(url)
        if not any(url.startswith(p) for p in prefixes) or url in visited:
            return None
        visited.add(url)

        try:
            async with semaphore:
                res = await crawler.arun(url=url, config=run_config)
            if not res.success:
                return None
        except Exception as e:
            print(f"[Crawler-_crawl_node] Warning: Failed to crawl {url}: {e}")
            return None

        internal_links = [
            normalize_url(i["href"])
            for i in (res.links or {}).get("internal", [])
            if i.get("href")
        ]

        node = {
            "url": url,
            "title": get_page_title(res.markdown or "", url),
            "depth_level": depth,
            "markdown_content": res.markdown or "",
            "sub_documents": [],
        }

        if depth < max_depth and internal_links:
            children_urls = [
                u
                for u in dict.fromkeys(internal_links)
                if any(u.startswith(p) for p in prefixes) and u not in visited
            ]
            if children_urls:
                children = await asyncio.gather(
                    *[_crawl_node(crawler, u, depth + 1) for u in children_urls]
                )
                node["sub_documents"] = [c for c in children if c is not None]

        return node

    print(
        f"[Crawler-crawl_site] Crawling {len(root_urls)} roots (max depth = {max_depth}, max concurrency = {max_concurrency})..."
    )
    async with AsyncWebCrawler() as crawler:
        results = await asyncio.gather(*[_crawl_node(crawler, u, 1) for u in root_urls])
        valid_trees = [r for r in results if r is not None]

    print(
        f"[Crawler-crawl_site] Completed: scraped {len(visited)} pages across {len(valid_trees)} trees."
    )
    return valid_trees


async def main() -> None:
    """Execute site crawling and export intermediate tree artifact."""
    root_urls = crawler_cfg.get("root_urls", [])
    max_depth = crawler_cfg.get("max_depth", 3)
    max_concurrency = crawler_cfg.get("max_concurrency", 30)
    css_selector = crawler_cfg.get("css_selector", "article")

    crawled_trees = await crawl_site(
        root_urls=root_urls,
        max_depth=max_depth,
        max_concurrency=max_concurrency,
        css_selector=css_selector,
    )
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(crawled_trees, f, ensure_ascii=False, indent=2)
    print(f"[Crawler-main] Saved document tree to '{OUTPUT_FILE.name}'")


if __name__ == "__main__":
    asyncio.run(main())
# endregion
