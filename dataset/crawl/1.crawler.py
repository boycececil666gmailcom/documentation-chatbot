# region Crawler
import asyncio
import json
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
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


def extract_sidebar_trees(seed_url: str, target_sections: list[str]) -> list[dict]:
    """Parses navigation sidebar from seed_url HTML and constructs exact hierarchical document trees."""
    req = urllib.request.Request(seed_url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req).read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")
    sidebar = soup.find("div", class_="sidebar-tree")
    if not sidebar:
        raise ValueError(f"Could not locate 'sidebar-tree' container in {seed_url}")

    top_ul = sidebar.find("ul")
    if not top_ul:
        raise ValueError(f"Could not locate top-level 'ul' in sidebar of {seed_url}")

    def _parse_li(li, depth: int = 1) -> dict:
        a = li.find("a", recursive=False) or li.find("a")
        href = a.get("href", "") if a else ""
        full_url = seed_url if href == "#" or not href else urljoin(seed_url, href)
        full_url = normalize_url(full_url)
        title = a.get_text(strip=True) if a else "Untitled"

        sub_ul = li.find("ul", recursive=False)
        sub_documents = []
        if sub_ul:
            for child_li in sub_ul.find_all("li", recursive=False):
                child_node = _parse_li(child_li, depth + 1)
                if child_node:
                    sub_documents.append(child_node)

        return {
            "url": full_url,
            "title": title,
            "depth_level": depth,
            "markdown_content": "",
            "sub_documents": sub_documents,
        }

    trees = []
    for li in top_ul.find_all("li", recursive=False):
        a = li.find("a", recursive=False) or li.find("a")
        if not a:
            continue
        text = a.get_text(strip=True)
        if any(ts.lower() in text.lower() for ts in target_sections):
            section_tree = _parse_li(li, depth=1)
            trees.append(section_tree)

    return trees


async def crawl_sidebar_site(
    seed_url: str,
    target_sections: list[str],
    max_concurrency: int = 30,
    css_selector: str = "article",
) -> list[dict]:
    """Crawls exact sidebar-defined hierarchical tree by caching unique pages and populating markdown."""
    print(
        f"[Crawler-crawl_sidebar_site] Extracting sidebar structure for sections: {target_sections}..."
    )
    trees = extract_sidebar_trees(seed_url, target_sections)

    unique_urls: set[str] = set()

    def _collect_urls(node: dict) -> None:
        if node.get("url"):
            unique_urls.add(node["url"])
        for child in node.get("sub_documents", []):
            _collect_urls(child)

    for tree in trees:
        _collect_urls(tree)

    print(
        f"[Crawler-crawl_sidebar_site] Discovered {len(unique_urls)} unique pages across {len(trees)} section trees."
    )

    semaphore = asyncio.Semaphore(max_concurrency)
    run_config = CrawlerRunConfig(css_selector=css_selector)
    page_cache: dict[str, dict] = {}

    async def _fetch_page(crawler: AsyncWebCrawler, url: str) -> None:
        async with semaphore:
            try:
                res = await crawler.arun(url=url, config=run_config)
                page_cache[url] = (res.markdown or "") if res.success else ""
            except Exception as e:
                print(f"[Crawler-_fetch_page] Warning: Failed to crawl {url}: {e}")
                page_cache[url] = ""

    print(f"[Crawler-crawl_sidebar_site] Scraping content with concurrency = {max_concurrency}...")
    async with AsyncWebCrawler() as crawler:
        tasks = [_fetch_page(crawler, url) for url in unique_urls]
        await asyncio.gather(*tasks)

    def _populate_content(node: dict) -> None:
        url = node.get("url", "")
        if url in page_cache:
            node["markdown_content"] = page_cache[url]
        for child in node.get("sub_documents", []):
            _populate_content(child)

    for tree in trees:
        _populate_content(tree)

    print(f"[Crawler-crawl_sidebar_site] Content populated across all hierarchical trees.")
    return trees


async def main() -> None:
    """Execute sidebar crawling and export canonical intermediate tree artifact."""
    seed_url = crawler_cfg.get(
        "seed_url",
        "https://docs.kanzi.com/4.1.0/en/working-with/performance-profiling/profiling-application-code.html",
    )
    target_sections = crawler_cfg.get(
        "target_sections",
        ["Best practices", "Working with", "References"],
    )
    max_concurrency = crawler_cfg.get("max_concurrency", 30)
    css_selector = crawler_cfg.get("css_selector", "article")

    crawled_trees = await crawl_sidebar_site(
        seed_url=seed_url,
        target_sections=target_sections,
        max_concurrency=max_concurrency,
        css_selector=css_selector,
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(crawled_trees, f, ensure_ascii=False, indent=2)
    print(f"[Crawler-main] Saved canonical document trees to '{OUTPUT_FILE.name}'")


if __name__ == "__main__":
    asyncio.run(main())
# endregion
