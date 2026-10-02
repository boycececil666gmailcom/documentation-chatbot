# region Imports
import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

# endregion

# region Configuration
MAX_DEPTH = 3
MAX_CONCURRENCY = 30
CSS_SELECTOR = "article"
OUTPUT_FILE = Path(__file__).resolve().parent / "1.scrape.json"
ROOT_URLS = [
    "https://docs.kanzi.com/4.1.0/en/kanzi-fundamentals/kanzi-fundamentals.html",
    "https://docs.kanzi.com/4.1.0/en/working-with/working-with.html",
    "https://docs.kanzi.com/4.1.0/en/best-practices/best-practices.html",
]
# endregion


# region Helpers
def normalize_url(url: str) -> str:
    return url.split("#")[0].strip()


def get_section_prefix(url: str) -> str:
    parts = urlparse(url).path.strip("/").split("/")
    return (
        f"https://{urlparse(url).netloc}/{'/'.join(parts[:3])}/"
        if len(parts) >= 3
        else url.rsplit("/", 1)[0] + "/"
    )


def get_page_title(markdown: str, fallback_url: str) -> str:
    for line in markdown.splitlines():
        if line.strip().startswith("#"):
            return line.replace("#", "").split("[¶]")[0].strip()
    slug = (
        fallback_url.rstrip("/").split("/")[-1].replace(".html", "").replace("-", " ")
    )
    return slug.title() or "Doc Page"


# endregion


# region Crawler
class DocCrawler:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        self.visited = set()
        self.prefixes = [get_section_prefix(u) for u in ROOT_URLS]
        self.run_config = CrawlerRunConfig(css_selector=CSS_SELECTOR)

    def is_allowed(self, url: str) -> bool:
        return any(url.startswith(p) for p in self.prefixes)

    async def crawl_node(
        self, crawler: AsyncWebCrawler, url: str, depth: int = 1
    ) -> dict | None:
        url = normalize_url(url)
        if not self.is_allowed(url) or url in self.visited:
            return None
        self.visited.add(url)

        print(f"{'  ' * (depth - 1)}[Depth {depth}/{MAX_DEPTH}] Crawling: {url}")
        async with self.semaphore:
            res = await crawler.arun(url=url, config=self.run_config)
            if not res.success:
                return None

        internal_links = [
            normalize_url(item["href"])
            for item in (res.links or {}).get("internal", [])
            if item.get("href")
        ]
        external_links = [
            item["href"].strip()
            for item in (res.links or {}).get("external", [])
            if item.get("href")
        ]

        node = {
            "url": url,
            "title": get_page_title(res.markdown or "", url),
            "depth_level": depth,
            "markdown_content": res.markdown or "",
            "links": {
                "internal": list(dict.fromkeys(internal_links)),
                "external": list(dict.fromkeys(external_links)),
            },
            "sub_documents": [],
        }

        if depth < MAX_DEPTH and internal_links:
            child_urls = [
                u for u in dict.fromkeys(internal_links) if self.is_allowed(u)
            ]
            children = await asyncio.gather(
                *[self.crawl_node(crawler, u, depth + 1) for u in child_urls]
            )
            node["sub_documents"] = [c for c in children if c]

        return node

    async def run(self) -> list[dict]:
        async with AsyncWebCrawler() as crawler:
            results = await asyncio.gather(
                *[self.crawl_node(crawler, u, 1) for u in ROOT_URLS]
            )
            return [r for r in results if r]


# endregion


# region Main
async def main() -> None:
    print(f"[Crawler] Scraping docs ({len(ROOT_URLS)} roots)...")
    crawler = DocCrawler()
    trees = await crawler.run()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(trees, f, ensure_ascii=False, indent=2)
    print(
        f"[Crawler] Completed: saved {len(trees)} trees ({len(crawler.visited)} pages) to {OUTPUT_FILE.name}"
    )


if __name__ == "__main__":
    asyncio.run(main())
# endregion
