import asyncio
import logging
import urllib.parse
from typing import Optional

logger = logging.getLogger(__name__)

_playwright_available = False
try:
    from playwright.async_api import async_playwright
    _playwright_available = True
except ImportError:
    pass


def is_playwright_available() -> bool:
    return _playwright_available


def detect_search_query(url: str) -> Optional[tuple[str, str]]:
    """URLから検索エンジン種別とクエリを検出。戻り値: (engine, query) or None"""
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    host = parsed.netloc.lower()

    if "google." in host and "/search" in parsed.path:
        q = params.get("q", [""])[0]
        return ("google", q) if q else None

    if "bing.com" in host and "/images" in parsed.path:
        q = params.get("q", [""])[0]
        return ("bing", q) if q else None

    if "bing.com" in host and "/search" in parsed.path:
        q = params.get("q", [""])[0]
        return ("bing", q) if q else None

    return None


async def scrape_bing_images(query: str, limit: int = 20) -> list[str]:
    """Bing画像検索からURLリストを取得"""
    if not _playwright_available:
        logger.warning("playwright not installed — run: pip install playwright && playwright install chromium")
        return []

    urls: list[str] = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            )
            page = await context.new_page()
            encoded = urllib.parse.quote(query)
            await page.goto(
                f"https://www.bing.com/images/search?q={encoded}&form=HDRSC3",
                wait_until="domcontentloaded",
                timeout=30000,
            )

            # Bing は `a.iusc` の `m` 属性に JSON で murl が入っている
            img_urls: list[str] = await page.eval_on_selector_all(
                "a.iusc",
                """els => els.map(el => {
                    try { return JSON.parse(el.getAttribute('m')).murl }
                    catch(e) { return null }
                }).filter(Boolean)""",
            )

            # フォールバック: img タグの src
            if not img_urls:
                img_urls = await page.eval_on_selector_all(
                    "img.mimg",
                    "els => els.map(el => el.src).filter(s => s && !s.startsWith('data:'))",
                )

            urls = [u for u in img_urls if u][:limit]
            await browser.close()
    except Exception as e:
        logger.error(f"Bing image scrape failed: {e}")

    return urls


async def scrape_google_images(query: str, limit: int = 20) -> list[str]:
    """Google画像検索からURLリストを取得（Bot対策があるため不安定）"""
    if not _playwright_available:
        logger.warning("playwright not installed")
        return []

    urls: list[str] = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
                locale="ja-JP",
            )
            page = await context.new_page()
            encoded = urllib.parse.quote(query)
            await page.goto(
                f"https://www.google.com/search?q={encoded}&udm=2",
                wait_until="networkidle",
                timeout=15000,
            )

            # Google 画像の実URLは data-src / src に入っている場合がある
            img_urls: list[str] = await page.eval_on_selector_all(
                "img",
                """els => els.map(el => el.src || el.getAttribute('data-src'))
                    .filter(s => s && !s.startsWith('data:') && s.startsWith('http'))""",
            )

            urls = [u for u in img_urls if u][:limit]
            await browser.close()
    except Exception as e:
        logger.error(f"Google image scrape failed: {e}")

    return urls


async def scrape_any_page_images(url: str, limit: int = 30) -> list[str]:
    """任意URLをPlaywrightでJSレンダリングして画像URL一覧を取得"""
    if not _playwright_available:
        return []

    urls: list[str] = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=20000)

            img_urls: list[str] = await page.eval_on_selector_all(
                "img",
                """els => els.map(el => el.src || el.getAttribute('data-src'))
                    .filter(s => s && !s.startsWith('data:') && s.startsWith('http'))""",
            )

            urls = [u for u in img_urls if u][:limit]
            await browser.close()
    except Exception as e:
        logger.error(f"Playwright scrape failed for {url}: {e}")

    return urls


async def scrape_images_from_url(url: str, limit: int = 30) -> list[str]:
    """URL種別を自動判定して最適な方法でスクレイピング"""
    detected = detect_search_query(url)
    if detected:
        engine, query = detected
        if engine == "bing":
            return await scrape_bing_images(query, limit)
        if engine == "google":
            # Google は Bot 対策が厳しいため Bing で代替検索
            bing_results = await scrape_bing_images(query, limit)
            if bing_results:
                return bing_results
            return await scrape_google_images(query, limit)

    return await scrape_any_page_images(url, limit)
