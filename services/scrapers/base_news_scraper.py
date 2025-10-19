# base_scraper.py
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed

import re
from typing import List, Optional, Dict, Iterable, Tuple
from urllib.parse import urlsplit, urlunsplit

import newspaper
import requests
from newspaper.network import session

from bs4 import BeautifulSoup
from pydantic import BaseModel

from models import Article
# from services.llm import gemini_client

from utils.logger import logger

class BaseScraper:
    """
    A reusable base for news scrapers.
    Subclasses should implement `get_article_links` and can optionally override `extract_article`.
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(
        self,
        limit: int = 10,
        max_workers: int = 6,
        async_scrape: bool = False,        
        base_url: Optional[str] = None,
        scraper_name: Optional[str] = None,
        gemini_client = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.gemini_client = gemini_client
        self.scraper_name = scraper_name
        self.limit = limit
        self.max_workers = max_workers
        self.async_scrape = async_scrape
        self.headers = {**(headers or self.DEFAULT_HEADERS)}
        self.base_url = base_url  # Optional hint for robots and link building


    def scrape(self) -> List[Article]:
        """
        Orchestrates: get links -> extract articles (seq or threaded).
        Subclasses can reuse as-is.
        """
        logger.info(f"{self.scraper_name or ''}: Fetching article links...")
            
        links = self.get_article_links()
        if not links:
            logger.info("No links found.")
            return []

        logger.info(f"{self.scraper_name or ''}: Found {len(links)} articles.")

        if self.async_scrape:
            logger.info(f"{self.scraper_name or ''}: Starting async extraction of articles...")

            articles = self._extract_many_threaded(links)
        else:
            articles = []
            for i, link in enumerate(links[: self.limit], start=1):
                logger.info(f"{self.scraper_name or ''}: [{i}/{min(len(links), self.limit)}] Scraping: {link}")
                art = self.extract_article(link)
                if art and art.content:
                    articles.append(art)

        logger.info(f"{self.scraper_name or ''}: Done. Scraped {len(articles)} article(s).")
        return articles

    # ---------- Methods for subclasses to implement or override ----------

    def get_article_links(self) -> List[str]:
        """
        Return a list of absolute article URLs. The base class enforces limit.
        Subclasses must implement this.
        """
        raise NotImplementedError

    def extract_article(self, url: str) -> Optional[Article]:
        """
        Extract one article. You can override this to use site-specific selectors.
        The default calls `extract_article_default` which tries common patterns.
        """
        return self.extract_article_default(url)

    # ---------- Helpful utilities for subclasses ----------

    # HTML fetch with retries, timeouts, robots, and per-host delay
    def fetch_html(self, url: str) -> Optional[str]:

        try:
            r = session.get(url, headers=self.headers)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None

    @staticmethod
    def normalize_url(url: str) -> str:
        parts = urlsplit(url)
        # Strip query and fragment
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    @staticmethod
    def soup(html: str) -> BeautifulSoup:
        # lxml is faster if installed, html.parser is fine too
        return BeautifulSoup(html, "lxml")

    @staticmethod
    def is_video_url(url: str) -> bool:
        return "/video/" in url or "/videos/" in url

    # ---------- Default extractor (override for per-site logic) ----------

    def extract_article_default(self, url: str) -> Optional[Article]:
        html = self.fetch_html(url)
        if not html:
            return None

        article = newspaper.article(url=url, input_html=html)
        article.nlp()
        
        title = article.title
        content = article.text.strip()
        single_line = re.sub(r"\s+", " ", content).strip()
        content = single_line
        publish_date = article.publish_date
        authors = article.authors
        summary = article.summary
        return Article(
            url=self.normalize_url(url),
            title=title or "",
            content=content or "",
            publish_date=str(publish_date) if publish_date else None,
            authors=authors or None,
            summary=summary,
        )

    # ---------- Internals ----------

    def _extract_many_threaded(self, links: List[str]) -> List[Article]:

        links = links[: self.limit]
        out_by_index: List[Optional[Article]] = [None] * len(links)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            fut_map = {pool.submit(self.extract_article, link): (i, link) for i, link in enumerate(links)}
            for fut in as_completed(fut_map):
                i, link = fut_map[fut]
                try:
                    art = fut.result()
                    if art and art.content:
                        out_by_index[i] = art
                except Exception as e:
                    logger.warning(f"Worker failed for {link}: {e}")

        return [a for a in out_by_index if a]
