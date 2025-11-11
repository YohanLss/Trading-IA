from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlencode

import requests

from services.scrapers.base_news_scraper import Article, BaseScraper
from utils.logger import logger


class NewsApiScraper(BaseScraper):
    """Scraper that bootstraps article URLs from NewsAPI before extracting full content."""

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        queries: Optional[Iterable[str]] = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 20,
        max_pages: int = 1,
        from_param: Optional[str | datetime] = None,
        to_param: Optional[str | datetime] = None,
        domains: Optional[Iterable[str]] = None,
        fetch_full_article: bool = True,
        request_timeout: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(base_url=self.BASE_URL, scraper_name="NewsAPI Scraper", **kwargs)
        self.api_key = api_key or os.getenv("NEWSAPI_KEY") or os.getenv("NEWSAPI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NewsAPI key missing. Provide `api_key` or set the NEWSAPI_KEY/NEWSAPI_API_KEY environment variable."
            )

        self.queries: List[str] = list(queries) if queries else [
            "stock market",
            "central bank",
            "earnings report",
        ]
        self.language = language
        self.sort_by = sort_by
        self.page_size = max(1, min(page_size, 100))
        self.max_pages = max(1, max_pages)
        self.from_param = self._format_datetime(from_param) or self._default_timespan(days=1)
        self.to_param = self._format_datetime(to_param)
        self.domains = ",".join(domains) if domains else None
        self.fetch_full_article = fetch_full_article
        self.request_timeout = request_timeout
        self._article_cache: Dict[str, Dict] = {}

    @staticmethod
    def _default_timespan(days: int) -> str:
        return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    @staticmethod
    def _format_datetime(value: Optional[str | datetime]) -> Optional[str]:
        if not value:
            return None
        if isinstance(value, str):
            return value
        return value.replace(microsecond=0).isoformat() + "Z"

    def _build_params(self, query: str, page: int) -> Dict[str, str | int]:
        params: Dict[str, str | int] = {
            "q": query,
            "language": self.language,
            "sortBy": self.sort_by,
            "pageSize": self.page_size,
            "page": page,
            "apiKey": self.api_key,
        }
        if self.from_param:
            params["from"] = self.from_param
        if self.to_param:
            params["to"] = self.to_param
        if self.domains:
            params["domains"] = self.domains
        return params

    def _build_url(self, query: str, page: int) -> str:
        params = self._build_params(query, page)
        return f"{self.BASE_URL}?{urlencode(params)}"

    def _call_newsapi(self, url: str) -> Optional[Dict]:
        headers = {"X-Api-Key": self.api_key}
        try:
            response = requests.get(url, headers=headers, timeout=self.request_timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning(f"NewsAPI request failed: {exc}")
            return None

        try:
            payload = response.json()
        except ValueError:
            logger.warning("NewsAPI returned non-JSON response.")
            return None

        if payload.get("status") != "ok":
            logger.warning(f"NewsAPI error: {payload.get('code')} - {payload.get('message')}")
            return None

        return payload

    def get_article_links(self) -> List[str]:
        links: List[str] = []
        seen: set[str] = set()
        self._article_cache = {}

        for query in self.queries:
            for page in range(1, self.max_pages + 1):
                url = self._build_url(query, page)
                payload = self._call_newsapi(url)
                if not payload:
                    break

                articles = payload.get("articles") or []
                if not articles:
                    break

                for raw_article in articles:
                    url = raw_article.get("url")
                    if not url:
                        continue

                    norm_url = self.normalize_url(url)
                    if norm_url in seen:
                        continue

                    seen.add(norm_url)
                    raw_article["_query"] = query
                    self._article_cache[norm_url] = raw_article
                    links.append(norm_url)

                    if len(links) >= self.limit:
                        return links

                if len(articles) < self.page_size:
                    break

            if len(links) >= self.limit:
                break

        return links

    def extract_article(self, url: str) -> Optional[Article]:
        normalized = self.normalize_url(url)
        cached = self._article_cache.get(normalized)

        if self.fetch_full_article:
            article = super().extract_article(url)
            if article and cached:
                if not article.summary and cached.get("description"):
                    article.summary = cached["description"]
                if not article.title and cached.get("title"):
                    article.title = cached["title"]
                if not article.publish_date and cached.get("publishedAt"):
                    article.publish_date = cached["publishedAt"]
                if cached.get("_query"):
                    article.keyword = cached["_query"]
                if not article.authors:
                    source_name = (cached.get("source") or {}).get("name")
                    if source_name:
                        article.authors = [source_name]
            return article

        if not cached:
            return None

        source_name = (cached.get("source") or {}).get("name")
        return Article(
            url=normalized,
            title=cached.get("title") or "",
            content=cached.get("content") or cached.get("description") or "",
            publish_date=cached.get("publishedAt"),
            authors=[source_name] if source_name else None,
            summary=cached.get("description"),
            keyword=cached.get("_query"),
        )


if __name__ == "__main__":
    scraper = NewsApiScraper()
    print(scraper.get_article_links())
