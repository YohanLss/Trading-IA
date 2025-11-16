from __future__ import annotations

from typing import List, Optional
from urllib.parse import urljoin

import newspaper
from bs4 import BeautifulSoup
import requests

from services.scrapers.base_news_scraper import BaseScraper, Article


class France24Scraper(BaseScraper):
    """
    Scrapes the Éco/Tech section published on france24.fr.
    """

    BASE_URL = "https://www.france24.com/fr/%C3%A9co-tech/"
    SITE_ROOT = "https://www.france24.com"
    HEADERS = {
        "User-Agent": "curl/8.4.0",
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        "Accept": "*/*",
    }

    def __init__(self, **kwargs):
        super().__init__(
            base_url=self.BASE_URL,
            scraper_name="France24 Scraper",
            headers=self.HEADERS,
            **kwargs,
        )

    def fetch_html(self, url: str) -> Optional[str]:
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException:
            return None

    def get_article_links(self) -> List[str]:
        html = self.fetch_html(self.BASE_URL)
        if not html:
            return []

        soup = BeautifulSoup(html, "lxml")
        candidates = soup.select("a[data-article-item-link][href]")

        links: List[str] = []
        seen: set[str] = set()

        for a_tag in candidates:
            href = a_tag.get("href")
            if not href:
                continue

            absolute = urljoin(self.SITE_ROOT, href)
            url = self.normalize_url(absolute)

            if url in seen or "/live/" in url or self.is_video_url(url):
                continue

            seen.add(url)
            links.append(url)

            if len(links) >= self.limit:
                break

        return links

    def extract_article(self, url: str) -> Optional[Article]:
        html = self.fetch_html(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "lxml")

        title = self._text_or_none(soup.select_one("h1.t-content__title"))
        summary = self._text_or_none(soup.select_one("p.t-content__chapo"))

        publish_time = soup.select_one("div.t-content__dates time")
        publish_date = publish_time.get("datetime") if publish_time else None

        authors = [
            self._text_or_none(tag)
            for tag in soup.select(".t-content__authors .m-author__name__title")
        ]
        authors = [a for a in authors if a]

        body = self._extract_body_text(soup)

        news_article = newspaper.article(url=url, input_html=html)
        news_article.nlp()

        title = title or news_article.title
        summary = summary or news_article.summary
        publish_date = publish_date or (
            str(news_article.publish_date) if news_article.publish_date else None
        )
        authors = authors or news_article.authors
        body = body or news_article.text

        article = Article(
            url=self.normalize_url(url),
            title=title or "",
            content=body or "",
            publish_date=publish_date,
            authors=authors or None,
            summary=summary,
        )

        if self.gemini_client:
            try:
                update = self.gemini_client.summarize_article(article)
                if update:
                    article = article.model_copy(update=update)
            except Exception:
                # Gemini summary failed; keep extracted summary
                pass

        return article

    @staticmethod
    def _text_or_none(node) -> Optional[str]:
        if not node:
            return None
        text = node.get_text(strip=True)
        return text or None

    def _extract_body_text(self, soup: BeautifulSoup) -> str:
        content_root = soup.select_one("[data-article-content]")
        if not content_root:
            return ""

        paragraphs: List[str] = []
        for p in content_root.select("p"):
            classes = p.get("class") or []
            if "t-content__chapo" in classes or "m-pub-dates" in classes:
                continue

            if p.find_parent(class_="t-content__metadata"):
                continue

            if p.find_parent("figure"):
                continue

            text = p.get_text(strip=True)
            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)

if __name__ == "__main__":
    scraper = France24Scraper()
    links = scraper.scrape()
    print(links)
