import re
from urllib.parse import urljoin
from itertools import count

import newspaper
from bs4 import BeautifulSoup
from typing import List, Optional
from services.scrapers.base_news_scraper import BaseScraper, Article
from utils.function_timer import function_timer
from utils.logger import logger

class MarketWatchScraper(BaseScraper):
    BASE_URL = "https://www.marketwatch.com/latest-news"
    
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    def __init__(self, **kwargs):
        super().__init__(base_url=self.BASE_URL, scraper_name="MarketWatch Scraper", **kwargs)
    
    @staticmethod
    def generate_latest_news_url(page_count=0) -> str:
        """Generate the latest news URL"""
        return f"https://www.marketwatch.com/latest-news?pageNumber={page_count}&position=1.1.0&partial=true"

    def get_article_links(self) -> List[str]:
        """Fetch Article Links from base url"""
        links, seen = [], set()

        for page_count in count(start=0):
            base_url = self.generate_latest_news_url(page_count)
            html = self.fetch_html(base_url)
            if not html:
                break

            soup = BeautifulSoup(html, "lxml")
            container_sel = "div.tab_pane.is-active[data-tab-pane='MarketWatch'] div.collection__elements"
            item_link_sel = (
                f"{container_sel} div.element.element--article "
                ".article__content h3.article__headline a.link"
            )

            candidates = soup.select(item_link_sel) or soup.select(
                "div.element.element--article h3.article__headline a.link"
            )

            for a in candidates:
                href = a.get("href")
                if not href:
                    continue

                url = self.normalize_url(urljoin(self.BASE_URL, href))
                if "www.marketwatch.com" not in url or url in seen:
                    continue

                seen.add(url)
                links.append(url)

                if len(links) >= self.limit:
                    return links
                
            if "see more" not in soup.text.lower():
                return links

        return links

    def extract_article(self, url: str) -> Optional[Article]:
        """
        Extract one article. 
        """
        html = self.fetch_html(url)
        if not html:
            return None

        news_article = newspaper.article(url=url, input_html=html)

        news_article.nlp() 
        content = news_article.text.strip()
        single_line = re.sub(r"\s+", " ", content).strip()
        content = single_line
        title = news_article.title
        publish_date = news_article.publish_date
        authors = news_article.authors
        
        article = Article(
            url=self.normalize_url(url),
            title=title or "",
            content=content or "",
            publish_date=str(publish_date) if publish_date else None,
            authors=authors or None,
            summary=None,
        )

        try:
            article.summary = self.gemini_client.summarize_article(article)
        except Exception as e:
            article.summary = news_article.summary

        return article
        
@function_timer
def fetch_marketwatch_articles():
    scraper = MarketWatchScraper()
    articles = scraper.scrape()
    for article in articles:
        logger.info(article)


@function_timer
def async_fetch_marketwatch_articles():
    scraper = MarketWatchScraper(limit=100, async_scrape=True)
    articles = scraper.scrape()
    # for article in articles:
    #     logger.info(article)


if __name__ == "__main__":
    # fetch_marketwatch_articles()
    async_fetch_marketwatch_articles()
    # url = "https://www.wsj.com/world/trump-says-hed-rather-end-war-than-send-tomahawks-to-ukraine-04956387"
    # scraper = MarketWatchScraper()
    # article = scraper.extract_article(url)
    # print(article)