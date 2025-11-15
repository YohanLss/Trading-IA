from typing import List
from urllib.parse import urlencode

import requests

from services.scrapers import BaseScraper


class BenzingaNewsScraper(BaseScraper):
    DEFAULT_HEADERS = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    BASE_API_URL = "https://www.benzinga.com/api"
    BASE_URL = "https://www.benzinga.com"
    
    def __init__(self, **kwargs):
        super().__init__(scraper_name="Benzinga Scraper", **kwargs)
        
        self.session = requests.Session()
        self.headers = dict(self.DEFAULT_HEADERS)

    
    def _build_news_url(
        self, limit: int = None
    ) -> str:
        if limit is None:
            limit = self.limit
            
        query = {
            "limit":limit
        }
        
        return f"{self.BASE_API_URL}/news?{urlencode(query)}"
    
    def get_article_links(self) -> List[str]:
        url = "https://www.benzinga.com/api/news"
        url = self._build_news_url()
        r = self.session.get(url, headers=self.headers, timeout=15)
        ct = r.headers.get("content-type", "")
        links = []
        if r.status_code == 200 and ct.startswith("application/json"):
            data = r.json()
            if isinstance(data, list) and len(data)> 0:
                for article in data:
                    links.append(article.get("url"))
        
        return links


if __name__ == "__main__":
    scraper = BenzingaNewsScraper()
    # links = scraper.get_article_links()
    # 
    # print(links)

    articles = scraper.scrape()
    for art in articles:
        print(
            f"Title: {art.title}\nURL: {art.url}\nPublish Date: {art.publish_date}\nAuthors: {art.authors}\n Content: {art.content}")
    pass