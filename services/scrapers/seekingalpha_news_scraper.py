import re
from typing import List, Optional
from urllib.parse import urlencode
import time

import newspaper
import requests

from models import Article
from services.scrapers.base_news_scraper import BaseScraper

class SeekingAlphaScraper(BaseScraper):
    BASE_API_URL = "https://seekingalpha.com/api/v3"
    BASE_URL = "https://seekingalpha.com"
    DEFAULT_HEADERS = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    def __init__(self, **kwargs):
        super().__init__(base_url=self.BASE_URL, scraper_name="Seeking Alpha Scraper", **kwargs)
        self.base_api_url = self.BASE_API_URL
        self.session = requests.Session()
        # instance-scoped headers to avoid class-level surprises
        self.headers = dict(self.DEFAULT_HEADERS)

    def _build_news_url(
        self,
        category: str = "market-news::all",
        since: int = 0,
        until: int = 0,
        page_size: int = 25,
        page_number: int = 1,
        include_tickers: bool = True,
    ) -> str:
        query = {
            "fields[news]": ",".join([
                "title","date","comment_count","content",
                "primaryTickers","secondaryTickers","tag","gettyImageUrl","publishOn",
            ]),
            "fields[tag]": "slug,name",
            "filter[category]": category,
            "filter[since]": since,
            "filter[until]": until,
            "isMounting": "true",
            "page[size]": page_size,
            "page[number]": page_number,
        }
        if include_tickers:
            query["include"] = "primaryTickers,secondaryTickers"
        return f"{self.BASE_API_URL}/news?{urlencode(query)}"

    def _fetch_json(self, url: str, max_retries: int = 1) -> Optional[dict]:
        backoff = 1.0
        for attempt in range(max_retries + 1):
            r = self.session.get(url, headers=self.headers, timeout=15)
            ct = r.headers.get("content-type", "")
            if r.status_code == 200 and ct.startswith("application/json"):
                data = r.json()
                # PerimeterX heuristic: look for known fields and bail
                if isinstance(data, dict) and ("appId" in data or "captcha" in str(data)):
                    return None
                return data
            if r.status_code in (403, 429):
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return None
            return None
        return None

    def get_article_links(self) -> List[str]:
        url = self._build_news_url(page_size=self.limit)
        payload = self._fetch_json(url)
        print(payload)
        if not payload or "data" not in payload or not isinstance(payload["data"], list):
            # handle fallback or just return empty list
            print("Error fetching data from Seeking Alpha.")
            return []
        print(f"fetched {len(payload['data'])} articles")
        headlines = [headline for headline in payload["data"]]
        return [self.base_url+item.get("links", {}).get("self", "") for item in payload["data"] if isinstance(item, dict)]

    def extract_article(self, url: str) -> Article | None:
        base, slug = url.split("/news/", 1)
        api_url = self.base_api_url + "/news/" + slug
                
        payload = self._fetch_json(api_url)
        
        if not payload or "data" not in payload or not isinstance(payload["data"], dict):
            # handle fallback or just return empty list
            print("Error fetching data from Seeking Alpha.")
            return None
        
        attributes = payload["data"].get("attributes", "")
        api_html = payload["data"].get("attributes", "").get("content", "")

        html = self.fetch_html(url)
        if not html:
            return None
        
        api_news_article = newspaper.article(url=url, input_html=api_html)
        news_article = newspaper.article(url=url, input_html=html)
        news_article.nlp() 
        
        content = api_news_article.text.strip()
        single_line = re.sub(r"\s+", " ", content).strip()
        content = single_line        
        title = attributes.get("title", "")
        publish_date = news_article.publish_date
        authors=news_article.authors
        
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


if __name__ == "__main__":
    scraper = SeekingAlphaScraper()
    # links = scraper.get_article_links()
    
    # articles = scraper.scrape()
    url = 'https://seekingalpha.com/news/4518980-lemonade-baldwin-insurance-surge-circle-internet-upstart-drop-weeks-financials-wrap'
    article = scraper.extract_article_data(url)
    # print(article)

    # url = "https://seekingalpha.com/api/v3/news?fields[news]=title%2Cdate%2Ccomment_count%2Ccontent%2CprimaryTickers%2CsecondaryTickers%2Ctag%2CgettyImageUrl%2CpublishOn&fields[tag]=slug%2Cname&filter[category]=market-news%3A%3Aall&filter[since]=0&filter[until]=0&include=primaryTickers%2CsecondaryTickers&isMounting=true&page[size]=25&page[number]=1"

    # ua = UserAgent()
    # headers = {'User-Agent': ua.random}
    # response = requests.get(url, headers=headers)
    pass
