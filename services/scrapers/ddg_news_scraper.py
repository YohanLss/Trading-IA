from __future__ import annotations

from urllib.parse import urlparse
from typing import List, Dict, Any, Iterable, Optional

from ddgs import DDGS

from services.scrapers.base_news_scraper import BaseScraper, Article
from utils.function_timer import function_timer
from utils.logger import logger


class DdgScraper(BaseScraper):
    

    BASE_URL = "https://duckduckgo.com"  # placeholder for BaseScraper's bookkeeping

    def __init__(
        self,
        *,
        queries: Optional[Iterable[str]] = None,
        allowed_sources: Optional[Iterable[str]] = None,
        region: str = "fr-fr",
        max_results_per_query: int = 10,
        timelimit: Optional[str] = "w",
        safesearch: str = "off",
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        
        
        super().__init__(base_url=self.BASE_URL, scraper_name="Ddg Scraper", **kwargs)
        self.region = region
        self.max_results_per_query = max_results_per_query
        self.timelimit = timelimit
        self.safesearch = safesearch
        self.verbose = verbose

        self.queries: List[str] = list(queries) if queries else [
            "actualités de bourses",
            
        ]

        self.allowed_sources: List[str] = [s.lower() for s in (allowed_sources or [
            "boursorama.com",
            "zonebourse.com", "lefigaro.fr", "france24.com",
        ])]

        self.final_query_list = []
        for q in self.queries:
            for source in self.allowed_sources:
                self.final_query_list.append(q + " site:" + source)

        print(self.final_query_list)
    @staticmethod
    #
    def _host_in_allowed(host: str, allowed_domains: Iterable[str]) -> bool:
        
        h = host.lower()
        for d in allowed_domains:
            d = d.lower()
            if h == d or h.endswith("." + d):
                return True
        return False

    @function_timer
    def get_article_links(self) -> List[str]:
        
        collected: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()

        # 1) retourne les resultats de la recherche
        with DDGS() as ddgs:
            for q in self.final_query_list:
                try:
                    results = ddgs.news(
                        q,
                        region=self.region,
                        max_results=self.max_results_per_query,
                        safesearch=self.safesearch,
                        timelimit=self.timelimit,
                    )
                except Exception as e:
                    logger.warning(f"DDG query failed for '{q}': {e}")
                    continue

                for r in results or []:
                    url = r.get("url")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    collected.append(r)

        # 2) filtrer les resutats trouves
        kept: List[str] = []
        for r in collected:
            url = r.get("url")
            if not url:
                continue
            

            host = urlparse(url).netloc.split(":")[0]  # strip port if present
            if self._host_in_allowed(host, self.allowed_sources):
                kept.append(url)
                if self.verbose:
                    title = r.get("title", "(titre manquant)")
                    source_label = r.get("source", "") or host
                    logger.info(f"Garde: {title} | {url} | Source: {source_label}")
            else:
                if self.verbose:
                    logger.info(f"Rejete: {host} not in allowed list")

        return kept



if __name__ == "__main__":
    scraper = DdgScraper(verbose=True, limit=50, async_scrape=True)
    links = scraper.scrape()
    # for u in links:
    #     print(u)

    print(f"Total articles fetched: {len(links)}")
    for art in links[:]:
        print(f"\nPublish date: {art.publish_date}, \nTitle: {art.title}, \nURL: {art.url}, \nSummary: {art.summary}\n"
              # f"Content: {art.content}\n"
              )
