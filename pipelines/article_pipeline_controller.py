# Pipeline for fetching articles and saving them in the database
from models import Article
from services.scrapers import BaseScraper, MarketWatchScraper, YahooScraper, DdgScraper
from utils import function_timer, logger
from services.llm import gemini_client
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.database import article_service
logger.setLevel("DEBUG")

class ArticlePipelineController:
    def __init__(self, article_limit=2, asynchronous=True, llm_summary=True, verify_db=True):
        self.asynchronous = asynchronous
        self.gemini_client = gemini_client
        self.article_db_service = article_service
        
        llm = self.gemini_client if llm_summary else None

        configs = {
            "limit": article_limit,
            "async_scrape": asynchronous,
            "gemini_client": llm,
            "article_db_service": self.article_db_service,
            "verify_db": verify_db,
        }

        self.yahoo_scraper = YahooScraper(**configs)
        self.marketwatch_scraper = MarketWatchScraper(**configs)
        self.base_scraper = BaseScraper(**configs)
        self.ddg_scraper = DdgScraper(**configs)
    
    def fetch_latest_news_articles(self):
        articles : list[Article] = []
        if not self.asynchronous:
            articles.extend(self.yahoo_scraper.scrape())
            articles.extend(self.marketwatch_scraper.scrape())
            # articles.extend(self.ddg_scraper.scrape())

        else:
            with ThreadPoolExecutor() as executor:
                futures = []
                threads = [
                    executor.submit(self.yahoo_scraper.scrape),
                    executor.submit(self.marketwatch_scraper.scrape),
                    # executor.submit(self.ddg_scraper.scrape),

                ]
                for t in threads:
                    futures.append(t)

                for future in as_completed(futures):
                    result = future.result()
                    articles.extend(result)

        sorted_articles = sorted(articles, key=lambda article: article.publish_date or "", reverse=True)
        return sorted_articles
    
    @function_timer
    def run(self):
        articles_fetched : list[Article] = self.fetch_latest_news_articles()
        logger.info(f"Total articles fetched: {len(articles_fetched)}")
        
        if articles_fetched:
            result = self.article_db_service.insert_many_articles(articles_fetched)
            logger.info(f"Result: {result}")
        else:
            logger.info("No new articles fetched.")
        
        
        

def main():
    pipeline = ArticlePipelineController(article_limit=40, verify_db=True)
    pipeline.run()

if __name__ == "__main__":
    main()