from services.scrapers import BaseScraper, MarketWatchScraper, YahooScraper, Article
from utils import function_timer, logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.llm import gemini_client


ARTICLE_LIMIT=5
ASYNCHRONOUS = True
llm_summary = True
llm = gemini_client if llm_summary else None
yahoo_scraper = YahooScraper(limit=ARTICLE_LIMIT, async_scrape=ASYNCHRONOUS, gemini_client=llm)
marketwatch_scraper = MarketWatchScraper(limit=ARTICLE_LIMIT, async_scrape=ASYNCHRONOUS, gemini_client=llm)


@function_timer
def fetch_latest_news():
    articles: list[Article] = []
    if not ASYNCHRONOUS:
        articles.extend(yahoo_scraper.scrape())
        articles.extend(marketwatch_scraper.scrape())
        
    else:
        with ThreadPoolExecutor() as executor:
            futures = []
            threads = [
                executor.submit(yahoo_scraper.scrape),
                executor.submit(marketwatch_scraper.scrape),
            ]
            for t in threads:
                futures.append(t)
                
            for future in as_completed(futures):
                result = future.result()
                articles.extend(result)
    
    sorted_articles = sorted(articles, key=lambda article: article.publish_date or "", reverse=True)
    return sorted_articles


def main():
    results = fetch_latest_news()
    print(f"Total articles fetched: {len(results)}")
    for art in results[:2]:
        print(f"\nPublish date: {art.publish_date}, \nTitle: {art.title}, \nURL: {art.url}, \nSummary: {art.summary}\n"
              # f"Content: {art.content}\n"
              )

if __name__ == "__main__":
    main()
    # print(gemini_client.send_text_request("hello"))
    # print("hello")
    pass