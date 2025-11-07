from models import Article
from services.scrapers import BaseScraper, MarketWatchScraper, YahooScraper, DdgScraper
from utils import function_timer, logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from services.llm import gemini_client

ARTICLE_LIMIT=2
ASYNCHRONOUS = True
llm_summary = True
llm = gemini_client if llm_summary else None
yahoo_scraper = YahooScraper(limit=ARTICLE_LIMIT, async_scrape=ASYNCHRONOUS, gemini_client=llm)
marketwatch_scraper = MarketWatchScraper(limit=ARTICLE_LIMIT, async_scrape=ASYNCHRONOUS, gemini_client=llm)
base_scraper = BaseScraper(limit=ARTICLE_LIMIT, async_scrape=ASYNCHRONOUS, gemini_client=llm)
ddg_scraper = DdgScraper(limit=ARTICLE_LIMIT, async_scrape=ASYNCHRONOUS, gemini_client=llm)
@function_timer
def fetch_ddg_news():
    from duckduckgo_search import DDGS

    links = []

    query = "stock market news"
    ddgs = DDGS(headers=base_scraper.headers)

    results = ddgs.news(query, max_results=100, safesearch="off", timelimit="d")
    links = [result["url"] for result in results]
    print(links)
    articles = []
    articles.extend(base_scraper.scrape(urls=links, manual_fetch=True))
    sorted_articles = sorted(articles, key=lambda article: article.publish_date or "", reverse=True)
    return sorted_articles


@function_timer
def fetch_latest_news():
    articles: list[Article] = []
    if not ASYNCHRONOUS:
        articles.extend(yahoo_scraper.scrape())
        articles.extend(marketwatch_scraper.scrape())
        articles.extend(ddg_scraper.scrape())

    else:
        with ThreadPoolExecutor() as executor:
            futures = []
            threads = [
                executor.submit(yahoo_scraper.scrape),
                executor.submit(marketwatch_scraper.scrape),
                executor.submit(ddg_scraper.scrape),

            ]
            for t in threads:
                futures.append(t)
                
            for future in as_completed(futures):
                result = future.result()
                articles.extend(result)
    
    sorted_articles = sorted(articles, key=lambda article: article.publish_date or "", reverse=True)
    return sorted_articles


def main():
    articles_fetched = []
    results = fetch_latest_news()
    articles_fetched.extend(results)
    
    # ddg_results = fetch_ddg_news()
    # articles_fetched.extend(ddg_results)

    
    print(f"Total articles fetched: {len(articles_fetched)}")
    for art in articles_fetched[:]:
        print(f"\nPublish date: {art.publish_date}, \nTitle: {art.title}, \nURL: {art.url}, \nSummary: {art.summary}\n"
              # f"Content: {art.content}\n"
              )

if __name__ == "__main__":
    main()
    # fetch_ddg_news()
    # print(gemini_client.send_text_request("hello"))
    # print("hello")
    pass