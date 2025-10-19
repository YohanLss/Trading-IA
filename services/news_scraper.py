from services.scrapers.yahoo_news_scraper import YahooNewsScraper 
from utils.custom_timer import CustomTimer  # optional if you want timing
from services.scrapers.marketwatch_news_scraper import MarketWatchNewsScraper
def test_yahoo_scraper():
    timer = CustomTimer()
    timer.start()

    scraper = YahooNewsScraper(limit=10, async_scrape=True)
    results = scraper.scrape_news()

    # Example print
    for art in results:
        print(f"\nTitle: {art['title']}\nURL: {art['url']}\nPublish Date: {art['publish_date']}\nAuthors: {art['authors']}\n\n")
        
        print(art["content"][:300], "...\n")

    timer.stop()


def test_marketwatch_scraper():
    timer = CustomTimer()
    timer.start()
    scraper = MarketWatchNewsScraper(limit=28, async_scrape=False)
    results = scraper.scrape_news()
    for art in results:
        print(f"\nTitle: {art['title']}\nURL: {art['url']}\nPublish Date: {art['publish_date']}\nAuthors: {art['authors']}\n")

        print("Content: ", art["content"][:300], "...\n")

    timer.stop()

if __name__ == "__main__":
    # content = extract_article_text("https://finance.yahoo.com/news/social-security-cola-increase-2026-133749845.html")
    # pass
    # newspaper_test()
    test_yahoo_scraper()
    # test_marketwatch_scraper()

