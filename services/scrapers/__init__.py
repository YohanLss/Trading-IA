import nltk
nltk.download('punkt_tab')
from .marketwatch_news_scraper import MarketWatchScraper
from .yahoo_news_scraper import YahooScraper
from .base_news_scraper import BaseScraper
from .ddg_news_scraper import DdgScraper