from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import math
import pandas as pd
import numpy as np

from services.marketdata import YahooStockMarket
from services.database import article_service, untracked_symbols_service
from utils import function_timer


def get_tickers(hours_back: int = 4, now=None, symbol_filter=None):
    if now is None:
        now = datetime.now(timezone.utc)

    start = now - timedelta(hours=hours_back)
    start_str = start.strftime("%Y-%m-%d 00:00:00")
    now_str = now.strftime("%Y-%m-%d 23:59:59")

    filter_stage = {
        '$match': {
            "created_at": {
                "$gte": start,
                "$lte": now
            }}
    }
    if symbol_filter:
        filter_stage["$match"]["tickers"] = {"$in": symbol_filter}
    pipeline = [
        filter_stage,
        {
            '$unwind': '$tickers'
        }, {
            '$group': {
                '_id': '$tickers',
                'count': {
                    '$sum': 1
                },
                'articles': {
                    '$push': {
                        '_id': '$_id',
                        'title': '$title',
                        'url': '$url',
                        'created_at': '$created_at',
                        'publish_date': '$publish_date',
                        'sentiment': '$sentiment',
                        'authors': '$authors',
                        'summary': '$summary'
                    }
                },
                'positive_articles': {
                    '$push': {
                        '$cond': [
                            {
                                '$eq': [
                                    '$sentiment', 'positive'
                                ]
                            }, {
                                '_id': '$_id',
                                'title': '$title',
                                'url': '$url',
                                'created_at': '$created_at',
                                'publish_date': '$publish_date',
                                'sentiment': '$sentiment'
                            }, '$Remove'
                        ]
                    }
                },
                'negative_articles': {
                    '$push': {
                        '$cond': [
                            {
                                '$eq': [
                                    '$sentiment', 'negative'
                                ]
                            }, {
                                '_id': '$_id',
                                'title': '$title',
                                'url': '$url',
                                'created_at': '$created_at',
                                'publish_date': '$publish_date',
                                'sentiment': '$sentiment'
                            }, '$Remove'
                        ]
                    }
                },
                'neutral_articles': {
                    '$push': {
                        '$cond': [
                            {
                                '$eq': [
                                    '$sentiment', 'neutral'
                                ]
                            }, {
                                '_id': '$_id',
                                'title': '$title',
                                'url': '$url',
                                'created_at': '$created_at',
                                'publish_date': '$publish_date',
                                'sentiment': '$sentiment'
                            }, '$Remove'
                        ]
                    }
                }
            }
        }, {
            '$project': {
                'ticker': '$_id',
                'count': 1,
                'articles': 1,
                'positive_articles': 1,
                'negative_articles': 1,
                'neutral_articles': 1,
                '_id': 0,

            }
        }, {
            '$sort': {
                'count': -1
            }
        }

    ]
    # articles = article_service.collection.find()
    res = list(article_service.collection.aggregate(pipeline))
    untracked_symbols = untracked_symbols_service.get_untracked_symbols()
    res = res[:]
    tickers = [r for r in res if r.get("ticker") not in untracked_symbols]

    symbols = [r.get("ticker") for r in tickers]
    symbols_dfs = YahooStockMarket().get_multiple_stock_df(ticker_symbols=symbols, interval="1h", length=hours_back,
                                                           end_time=now, prepost=True)
    failed_symbols = []
    for r in tickers:
        article_count = r.get("count")
        neutral_count = len(r.get("neutral_articles"))
        positive_count = len(r.get("positive_articles"))
        negative_count = len(r.get("negative_articles"))

        symbol_score = 0
        coefficient = 0.02
        for article in r.get("positive_articles"):
            created_at = article.get("created_at")
            age = (now.replace(tzinfo=None) - created_at).total_seconds() / 60
            weight = math.exp(-age * coefficient)

            symbol_score += weight

        # for article in r.get("neutral_articles"):
        #     created_at = article.get("created_at").replace(tzinfo=timezone.utc)
        #     print(created_at)
        #     age = (now - created_at).total_seconds() / 60
        #     weight = math.exp(-age * coefficient)*0.5
        # 
        #     symbol_score += weight

        for article in r.get("negative_articles"):
            created_at = article.get("created_at")
            age = (now.replace(tzinfo=None) - created_at).total_seconds() / 60
            weight = math.exp(-age * coefficient)

            symbol_score -= weight

        r["symbol_score"] = round(symbol_score / article_count, 3)
        r["neutral_count"] = neutral_count
        r["positive_count"] = positive_count
        r["negative_count"] = negative_count
        r["neutral_percent"] = round(neutral_count / article_count, 3)
        r["positive_percent"] = round(positive_count / article_count, 3)
        r["negative_percent"] = round(negative_count / article_count, 3)
        try:
            r["return"] = round(
                (symbols_dfs[r.get("ticker")].Close.iloc[-1] / symbols_dfs[r.get("ticker")].Close.iloc[0] - 1) * 100, 3)
        except Exception as e:
            r["return"] = None
            # failed_symbols.append(r.get("ticker"))

        pass
    if failed_symbols:
        untracked_symbols_service.add_untracked_symbols(failed_symbols)
    df = pd.DataFrame([
        {
            "symbol": r.get("ticker"),
            "article_count": r.get("count"),
            "positive_sentiment": r.get("positive_percent"),
            "negative_sentiment": r.get("negative_percent"),
            "neutral_sentiment": r.get("neutral_percent"),
            "weighted_sentiment_score": (r.get("positive_percent") - r.get("negative_percent")) * math.log(
                r.get("count")),
            "symbol_score": r.get("symbol_score") * math.log(1 + r.get("count")),
            "return": r.get("return")
        }
        for r in tickers
    ])

    # df["score"] = df["weighted_sentiment_score"] * math.log(1+ df["return"])
    df["score"] = df["weighted_sentiment_score"] * df["return"]
    df["score2"] = df["symbol_score"] * df["return"]

    df = df.sort_values(by="score", ascending=False)

    return df


class TradingAgent:
    def __init__(self):
        self.today = "2025-11-18"
        self.starting_time = datetime.strptime(self.today, "%Y-%m-%d", )
        self.now = self.starting_time
        self.end_time = self.starting_time + timedelta(days=1)
        self.premarket_start = datetime.strptime(f"{self.today} 04:00", "%Y-%m-%d %H:%M")
        self.postmarket_start = datetime.strptime(f"{self.today} 20:00", "%Y-%m-%d %H:%M")
        self.increment = timedelta(minutes=1)

        self.opening_time = datetime.strptime(f"{self.today} 09:30", "%Y-%m-%d %H:%M")
        self.closing_time = datetime.strptime(f"{self.today} 16:00", "%Y-%m-%d %H:%M")
        
        self.holdings_details = {}
        
        self.current_holdings: list = []
        self.history = YahooStockMarket().get_stock_history("NVDA", return_df=True, start_time=self.starting_time,
                                                            end_time=self.end_time, interval="1m", prepost=False)
        pass

    def market_open(self):
        return self.opening_time <= self.now <= self.closing_time

    def premarket_open(self):
        return self.premarket_start <= self.now <= self.opening_time

    def postmarket_open(self):
        return self.closing_time <= self.now <= self.postmarket_start

    def get_history_row(self, time: datetime):
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M")
            row = self.history.loc[timestamp]
            return row
        except Exception as e:
            return None

    def sleep(self, x=1):
        self.now += self.increment * x

    def symbols_analysis(self):
        print("Analyzing symbols...")
        if len(self.current_holdings) == 10:
            symbols = get_tickers(hours_back=4, now=self.now, symbol_filter=self.current_holdings)
        else:
            symbols = get_tickers(hours_back=4, now=self.now)

        for index, row in symbols.iterrows():
            symbol = row["symbol"]
            score = row["score"]
            return_percentage = row["return"]
            score_2 = row["score2"]
            weighted_sentiment_score = row["weighted_sentiment_score"]

            if score >= 0 and return_percentage > 0 and score_2 >= 0 and weighted_sentiment_score > 0 and len(self.current_holdings) < 10:
                ticker_info = YahooStockMarket().get_stock_info(symbol)
                print(f"Buying {ticker_info.name} ({symbol}) at ${ticker_info.regularMarketPrice}")
                self.current_holdings.append(symbol)

            if len(self.current_holdings) == 10:
                break

        pass

    def run(self):
        print(self.now)
        print(self.end_time)

        while self.now < self.end_time:
            if self.premarket_open():
                print(f"Current time: {self.now}, Premarket open: {self.premarket_open()}")
                self.sleep(60)
                continue


            elif self.postmarket_open():
                print(f"Current time: {self.now}, Postmarket open: {self.postmarket_open()}")
                self.sleep(60)
                continue

            elif self.market_open():
                row = self.get_history_row(self.now)
                print(f"""Market open -- Time: {self.now.strftime("%H:%M")}\n Holdings: {self.current_holdings}\n""")

                if len(self.current_holdings) < 10:
                    self.symbols_analysis()
                    pass

            self.sleep(30)


if __name__ == "__main__":
    agent = TradingAgent()
    agent.run()
    # symbols = get_tickers(hours_back=24, symbol_filter=["NVDA"])
    pass
