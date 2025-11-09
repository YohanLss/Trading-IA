from typing import Optional, Any, List, Dict, Tuple
from datetime import datetime, timezone, timedelta, time as dt_time, date as dt_date
import time
from urllib.parse import urlencode

import requests
from fake_useragent import UserAgent
from pydantic import BaseModel
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils import function_timer
from utils import logger
from zoneinfo import ZoneInfo

DEFAULT_HEADERS = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
class YahooFinanceError(RuntimeError):
    """Raised when the Yahoo Finance API responds with an error payload."""


class Ticker(BaseModel):
    symbol: str
    name: str
    currency: Optional[str] = None
    regularMarketTime : Optional[Any] = None
    regularMarketPrice : Optional[float] = None
    regularMarketDayHigh : Optional[float] = None
    regularMarketDayLow : Optional[float] = None
    fiftyTwoWeekHigh : Optional[float] = None
    fiftyTwoWeekLow : Optional[float] = None


class MarketCandle(BaseModel):
    date: str 
    open: float 
    high: float 
    low: float
    close: float 
    volume: Optional[int] 

def timestamp_to_datetime(timestamp: Optional[int]) -> Optional[str]:
    """
    Convert a Unix timestamp to a human readable UTC datetime string.

    Yahoo can occasionally return null timestamps, so we guard against that.
    """
    if timestamp is None:
        return None
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

class YahooStockMarket:
    """
    Small wrapper around Yahoo's public chart endpoint with simple retries and
    optional user-agent rotation to avoid being throttled.
    """

    RETRY_STATUS = (429, 500, 502, 503, 504)

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        retries: int = 2,
        backoff_factor: float = 0.4,
        rotate_user_agent: bool = False,
        session: Optional[Session] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = session or self._build_session(retries, backoff_factor)
        self.rotate_user_agent = rotate_user_agent
        self._ua_provider = self._init_user_agent_provider() if rotate_user_agent else None

    @staticmethod
    def _build_session(retries: int, backoff_factor: float) -> Session:
        session = Session()
        if retries > 0:
            retry = Retry(
                total=retries,
                read=retries,
                connect=retries,
                status=retries,
                respect_retry_after_header=True,
                backoff_factor=backoff_factor,
                status_forcelist=YahooStockMarket.RETRY_STATUS,
                allowed_methods=("GET",),
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        return session

    @staticmethod
    def _init_user_agent_provider() -> Optional[UserAgent]:
        try:
            return UserAgent()
        except Exception as exc:  # pragma: no cover - network issues are best-effort
            logger.warning("Unable to initialize fake user agent: %s", exc)
            return None

    def _get_user_agent(self) -> str:
        if self._ua_provider is not None:
            try:
                return self._ua_provider.random
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to get rotating user agent: %s", exc)
        return DEFAULT_HEADERS["User-Agent"]

    def _build_headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        built_headers = {**DEFAULT_HEADERS, "User-Agent": self._get_user_agent()}
        if extra:
            built_headers.update(extra)
        return built_headers

    def _request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        response = self.session.get(
            url,
            params=params,
            headers=self._build_headers(extra_headers),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise YahooFinanceError("Invalid JSON returned by Yahoo Finance") from exc

    def _fetch_chart_result(
        self,
        symbol: str,
        *,
        period1: Optional[int] = None,
        period2: Optional[int] = None,
        interval: str = "1d",
        events: str = "capitalGain|div|split",
        include_pre_post: bool = False,
        lang: str = "en-US",
        region: str = "US",
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        if period1 is not None and period2 is not None:
            url = self.build_yahoo_chart_url(
                symbol,
                period1=period1,
                period2=period2,
                interval=interval,
                events=events,
                include_pre_post=include_pre_post,
                lang=lang,
                region=region,
                source=source,
            )
        else:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

        payload = self._request(url)
        chart = payload.get("chart")
        if not chart:
            raise YahooFinanceError("Missing 'chart' key in Yahoo Finance response")

        error = chart.get("error")
        if error:
            raise YahooFinanceError(error.get("description") or str(error))

        results = chart.get("result")
        if not results:
            raise YahooFinanceError("Yahoo Finance returned an empty result")

        return results[0]
    
    def get_stock_info(self, symbol: str = "AVGO") -> Optional[Ticker]:
        """Fetch metadata for a ticker."""
        try:
            result = self._fetch_chart_result(symbol)
        except (requests.RequestException, YahooFinanceError, ValueError) as exc:
            logger.error("Error fetching %s info: %s", symbol, exc)
            return None

        meta = result.get("meta") or {}

        return Ticker(
            symbol=meta.get("symbol", symbol),
            name=meta.get("shortName") or meta.get("longName") or symbol,
            currency=meta.get("currency"),
            regularMarketPrice=meta.get("regularMarketPrice"),
            regularMarketTime=timestamp_to_datetime(meta.get("regularMarketTime")),
            regularMarketDayHigh=meta.get("regularMarketDayHigh"),
            regularMarketDayLow=meta.get("regularMarketDayLow"),
            fiftyTwoWeekHigh=meta.get("fiftyTwoWeekHigh"),
            fiftyTwoWeekLow=meta.get("fiftyTwoWeekLow"),
        )
    
    
    
    @staticmethod
    def build_yahoo_chart_url(
            symbol: str,
            period1: int,
            period2: int,
            interval: str = "1d",
            events: str = "capitalGain|div|split",
            formatted: bool = True,
            include_adjusted_close: bool = True,
            include_pre_post: bool = False,
            lang: str = "en-US",
            region: str = "US",
            source: Optional[str] = None,
    ) -> str:
        base_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

        params = {
            "events": events,
            "formatted": str(formatted).lower(),
            "includeAdjustedClose": str(include_adjusted_close).lower(),
            "interval": interval,
            "period1": period1,
            "period2": period2,
            "symbol": symbol,
            "userYfid": "true",
            "lang": lang,
            "region": region,
        }

        if include_pre_post:
            params["includePrePost"] = "true"
        if source:
            params["source"] = source

        return f"{base_url}?{urlencode(params)}"
    
    @function_timer
    def get_stock_history(
        self,
        symbol: str = "AVGO",
        days: int = 7,
        interval: str = "1d",
    ) -> Optional[List[MarketCandle]]:
        """Return historic candles for the past `days`."""
        now_utc = datetime.now(timezone.utc)
        period2 = int(now_utc.timestamp())
        period1 = int((now_utc - timedelta(days=days)).timestamp())

        try:
            result = self._fetch_chart_result(
                symbol,
                period1=period1,
                period2=period2,
                interval=interval,
            )
            return self._build_market_candles(result)
        except (requests.RequestException, YahooFinanceError) as exc:
            logger.error("Error fetching %s history: %s", symbol, exc)
            return None

    @function_timer
    def get_intraday_history(
        self,
        symbol: str = "AVGO",
        interval: str = "1m",
        include_pre_post: bool = True,
        tz_name: str = "US/Eastern",
        source: Optional[str] = None,
    ) -> Optional[List[MarketCandle]]:
        """
        Fetch 1-minute candles for the most recent completed trading day.
        """
        period1, period2 = self._most_recent_trading_day_range(tz_name=tz_name)
        try:
            result = self._fetch_chart_result(
                symbol,
                period1=period1,
                period2=period2,
                interval=interval,
                include_pre_post=include_pre_post,
                source=source,
            )
            return self._build_market_candles(result)
        except (requests.RequestException, YahooFinanceError) as exc:
            logger.error("Error fetching %s intraday data: %s", symbol, exc)
            return None

    def get_most_active_stocks(self):
        url = "https://query1.finance.yahoo.com/v7/finance/desktop/portfolio/all?formatted=true&amp;includeBetaVersion=1&amp;lang=en-US&amp;region=US&amp;crumb=TM.fQv3NKyH"
        
        payload = self._request(url)
        
        pass

    def _build_market_candles(self, result: Dict[str, Any]) -> List[MarketCandle]:
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators", {})
        quote_list = indicators.get("quote") or []

        if not timestamps or not quote_list:
            raise YahooFinanceError("Missing timestamps or quote data in Yahoo response")

        quote = quote_list[0]
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []

        length = min(
            len(timestamps),
            len(opens),
            len(highs),
            len(lows),
            len(closes),
            len(volumes),
        )

        market_history: List[MarketCandle] = []
        for i in range(length):
            if opens[i] is None or closes[i] is None:
                continue
            date_str = timestamp_to_datetime(timestamps[i])
            if date_str is None:
                continue
            market_history.append(
                MarketCandle(
                    date=date_str,
                    open=opens[i],
                    high=highs[i],
                    low=lows[i],
                    close=closes[i],
                    volume=volumes[i],
                )
            )

        return market_history

    @staticmethod
    def _most_recent_trading_day_range(tz_name: str = "US/Eastern") -> Tuple[int, int]:
        """
        Determine the most recent completed US trading day and return its UTC bounds.
        """
        eastern = ZoneInfo(tz_name)
        now = datetime.now(eastern)
        market_open = dt_time(hour=9, minute=30)

        def previous_weekday(date_val: dt_date) -> dt_date:
            while date_val.weekday() >= 5:
                date_val -= timedelta(days=1)
            return date_val

        trading_date = previous_weekday(now.date())

        if now.weekday() < 5 and now.time() < market_open:
            trading_date = previous_weekday(trading_date - timedelta(days=1))

        start_dt = datetime.combine(trading_date, dt_time.min, tzinfo=eastern)
        end_dt = start_dt + timedelta(days=1)

        return int(start_dt.timestamp()), int(end_dt.timestamp())

    def get_most_active_symbols(
        self,
        *,
        count: int = 200,
        start: int = 0,
        fields: Optional[List[str]] = None,
        lang: str = "en-US",
        region: str = "US",
        screener_id: str = "MOST_ACTIVES",
        use_records_response: bool = True,
    ) -> List[Dict[str, Optional[str]]]:
        """
        Fetch the list of most active tickers from Yahoo's predefined screener.
        """
        fields = fields or ["symbol", "shortName"]
        params = {
            "count": count,
            "formatted": "true",
            "scrIds": screener_id,
            "sortField": "",
            "sortType": "",
            "start": start,
            "useRecordsResponse": str(use_records_response).lower(),
            "fields": ",".join(fields),
            "lang": lang,
            "region": region,
        }

        try:
            payload = self._request(
                "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
                params=params,
            )
        except (requests.RequestException, YahooFinanceError) as exc:
            logger.error("Error fetching most active symbols: %s", exc)
            return []

        finance = payload.get("finance") or {}
        results = finance.get("result") or []
        if not results:
            logger.error("Yahoo screener returned no result payload: %s", finance.get("error"))
            return []

        first_result = results[0]
        entries = first_result.get("records") if use_records_response else first_result.get("quotes")
        if not entries:
            entries = first_result.get("quotes") or first_result.get("records") or []

        symbols: List[Dict[str, Optional[str]]] = []
        for entry in entries:
            symbol = entry.get("symbol") or entry.get("ticker")
            if not symbol:
                continue
            symbols.append(
                {
                    "symbol": symbol,
                    "shortName": entry.get("shortName") or entry.get("companyName"),
                }
            )

        return symbols

def continuous_fetch(symbol: str = "AVGO", days: int = 5, delay_seconds: float = 2.0) -> None:
    """
    Simple helper to repeatedly poll Yahoo for manual profiling or debugging.
    """
    stock_market = YahooStockMarket(rotate_user_agent=True)

    num = 0
    stock_cache: Optional[List[MarketCandle]] = None
    start_time = time.perf_counter()

    while True:
        stock = stock_market.get_stock_history(symbol=symbol, days=days)
        if stock:
            num += 1
            stock_cache = stock
            print(f"Fetch number: {num}")
            print(f"Elapsed time: {time.perf_counter() - start_time:.4f} seconds")
            time.sleep(delay_seconds)
        else:
            break

    elapsed_time = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Number of fetches: {num}")
    if stock_cache:
        print(f"Number of candles: {len(stock_cache)}")
    
    
def main() -> None:
    symbol = "AVGO"
    stock_market = YahooStockMarket()

    ticker = stock_market.get_stock_info(symbol=symbol)
    daily_candles = stock_market.get_stock_history(symbol=symbol, days=5) or []
    intraday = stock_market.get_intraday_history(symbol=symbol, include_pre_post=True) or []
    most_active = stock_market.get_most_active_symbols(count=10)
    # 
    print(ticker)
    print(f"Fetched {len(daily_candles)} daily candles for {symbol}")
    print(f"Fetched {len(intraday)} intraday candles for {symbol}")
    print("Most active symbols:", [entry["symbol"] for entry in most_active if entry.get("symbol")])


if __name__ == "__main__":
    main()
