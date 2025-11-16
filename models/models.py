from typing import Optional, List

from pydantic import BaseModel


class Article(BaseModel):
    url: str
    title: str = ""
    content: str = ""
    publish_date: Optional[str] = None  # ISO string if available
    authors: Optional[List[str]] = None
    summary: Optional[str] = None,
    keyword: Optional[str] = None
    sectors: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    sentiment: Optional[str] = None
    tickers: Optional[List[str]] = None
