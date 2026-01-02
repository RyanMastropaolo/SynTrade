"""
Unified News Scraper Tool

Combines news from multiple sources:
- CNBC (via NewsAPI.ai)
- MarketWatch (via NewsAPI.ai)
- Yahoo Finance

Fetches latest news articles for a specific stock ticker.
"""

import os
import time
import requests
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any
import yfinance as yf


_LEGAL_SUFFIXES = {
    "inc", "inc.", "incorporated",
    "corp", "corp.", "corporation",
    "co", "co.", "company",
    "ltd", "ltd.", "limited",
    "plc",
    "group", "holdings", "holding",
}


def _strip_legal_suffixes(name: str) -> str:
    if not name:
        return name
    cleaned = name.replace(",", " ").strip()
    parts = [p for p in cleaned.split() if p]
    while parts and parts[-1].lower() in _LEGAL_SUFFIXES:
        parts.pop()
    return " ".join(parts).strip()


def build_search_terms_from_ticker(ticker: str) -> List[str]:
    """
    Build a list of keyword terms for Event Registry.
    We avoid boolean strings like '"Company" OR TICKER' because Event Registry's `keyword`
    operator often treats those as a literal phrase (leading to many false 0-results).
    """
    terms: List[str] = []
    t = (ticker or "").strip().upper()
    if t:
        terms.append(t)

    try:
        yf_ticker = yf.Ticker(t)
        info = getattr(yf_ticker, "info", {}) or {}
        company_name = (
            info.get("shortName")
            or info.get("longName")
            or info.get("displayName")
            or info.get("name")
        )
        if company_name:
            company_name = str(company_name).strip()
            if company_name:
                terms.append(company_name)

                stripped = _strip_legal_suffixes(company_name)
                if stripped and stripped.lower() != company_name.lower():
                    terms.append(stripped)

                # Often articles use the brand/first token (e.g., "Microsoft"), not the legal name.
                first_token = stripped.split()[0] if stripped else company_name.split()[0]
                if first_token and len(first_token) >= 3:
                    terms.append(first_token)
    except Exception:
        pass

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for term in terms:
        norm = term.strip()
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def build_search_term_from_ticker(ticker: str) -> str:
    """
    Backwards-compatible helper (kept because other modules may import it).
    Returns a single "best" term; internally we prefer `build_search_terms_from_ticker`.
    """
    terms = build_search_terms_from_ticker(ticker)
    return terms[1] if len(terms) > 1 else (terms[0] if terms else ticker)


def _parse_source_from_eventregistry_item(item: Dict[str, Any]) -> str:
    src = item.get("source")
    if isinstance(src, dict):
        return src.get("title") or src.get("name") or src.get("uri") or src.get("id") or "Event Registry"
    if isinstance(src, str) and src.strip():
        return src.strip()
    # Fall back to a couple of other common keys if present
    return item.get("sourceTitle") or item.get("sourceUri") or "Event Registry"


def fetch_eventregistry_news(
    ticker: str,
    api_key: str,
    sources: List[str],
    max_articles: int = 40,
    max_retries: int = 3,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch news using Event Registry (NewsAPI.ai).

    We use `$or` across:
    - multiple `sourceUri` domains
    - multiple keyword variants (ticker, company short name, stripped legal name)
    """
    articles: List[Dict[str, Any]] = []
    if not api_key:
        return articles

    keywords = build_search_terms_from_ticker(ticker)
    url = "https://eventregistry.org/api/v1/article/getArticles"

    query_conditions: List[Dict[str, Any]] = [{"lang": "eng"}]

    srcs = [s.strip() for s in (sources or []) if isinstance(s, str) and s.strip()]
    if srcs:
        query_conditions.append({"$or": [{"sourceUri": s} for s in srcs]})

    if keywords:
        query_conditions.append({"$or": [{"keyword": k} for k in keywords]})

    if from_date:
        if isinstance(from_date, str):
            from_date = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
        if from_date.tzinfo is None:
            from_date = from_date.replace(tzinfo=timezone.utc)
        else:
            from_date = from_date.astimezone(timezone.utc)
        query_conditions.append({"dateStart": from_date.strftime("%Y-%m-%d")})

    if to_date:
        if isinstance(to_date, str):
            to_date = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
        if to_date.tzinfo is None:
            to_date = to_date.replace(tzinfo=timezone.utc)
        else:
            to_date = to_date.astimezone(timezone.utc)
        query_conditions.append({"dateEnd": to_date.strftime("%Y-%m-%d")})

    payload = {
        "query": {"$query": {"$and": query_conditions}},
        "resultType": "articles",
        "articlesCount": min(max_articles, 100),
        "apiKey": api_key,
    }

    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15)

            if resp.status_code == 401:
                print("NewsAPI returned 401 Unauthorized. Check your API key.")
                break

            if resp.status_code == 429:
                wait_seconds = 2 ** attempt
                print(f"Event Registry rate limited (429). Waiting {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                continue

            resp.raise_for_status()
            data = resp.json()

            news_items: List[Dict[str, Any]] = []
            if isinstance(data, dict):
                if "articles" in data and isinstance(data["articles"], dict) and "results" in data["articles"]:
                    news_items = data["articles"]["results"] or []
                elif "articles" in data and isinstance(data["articles"], list):
                    news_items = data["articles"]

            for item in news_items[:max_articles]:
                published_at = item.get("date") or item.get("datePublished") or item.get("publishedAt")
                if not published_at:
                    continue

                try:
                    if isinstance(published_at, (int, float)):
                        ts_datetime = datetime.fromtimestamp(
                            published_at / 1000 if published_at > 1e10 else published_at, tz=timezone.utc
                        )
                    else:
                        ts_datetime = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
                        if ts_datetime.tzinfo is None:
                            ts_datetime = ts_datetime.replace(tzinfo=timezone.utc)
                except Exception:
                    continue

                source_name = _parse_source_from_eventregistry_item(item)
                uri = item.get("uri") or item.get("url") or ""
                url_val = item.get("url") or uri
                articles.append(
                    {
                        "headline": item.get("title", "") or item.get("headline", ""),
                        "summary": item.get("body", "") or item.get("description", "") or item.get("summary", ""),
                        "url": url_val,
                        "source": source_name,
                        "datetime": ts_datetime.isoformat(),
                        "category": "news",
                        "id": uri.split("/")[-1] if uri else (url_val.split("/")[-1] if url_val else ""),
                    }
                )

            break

        except Exception as e:
            print(f"Error fetching Event Registry news for {ticker}: {e}")
            if attempt == max_retries - 1:
                break
            time.sleep(2 ** attempt)

    return articles


def fetch_cnbc_news(
    ticker: str, 
    api_key: str, 
    max_articles: int = 20, 
    max_retries: int = 3,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Fetch CNBC news articles for a ticker using NewsAPI.
    
    Args:
        ticker: Stock ticker symbol
        api_key: NewsAPI key
        max_articles: Maximum number of articles to fetch
        max_retries: Number of retry attempts
        from_date: Optional start date for historical articles (ISO format string or datetime)
        to_date: Optional end date for historical articles (ISO format string or datetime)
        
    Returns:
        List of news article dictionaries
    """
    # Keep this wrapper for backwards compatibility, but implement it using the new OR-query.
    return fetch_eventregistry_news(
        ticker=ticker,
        api_key=api_key,
        sources=["cnbc.com"],
        max_articles=max_articles,
        max_retries=max_retries,
        from_date=from_date,
        to_date=to_date,
    )


def fetch_marketwatch_news(
    ticker: str, 
    api_key: str, 
    max_articles: int = 20, 
    max_retries: int = 3,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Fetch MarketWatch news articles for a ticker using NewsAPI.
    
    Args:
        ticker: Stock ticker symbol
        api_key: NewsAPI key
        max_articles: Maximum number of articles to fetch
        max_retries: Number of retry attempts
        from_date: Optional start date for historical articles (ISO format string or datetime)
        to_date: Optional end date for historical articles (ISO format string or datetime)
        
    Returns:
        List of news article dictionaries
    """
    # Keep this wrapper for backwards compatibility, but implement it using the new OR-query.
    return fetch_eventregistry_news(
        ticker=ticker,
        api_key=api_key,
        sources=["marketwatch.com"],
        max_articles=max_articles,
        max_retries=max_retries,
        from_date=from_date,
        to_date=to_date,
    )


def fetch_yahoo_news(ticker: str, max_articles: int = 20, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Fetch Yahoo Finance news articles for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        max_articles: Maximum number of articles to fetch
        max_retries: Number of retry attempts
        
    Returns:
        List of news article dictionaries
    """
    articles = []
    
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}&newsCount={max_articles}"

    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url,
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; SynTrade2.0/1.0)"
                },
            )

            if resp.status_code == 429:
                wait_seconds = 2 ** attempt
                print(f"Yahoo Finance rate limited (429). Waiting {wait_seconds} seconds...")
                time.sleep(wait_seconds)
                continue

            resp.raise_for_status()
            data = resp.json()
            news_items = data.get("news", [])

            for item in news_items[:max_articles]:
                ts_unix = item.get("providerPublishTime")
                if ts_unix is None:
                    continue
                
                try:
                    ts_datetime = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
                except:
                    continue
                
                articles.append({
                    "headline": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "url": item.get("link", ""),
                    "source": item.get("publisher", "Yahoo Finance"),
                    "datetime": ts_datetime.isoformat(),
                    "category": "news",
                    "id": item.get("uuid", "")
                })
            
            break  # Success, exit retry loop
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance news for {ticker}: {e}")
            if attempt == max_retries - 1:
                break
            time.sleep(2 ** attempt)
    
    return articles


def news_scraper(
    ticker: str,
    newsapi_key: Optional[str] = None,
    max_articles_per_source: int = 20,
    days_back: int = 7,
    as_of_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Main function to scrape news from multiple sources (CNBC, MarketWatch, Yahoo Finance).
    
    Args:
        ticker: Stock ticker symbol
        newsapi_key: NewsAPI key (for CNBC and MarketWatch, from environment or parameter)
        max_articles_per_source: Maximum articles to fetch from each source
        days_back: Number of days to look back (not strictly enforced, used for filtering)
        as_of_date: Optional historical date (ISO format string or datetime). If provided, fetches
                   articles from around that date. If None, uses current time.
        
    Returns:
        Combined list of news articles from all sources, sorted by datetime (newest first)
    """
    # Get API key from environment if not provided
    api_key = newsapi_key or os.getenv("NEWSAPI_KEY")
    
    # Determine the reference date (historical or current)
    if as_of_date:
        if isinstance(as_of_date, str):
            reference_date = datetime.fromisoformat(as_of_date.replace("Z", "+00:00"))
        else:
            reference_date = as_of_date
        if reference_date.tzinfo is None:
            reference_date = reference_date.replace(tzinfo=timezone.utc)
    else:
        reference_date = datetime.now(timezone.utc)
    
    # Calculate date range for NewsAPI (from_date and to_date)
    # For historical queries, fetch articles from a window around the target date
    from_date = reference_date - timedelta(days=days_back)
    to_date = reference_date + timedelta(days=1)  # Include articles up to 1 day after
    
    all_articles = []
    
    # Fetch from Event Registry (newsapi.ai) using a less strict query:
    # - multi-keyword OR (ticker + company name variants)
    # - configurable multi-source OR
    #
    # NOTE: defaults still include CNBC + MarketWatch, but you can broaden via NEWS_SOURCES env var
    # e.g. NEWS_SOURCES="cnbc.com,marketwatch.com,reuters.com,bloomberg.com,ft.com,wsj.com"
    if api_key:
        try:
            sources_env = os.getenv("NEWS_SOURCES", "").strip()
            if sources_env:
                sources = [s.strip() for s in sources_env.split(",") if s.strip()]
            else:
                sources = ["cnbc.com", "marketwatch.com", "reuters.com", "bloomberg.com", "ft.com", "wsj.com"]

            er_articles = fetch_eventregistry_news(
                ticker=ticker,
                api_key=api_key,
                sources=sources,
                # Request more than the final cap to survive dedupe/window filtering,
                # but avoid always hitting Event Registry's 100 hard cap.
                max_articles=min(100, max_articles_per_source * 4),
                from_date=from_date,
                to_date=to_date,
            )
            all_articles.extend(er_articles)
        except Exception as e:
            print(f"Error fetching Event Registry news: {e}")
    
    # Fetch from Yahoo Finance (no API key needed, doesn't support historical dates)
    # For historical queries, skip Yahoo Finance or filter strictly by date
    if as_of_date:
        # For historical dates, skip Yahoo Finance since it doesn't support historical queries
        # This prevents mixing current articles with historical ones
        print(f"Note: Skipping Yahoo Finance for historical date {as_of_date.strftime('%Y-%m-%d')} (not supported)")
    else:
        # Only fetch Yahoo Finance for current date queries
        try:
            yahoo_articles = fetch_yahoo_news(ticker, max_articles_per_source)
            all_articles.extend(yahoo_articles)
        except Exception as e:
            print(f"Error fetching Yahoo Finance news: {e}")
    
    # Sort by datetime (newest first) and remove duplicates based on URL
    seen_urls = set()
    unique_articles = []
    
    # Calculate cutoff time for filtering (relative to reference_date, not current time)
    # Ensure timezone-aware for proper comparison
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)
    cutoff_time = reference_date - timedelta(days=days_back)
    
    # Sort by datetime descending (ISO-8601 strings sort correctly)
    all_articles.sort(key=lambda x: x.get("datetime", ""), reverse=True)
    
    # Filter by days_back and remove duplicates
    # For historical dates, also enforce upper bound to prevent future articles
    upper_bound = reference_date + timedelta(days=1) if as_of_date else None
    
    for article in all_articles:
        url = article.get("url", "")
        if url and url not in seen_urls:
            # Check if article is within the time window
            article_datetime_str = article.get("datetime", "")
            if article_datetime_str:
                try:
                    article_time = datetime.fromisoformat(article_datetime_str.replace("Z", "+00:00"))
                    # Ensure timezone-aware for comparison
                    if article_time.tzinfo is None:
                        article_time = article_time.replace(tzinfo=timezone.utc)
                    
                    # Check lower bound (cutoff_time)
                    if article_time >= cutoff_time:
                        # For historical dates, also check upper bound to prevent future articles
                        if upper_bound is None or article_time <= upper_bound:
                            seen_urls.add(url)
                            unique_articles.append(article)
                except Exception as e:
                    # If datetime parsing fails, skip the article for historical dates
                    # (to avoid including articles with invalid dates)
                    if not as_of_date:
                        seen_urls.add(url)
                        unique_articles.append(article)

    # Hard cap: keep only the newest N after filtering/deduping.
    # This ensures `news_count` is bounded (and downstream LLM cost stays controlled).
    return unique_articles[:max(0, int(max_articles_per_source or 0))]

