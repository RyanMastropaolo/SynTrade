"""
A1 Scraper Agent - Main Entry Point.

Aggregates raw market data, fundamentals, macro indicators, and technical signals
into a unified MarketSnapshot per ticker.
"""

import os
import time
import requests
from datetime import datetime
from typing import Dict, Optional, List, Any
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root (2 levels up from this file)
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        # Try loading from current directory as fallback
        load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system env vars
    pass

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.news_scraper import news_scraper
from tools.filings_scraper import filings_scraper
from tools.fundamentals_fetcher import fundamentals_fetcher
from tools.macro_fetcher import macro_fetcher
from tools.technicals_computer import technicals_computer


def _fetch_with_retry(
    tool_func,
    tool_name: str,
    max_retries: int = 3,
    retry_delay: int = 1,
    *args,
    **kwargs
) -> Any:
    """
    Fetch data from a tool with exponential backoff retry logic.
    
    Args:
        tool_func: The tool function to call
        tool_name: Name of the tool for error messages
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay in seconds (exponentially increases)
        *args, **kwargs: Arguments to pass to tool_func
        
    Returns:
        Result from tool_func, or None if all retries fail
    """
    # Extract ticker from kwargs for error messages if available
    ticker = kwargs.get('ticker', 'unknown')
    
    for attempt in range(max_retries):
        try:
            return tool_func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            if e.response and e.response.status_code == 429:
                # Rate limit error
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⚠️  Rate limit (429) for {tool_name} on {ticker}. "
                          f"Waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Rate limit exceeded for {tool_name} after {max_retries} attempts")
                    print(f"   💡 Suggestions:")
                    print(f"      - Wait a few minutes and try again")
                    print(f"      - Check your API quota/limits")
                    return None
            elif e.response and e.response.status_code == 401:
                # Authentication error
                print(f"❌ Authentication error (401) for {tool_name}")
                print(f"   💡 Check your API key is correct and valid")
                return None
            else:
                # Other HTTP errors
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"⚠️  HTTP error {e.response.status_code if e.response else 'unknown'} "
                          f"for {tool_name}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ HTTP error for {tool_name} after {max_retries} attempts: {e}")
                    return None
        except requests.exceptions.RequestException as e:
            # Network errors
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"⚠️  Network error for {tool_name}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"❌ Network error for {tool_name} after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            # Other errors
            print(f"❌ Error in {tool_name} for {ticker}: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                return None
    
    return None


def gather_market_snapshot(
    ticker: str,
    as_of: str,
    requested_sources: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Build a unified MarketSnapshot dictionary for each ticker.
    
    This is the main entrypoint for A1_Scraper_Agent. It orchestrates all tools
    to gather comprehensive market data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "MSFT")
        as_of: ISO-8601 timestamp indicating when the snapshot should be taken
        requested_sources: Optional list of sources to fetch. If None, fetches all.
                          Valid values: ["news", "filings", "fundamentals", 
                                        "macro", "technicals"]
    
    Returns:
        Dictionary matching the MarketSnapshot contract:
        {
            "ticker": str,
            "as_of": str (ISO-8601),
            "sources": {
                "news": array<object>,
                "filings_links": array<object>
            },
            "fundamentals": {
                "revenue_growth_ttm": number | null,
                "eps_growth_ttm": number | null,
                "gross_margin": number | null,
                "operating_margin": number | null,
                "debt_to_equity": number | null,
                "free_cash_flow_margin": number | null,
                "dividend_yield": number | null,
                "next_earnings_date": ISO-8601 date | null
            },
            "macro": {
                "policy_rate": number,
                "policy_rate_change_3m": number,
                "inflation_yoy": number,
                "unemployment_rate": number,
                "risk_regime": string
            },
            "technicals": {
                "rsi_14": number,
                "macd": number,
                "macd_signal": number,
                "sma_20": number,
                "sma_50": number,
                "sma_200": number,
                "volume_zscore_20": number,
                "atr_14": number,
                "trend_strength": number
            },
            "freshness": {
                "news_max_age_min": number,
                "fundamentals_as_of": ISO-8601 date,
                "macro_as_of": ISO-8601 date,
                "technicals_as_of": ISO-8601 date
            }
        }
    """
    # Parse as_of timestamp
    try:
        snapshot_time = datetime.fromisoformat(as_of.replace('Z', '+00:00'))
    except:
        snapshot_time = datetime.now()
        as_of = snapshot_time.isoformat()
    
    # Determine which sources to fetch
    fetch_all = requested_sources is None
    fetch_news = fetch_all or "news" in requested_sources
    fetch_filings = False  # Disabled - SEC filings scraper not fully implemented yet
    fetch_fundamentals = fetch_all or "fundamentals" in requested_sources
    fetch_macro = fetch_all or "macro" in requested_sources
    fetch_technicals = fetch_all or "technicals" in requested_sources
    
    # Initialize snapshot structure
    snapshot = {
        "ticker": ticker.upper(),
        "as_of": as_of,
        "sources": {
            "news": [],
            "filings_links": []
        },
        "fundamentals": {
            "revenue_growth_ttm": None,
            "eps_growth_ttm": None,
            "gross_margin": None,
            "operating_margin": None,
            "debt_to_equity": None,
            "free_cash_flow_margin": None,
            "dividend_yield": None,
            "next_earnings_date": None
        },
        "macro": {
            "policy_rate": 0.0,
            "policy_rate_change_3m": 0.0,
            "inflation_yoy": 0.0,
            "unemployment_rate": 0.0,
            "risk_regime": "neutral"
        },
        "technicals": {
            "rsi_14": 50.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "sma_20": 0.0,
            "sma_50": 0.0,
            "sma_200": 0.0,
            "volume_zscore_20": 0.0,
            "atr_14": 0.0,
            "trend_strength": 0.0
        },
        "freshness": {
            "news_max_age_min": None,
            "fundamentals_as_of": None,
            "macro_as_of": None,
            "technicals_as_of": None
        }
    }
    
    # Fetch news from unified news scraper (CNBC, MarketWatch, Yahoo Finance)
    if fetch_news:
        newsapi_key = os.getenv("NEWSAPI_KEY")
        # Pass the historical date (as_of) to news_scraper for historical article fetching
        news_articles = _fetch_with_retry(
            news_scraper,
            "news_scraper",
            max_retries=3,
            retry_delay=1,
            ticker=ticker,
            newsapi_key=newsapi_key,
            max_articles_per_source=20,
            days_back=7,
            as_of_date=snapshot_time  # Pass the historical date
        )
        
        if news_articles is not None:
            snapshot["sources"]["news"] = news_articles
            
            # Calculate news max age
            if news_articles:
                # Use snapshot_time instead of current time for historical accuracy
                reference_time = snapshot_time
                # Convert to naive datetime for comparison if needed
                if reference_time.tzinfo:
                    reference_time = reference_time.replace(tzinfo=None)
                max_age = 0
                for article in news_articles:
                    if "datetime" in article and article["datetime"]:
                        try:
                            # Parse ISO-8601 datetime string
                            article_time = datetime.fromisoformat(
                                article["datetime"].replace("Z", "+00:00")
                            )
                            # Convert to naive datetime for comparison if needed
                            if article_time.tzinfo:
                                article_time = article_time.replace(tzinfo=None)
                            age_minutes = (reference_time - article_time).total_seconds() / 60
                            max_age = max(max_age, age_minutes)
                        except:
                            pass
                snapshot["freshness"]["news_max_age_min"] = max_age if max_age > 0 else None
        
        # Rate limit delay between tool calls
        time.sleep(1)
    
    # Fetch filings
    if fetch_filings:
        filings_data = _fetch_with_retry(
            filings_scraper,
            "filings_scraper",
            max_retries=3,
            retry_delay=1,
            ticker=ticker
        )
        
        if filings_data is not None:
            snapshot["sources"]["filings_links"] = filings_data.get("filings_links", [])
        
        # Rate limit delay between tool calls
        time.sleep(1)
    
    # Fetch fundamentals
    if fetch_fundamentals:
        # Pass the historical date (as_of) to fundamentals_fetcher for historical data
        fundamentals_data = _fetch_with_retry(
            fundamentals_fetcher,
            "fundamentals_fetcher",
            max_retries=3,
            retry_delay=1,
            ticker=ticker,
            finnhub_api_key=None,
            as_of_date=snapshot_time  # Pass the historical date
        )
        
        if fundamentals_data is not None:
            snapshot["fundamentals"].update(fundamentals_data)
            snapshot["freshness"]["fundamentals_as_of"] = snapshot_time.strftime("%Y-%m-%d")
        
        # Rate limit delay between tool calls
        time.sleep(1)
    
    # Fetch macro indicators (not ticker-specific)
    if fetch_macro:
        # Pass the historical date (as_of) to macro_fetcher for historical data
        macro_data = _fetch_with_retry(
            macro_fetcher,
            "macro_fetcher",
            max_retries=3,
            retry_delay=1,
            fred_api_key=None,
            as_of_date=snapshot_time  # Pass the historical date
        )
        
        if macro_data is not None:
            snapshot["macro"].update(macro_data)
            snapshot["freshness"]["macro_as_of"] = snapshot_time.strftime("%Y-%m-%d")
        
        # Rate limit delay between tool calls
        time.sleep(1)
    
    # Compute technicals
    if fetch_technicals:
        # Pass the historical date (as_of) to technicals_computer for historical calculations
        technicals_data = _fetch_with_retry(
            technicals_computer,
            "technicals_computer",
            max_retries=3,
            retry_delay=1,
            ticker=ticker,
            as_of_date=snapshot_time  # Pass the historical date
        )
        
        if technicals_data is not None:
            snapshot["technicals"].update(technicals_data)
            snapshot["freshness"]["technicals_as_of"] = snapshot_time.strftime("%Y-%m-%d")
    
    return snapshot

