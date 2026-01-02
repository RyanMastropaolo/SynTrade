"""
Filings Scraper Tool (A1_Scraper).

Fetches:
- SEC EDGAR filings links

Note: News fetching is handled by news_scraper.py (CNBC, MarketWatch, Yahoo Finance)
"""

import os
import requests
from typing import List, Dict, Optional, Any


def fetch_filings_links(ticker: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch SEC EDGAR filings links.
    
    Uses SEC EDGAR API to get recent filings for a company.
    
    Args:
        ticker: Stock ticker symbol
        api_key: Optional FinnHub API key (for company profile lookup)
        
    Returns:
        List of filing dictionaries with links
    """
    filings = []
    try:
        # First, get CIK from ticker (simplified - in production, use a mapping)
        # SEC EDGAR API endpoint
        url = "https://data.sec.gov/submissions/CIK0000000000.json"
        
        # For now, we'll use a simplified approach
        # In production, you'd need a ticker->CIK mapping
        # This is a placeholder that would need the actual CIK
        
        # Alternative: Use FinnHub for company profile to get CIK
        if api_key:
            try:
                profile_url = "https://finnhub.io/api/v1/stock/profile2"
                profile_params = {"symbol": ticker, "token": api_key}
                profile_response = requests.get(profile_url, params=profile_params, timeout=10)
                profile_data = profile_response.json()
                
                # FinnHub doesn't directly provide CIK, so we'll use a different approach
                # For now, return empty list and note that CIK mapping is needed
                pass
            except:
                pass
        
        # Placeholder: Return structure but note that CIK mapping is required
        # In production, implement proper ticker->CIK lookup
        filings.append({
            "note": "CIK mapping required for SEC EDGAR access",
            "ticker": ticker,
            "filings_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=&dateb=&owner=exclude&count=40"
        })
        
    except Exception as e:
        print(f"Error fetching filings for {ticker}: {e}")
    
    return filings


def filings_scraper(
    ticker: str,
    finnhub_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to scrape filings links.
    
    Args:
        ticker: Stock ticker symbol
        finnhub_api_key: FinnHub API key (optional, for filings lookup)
        
    Returns:
        Dictionary with filings_links
    """
    finnhub_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
    
    filings_links = fetch_filings_links(ticker, finnhub_key)
    
    return {
        "filings_links": filings_links
    }

