"""
Macro Fetcher Tool (A1c_Macro_Fetcher).

Fetches macroeconomic indicators from FRED (Federal Reserve Economic Data).
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple


# FRED series IDs for key economic indicators
FRED_SERIES = {
    "policy_rate": "FEDFUNDS",  # Federal Funds Effective Rate
    "inflation_yoy": "CPIAUCSL",  # CPI for All Urban Consumers
    "unemployment_rate": "UNRATE"  # Unemployment Rate
}


def fetch_fred_series(series_id: str, api_key: str, end_date: Optional[datetime] = None) -> Optional[float]:
    """
    Fetch a single FRED series value.
    
    Args:
        series_id: FRED series ID
        api_key: FRED API key
        end_date: End date (datetime object or ISO string, defaults to today)
        
    Returns:
        Most recent value as float or None
    """
    try:
        # Determine reference date (historical or current)
        if end_date:
            if isinstance(end_date, str):
                ref_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            else:
                ref_date = end_date
            if ref_date.tzinfo is not None:
                ref_date = ref_date.replace(tzinfo=None)
        else:
            ref_date = datetime.now()
        
        end_date_str = ref_date.strftime("%Y-%m-%d")
        start_date_str = (ref_date - timedelta(days=365)).strftime("%Y-%m-%d")
        
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
            "observation_start": start_date_str,
            "observation_end": end_date_str
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "observations" in data and len(data["observations"]) > 0:
            value_str = data["observations"][0].get("value", ".")
            if value_str != ".":
                return float(value_str)
                
    except Exception as e:
        print(f"Error fetching FRED series {series_id}: {e}")
    
    return None


def fetch_policy_rate(api_key: str, as_of_date: Optional[datetime] = None) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch policy rate and 3-month change as of a specific date.
    
    Args:
        api_key: FRED API key
        as_of_date: Optional historical date (datetime object or ISO string). If None, uses current date.
        
    Returns:
        Tuple of (rate, 3_month_change)
    """
    try:
        # Determine reference date (historical or current)
        if as_of_date:
            if isinstance(as_of_date, str):
                ref_date = datetime.fromisoformat(as_of_date.replace("Z", "+00:00"))
            else:
                ref_date = as_of_date
            if ref_date.tzinfo is not None:
                ref_date = ref_date.replace(tzinfo=None)
        else:
            ref_date = datetime.now()
        
        current_rate = fetch_fred_series(FRED_SERIES["policy_rate"], api_key, ref_date)
        
        # Get rate from 3 months ago (relative to reference date)
        three_months_ago = ref_date - timedelta(days=90)
        rate_3m_ago = fetch_fred_series(FRED_SERIES["policy_rate"], api_key, three_months_ago)
        
        if current_rate is not None and rate_3m_ago is not None:
            change_3m = current_rate - rate_3m_ago
            return current_rate, change_3m
        elif current_rate is not None:
            return current_rate, None
            
    except Exception as e:
        print(f"Error fetching policy rate: {e}")
    
    return None, None


def fetch_inflation_yoy(api_key: str, as_of_date: Optional[datetime] = None) -> Optional[float]:
    """
    Fetch year-over-year inflation rate as of a specific date.
    
    Args:
        api_key: FRED API key
        as_of_date: Optional historical date (datetime object or ISO string). If None, uses current date.
        
    Returns:
        YoY inflation rate as float or None
    """
    try:
        # Determine reference date (historical or current)
        if as_of_date:
            if isinstance(as_of_date, str):
                ref_date = datetime.fromisoformat(as_of_date.replace("Z", "+00:00"))
            else:
                ref_date = as_of_date
            if ref_date.tzinfo is not None:
                ref_date = ref_date.replace(tzinfo=None)
        else:
            ref_date = datetime.now()
        
        # Get CPI as of reference date
        current_cpi = fetch_fred_series(FRED_SERIES["inflation_yoy"], api_key, ref_date)
        
        # Get CPI from 12 months ago (relative to reference date)
        one_year_ago = ref_date - timedelta(days=365)
        cpi_1y_ago = fetch_fred_series(FRED_SERIES["inflation_yoy"], api_key, one_year_ago)
        
        if current_cpi is not None and cpi_1y_ago is not None and cpi_1y_ago > 0:
            inflation_yoy = ((current_cpi - cpi_1y_ago) / cpi_1y_ago) * 100
            return inflation_yoy
            
    except Exception as e:
        print(f"Error fetching inflation: {e}")
    
    return None


def fetch_unemployment_rate(api_key: str, as_of_date: Optional[datetime] = None) -> Optional[float]:
    """
    Fetch unemployment rate as of a specific date.
    
    Args:
        api_key: FRED API key
        as_of_date: Optional historical date (datetime object or ISO string). If None, uses current date.
        
    Returns:
        Unemployment rate as float or None
    """
    return fetch_fred_series(FRED_SERIES["unemployment_rate"], api_key, as_of_date)


def determine_risk_regime(
    policy_rate: Optional[float],
    inflation_yoy: Optional[float],
    unemployment_rate: Optional[float]
) -> str:
    """
    Determine risk regime based on macro indicators.
    
    Args:
        policy_rate: Current policy rate
        inflation_yoy: Year-over-year inflation
        unemployment_rate: Unemployment rate
        
    Returns:
        Risk regime string: "risk_on", "risk_off", or "neutral"
    """
    # Simple heuristic for risk regime
    # Risk-on: Low rates, low inflation, low unemployment
    # Risk-off: High rates, high inflation, high unemployment
    
    risk_score = 0
    
    if policy_rate is not None:
        if policy_rate < 2.0:
            risk_score += 1  # Low rates = risk on
        elif policy_rate > 4.0:
            risk_score -= 1  # High rates = risk off
    
    if inflation_yoy is not None:
        if inflation_yoy < 2.0:
            risk_score += 1  # Low inflation = risk on
        elif inflation_yoy > 4.0:
            risk_score -= 1  # High inflation = risk off
    
    if unemployment_rate is not None:
        if unemployment_rate < 4.0:
            risk_score += 1  # Low unemployment = risk on
        elif unemployment_rate > 6.0:
            risk_score -= 1  # High unemployment = risk off
    
    if risk_score > 0:
        return "risk_on"
    elif risk_score < 0:
        return "risk_off"
    else:
        return "neutral"


def macro_fetcher(
    fred_api_key: Optional[str] = None,
    as_of_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Main function to fetch all macro indicators.
    
    Args:
        fred_api_key: FRED API key (from environment or parameter)
        as_of_date: Optional historical date (datetime object or ISO string). If provided,
                   fetches macro data as of that date. If None, uses current date.
        
    Returns:
        Dictionary with all macro indicators
    """
    # Get API key from environment if not provided
    api_key = fred_api_key or os.getenv("FRED_API_KEY")
    
    if not api_key:
        print("Warning: FRED_API_KEY not found. Macro data will be missing.")
        return {
            "policy_rate": None,
            "policy_rate_change_3m": None,
            "inflation_yoy": None,
            "unemployment_rate": None,
            "risk_regime": "neutral"
        }
    
    # Fetch all indicators using historical date if provided
    policy_rate, policy_rate_change_3m = fetch_policy_rate(api_key, as_of_date)
    inflation_yoy = fetch_inflation_yoy(api_key, as_of_date)
    unemployment_rate = fetch_unemployment_rate(api_key, as_of_date)
    
    # Determine risk regime
    risk_regime = determine_risk_regime(policy_rate, inflation_yoy, unemployment_rate)
    
    return {
        "policy_rate": policy_rate if policy_rate is not None else 0.0,
        "policy_rate_change_3m": policy_rate_change_3m if policy_rate_change_3m is not None else 0.0,
        "inflation_yoy": inflation_yoy if inflation_yoy is not None else 0.0,
        "unemployment_rate": unemployment_rate if unemployment_rate is not None else 0.0,
        "risk_regime": risk_regime
    }

