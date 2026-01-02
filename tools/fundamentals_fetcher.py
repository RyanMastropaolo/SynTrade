"""
Fundamentals Fetcher Tool (A1b_Fundamentals_Fetcher).

Fetches:
- Financial statements from Alpha Vantage API (historical), FinnHub, and yfinance
- Earnings calendar from FinnHub
- Dividend data from yfinance
"""

import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import pandas as pd


def calculate_ttm_metrics_from_quarterly(quarterly_financials: pd.DataFrame, quarterly_income: pd.DataFrame, 
                                         quarterly_balance: pd.DataFrame, quarterly_cashflow: pd.DataFrame,
                                         as_of_date: datetime) -> Dict[str, Any]:
    """
    Calculate TTM (Trailing Twelve Months) metrics from quarterly financial statements.
    
    Args:
        quarterly_financials: Quarterly financials DataFrame
        quarterly_income: Quarterly income statement DataFrame
        quarterly_balance: Quarterly balance sheet DataFrame
        quarterly_cashflow: Quarterly cash flow statement DataFrame
        as_of_date: Date to calculate metrics as of
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {
        "revenue_growth_ttm": None,
        "eps_growth_ttm": None,
        "gross_margin": None,
        "operating_margin": None,
        "debt_to_equity": None,
        "free_cash_flow_margin": None
    }
    
    try:
        # Helper function to convert column dates to datetime for comparison
        def parse_quarter_date(col):
            """Convert yfinance quarter column to datetime."""
            if isinstance(col, datetime):
                return col
            if isinstance(col, pd.Timestamp):
                return col.to_pydatetime()
            # Try parsing string dates (yfinance often uses string dates like "2023-12-31")
            if isinstance(col, str):
                try:
                    # Try common date formats
                    for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                        try:
                            return datetime.strptime(col, fmt)
                        except:
                            continue
                    # Fallback to pandas parsing
                    return pd.to_datetime(col).to_pydatetime()
                except:
                    pass
            return None
        
        # Filter quarters up to as_of_date
        if quarterly_income is not None and not quarterly_income.empty:
            # Convert column dates and filter quarters up to as_of_date
            valid_quarter_cols = []
            all_quarter_cols = []
            for col in quarterly_income.columns:
                col_date = parse_quarter_date(col)
                if col_date:
                    all_quarter_cols.append((col, col_date))
                    if col_date <= as_of_date:
                        valid_quarter_cols.append(col)
            
            # If no quarters found before the date, use the earliest available quarters
            # (yfinance may only return recent quarters, not historical ones)
            if len(valid_quarter_cols) == 0 and len(all_quarter_cols) > 0:
                print(f"⚠️ No quarters found before {as_of_date.date()}, using earliest available quarters")
                # Sort all quarters and use the earliest ones
                sorted_all = sorted(all_quarter_cols, key=lambda x: x[1])
                # Use up to 4 earliest quarters
                valid_quarter_cols = [col for col, _ in sorted_all[:4]]
                print(f"Using quarters: {[str(parse_quarter_date(col).date()) for col in valid_quarter_cols]}")
            
            if len(valid_quarter_cols) >= 2:
                # Get last 4 quarters (or available quarters) - sorted by date
                sorted_quarters = sorted(valid_quarter_cols, key=lambda x: parse_quarter_date(x) or datetime.min)
                quarters_to_use = sorted_quarters[-4:] if len(sorted_quarters) >= 4 else sorted_quarters
                
                # Calculate TTM revenue (sum of last 4 quarters)
                if "Total Revenue" in quarterly_income.index:
                    try:
                        ttm_revenue = quarterly_income.loc["Total Revenue", quarters_to_use].sum()
                        # Previous TTM (quarters 5-8, or previous 4 if available)
                        # Need at least 5 quarters to calculate growth (4 for current TTM + 1 for previous period)
                        if len(sorted_quarters) >= 5:
                            # Get previous 4 quarters (quarters -8 to -4 if we have 8+, otherwise first 4)
                            if len(sorted_quarters) >= 8:
                                prev_quarters = sorted_quarters[-8:-4]
                            else:
                                # If we have 5-7 quarters, use the first 4 for comparison
                                prev_quarters = sorted_quarters[:4]
                            # Verify these quarters exist in the DataFrame
                            available_prev_quarters = [q for q in prev_quarters if q in quarterly_income.columns]
                            if len(available_prev_quarters) >= 1:
                                prev_ttm_revenue = quarterly_income.loc["Total Revenue", available_prev_quarters].sum()
                                if prev_ttm_revenue > 0:
                                    metrics["revenue_growth_ttm"] = (ttm_revenue - prev_ttm_revenue) / prev_ttm_revenue
                    except Exception as e:
                        print(f"Error calculating revenue growth: {e}")
                
                # Calculate gross margin (TTM)
                if "Total Revenue" in quarterly_income.index and "Cost Of Revenue" in quarterly_income.index:
                    ttm_revenue = quarterly_income.loc["Total Revenue", quarters_to_use].sum()
                    ttm_cost = quarterly_income.loc["Cost Of Revenue", quarters_to_use].sum()
                    if ttm_revenue > 0:
                        metrics["gross_margin"] = (ttm_revenue - ttm_cost) / ttm_revenue
                
                # Calculate operating margin (TTM)
                if "Total Revenue" in quarterly_income.index and "Operating Income" in quarterly_income.index:
                    ttm_revenue = quarterly_income.loc["Total Revenue", quarters_to_use].sum()
                    ttm_operating_income = quarterly_income.loc["Operating Income", quarters_to_use].sum()
                    if ttm_revenue > 0:
                        metrics["operating_margin"] = ttm_operating_income / ttm_revenue
                
                # Calculate EPS growth (if available)
                if "Basic EPS" in quarterly_income.index or "Diluted EPS" in quarterly_income.index:
                    eps_key = "Diluted EPS" if "Diluted EPS" in quarterly_income.index else "Basic EPS"
                    try:
                        ttm_eps = quarterly_income.loc[eps_key, quarters_to_use].sum()
                        if len(sorted_quarters) >= 5:
                            # Get previous 4 quarters for comparison
                            if len(sorted_quarters) >= 8:
                                prev_quarters = sorted_quarters[-8:-4]
                            else:
                                prev_quarters = sorted_quarters[:4]
                            # Verify these quarters exist in the DataFrame
                            available_prev_quarters = [q for q in prev_quarters if q in quarterly_income.columns]
                            if len(available_prev_quarters) >= 1:
                                prev_ttm_eps = quarterly_income.loc[eps_key, available_prev_quarters].sum()
                                if prev_ttm_eps > 0:
                                    metrics["eps_growth_ttm"] = (ttm_eps - prev_ttm_eps) / prev_ttm_eps
                    except Exception as e:
                        print(f"Error calculating EPS growth: {e}")
        
        # Calculate debt-to-equity from balance sheet (use most recent quarter)
        if quarterly_balance is not None and not quarterly_balance.empty:
            valid_quarter_cols = []
            all_quarter_cols = []
            for col in quarterly_balance.columns:
                col_date = parse_quarter_date(col)
                if col_date:
                    all_quarter_cols.append((col, col_date))
                    if col_date <= as_of_date:
                        valid_quarter_cols.append(col)
            
            # If no quarters found before the date, use earliest available
            if len(valid_quarter_cols) == 0 and len(all_quarter_cols) > 0:
                sorted_all = sorted(all_quarter_cols, key=lambda x: x[1])
                valid_quarter_cols = [col for col, _ in sorted_all[:1]]  # Just use earliest for balance sheet
            
            if len(valid_quarter_cols) > 0:
                sorted_quarters = sorted(valid_quarter_cols, key=lambda x: parse_quarter_date(x) or datetime.min)
                latest_quarter = sorted_quarters[-1]
                # Try different possible field names
                debt_fields = ["Total Debt", "Total Liabilities Net Minority Interest", "Total Debt NonCurrent"]
                equity_fields = ["Stockholders Equity", "Total Stockholders Equity", "Common Stock Equity"]
                
                total_debt = None
                total_equity = None
                
                for field in debt_fields:
                    if field in quarterly_balance.index:
                        total_debt = quarterly_balance.loc[field, latest_quarter]
                        break
                
                for field in equity_fields:
                    if field in quarterly_balance.index:
                        total_equity = quarterly_balance.loc[field, latest_quarter]
                        break
                
                if total_debt is not None and total_equity is not None and total_equity > 0:
                    metrics["debt_to_equity"] = total_debt / total_equity
        
        # Calculate free cash flow margin (TTM)
        if quarterly_cashflow is not None and not quarterly_cashflow.empty:
            valid_quarter_cols = []
            all_quarter_cols = []
            for col in quarterly_cashflow.columns:
                col_date = parse_quarter_date(col)
                if col_date:
                    all_quarter_cols.append((col, col_date))
                    if col_date <= as_of_date:
                        valid_quarter_cols.append(col)
            
            # If no quarters found before the date, use earliest available
            if len(valid_quarter_cols) == 0 and len(all_quarter_cols) > 0:
                sorted_all = sorted(all_quarter_cols, key=lambda x: x[1])
                valid_quarter_cols = [col for col, _ in sorted_all[:4]]  # Use up to 4 earliest
            
            if len(valid_quarter_cols) >= 1:
                sorted_quarters = sorted(valid_quarter_cols, key=lambda x: parse_quarter_date(x) or datetime.min)
                quarters_to_use = sorted_quarters[-4:] if len(sorted_quarters) >= 4 else sorted_quarters
                
                # Get FCF and revenue
                fcf_fields = ["Free Cash Flow", "Operating Cash Flow"]
                ttm_fcf = None
                for field in fcf_fields:
                    if field in quarterly_cashflow.index:
                        ttm_fcf = quarterly_cashflow.loc[field, quarters_to_use].sum()
                        break
                
                if ttm_fcf is not None and quarterly_income is not None and not quarterly_income.empty:
                    if "Total Revenue" in quarterly_income.index:
                        ttm_revenue = quarterly_income.loc["Total Revenue", quarters_to_use].sum()
                        if ttm_revenue > 0:
                            metrics["free_cash_flow_margin"] = ttm_fcf / ttm_revenue
                            
    except Exception as e:
        print(f"Error calculating TTM metrics from quarterly data: {e}")
    
    return metrics


def fetch_alpha_vantage_quarterly_data(ticker: str, api_key: str, as_of_date: datetime) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Fetch historical quarterly financial statements from Alpha Vantage API.
    
    Args:
        ticker: Stock ticker symbol
        api_key: Alpha Vantage API key
        as_of_date: Date to fetch data up to
        
    Returns:
        Dictionary with 'income', 'balance', 'cashflow' DataFrames, or None if failed
    """
    base_url = "https://www.alphavantage.co/query"
    
    try:
        # Fetch income statements (quarterly)
        income_params = {
            "function": "INCOME_STATEMENT",
            "symbol": ticker,
            "apikey": api_key
        }
        income_resp = requests.get(base_url, params=income_params, timeout=10)
        
        # Fetch balance sheet statements (quarterly)
        balance_params = {
            "function": "BALANCE_SHEET",
            "symbol": ticker,
            "apikey": api_key
        }
        balance_resp = requests.get(base_url, params=balance_params, timeout=10)
        
        # Fetch cash flow statements (quarterly)
        cashflow_params = {
            "function": "CASH_FLOW",
            "symbol": ticker,
            "apikey": api_key
        }
        cashflow_resp = requests.get(base_url, params=cashflow_params, timeout=10)
        
        if income_resp.status_code != 200 or balance_resp.status_code != 200 or cashflow_resp.status_code != 200:
            return None
        
        income_data = income_resp.json()
        balance_data = balance_resp.json()
        cashflow_data = cashflow_resp.json()
        
        # Check for API errors
        if "Error Message" in income_data or "Note" in income_data:
            return None
        if "Error Message" in balance_data or "Note" in balance_data:
            return None
        if "Error Message" in cashflow_data or "Note" in cashflow_data:
            return None
        
        # Alpha Vantage returns quarterlyReports array
        income_reports = income_data.get("quarterlyReports", [])
        balance_reports = balance_data.get("quarterlyReports", [])
        cashflow_reports = cashflow_data.get("quarterlyReports", [])
        
        if not income_reports:
            return None
        
        # Convert to DataFrames and filter by date
        def filter_by_date(reports: List[Dict], date_field: str = "fiscalDateEnding") -> pd.DataFrame:
            """Convert Alpha Vantage reports to DataFrame and filter quarters up to as_of_date."""
            if not reports:
                return pd.DataFrame()
            
            # Filter quarters up to as_of_date
            filtered = []
            for item in reports:
                try:
                    quarter_date = datetime.strptime(item[date_field], "%Y-%m-%d")
                    if quarter_date <= as_of_date:
                        filtered.append(item)
                except:
                    continue
            
            if not filtered:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(filtered)
            # Set date as index and sort
            df['date'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df.sort_values('date')
            # Transpose so dates are columns and fields are rows (like yfinance format)
            df = df.set_index('date').T
            # Convert column names (dates) back to Timestamp if needed
            df.columns = pd.to_datetime(df.columns)
            return df
        
        income_df = filter_by_date(income_reports)
        balance_df = filter_by_date(balance_reports)
        cashflow_df = filter_by_date(cashflow_reports)
        
        if income_df.empty:
            return None
        
        return {
            "income": income_df,
            "balance": balance_df,
            "cashflow": cashflow_df
        }
        
    except Exception as e:
        print(f"Error fetching Alpha Vantage quarterly data for {ticker}: {e}")
        return None


def calculate_ttm_from_alpha_vantage_data(av_data: Dict[str, pd.DataFrame], as_of_date: datetime) -> Dict[str, Any]:
    """
    Calculate TTM metrics from Alpha Vantage API quarterly data.
    
    Args:
        av_data: Dictionary with 'income', 'balance', 'cashflow' DataFrames
        as_of_date: Date to calculate metrics as of
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {
        "revenue_growth_ttm": None,
        "eps_growth_ttm": None,
        "gross_margin": None,
        "operating_margin": None,
        "debt_to_equity": None,
        "free_cash_flow_margin": None
    }
    
    try:
        income_df = av_data.get("income")
        balance_df = av_data.get("balance")
        cashflow_df = av_data.get("cashflow")
        
        if income_df is None or income_df.empty:
            return metrics
        
        # Get quarters up to as_of_date (already filtered, but ensure we have enough)
        quarters = sorted([col for col in income_df.columns if isinstance(col, pd.Timestamp) and col <= as_of_date])
        
        if len(quarters) < 2:
            return metrics
        
        # Use last 4 quarters for TTM
        quarters_to_use = quarters[-4:] if len(quarters) >= 4 else quarters
        
        # Alpha Vantage API field names (values are strings, need to convert to float)
        def get_float_value(df, field, quarter):
            """Get float value from Alpha Vantage DataFrame (handles string values)."""
            try:
                if field in df.index:
                    val = df.loc[field, quarter]
                    if isinstance(val, str):
                        # Remove commas and convert
                        val = val.replace(',', '')
                    return float(val) if val else 0.0
            except:
                pass
            return 0.0
        
        revenue_field = "totalRevenue"
        cost_field = "costOfRevenue"
        operating_income_field = "operatingIncome"
        eps_field = "reportedEPS"
        
        # Calculate TTM revenue
        if revenue_field in income_df.index:
            ttm_revenue = sum([get_float_value(income_df, revenue_field, q) for q in quarters_to_use])
            
            # Calculate revenue growth (need previous 4 quarters)
            if len(quarters) >= 5:
                prev_quarters = quarters[-8:-4] if len(quarters) >= 8 else quarters[:4]
                if len(prev_quarters) > 0:
                    prev_ttm_revenue = sum([get_float_value(income_df, revenue_field, q) for q in prev_quarters])
                    if prev_ttm_revenue > 0:
                        metrics["revenue_growth_ttm"] = (ttm_revenue - prev_ttm_revenue) / prev_ttm_revenue
            
            # Calculate gross margin (TTM)
            if cost_field in income_df.index:
                ttm_cost = sum([get_float_value(income_df, cost_field, q) for q in quarters_to_use])
                if ttm_revenue > 0:
                    metrics["gross_margin"] = (ttm_revenue - ttm_cost) / ttm_revenue
            
            # Calculate operating margin (TTM)
            if operating_income_field in income_df.index:
                ttm_operating_income = sum([get_float_value(income_df, operating_income_field, q) for q in quarters_to_use])
                if ttm_revenue > 0:
                    metrics["operating_margin"] = ttm_operating_income / ttm_revenue
            
            # Calculate EPS growth (TTM)
            if eps_field in income_df.index:
                ttm_eps = sum([get_float_value(income_df, eps_field, q) for q in quarters_to_use])
                if len(quarters) >= 5:
                    prev_quarters = quarters[-8:-4] if len(quarters) >= 8 else quarters[:4]
                    if len(prev_quarters) > 0:
                        prev_ttm_eps = sum([get_float_value(income_df, eps_field, q) for q in prev_quarters])
                        if prev_ttm_eps > 0:
                            metrics["eps_growth_ttm"] = (ttm_eps - prev_ttm_eps) / prev_ttm_eps
        
        # Calculate debt-to-equity from balance sheet (most recent quarter)
        if balance_df is not None and not balance_df.empty:
            quarters_balance = sorted([col for col in balance_df.columns if isinstance(col, pd.Timestamp) and col <= as_of_date])
            if len(quarters_balance) > 0:
                latest_quarter = quarters_balance[-1]
                # Alpha Vantage field names - calculate total debt from components
                short_term_debt = get_float_value(balance_df, "shortTermDebt", latest_quarter)
                long_term_debt = get_float_value(balance_df, "longTermDebt", latest_quarter)
                total_debt = short_term_debt + long_term_debt
                total_equity = get_float_value(balance_df, "totalShareholderEquity", latest_quarter)
                
                if total_debt > 0 and total_equity > 0:
                    metrics["debt_to_equity"] = total_debt / total_equity
        
        # Calculate free cash flow margin (TTM)
        if cashflow_df is not None and not cashflow_df.empty:
            quarters_cashflow = sorted([col for col in cashflow_df.columns if isinstance(col, pd.Timestamp) and col <= as_of_date])
            if len(quarters_cashflow) >= 1:
                quarters_to_use_cf = quarters_cashflow[-4:] if len(quarters_cashflow) >= 4 else quarters_cashflow
                
                # Alpha Vantage field name
                fcf_field = "operatingCashflow"  # Alpha Vantage uses operatingCashflow, not freeCashFlow
                # We'll approximate FCF as operatingCashflow - capitalExpenditures
                if "operatingCashflow" in cashflow_df.index and "capitalExpenditures" in cashflow_df.index:
                    ttm_ocf = sum([get_float_value(cashflow_df, "operatingCashflow", q) for q in quarters_to_use_cf])
                    ttm_capex = sum([get_float_value(cashflow_df, "capitalExpenditures", q) for q in quarters_to_use_cf])
                    ttm_fcf = ttm_ocf - abs(ttm_capex)  # Capex is negative, so use abs
                    if ttm_revenue > 0:
                        metrics["free_cash_flow_margin"] = ttm_fcf / ttm_revenue
        
    except Exception as e:
        print(f"Error calculating TTM metrics from Alpha Vantage data: {e}")
    
    return metrics


def fetch_financial_statements(ticker: str, api_key: Optional[str] = None, as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Fetch financial statements from FinnHub and/or yfinance.
    
    Args:
        ticker: Stock ticker symbol
        api_key: FinnHub API key (optional)
        as_of_date: Optional historical date (datetime object or ISO string). If provided,
                   attempts to fetch historical fundamentals as of that date.
        
    Returns:
        Dictionary with financial metrics
    """
    metrics = {
        "revenue_growth_ttm": None,
        "eps_growth_ttm": None,
        "gross_margin": None,
        "operating_margin": None,
        "debt_to_equity": None,
        "free_cash_flow_margin": None
    }
    
    # If historical date is provided, try to get historical financials from Alpha Vantage API first, then yfinance
    historical_calculation_successful = False
    if as_of_date:
        # Try Alpha Vantage API first (best for historical data)
        av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if av_api_key:
            try:
                if isinstance(as_of_date, str):
                    ref_date = datetime.fromisoformat(as_of_date.replace("Z", "+00:00"))
                else:
                    ref_date = as_of_date
                if ref_date.tzinfo is not None:
                    ref_date = ref_date.replace(tzinfo=None)
                
                av_data = fetch_alpha_vantage_quarterly_data(ticker, av_api_key, ref_date)
                if av_data:
                    historical_metrics = calculate_ttm_from_alpha_vantage_data(av_data, ref_date)
                    metrics_calculated = sum(1 for v in historical_metrics.values() if v is not None)
                    if metrics_calculated > 0:
                        historical_calculation_successful = True
                        for key, value in historical_metrics.items():
                            if value is not None:
                                metrics[key] = value
                        print(f"✓ Calculated {metrics_calculated} historical metrics from Alpha Vantage API for {ticker} as of {ref_date.date()}")
            except Exception as e:
                print(f"Error fetching Alpha Vantage data for {ticker}: {e}")
        
        # Fallback to yfinance if Alpha Vantage didn't work
        if not historical_calculation_successful:
            try:
                if isinstance(as_of_date, str):
                    ref_date = datetime.fromisoformat(as_of_date.replace("Z", "+00:00"))
                else:
                    ref_date = as_of_date
                if ref_date.tzinfo is not None:
                    ref_date = ref_date.replace(tzinfo=None)
                
                stock = yf.Ticker(ticker)
                
                # Try to get quarterly financials
                try:
                    quarterly_income = stock.quarterly_financials
                    quarterly_balance = stock.quarterly_balance_sheet
                    quarterly_cashflow = stock.quarterly_cashflow
                    
                    if quarterly_income is not None and not quarterly_income.empty:
                        # Calculate metrics from historical quarterly data
                        historical_metrics = calculate_ttm_metrics_from_quarterly(
                            None, quarterly_income, quarterly_balance, quarterly_cashflow, ref_date
                        )
                        # Check if we got at least some metrics
                        metrics_calculated = sum(1 for v in historical_metrics.values() if v is not None)
                        if metrics_calculated > 0:
                            historical_calculation_successful = True
                            # Update metrics with historical values (only if calculated)
                            for key, value in historical_metrics.items():
                                if value is not None:
                                    metrics[key] = value
                            print(f"✓ Calculated {metrics_calculated} historical metrics for {ticker} as of {ref_date.date()}")
                except Exception as e:
                    print(f"Error fetching historical quarterly financials for {ticker}: {e}")
            except Exception as e:
                print(f"Error processing historical date for {ticker}: {e}")
    
    # Fallback to current data ONLY if:
    # 1. No historical date was requested (as_of_date is None), OR
    # 2. Historical calculation failed or returned no metrics
    if not as_of_date or not historical_calculation_successful:
        try:
            # Try FinnHub first (only if API key available)
            if api_key:
                url = "https://finnhub.io/api/v1/stock/metric"
                params = {"symbol": ticker, "metric": "all", "token": api_key}
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json() or {}

                metric = data.get("metric") or {}
                if isinstance(metric, dict):
                    # Fill in missing metrics from FinnHub
                    if metrics["revenue_growth_ttm"] is None and "revenueGrowth" in metric:
                        metrics["revenue_growth_ttm"] = metric.get("revenueGrowth")

                    if metrics["eps_growth_ttm"] is None and "epsGrowth" in metric:
                        metrics["eps_growth_ttm"] = metric.get("epsGrowth")

                    if metrics["gross_margin"] is None and "grossMargin" in metric:
                        metrics["gross_margin"] = metric.get("grossMargin")

                    if metrics["operating_margin"] is None and "operatingMargin" in metric:
                        metrics["operating_margin"] = metric.get("operatingMargin")

                    if metrics["debt_to_equity"] is None and "debtEquity" in metric:
                        metrics["debt_to_equity"] = metric.get("debtEquity")
            
                    # Approximate "free cash flow margin" from per-share values if present
                    if metrics["free_cash_flow_margin"] is None:
                        fcf_per_share = metric.get("freeCashflowPerShare")
                        price = metric.get("price")
                        try:
                            if fcf_per_share is not None and price is not None and float(price) > 0:
                                metrics["free_cash_flow_margin"] = float(fcf_per_share) / float(price)
                        except Exception:
                            pass
            
            # Also try yfinance current info as backup (fills any remaining None fields)
            try:
                stock = yf.Ticker(ticker)
                info = getattr(stock, "info", {}) or {}
                
                if metrics["revenue_growth_ttm"] is None and "revenueGrowth" in info:
                    metrics["revenue_growth_ttm"] = info.get("revenueGrowth")
                
                if metrics["gross_margin"] is None and "grossMargins" in info:
                    metrics["gross_margin"] = info.get("grossMargins")
                
                if metrics["operating_margin"] is None and "operatingMargins" in info:
                    metrics["operating_margin"] = info.get("operatingMargins")
                
                if metrics["debt_to_equity"] is None and "debtToEquity" in info:
                    metrics["debt_to_equity"] = info.get("debtToEquity")
                    
            except Exception as e:
                print(f"Error fetching yfinance current data for {ticker}: {e}")
        except Exception as e:
            print(f"Error fetching financial statements for {ticker}: {e}")
    
    return metrics


def fetch_earnings_calendar(ticker: str, api_key: str, as_of_date: Optional[datetime] = None) -> Optional[str]:
    """
    Fetch next earnings date from FinnHub as of a specific date.
    
    Args:
        ticker: Stock ticker symbol
        api_key: FinnHub API key
        as_of_date: Optional historical date (datetime object or ISO string). If provided,
                   finds next earnings date relative to that date. If None, uses current date.
        
    Returns:
        Next earnings date as ISO-8601 date string or None
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
            reference_date = ref_date.date()
        else:
            reference_date = datetime.now().date()
        
        url = "https://finnhub.io/api/v1/calendar/earnings"
        params = {
            "symbol": ticker,
            "token": api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "earningsCalendar" in data and len(data["earningsCalendar"]) > 0:
            # Get the next upcoming earnings date relative to reference date
            for earnings in data["earningsCalendar"]:
                if "date" in earnings:
                    earnings_date = datetime.strptime(earnings["date"], "%Y-%m-%d").date()
                    if earnings_date >= reference_date:
                        return earnings["date"]
        
        # Fallback to yfinance (note: yfinance calendar is usually current, not historical)
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            if calendar is not None and not calendar.empty:
                next_earnings = calendar.index[0] if len(calendar) > 0 else None
                if next_earnings:
                    return next_earnings.strftime("%Y-%m-%d")
        except:
            pass
            
    except Exception as e:
        print(f"Error fetching earnings calendar for {ticker}: {e}")
    
    return None


def fetch_dividend_data(ticker: str, as_of_date: Optional[datetime] = None) -> Optional[float]:
    """
    Fetch dividend yield from yfinance as of a specific date.
    
    Args:
        ticker: Stock ticker symbol
        as_of_date: Optional historical date (datetime object or ISO string). If provided,
                   attempts to get dividend data as of that date. If None, uses current data.
                   Note: yfinance info is usually current, so historical may be limited.
        
    Returns:
        Dividend yield as float or None
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try to get historical dividend data if as_of_date is provided
        if as_of_date:
            if isinstance(as_of_date, str):
                ref_date = datetime.fromisoformat(as_of_date.replace("Z", "+00:00"))
            else:
                ref_date = as_of_date
            if ref_date.tzinfo is not None:
                ref_date = ref_date.replace(tzinfo=None)
            
            # Try to get historical dividend data
            try:
                # Get dividend history up to the reference date
                end_date = ref_date
                start_date = ref_date - timedelta(days=365)
                hist = stock.history(start=start_date, end=end_date, interval="1d")
                
                # Get dividends from history
                dividends = stock.dividends
                if not dividends.empty:
                    # Convert ref_date to timezone-aware if dividends index is timezone-aware
                    if dividends.index.tz is not None:
                        # Make ref_date timezone-aware (use same timezone as dividends)
                        ref_date_tz = ref_date.replace(tzinfo=dividends.index.tz)
                    else:
                        ref_date_tz = ref_date
                    
                    # Get most recent dividend before or on reference date
                    dividends_up_to_date = dividends[dividends.index <= ref_date_tz]
                    if not dividends_up_to_date.empty:
                        # Calculate annual dividend yield from most recent dividend
                        latest_dividend = dividends_up_to_date.iloc[-1]
                        # Get price at reference date
                        if not hist.empty:
                            price_at_date = hist["Close"].iloc[-1]
                            if price_at_date > 0:
                                # Estimate annual yield (assuming quarterly dividends)
                                annual_dividend = latest_dividend * 4
                                return annual_dividend / price_at_date
            except Exception as e:
                print(f"Error calculating historical dividend yield: {e}")
                pass  # Fall back to current info
        
        # Fallback to current info
        if "dividendYield" in info and info["dividendYield"] is not None:
            return float(info["dividendYield"])
        
        # Alternative: calculate from dividend rate and price
        if "dividendRate" in info and "currentPrice" in info:
            dividend_rate = info.get("dividendRate", 0)
            current_price = info.get("currentPrice", 1)
            if current_price > 0:
                return dividend_rate / current_price
                
    except Exception as e:
        print(f"Error fetching dividend data for {ticker}: {e}")
    
    return None


def fundamentals_fetcher(
    ticker: str,
    finnhub_api_key: Optional[str] = None,
    as_of_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Main function to fetch all fundamentals data.
    
    Args:
        ticker: Stock ticker symbol
        finnhub_api_key: FinnHub API key (from environment or parameter)
        as_of_date: Optional historical date (datetime object or ISO string). If provided,
                   attempts to fetch fundamentals as of that date. If None, uses current data.
                   Note: FinnHub and yfinance may have limited historical support.
        
    Returns:
        Dictionary with all fundamental metrics
    """
    # Get API key from environment if not provided
    api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
    
    if not api_key:
        print("Warning: FINNHUB_API_KEY not found. Some fundamentals may be missing.")
    
    # Fetch financial statements (with historical date support via yfinance)
    financials = fetch_financial_statements(ticker, api_key, as_of_date)
    
    # Fetch earnings calendar (with historical date support)
    next_earnings_date = fetch_earnings_calendar(ticker, api_key, as_of_date) if api_key else None
    
    # Fetch dividend data (with historical date support)
    dividend_yield = fetch_dividend_data(ticker, as_of_date)
    
    return {
        "revenue_growth_ttm": financials.get("revenue_growth_ttm"),
        "eps_growth_ttm": financials.get("eps_growth_ttm"),
        "gross_margin": financials.get("gross_margin"),
        "operating_margin": financials.get("operating_margin"),
        "debt_to_equity": financials.get("debt_to_equity"),
        "free_cash_flow_margin": financials.get("free_cash_flow_margin"),
        "dividend_yield": dividend_yield,
        "next_earnings_date": next_earnings_date
    }

