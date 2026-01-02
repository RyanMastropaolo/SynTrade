"""
Technicals Computer Tool (A1d_Technicals_Computer).

Computes OHLCV-based technical indicators using yfinance data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple


def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Series of closing prices
        period: RSI period (default 14)
        
    Returns:
        RSI value as float
    """
    if len(prices) < period + 1:
        return 50.0  # Default neutral RSI
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
    """
    Calculate MACD and MACD Signal.
    
    Args:
        prices: Series of closing prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
        
    Returns:
        Tuple of (MACD, MACD Signal)
    """
    if len(prices) < slow + signal:
        return 0.0, 0.0
    
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    macd_value = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
    signal_value = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
    
    return macd_value, signal_value


def calculate_sma(prices: pd.Series, period: int) -> float:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        prices: Series of closing prices
        period: SMA period
        
    Returns:
        SMA value as float
    """
    if len(prices) < period:
        return float(prices.iloc[-1]) if len(prices) > 0 else 0.0
    
    sma = prices.rolling(window=period).mean()
    return float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else float(prices.iloc[-1])


def calculate_volume_zscore(volumes: pd.Series, period: int = 20) -> float:
    """
    Calculate volume z-score.
    
    Args:
        volumes: Series of volume data
        period: Period for rolling mean/std (default 20)
        
    Returns:
        Volume z-score as float
    """
    if len(volumes) < period + 1:
        return 0.0
    
    rolling_mean = volumes.rolling(window=period).mean()
    rolling_std = volumes.rolling(window=period).std()
    
    current_volume = volumes.iloc[-1]
    mean_volume = rolling_mean.iloc[-1]
    std_volume = rolling_std.iloc[-1]
    
    if pd.isna(std_volume) or std_volume == 0:
        return 0.0
    
    zscore = (current_volume - mean_volume) / std_volume
    return float(zscore) if not pd.isna(zscore) else 0.0


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: ATR period (default 14)
        
    Returns:
        ATR value as float
    """
    if len(high) < period + 1:
        return 0.0
    
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0


def calculate_trend_strength(prices: pd.Series, sma_20: float, sma_50: float, sma_200: float) -> float:
    """
    Calculate trend strength based on SMA alignment.
    
    Args:
        prices: Series of closing prices
        sma_20: 20-period SMA
        sma_50: 50-period SMA
        sma_200: 200-period SMA
        current_price: Current price
        
    Returns:
        Trend strength (-1 to 1, where 1 is strong uptrend, -1 is strong downtrend)
    """
    if len(prices) == 0:
        return 0.0
    
    current_price = float(prices.iloc[-1])
    
    # Check alignment of SMAs and price
    # Strong uptrend: price > SMA20 > SMA50 > SMA200
    # Strong downtrend: price < SMA20 < SMA50 < SMA200
    
    trend_score = 0.0
    
    if current_price > sma_20:
        trend_score += 0.33
    elif current_price < sma_20:
        trend_score -= 0.33
    
    if sma_20 > sma_50:
        trend_score += 0.33
    elif sma_20 < sma_50:
        trend_score -= 0.33
    
    if sma_50 > sma_200:
        trend_score += 0.34
    elif sma_50 < sma_200:
        trend_score -= 0.34
    
    return max(-1.0, min(1.0, trend_score))


def technicals_computer(ticker: str, period_days: int = 252, as_of_date: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Main function to compute all technical indicators.
    
    Args:
        ticker: Stock ticker symbol
        period_days: Number of days of historical data to fetch (default 252 = 1 year)
        as_of_date: Optional historical date (ISO format string or datetime). If provided,
                   calculates technicals as of that date. If None, uses current date.
        
    Returns:
        Dictionary with all technical indicators
    """
    try:
        # Determine the reference date (historical or current)
        if as_of_date:
            if isinstance(as_of_date, str):
                end_date = datetime.fromisoformat(as_of_date.replace("Z", "+00:00"))
            else:
                end_date = as_of_date
            # Remove timezone for yfinance (it expects naive datetime or handles it internally)
            if end_date.tzinfo is not None:
                end_date = end_date.replace(tzinfo=None)
        else:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=period_days)
        
        # Fetch historical data up to the reference date
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, interval="1d")
        
        if hist.empty or len(hist) < 50:
            # Return default values if insufficient data
            return {
                "rsi_14": 50.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "sma_20": 0.0,
                "sma_50": 0.0,
                "sma_200": 0.0,
                "volume_zscore_20": 0.0,
                "atr_14": 0.0,
                "trend_strength": 0.0
            }
        
        close = hist["Close"]
        high = hist["High"]
        low = hist["Low"]
        volume = hist["Volume"]
        
        # Calculate all indicators
        rsi_14 = calculate_rsi(close, period=14)
        macd, macd_signal = calculate_macd(close)
        sma_20 = calculate_sma(close, period=20)
        sma_50 = calculate_sma(close, period=50)
        sma_200 = calculate_sma(close, period=200)
        volume_zscore_20 = calculate_volume_zscore(volume, period=20)
        atr_14 = calculate_atr(high, low, close, period=14)
        trend_strength = calculate_trend_strength(close, sma_20, sma_50, sma_200)
        
        return {
            "rsi_14": rsi_14,
            "macd": macd,
            "macd_signal": macd_signal,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "volume_zscore_20": volume_zscore_20,
            "atr_14": atr_14,
            "trend_strength": trend_strength
        }
        
    except Exception as e:
        print(f"Error computing technicals for {ticker}: {e}")
        # Return default values on error
        return {
            "rsi_14": 50.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "sma_20": 0.0,
            "sma_50": 0.0,
            "sma_200": 0.0,
            "volume_zscore_20": 0.0,
            "atr_14": 0.0,
            "trend_strength": 0.0
        }

