"""
Decision Log Loader - Load and prepare decision logs for backtesting
"""

import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DecisionLoader:
    """Load and prepare decision logs for backtesting."""
    
    def __init__(self, log_file: Optional[Path] = None):
        """
        Initialize the decision loader.
        
        Args:
            log_file: Path to decisions.jsonl file. Defaults to decision_logs/decisions.jsonl
        """
        if log_file is None:
            # Go up one level from utils/ to project root
            project_root = Path(__file__).parent.parent
            log_file = project_root / "decision_logs" / "decisions.jsonl"
        
        self.log_file = Path(log_file)
        if not self.log_file.exists():
            log_dir = self.log_file.parent
            if not log_dir.exists():
                error_msg = (
                    f"Decision log directory not found: {log_dir}\n"
                    f"Please run the pipeline first to generate decision logs:\n"
                    f"  python run_full_pipeline.py AAPL\n"
                    f"  python run_full_pipeline.py TSLA\n"
                    f"  # ... run for multiple tickers to build decision history\n"
                    f"\n"
                    f"The pipeline will automatically create the '{log_dir.name}' directory\n"
                    f"and generate '{self.log_file.name}' with your trading decisions."
                )
            else:
                error_msg = (
                    f"Decision log file not found: {self.log_file}\n"
                    f"Please run the pipeline first to generate decision logs:\n"
                    f"  python run_full_pipeline.py AAPL\n"
                    f"  python run_full_pipeline.py TSLA\n"
                    f"  # ... run for multiple tickers to build decision history"
                )
            raise FileNotFoundError(error_msg)
    
    def load_decisions(self) -> List[Dict]:
        """Load all decisions from the log file."""
        decisions = []
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        decisions.append(json.loads(line))
            
            logger.info(f"Loaded {len(decisions)} decisions from {self.log_file}")
            return decisions
            
        except Exception as e:
            logger.error(f"Failed to load decisions: {e}")
            raise
    
    def prepare_decisions_df(self, decisions: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Convert decisions to a pandas DataFrame for easier manipulation.
        
        Args:
            decisions: Optional list of decisions. If None, loads from file.
        
        Returns:
            DataFrame with columns: timestamp, ticker, action, position_size_pct, 
            sentiment_score, credibility_score, model_confidence, verdict, etc.
        """
        if decisions is None:
            decisions = self.load_decisions()
        
        if not decisions:
            return pd.DataFrame()
        
        rows = []
        for decision in decisions:
            summary = decision.get("decision_summary", {})
            pipeline = decision.get("pipeline", {})
            a3_output = pipeline.get("a3_output", {})
            a4_output = pipeline.get("a4_output", {})
            
            # Extract technicals from A1 snapshot for exit conditions
            a1_snapshot = pipeline.get("a1_snapshot", {})
            technicals = a1_snapshot.get("technicals", {}) if isinstance(a1_snapshot, dict) else {}
            
            row = {
                "decision_id": decision.get("decision_id"),
                "timestamp": decision.get("timestamp"),
                "ticker": decision.get("ticker"),
                "action": summary.get("action", "hold"),
                "position_size_pct": summary.get("position_size_pct", 0.0),
                "sentiment": summary.get("sentiment"),
                "sentiment_score": summary.get("sentiment_score", 0.0),
                "credibility": summary.get("credibility"),
                "credibility_score": summary.get("credibility_score", 0.0),
                "model_confidence": summary.get("model_confidence", 0.0),
                "verdict": summary.get("verification_verdict", "approve"),
                "verification_confidence": summary.get("verification_confidence", 1.0),
                "rationale": a4_output.get("rationale", []),
                "technicals": technicals,  # Add technicals for exit conditions
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Convert timestamp to datetime
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        
        return df
    
    def get_price_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for price data
            end_date: End date for price data
            interval: Data interval (1d, 1h, etc.)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(ticker)
            # Add buffer to ensure we get data
            buffer_start = start_date - timedelta(days=30)
            hist = stock.history(start=buffer_start, end=end_date, interval=interval)
            
            if hist.empty:
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()
            
            # Rename columns to lowercase for Backtrader compatibility
            hist.columns = [col.lower() for col in hist.columns]
            hist.index.name = "Date"
            
            return hist
            
        except Exception as e:
            logger.error(f"Failed to fetch price data for {ticker}: {e}")
            return pd.DataFrame()
    
    def prepare_backtest_data(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Prepare decisions and price data for backtesting.
        
        Args:
            ticker: Optional filter by ticker
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            Tuple of (decisions_df, price_data_dict) where price_data_dict maps
            ticker -> price DataFrame
        """
        # Load decisions
        decisions_df = self.prepare_decisions_df()
        
        if decisions_df.empty:
            logger.warning("No decisions found for backtesting")
            return pd.DataFrame(), {}
        
        # Apply filters
        if ticker:
            decisions_df = decisions_df[decisions_df["ticker"] == ticker.upper()]
        
        if start_date:
            decisions_df = decisions_df[decisions_df["timestamp"] >= start_date]
        
        if end_date:
            decisions_df = decisions_df[decisions_df["timestamp"] <= end_date]
        
        if decisions_df.empty:
            logger.warning("No decisions found after filtering")
            return pd.DataFrame(), {}
        
        # Get unique tickers and date range
        unique_tickers = decisions_df["ticker"].unique()
        min_date = decisions_df["timestamp"].min()
        max_date = decisions_df["timestamp"].max()
        
        # Fetch price data for all tickers
        price_data = {}
        for ticker in unique_tickers:
            logger.info(f"Fetching price data for {ticker}...")
            prices = self.get_price_data(ticker, min_date, max_date)
            if not prices.empty:
                price_data[ticker] = prices
            else:
                logger.warning(f"Skipping {ticker} - no price data available")
        
        logger.info(f"Prepared backtest data: {len(decisions_df)} decisions, {len(price_data)} tickers")
        
        return decisions_df, price_data

