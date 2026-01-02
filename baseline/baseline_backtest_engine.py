"""
Backtest Engine for Baseline Strategies
Uses the same methodology as SynTrade's backtest engine
"""

import backtrader as bt
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaselineBacktestEngine:
    """
    Backtest engine for baseline strategies using SynTrade's exact methodology.
    """
    
    def __init__(
        self,
        initial_cash: float = 1000000.0,
        commission: float = 0.001,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_cash: Starting capital
            commission: Commission rate (0.001 = 0.1%)
            start_date: Start date for backtest
            end_date: End date for backtest
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.start_date = start_date
        self.end_date = end_date
    
    def run_single_backtest(
        self,
        ticker: str,
        strategy_class: bt.Strategy,
        strategy_params: Dict = None
    ) -> Dict:
        """
        Run backtest for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            strategy_class: Backtrader strategy class
            strategy_params: Parameters for the strategy
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for {ticker}...")
        
        # Fetch price data
        stock = yf.Ticker(ticker)
        
        if self.start_date and self.end_date:
            # Fetch data starting 60 days before start_date to ensure indicators are ready
            from datetime import timedelta
            data_start_date = self.start_date - timedelta(days=60)
            hist = stock.history(start=data_start_date, end=self.end_date, interval="1d")
        else:
            # Default to 2 years
            hist = stock.history(period="2y", interval="1d")
        
        if hist.empty:
            logger.warning(f"No price data for {ticker}")
            return {}
        
        # Prepare data for Backtrader
        price_df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        price_df.columns = [col.lower() for col in price_df.columns]
        price_df.index.name = "Date"
        
        # Create Cerebro engine
        cerebro = bt.Cerebro()
        
        # Add data feed
        data_feed = bt.feeds.PandasData(
            dataname=price_df,
            datetime=None,
            open=0,
            high=1,
            low=2,
            close=3,
            volume=4,
            openinterest=-1
        )
        cerebro.adddata(data_feed)
        
        # Add strategy
        if strategy_params:
            cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            cerebro.addstrategy(strategy_class)
        
        # Set initial cash (1 million per ticker)
        cash_per_ticker = 1000000.0  # 1 million per stock
        cerebro.broker.setcash(cash_per_ticker)
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers (same as SynTrade)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        
        # Run backtest
        logger.info(f"Starting backtest for {ticker}...")
        strategies = cerebro.run()
        strategy = strategies[0]
        
        # Extract results (same methodology as SynTrade)
        analyzers = strategy.analyzers
        
        # Get final portfolio value
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - cash_per_ticker) / cash_per_ticker
        
        # Extract analyzer results (same as SynTrade)
        sharpe = analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
        if sharpe is None or np.isnan(sharpe):
            sharpe = 0.0
        
        drawdown = analyzers.drawdown.get_analysis()
        max_drawdown = drawdown.get('max', {}).get('drawdown', 0.0) / 100.0
        
        returns = analyzers.returns.get_analysis()
        annual_return = returns.get('rnorm100', 0.0)
        
        trades_analysis = analyzers.trades.get_analysis()
        
        # Get trades from strategy
        trades = strategy.trades if hasattr(strategy, 'trades') else []
        
        # Get portfolio value history for equity curve
        portfolio_values = strategy.portfolio_values if hasattr(strategy, 'portfolio_values') else []
        portfolio_dates = strategy.portfolio_dates if hasattr(strategy, 'portfolio_dates') else []
        
        results = {
            "ticker": ticker,
            "initial_cash": cash_per_ticker,
            "final_value": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_trades": len(trades),
            "trades": trades,
            "trades_analysis": trades_analysis,
            "portfolio_values": portfolio_values,  # For equity curve
            "portfolio_dates": portfolio_dates,    # For equity curve
        }
        
        logger.info(f"Backtest complete for {ticker}: Return={total_return:.2%}, Sharpe={sharpe:.2f}")
        
        return results
    
    def run_portfolio_backtest(
        self,
        tickers: List[str],
        strategy_class: bt.Strategy,
        strategy_params: Dict = None
    ) -> Dict:
        """
        Run backtest across multiple tickers (portfolio).
        
        Args:
            tickers: List of ticker symbols
            strategy_class: Backtrader strategy class
            strategy_params: Parameters for the strategy
        
        Returns:
            Dictionary with portfolio backtest results
        """
        logger.info("Running portfolio backtest...")
        
        # Run backtest for each ticker
        ticker_results = {}
        for ticker in tickers:
            result = self.run_single_backtest(ticker, strategy_class, strategy_params)
            if result:
                ticker_results[ticker] = result
        
        # Aggregate portfolio results (same methodology as SynTrade)
        if not ticker_results:
            logger.warning("No valid backtest results")
            return {}
        
        # Calculate portfolio metrics
        total_initial = sum(r["initial_cash"] for r in ticker_results.values())
        total_final = sum(r["final_value"] for r in ticker_results.values())
        portfolio_return = (total_final - total_initial) / total_initial if total_initial > 0 else 0.0
        
        total_trades = sum(r["total_trades"] for r in ticker_results.values())
        
        # Aggregate trades across tickers
        portfolio_trades: List[Dict] = []
        for tkr, tr in ticker_results.items():
            for trade in tr.get("trades", []) or []:
                if isinstance(trade, dict):
                    trade_with_ticker = dict(trade)
                    trade_with_ticker.setdefault("ticker", tkr)
                    portfolio_trades.append(trade_with_ticker)
        
        # Build portfolio equity curve by summing per-ticker equity curves across aligned dates
        equity_series_by_ticker = {}
        for tkr, tr in ticker_results.items():
            dates = tr.get("portfolio_dates") or []
            values = tr.get("portfolio_values") or []
            if dates and values and len(dates) == len(values):
                try:
                    idx = pd.to_datetime(dates)
                    s = pd.Series(values, index=idx).sort_index()
                    equity_series_by_ticker[tkr] = s
                except Exception:
                    continue
        
        portfolio_values: List[float] = []
        portfolio_dates: List[str] = []
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        annual_return = 0.0
        
        if equity_series_by_ticker:
            eq_df = pd.concat(equity_series_by_ticker.values(), axis=1)
            eq_df = eq_df.sort_index().ffill()
            # If any series starts later, fill initial gaps with its first available value
            eq_df = eq_df.apply(lambda col: col.bfill())
            
            portfolio_eq = eq_df.sum(axis=1)
            # Save equity curve for reporting/plots
            portfolio_dates = [d.date().isoformat() for d in portfolio_eq.index]
            portfolio_values = [float(v) for v in portfolio_eq.values]
            
            # Daily returns for Sharpe (same as SynTrade)
            rets = portfolio_eq.pct_change().dropna()
            if len(rets) > 2 and rets.std(ddof=1) and float(rets.std(ddof=1)) > 0:
                sharpe_ratio = float((rets.mean() / rets.std(ddof=1)) * np.sqrt(252))
            
            # Max drawdown from equity curve (same as SynTrade)
            running_max = portfolio_eq.cummax()
            drawdowns = (portfolio_eq / running_max) - 1.0
            max_drawdown = float(abs(drawdowns.min())) if not drawdowns.empty else 0.0
            
            # Annualized return (percent) from start/end (same as SynTrade)
            try:
                start_dt = portfolio_eq.index.min()
                end_dt = portfolio_eq.index.max()
                days = (end_dt - start_dt).days
                if days > 0 and total_initial > 0:
                    years = days / 365.25
                    if years > 0:
                        annual_return = float(((1 + portfolio_return) ** (1 / years) - 1) * 100.0)
            except Exception:
                pass
        
        # Get start/end dates from ticker results
        start_dates = [r.get("portfolio_dates", [])[0] if r.get("portfolio_dates") else None for r in ticker_results.values()]
        end_dates = [r.get("portfolio_dates", [])[-1] if r.get("portfolio_dates") else None for r in ticker_results.values()]
        start_dates = [d for d in start_dates if d is not None]
        end_dates = [d for d in end_dates if d is not None]
        
        portfolio_results = {
            "tickers": list(ticker_results.keys()),
            "initial_cash": total_initial,
            "final_value": total_final,
            "total_return": portfolio_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "trades": portfolio_trades,
            "ticker_results": ticker_results,
            "start_date": min(start_dates) if start_dates else None,
            "end_date": max(end_dates) if end_dates else None,
            "portfolio_dates": portfolio_dates,
            "portfolio_values": portfolio_values,
        }
        
        logger.info(f"Portfolio backtest complete: Return={portfolio_return:.2%}, Trades={total_trades}")
        
        return portfolio_results
