"""
Backtrader Backtesting Engine for SynTrade
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.decision_loader import DecisionLoader
from backtesting.exit_conditions import ExitConditions

logger = logging.getLogger(__name__)


class SynTradeStrategy(bt.Strategy):
    """
    Backtrader strategy that executes trades based on SynTrade A4 decisions.
    """
    
    params = (
        ('decisions', None),  # DataFrame with decisions
        ('initial_cash', 100000.0),
        ('commission', 0.001),  # 0.1% commission
        ('disable_exits', False),  # If True, ignore exit_conditions (stop-loss/profit-taking/trend)
        # Backtest-only overlay to adjust exposure without regenerating decisions:
        # effective_position_size_pct = clamp(position_size_pct * position_scale, min_position_pct, max_position_pct)
        ('position_scale', 1.0),
        ('min_position_pct', 0.01),
        ('max_position_pct', 0.15),
    )
    
    def __init__(self):
        """Initialize the strategy."""
        self.decisions = self.params.decisions
        self.current_position = 0.0
        self.trades = []
        self.order = None
        
        # Track decision index
        self.decision_idx = 0
        
        # Exit conditions manager
        self.exit_conditions = ExitConditions()
        
        # Performance tracking
        self.entry_prices = []
        self.entry_dates = []
        self.exit_prices = []
        self.exit_dates = []
        self.position_sizes = []
        
        # Track portfolio value over time for equity curve
        self.portfolio_values = []
        self.portfolio_dates = []
        
        # Track processed decision IDs to avoid duplicates
        self.processed_decisions = set()
    
    def next(self):
        """Called for each bar. Execute trades based on decisions."""
        current_date = self.data.datetime.date(0)
        current_datetime = datetime.combine(current_date, datetime.min.time())
        
        # Track portfolio value at each bar for equity curve
        portfolio_value = self.broker.getvalue()
        self.portfolio_values.append(portfolio_value)
        self.portfolio_dates.append(current_date)
        
        current_price = self.data.close[0]
        
        # Get ticker from decisions (assume single ticker per backtest for now)
        ticker = "UNKNOWN"
        if not self.decisions.empty:
            date_decisions = self.decisions[
                self.decisions["timestamp"].dt.date <= current_date
            ]
            if not date_decisions.empty:
                ticker = date_decisions.iloc[-1].get("ticker", "UNKNOWN")
        
        # Step 1: Check exit conditions for existing positions (before A4 decisions)
        if not self.p.disable_exits and self.exit_conditions.has_position(ticker):
            # Get technicals from most recent decision (if available)
            technicals = None
            if not self.decisions.empty:
                date_decisions = self.decisions[
                    self.decisions["timestamp"].dt.date <= current_date
                ]
                if not date_decisions.empty:
                    latest_decision = date_decisions.iloc[-1]
                    # Extract technicals from decision DataFrame
                    if "technicals" in latest_decision.index:
                        tech_data = latest_decision["technicals"]
                        if isinstance(tech_data, dict) and tech_data:
                            technicals = tech_data
                        elif pd.notna(tech_data) and isinstance(tech_data, dict):
                            technicals = tech_data if tech_data else None
                        else:
                            technicals = None
            
            # Check exit conditions
            exit_action, exit_reason, sell_pct = self.exit_conditions.check_exit_conditions(
                ticker=ticker,
                current_price=current_price,
                current_date=current_datetime,
                technicals=technicals
            )
            
            if exit_action:
                shares_to_sell = self.exit_conditions.execute_sell(
                    ticker=ticker,
                    sell_pct=sell_pct,
                    exit_date=current_datetime,
                    reason=exit_reason
                )
                
                if shares_to_sell > 0:
                    # Execute the sell
                    if self.position.size > 0:
                        sell_size = min(int(shares_to_sell), int(self.position.size))
                        if sell_size > 0:
                            self.sell(size=sell_size)
                            self.trades.append({
                                "decision_id": f"exit_{exit_reason}_{current_date}",
                                "date": current_date,
                                "action": "sell" if sell_pct >= 1.0 else "sell_partial",
                                "shares": sell_size,
                                "price": current_price,
                                "position_size_pct": 0.0,
                                "exit_reason": exit_reason,
                                "sentiment_score": 0.0,
                                "credibility_score": 0.0,
                                "verdict": "exit_condition",
                            })
                            logger.info(f"Exit condition executed: {exit_reason} for {ticker}, sold {sell_size} shares")
        
        # Step 2: Process A4 decisions for new entries/adjustments
        if self.decisions is not None and not self.decisions.empty:
            # Get decisions for this date (or closest previous date)
            date_decisions = self.decisions[
                self.decisions["timestamp"].dt.date <= current_date
            ]
            
            if not date_decisions.empty:
                # Get the most recent decision for this date
                latest_decision = date_decisions.iloc[-1]
                
                # Only execute if this is a new decision (not already processed)
                decision_id = latest_decision.get("decision_id")
                if decision_id not in self.processed_decisions:
                    self.execute_decision(latest_decision, current_date, current_datetime, current_price)
                    self.processed_decisions.add(decision_id)
    
    def execute_decision(
        self, 
        decision: pd.Series, 
        current_date: datetime.date,
        current_datetime: datetime,
        current_price: float
    ):
        """Execute a trading decision from A4."""
        action = decision.get("action", "hold").lower()
        position_size_pct = float(decision.get("position_size_pct", 0.0) or 0.0)
        ticker = decision.get("ticker", "UNKNOWN")
        
        # Apply backtest-only exposure overlay (keeps historical decision logs unchanged).
        position_scale = float(getattr(self.p, "position_scale", 1.0) or 1.0)
        min_position_pct = float(getattr(self.p, "min_position_pct", 0.0) or 0.0)
        max_position_pct = float(getattr(self.p, "max_position_pct", 0.15) or 0.15)
        max_position_pct = max(0.0, min(max_position_pct, 1.0))
        min_position_pct = max(0.0, min(min_position_pct, max_position_pct))

        effective_position_size_pct = position_size_pct
        if action == "buy":
            scaled = position_size_pct * position_scale
            if scaled <= 0:
                return
            effective_position_size_pct = min(max(scaled, min_position_pct), max_position_pct)
        else:
            # For non-buy actions, keep original sizing (sells are handled via shares/position).
            effective_position_size_pct = position_size_pct
        
        # Skip if position size is too small to execute meaningfully
        if action == "buy" and effective_position_size_pct <= 0:
            return
        
        # Get decision score from rationale (if available)
        decision_score = 0.0
        rationale = decision.get("rationale", [])
        if isinstance(rationale, list) and rationale:
            # Try to extract decision score from rationale
            for r in rationale:
                if "Decision score:" in str(r):
                    try:
                        score_str = str(r).split("Decision score:")[1].split("(")[0].strip()
                        decision_score = float(score_str)
                        break
                    except:
                        pass
        
        # Get technicals from decision (if available)
        technicals = None
        if "technicals" in decision.index:
            tech_data = decision["technicals"]
            if isinstance(tech_data, dict) and tech_data:
                technicals = tech_data
            elif pd.notna(tech_data) and isinstance(tech_data, dict):
                technicals = tech_data if tech_data else None
            else:
                technicals = None
        
        # Check re-entry conditions (Rule 5)
        if action == "buy":
            if not self.exit_conditions.can_reenter(
                ticker=ticker,
                current_date=current_datetime,
                current_price=current_price,
                technicals=technicals,
                decision_score=decision_score
            ):
                logger.info(f"{ticker}: Re-entry blocked (cooldown or conditions not met)")
            return
        
        # Calculate position size in shares
        portfolio_value = self.broker.getvalue()
        target_value = portfolio_value * effective_position_size_pct
        
        # Enforce maximum position size (Rule 6)
        max_position_value = portfolio_value * max_position_pct
        target_value = min(target_value, max_position_value)
        
        shares = int(target_value / current_price)
        
        if shares == 0:
            return
        
        # Execute trade based on action
        if action == "buy":
            # Close any existing short position first
            if self.position.size < 0:
                self.close()
            
            # Check if we already have a position (for scale-in)
                current_shares = self.position.size if self.position.size > 0 else 0
            
            if current_shares == 0:
                # New position
                if shares > 0:
                    self.buy(size=shares)
                    self.exit_conditions.add_position(
                        ticker=ticker,
                        entry_price=current_price,
                        entry_date=current_datetime,
                        shares=float(shares),
                        decision_score=decision_score
                    )
                    self.trades.append({
                        "decision_id": decision.get("decision_id"),
                        "date": current_date,
                        "action": "buy",
                        "shares": shares,
                        "price": current_price,
                        "position_size_pct": effective_position_size_pct,
                        "sentiment_score": decision.get("sentiment_score", 0.0),
                        "credibility_score": decision.get("credibility_score", 0.0),
                        "verdict": decision.get("verdict", "approve"),
                    })
            else:
                # Scale in (Rule 6)
                shares_to_buy = max(0, shares - current_shares)
                if shares_to_buy > 0:
                    self.buy(size=shares_to_buy)
                    self.exit_conditions.add_position(
                        ticker=ticker,
                        entry_price=current_price,
                        entry_date=current_datetime,
                        shares=float(shares_to_buy),
                        decision_score=decision_score
                    )
                    self.trades.append({
                        "decision_id": decision.get("decision_id"),
                        "date": current_date,
                        "action": "buy_scale_in",
                        "shares": shares_to_buy,
                        "price": current_price,
                        "position_size_pct": effective_position_size_pct,
                        "sentiment_score": decision.get("sentiment_score", 0.0),
                        "credibility_score": decision.get("credibility_score", 0.0),
                        "verdict": decision.get("verdict", "approve"),
                    })
        
        elif action == "sell":
            # A4 sell decision (if not already handled by exit conditions)
            if self.position.size > 0 and not self.exit_conditions.has_position(ticker):
                # Position was already closed by exit conditions, skip
                return
            
            if self.position.size > 0:
                sell_size = min(shares, int(self.position.size))
                if sell_size > 0:
                    self.sell(size=sell_size)
                    # Update exit conditions
                    if self.exit_conditions.has_position(ticker):
                        self.exit_conditions.execute_sell(
                            ticker=ticker,
                            sell_pct=1.0,
                            exit_date=current_datetime,
                            reason="a4_sell_decision"
                        )
                self.trades.append({
                    "decision_id": decision.get("decision_id"),
                    "date": current_date,
                    "action": "sell",
                        "shares": sell_size,
                    "price": current_price,
                    "position_size_pct": position_size_pct,
                    "sentiment_score": decision.get("sentiment_score", 0.0),
                    "credibility_score": decision.get("credibility_score", 0.0),
                    "verdict": decision.get("verdict", "approve"),
                })
        
        # "hold" action - do nothing
    
    def notify_order(self, order):
        """Notify about order status."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.debug(f"BUY EXECUTED: {order.executed.size} shares @ {order.executed.price:.2f}")
            elif order.issell():
                logger.debug(f"SELL EXECUTED: {order.executed.size} shares @ {order.executed.price:.2f}")
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"Order {order.status}: {order.info}")
    
    def stop(self):
        """Called when backtest ends. Close any open positions."""
        if self.position.size != 0:
            self.close()


class BacktestEngine:
    """
    Main backtesting engine using Backtrader.
    """
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        log_file: Optional[str] = None,
        position_scale: float = 1.0,
        min_position_pct: float = 0.01,
        max_position_pct: float = 0.15,
        disable_exits: bool = False,
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_cash: Starting capital
            commission: Commission rate (0.001 = 0.1%)
            log_file: Path to decisions.jsonl file
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.position_scale = position_scale
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct
        self.disable_exits = disable_exits
        self.decision_loader = DecisionLoader(log_file)
        self.results = {}
    
    def run_backtest(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Run backtest for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date
            end_date: Optional end date
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for {ticker}...")
        
        # Prepare data
        decisions_df, price_data = self.decision_loader.prepare_backtest_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if decisions_df.empty:
            logger.warning(f"No decisions found for {ticker}")
            return {}
        
        if ticker not in price_data:
            logger.warning(f"No price data found for {ticker}")
            return {}
        
        # Filter decisions for this ticker
        ticker_decisions = decisions_df[decisions_df["ticker"] == ticker.upper()].copy()
        
        if ticker_decisions.empty:
            logger.warning(f"No decisions found for {ticker} after filtering")
            return {}
        
        # Prepare price data for Backtrader
        price_df = price_data[ticker].copy()
        price_df = price_df[["open", "high", "low", "close", "volume"]]
        
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
        
        # Add strategy with decisions
        cerebro.addstrategy(
            SynTradeStrategy,
            decisions=ticker_decisions,
            initial_cash=self.initial_cash,
            commission=self.commission,
            position_scale=self.position_scale,
            min_position_pct=self.min_position_pct,
            max_position_pct=self.max_position_pct,
            disable_exits=self.disable_exits,
        )
        
        # Set initial cash
        cerebro.broker.setcash(self.initial_cash)
        
        # Set commission
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        
        # Run backtest
        logger.info(f"Starting backtest for {ticker}...")
        strategies = cerebro.run()
        strategy = strategies[0]
        
        # Extract results
        analyzers = strategy.analyzers
        
        # Get final portfolio value
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        # Extract analyzer results
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
            "initial_cash": self.initial_cash,
            "final_value": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "total_trades": len(trades),
            "trades": trades,
            "trades_analysis": trades_analysis,
            "decisions_count": len(ticker_decisions),
            "start_date": ticker_decisions["timestamp"].min(),
            "end_date": ticker_decisions["timestamp"].max(),
            "portfolio_values": portfolio_values,  # For equity curve
            "portfolio_dates": portfolio_dates,    # For equity curve
        }
        
        logger.info(f"Backtest complete for {ticker}: Return={total_return:.2%}, Sharpe={sharpe:.2f}")
        
        return results
    
    def run_portfolio_backtest(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Run backtest across multiple tickers (portfolio).
        
        Args:
            tickers: Optional list of tickers. If None, uses all tickers in decisions.
            start_date: Optional start date
            end_date: Optional end date
        
        Returns:
            Dictionary with portfolio backtest results
        """
        logger.info("Running portfolio backtest...")
        
        # Prepare data
        decisions_df, price_data = self.decision_loader.prepare_backtest_data(
            ticker=None,
            start_date=start_date,
            end_date=end_date
        )
        
        if decisions_df.empty:
            logger.warning("No decisions found for portfolio backtest")
            return {}
        
        # Get tickers to backtest
        if tickers is None:
            tickers = decisions_df["ticker"].unique().tolist()
        
        # Run backtest for each ticker
        ticker_results = {}
        for ticker in tickers:
            if ticker in price_data:
                result = self.run_backtest(ticker, start_date, end_date)
                if result:
                    ticker_results[ticker] = result
        
        # Aggregate portfolio results
        if not ticker_results:
            logger.warning("No valid backtest results")
            return {}
        
        # Calculate portfolio metrics
        total_initial = sum(r["initial_cash"] for r in ticker_results.values())
        total_final = sum(r["final_value"] for r in ticker_results.values())
        portfolio_return = (total_final - total_initial) / total_initial if total_initial > 0 else 0.0
        
        total_trades = sum(r["total_trades"] for r in ticker_results.values())

        # Aggregate trades across tickers (optional but useful for metrics)
        portfolio_trades: List[Dict] = []
        for tkr, tr in ticker_results.items():
            for trade in tr.get("trades", []) or []:
                if isinstance(trade, dict):
                    trade_with_ticker = dict(trade)
                    trade_with_ticker.setdefault("ticker", tkr)
                    portfolio_trades.append(trade_with_ticker)

        # Build portfolio equity curve by summing per-ticker equity curves across aligned dates.
        # Each ticker backtest is run with its own cash bucket, so summing matches the
        # portfolio totals used above (total_initial/total_final).
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
        annual_return = 0.0  # percent, to match baseline outputs

        if equity_series_by_ticker:
            eq_df = pd.concat(equity_series_by_ticker.values(), axis=1)
            eq_df = eq_df.sort_index().ffill()
            # If any series starts later, fill initial gaps with its first available value
            eq_df = eq_df.apply(lambda col: col.fillna(method="bfill"))

            portfolio_eq = eq_df.sum(axis=1)
            # Save equity curve for reporting/plots
            portfolio_dates = [d.date().isoformat() for d in portfolio_eq.index]
            portfolio_values = [float(v) for v in portfolio_eq.values]

            # Daily returns for Sharpe
            rets = portfolio_eq.pct_change().dropna()
            if len(rets) > 2 and rets.std(ddof=1) and float(rets.std(ddof=1)) > 0:
                sharpe_ratio = float((rets.mean() / rets.std(ddof=1)) * np.sqrt(252))

            # Max drawdown from equity curve
            running_max = portfolio_eq.cummax()
            drawdowns = (portfolio_eq / running_max) - 1.0
            max_drawdown = float(abs(drawdowns.min())) if not drawdowns.empty else 0.0

            # Annualized return (percent) from start/end
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
            "start_date": min(r["start_date"] for r in ticker_results.values() if "start_date" in r),
            "end_date": max(r["end_date"] for r in ticker_results.values() if "end_date" in r),
            "portfolio_dates": portfolio_dates,
            "portfolio_values": portfolio_values,
        }
        
        logger.info(f"Portfolio backtest complete: Return={portfolio_return:.2%}, Trades={total_trades}")
        
        return portfolio_results
