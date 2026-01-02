"""
Visualization functions for backtesting results
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BacktestVisualizer:
    """Create visualizations for backtest results."""
    
    @staticmethod
    def plot_equity_curve(
        backtest_results: Dict,
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Plot equity curve (portfolio value over time).
        
        Uses actual portfolio values tracked during backtest execution.
        Falls back to linear interpolation if portfolio values are not available.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        initial_cash = backtest_results.get("initial_cash", 0.0)
        final_value = backtest_results.get("final_value", 0.0)
        start_date = backtest_results.get("start_date")
        end_date = backtest_results.get("end_date")
        
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Try to use actual portfolio values if available
        portfolio_values = backtest_results.get("portfolio_values", [])
        portfolio_dates = backtest_results.get("portfolio_dates", [])
        
        if portfolio_values and portfolio_dates and len(portfolio_values) == len(portfolio_dates):
            # Use actual portfolio values tracked during backtest
            dates = pd.to_datetime(portfolio_dates)
            values = np.array(portfolio_values)
            
            ax.plot(dates, values, linewidth=2, label='Portfolio Value', color='#2E86AB')
            
            # Add fill under curve for better visualization
            ax.fill_between(dates, initial_cash, values, alpha=0.2, color='#2E86AB', 
                          where=(values >= initial_cash), label='Gain')
            ax.fill_between(dates, initial_cash, values, alpha=0.2, color='#A23B72', 
                          where=(values < initial_cash), label='Loss')
        else:
            # Fallback: Create simple equity curve (linear interpolation)
            logger.warning("Portfolio values not available, using linear interpolation")
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            if len(dates) > 0:
                # Simple linear interpolation (fallback)
                values = np.linspace(initial_cash, final_value, len(dates))
                ax.plot(dates, values, linewidth=2, label='Portfolio Value (Interpolated)', 
                       color='#2E86AB', linestyle='--', alpha=0.7)
        
        # Add initial capital reference line
        ax.axhline(y=initial_cash, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Initial Capital')
        
        # Add final value annotation
        if portfolio_values and portfolio_dates:
            final_date = pd.to_datetime(portfolio_dates[-1])
            ax.annotate(f'Final: ${final_value:,.2f}', 
                       xy=(final_date, final_value),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(portfolio_dates) // 12) if portfolio_dates else 3))
        plt.xticks(rotation=45, ha='right')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Equity curve saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_metrics_comparison(
        metrics: Dict,
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """Create a bar chart comparing key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Return metrics
        ax = axes[0, 0]
        return_metrics = {}
        if "total_return" in metrics:
            return_metrics["Total Return"] = metrics["total_return"] * 100
        if "annualized_return" in metrics:
            return_metrics["Annualized Return"] = metrics["annualized_return"] * 100
        
        if return_metrics:
            bars = ax.bar(return_metrics.keys(), return_metrics.values(), color='green', alpha=0.7)
            ax.set_ylabel('Return (%)')
            ax.set_title('Return Metrics')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Risk metrics
        ax = axes[0, 1]
        risk_metrics = {}
        if "sharpe_ratio" in metrics:
            risk_metrics["Sharpe Ratio"] = metrics["sharpe_ratio"]
        if "max_drawdown" in metrics:
            risk_metrics["Max Drawdown"] = abs(metrics["max_drawdown"]) * 100
        
        if risk_metrics:
            colors = ['blue' if 'Sharpe' in k else 'red' for k in risk_metrics.keys()]
            bars = ax.bar(risk_metrics.keys(), risk_metrics.values(), color=colors, alpha=0.7)
            ax.set_ylabel('Value')
            ax.set_title('Risk Metrics')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Trade metrics
        ax = axes[1, 0]
        trade_metrics = {}
        if "total_trades" in metrics:
            trade_metrics["Total Trades"] = metrics["total_trades"]
        if "buy_trades" in metrics:
            trade_metrics["Buy Trades"] = metrics["buy_trades"]
        if "sell_trades" in metrics:
            trade_metrics["Sell Trades"] = metrics["sell_trades"]
        
        if trade_metrics:
            bars = ax.bar(trade_metrics.keys(), trade_metrics.values(), color='orange', alpha=0.7)
            ax.set_ylabel('Count')
            ax.set_title('Trade Metrics')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Position sizing
        ax = axes[1, 1]
        if "avg_position_size_pct" in metrics:
            pos_metrics = {
                "Avg Position": metrics["avg_position_size_pct"] * 100
            }
            if "max_position_size_pct" in metrics:
                pos_metrics["Max Position"] = metrics["max_position_size_pct"] * 100
            if "min_position_size_pct" in metrics:
                pos_metrics["Min Position"] = metrics["min_position_size_pct"] * 100
            
            bars = ax.bar(pos_metrics.keys(), pos_metrics.values(), color='purple', alpha=0.7)
            ax.set_ylabel('Position Size (%)')
            ax.set_title('Position Sizing')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%', ha='center', va='bottom')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_agent_analysis(
        agent_metrics: Dict,
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """Plot agent-specific analysis (A3 verdict impact, etc.)."""
        if not agent_metrics:
            logger.warning("No agent metrics to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # A3 Verdict breakdown
        ax = axes[0]
        if "a3_verdict_breakdown" in agent_metrics:
            verdict_data = agent_metrics["a3_verdict_breakdown"]
            verdicts = list(verdict_data.keys())
            counts = [v["count"] for v in verdict_data.values()]
            
            bars = ax.bar(verdicts, counts, color=['green', 'orange', 'red'], alpha=0.7)
            ax.set_ylabel('Decision Count')
            ax.set_title('A3 Verification Verdict Distribution')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
        
        # Sentiment distribution
        ax = axes[1]
        if "sentiment_distribution" in agent_metrics:
            sent_data = agent_metrics["sentiment_distribution"]
            sentiments = list(sent_data.keys())
            counts = list(sent_data.values())
            
            colors = {'bullish': 'green', 'bearish': 'red', 'neutral': 'gray'}
            bar_colors = [colors.get(s.lower(), 'blue') for s in sentiments]
            
            bars = ax.bar(sentiments, counts, color=bar_colors, alpha=0.7)
            ax.set_ylabel('Count')
            ax.set_title('Sentiment Distribution')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Agent analysis saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def create_summary_report(
        backtest_results: Dict,
        metrics: Dict,
        agent_metrics: Optional[Dict] = None,
        output_path: Optional[Path] = None
    ) -> str:
        """Create a text summary report of backtest results."""
        lines = []
        lines.append("=" * 70)
        lines.append("SYNTRADE BACKTEST SUMMARY REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # Basic info
        lines.append("BACKTEST PARAMETERS:")
        lines.append(f"  Ticker: {backtest_results.get('ticker', 'N/A')}")
        lines.append(f"  Initial Cash: ${backtest_results.get('initial_cash', 0):,.2f}")
        lines.append(f"  Final Value: ${backtest_results.get('final_value', 0):,.2f}")
        if "start_date" in backtest_results:
            lines.append(f"  Start Date: {backtest_results['start_date']}")
        if "end_date" in backtest_results:
            lines.append(f"  End Date: {backtest_results['end_date']}")
        lines.append("")
        
        # Performance metrics
        lines.append("PERFORMANCE METRICS:")
        if "total_return" in metrics:
            lines.append(f"  Total Return: {metrics['total_return']:.2%}")
        if "annualized_return" in metrics:
            lines.append(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        if "sharpe_ratio" in metrics:
            lines.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        if "max_drawdown" in metrics:
            lines.append(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        lines.append("")
        
        # Trade metrics
        lines.append("TRADE STATISTICS:")
        if "total_trades" in metrics:
            lines.append(f"  Total Trades: {metrics['total_trades']}")
        if "buy_trades" in metrics:
            lines.append(f"  Buy Trades: {metrics['buy_trades']}")
        if "sell_trades" in metrics:
            lines.append(f"  Sell Trades: {metrics['sell_trades']}")
        lines.append("")
        
        # Agent metrics
        if agent_metrics:
            lines.append("AGENT ANALYSIS:")
            if "a3_verdict_breakdown" in agent_metrics:
                lines.append("  A3 Verification Breakdown:")
                for verdict, data in agent_metrics["a3_verdict_breakdown"].items():
                    lines.append(f"    {verdict.upper()}: {data['count']} decisions")
            lines.append("")
        
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Summary report saved to {output_path}")
        
        return report
    
    @staticmethod
    def plot_baseline_equity_curve(
        baseline_results_path: str,
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Plot equity curve for baseline strategies (Buy-and-Hold or Technical-Only).
        
        Reconstructs portfolio value over time from baseline results JSON file.
        
        Args:
            baseline_results_path: Path to baseline results JSON file
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        import json
        
        # Load baseline results
        with open(baseline_results_path, 'r') as f:
            results = json.load(f)
        
        strategy_name = results.get("strategy", "unknown")
        tickers = results.get("tickers", [])
        initial_cash = results.get("initial_cash", 100000.0)
        final_value = results.get("final_value", 0.0)
        ticker_results = results.get("ticker_results", {})
        
        if not ticker_results:
            logger.warning("No ticker results found in baseline results")
            return
        
        # Reconstruct portfolio value over time
        # For each ticker, fetch price data and calculate portfolio value
        import yfinance as yf
        from datetime import datetime
        
        # Get date range from first ticker's results
        first_ticker = list(ticker_results.keys())[0]
        first_result = ticker_results[first_ticker]
        
        # Fetch price data for all tickers to get date range
        all_dates = set()
        price_data = {}
        
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y", interval="1d")
            if not hist.empty:
                price_data[ticker] = hist
                all_dates.update(hist.index.date)
        
        if not all_dates:
            logger.warning("No price data available for equity curve reconstruction")
            return
        
        # Sort dates
        sorted_dates = sorted(all_dates)
        if not sorted_dates:
            return
        
        # Calculate portfolio value over time
        portfolio_values = []
        portfolio_dates = []
        
        cash_per_ticker = initial_cash / len(tickers) if tickers else initial_cash
        
        for date in sorted_dates:
            total_value = 0.0
            
            for ticker in tickers:
                if ticker not in price_data:
                    continue
                
                ticker_prices = price_data[ticker]
                date_prices = ticker_prices[ticker_prices.index.date <= date]
                
                if date_prices.empty:
                    # Use initial cash if no price data yet
                    total_value += cash_per_ticker
                    continue
                
                # Get latest price for this date
                latest_price = date_prices["Close"].iloc[-1]
                
                # Calculate position value based on strategy
                ticker_result = ticker_results.get(ticker, {})
                
                if strategy_name == "buy_and_hold_equal_weight":
                    # Buy-and-hold: constant position
                    # Calculate shares from initial cash and initial price
                    initial_price_data = ticker_prices.iloc[0] if len(ticker_prices) > 0 else None
                    if initial_price_data is not None:
                        initial_price = initial_price_data["Close"]
                        shares = int(cash_per_ticker / initial_price) if initial_price > 0 else 0
                        position_value = shares * latest_price
                        # Add remaining cash
                        remaining_cash = cash_per_ticker - (shares * initial_price)
                        total_value += position_value + remaining_cash
                    else:
                        total_value += cash_per_ticker
                
                elif strategy_name == "technical_only":
                    # Technical-only: reconstruct from trades
                    trades = ticker_result.get("trades", [])
                    shares = 0
                    cash = cash_per_ticker
                    
                    # Process trades up to this date
                    for trade in trades:
                        trade_date_str = trade.get("date", "")
                        if isinstance(trade_date_str, str):
                            try:
                                trade_date = datetime.strptime(trade_date_str, "%Y-%m-%d").date()
                            except:
                                continue
                        else:
                            trade_date = trade_date_str
                        
                        if trade_date <= date:
                            action = trade.get("action", "").lower()
                            trade_shares = trade.get("shares", 0)
                            trade_price = trade.get("price", 0)
                            
                            if action == "buy":
                                cost = trade_shares * trade_price
                                if cash >= cost:
                                    shares += trade_shares
                                    cash -= cost
                            elif action == "sell":
                                if shares >= trade_shares:
                                    shares -= trade_shares
                                    cash += trade_shares * trade_price
                    
                    # Calculate current position value
                    position_value = shares * latest_price + cash
                    total_value += position_value
                else:
                    # Fallback: use initial cash
                    total_value += cash_per_ticker
            
            portfolio_values.append(total_value)
            portfolio_dates.append(date)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if portfolio_values and portfolio_dates:
            dates = pd.to_datetime(portfolio_dates)
            values = np.array(portfolio_values)
            
            # Determine strategy color
            if strategy_name == "buy_and_hold_equal_weight":
                color = '#1f77b4'  # Blue
                label = 'Buy-and-Hold Equal-Weight'
            elif strategy_name == "technical_only":
                color = '#ff7f0e'  # Orange
                label = 'Technical-Only'
            else:
                color = '#2E86AB'
                label = strategy_name.replace('_', ' ').title()
            
            ax.plot(dates, values, linewidth=2, label=label, color=color)
            
            # Add fill under curve
            ax.fill_between(dates, initial_cash, values, alpha=0.2, color=color,
                          where=(values >= initial_cash), label='Gain' if values[-1] >= initial_cash else '')
            ax.fill_between(dates, initial_cash, values, alpha=0.2, color='#A23B72',
                          where=(values < initial_cash), label='Loss' if values[-1] < initial_cash else '')
        
        # Add initial capital reference line
        ax.axhline(y=initial_cash, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Initial Capital')
        
        # Add final value annotation
        if portfolio_values and portfolio_dates:
            final_date = pd.to_datetime(portfolio_dates[-1])
            final_val = portfolio_values[-1]
            ax.annotate(f'Final: ${final_val:,.2f}', 
                       xy=(final_date, final_val),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title(f'{label} Equity Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(portfolio_dates) // 12) if portfolio_dates else 3))
        plt.xticks(rotation=45, ha='right')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Baseline equity curve saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_strategy_comparison(
        syntrade_results: Dict,
        buy_and_hold_path: Optional[str] = None,
        technical_only_path: Optional[str] = None,
        save_path: Optional[Path] = None,
        show: bool = True
    ):
        """
        Plot comparison of all three strategies on a single equity curve.
        
        Args:
            syntrade_results: SynTrade backtest results dictionary
            buy_and_hold_path: Optional path to buy-and-hold results JSON
            technical_only_path: Optional path to technical-only results JSON
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        import json
        
        # Plot SynTrade
        syntrade_values = syntrade_results.get("portfolio_values", [])
        syntrade_dates = syntrade_results.get("portfolio_dates", [])
        
        if syntrade_values and syntrade_dates:
            dates = pd.to_datetime(syntrade_dates)
            values = np.array(syntrade_values)
            ax.plot(dates, values, linewidth=2.5, label='SynTrade', color='#2E86AB', alpha=0.9)
        
        # Plot Buy-and-Hold if provided
        if buy_and_hold_path:
            try:
                with open(buy_and_hold_path, 'r') as f:
                    bh_results = json.load(f)
                
                # Reconstruct buy-and-hold equity curve
                bh_values, bh_dates = BacktestVisualizer._reconstruct_baseline_equity_curve(bh_results)
                if bh_values and bh_dates:
                    dates = pd.to_datetime(bh_dates)
                    values = np.array(bh_values)
                    ax.plot(dates, values, linewidth=2.5, label='Buy-and-Hold Equal-Weight', 
                           color='#1f77b4', linestyle='-', alpha=0.9)
            except Exception as e:
                logger.warning(f"Could not plot buy-and-hold: {e}")
        
        # Plot Technical-Only if provided
        if technical_only_path:
            try:
                with open(technical_only_path, 'r') as f:
                    to_results = json.load(f)
                
                # Reconstruct technical-only equity curve
                to_values, to_dates = BacktestVisualizer._reconstruct_baseline_equity_curve(to_results)
                if to_values and to_dates:
                    dates = pd.to_datetime(to_dates)
                    values = np.array(to_values)
                    ax.plot(dates, values, linewidth=2.5, label='Technical-Only', 
                           color='#ff7f0e', linestyle='-', alpha=0.9)
            except Exception as e:
                logger.warning(f"Could not plot technical-only: {e}")
        
        # Add initial capital reference line
        initial_cash = syntrade_results.get("initial_cash", 100000.0)
        ax.axhline(y=initial_cash, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Initial Capital')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title('Strategy Comparison: Equity Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha='right')
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Strategy comparison plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def _reconstruct_baseline_equity_curve(results: Dict) -> tuple:
        """
        Helper method to reconstruct equity curve from baseline results.
        
        Returns:
            Tuple of (portfolio_values list, portfolio_dates list)
        """
        import yfinance as yf
        
        tickers = results.get("tickers", [])
        initial_cash = results.get("initial_cash", 100000.0)
        ticker_results = results.get("ticker_results", {})
        strategy_name = results.get("strategy", "")
        
        if not ticker_results:
            return [], []
        
        # Fetch price data for all tickers
        all_dates = set()
        price_data = {}
        
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y", interval="1d")
            if not hist.empty:
                price_data[ticker] = hist
                all_dates.update(hist.index.date)
        
        if not all_dates:
            return [], []
        
        sorted_dates = sorted(all_dates)
        cash_per_ticker = initial_cash / len(tickers) if tickers else initial_cash
        
        portfolio_values = []
        portfolio_dates = []
        
        for date in sorted_dates:
            total_value = 0.0
            
            for ticker in tickers:
                if ticker not in price_data:
                    total_value += cash_per_ticker
                    continue
                
                ticker_prices = price_data[ticker]
                date_prices = ticker_prices[ticker_prices.index.date <= date]
                
                if date_prices.empty:
                    total_value += cash_per_ticker
                    continue
                
                latest_price = date_prices["Close"].iloc[-1]
                ticker_result = ticker_results.get(ticker, {})
                
                if strategy_name == "buy_and_hold_equal_weight":
                    initial_price_data = ticker_prices.iloc[0] if len(ticker_prices) > 0 else None
                    if initial_price_data is not None:
                        initial_price = initial_price_data["Close"]
                        shares = int(cash_per_ticker / initial_price) if initial_price > 0 else 0
                        position_value = shares * latest_price
                        remaining_cash = cash_per_ticker - (shares * initial_price)
                        total_value += position_value + remaining_cash
                    else:
                        total_value += cash_per_ticker
                
                elif strategy_name == "technical_only":
                    trades = ticker_result.get("trades", [])
                    shares = 0
                    cash = cash_per_ticker
                    
                    for trade in trades:
                        trade_date_str = trade.get("date", "")
                        if isinstance(trade_date_str, str):
                            try:
                                from datetime import datetime
                                trade_date = datetime.strptime(trade_date_str, "%Y-%m-%d").date()
                            except:
                                continue
                        else:
                            trade_date = trade_date_str
                        
                        if trade_date <= date:
                            action = trade.get("action", "").lower()
                            trade_shares = trade.get("shares", 0)
                            trade_price = trade.get("price", 0)
                            
                            if action == "buy":
                                cost = trade_shares * trade_price
                                if cash >= cost:
                                    shares += trade_shares
                                    cash -= cost
                            elif action == "sell":
                                if shares >= trade_shares:
                                    shares -= trade_shares
                                    cash += trade_shares * trade_price
                    
                    position_value = shares * latest_price + cash
                    total_value += position_value
                else:
                    total_value += cash_per_ticker
            
            portfolio_values.append(total_value)
            portfolio_dates.append(date)
        
        return portfolio_values, portfolio_dates
