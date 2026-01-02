"""
Performance Metrics Calculator for Backtesting Results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate comprehensive performance metrics from backtest results."""
    
    @staticmethod
    def calculate_all_metrics(backtest_results: Dict) -> Dict:
        """
        Calculate all performance metrics from backtest results.
        
        Args:
            backtest_results: Results from BacktestEngine
        
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Basic return metrics
        metrics.update(MetricsCalculator.calculate_return_metrics(backtest_results))
        
        # Risk metrics
        metrics.update(MetricsCalculator.calculate_risk_metrics(backtest_results))
        
        # Trade-level metrics
        metrics.update(MetricsCalculator.calculate_trade_metrics(backtest_results))
        
        return metrics
    
    @staticmethod
    def calculate_return_metrics(results: Dict) -> Dict:
        """Calculate return-related metrics."""
        metrics = {}
        
        initial_cash = results.get("initial_cash", 0.0)
        final_value = results.get("final_value", 0.0)
        
        if initial_cash > 0:
            total_return = (final_value - initial_cash) / initial_cash
            metrics["total_return"] = total_return
            
            # Annualized return
            start_date = results.get("start_date")
            end_date = results.get("end_date")
            
            if start_date and end_date:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
                
                days = (end_date - start_date).days
                if days > 0:
                    years = days / 365.25
                    if years > 0:
                        annualized_return = (1 + total_return) ** (1 / years) - 1
                        metrics["annualized_return"] = annualized_return
                        metrics["period_days"] = days
                        metrics["period_years"] = years
        
        # Get annual return from analyzer if available
        if "annual_return" in results:
            metrics["annual_return_analyzer"] = results["annual_return"] / 100.0
        
        return metrics
    
    @staticmethod
    def calculate_risk_metrics(results: Dict) -> Dict:
        """Calculate risk-related metrics."""
        metrics = {}
        
        # Sharpe ratio
        if "sharpe_ratio" in results:
            sharpe = results["sharpe_ratio"]
            if sharpe is not None and not np.isnan(sharpe):
                metrics["sharpe_ratio"] = sharpe
        
        # Max drawdown
        if "max_drawdown" in results:
            max_dd = results["max_drawdown"]
            if max_dd is not None and not np.isnan(max_dd):
                metrics["max_drawdown"] = max_dd
        
        # Calmar ratio (annual return / max drawdown)
        annual_return = results.get("annual_return", 0.0)
        if isinstance(annual_return, (int, float)):
            annual_return = annual_return / 100.0 if annual_return > 1 else annual_return
        
        max_dd = metrics.get("max_drawdown", 0.0)
        if max_dd > 0 and annual_return != 0:
            calmar = annual_return / abs(max_dd)
            metrics["calmar_ratio"] = calmar
        
        return metrics
    
    @staticmethod
    def calculate_trade_metrics(results: Dict) -> Dict:
        """Calculate trade-level metrics."""
        metrics = {}
        
        trades = results.get("trades", [])
        if not trades:
            return metrics
        
        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades)
        
        if trades_df.empty:
            return metrics
        
        # Total trades
        metrics["total_trades"] = len(trades)
        
        # Win rate (simplified - would need exit prices for accurate calculation)
        buy_trades = trades_df[trades_df["action"] == "buy"]
        sell_trades = trades_df[trades_df["action"] == "sell"]
        
        metrics["buy_trades"] = len(buy_trades)
        metrics["sell_trades"] = len(sell_trades)
        
        # Average position size
        if "position_size_pct" in trades_df.columns:
            metrics["avg_position_size_pct"] = trades_df["position_size_pct"].mean()
            metrics["max_position_size_pct"] = trades_df["position_size_pct"].max()
            metrics["min_position_size_pct"] = trades_df["position_size_pct"].min()
        
        # Trades by verdict
        if "verdict" in trades_df.columns:
            verdict_counts = trades_df["verdict"].value_counts().to_dict()
            metrics["trades_by_verdict"] = verdict_counts
        
        # Average sentiment/credibility scores
        if "sentiment_score" in trades_df.columns:
            metrics["avg_sentiment_score"] = trades_df["sentiment_score"].mean()
        
        if "credibility_score" in trades_df.columns:
            metrics["avg_credibility_score"] = trades_df["credibility_score"].mean()
        
        return metrics
    
    @staticmethod
    def calculate_agent_metrics(decisions_df: pd.DataFrame, backtest_results: Dict) -> Dict:
        """
        Calculate agent-specific performance metrics.
        
        Args:
            decisions_df: DataFrame with all decisions
            backtest_results: Backtest results
        
        Returns:
            Dictionary with agent-specific metrics
        """
        metrics = {}
        
        if decisions_df.empty:
            return metrics
        
        # A3 Verification Impact
        if "verdict" in decisions_df.columns:
            verdict_performance = {}
            for verdict in ["approve", "revise", "reject"]:
                verdict_decisions = decisions_df[decisions_df["verdict"] == verdict]
                if not verdict_decisions.empty:
                    verdict_performance[verdict] = {
                        "count": len(verdict_decisions),
                        "avg_sentiment": verdict_decisions["sentiment_score"].mean(),
                        "avg_credibility": verdict_decisions["credibility_score"].mean(),
                    }
            metrics["a3_verdict_breakdown"] = verdict_performance
        
        # Sentiment accuracy (would need forward returns for full analysis)
        if "sentiment" in decisions_df.columns and "sentiment_score" in decisions_df.columns:
            sentiment_dist = decisions_df["sentiment"].value_counts().to_dict()
            metrics["sentiment_distribution"] = sentiment_dist
            
            avg_sentiment_by_label = decisions_df.groupby("sentiment")["sentiment_score"].mean().to_dict()
            metrics["avg_sentiment_by_label"] = avg_sentiment_by_label
        
        # Credibility impact
        if "credibility" in decisions_df.columns:
            credibility_dist = decisions_df["credibility"].value_counts().to_dict()
            metrics["credibility_distribution"] = credibility_dist
        
        return metrics
    
    @staticmethod
    def format_metrics_report(metrics: Dict) -> str:
        """Format metrics as a readable report string."""
        lines = []
        lines.append("=" * 70)
        lines.append("BACKTEST PERFORMANCE METRICS")
        lines.append("=" * 70)
        lines.append("")
        
        # Return metrics
        lines.append("RETURN METRICS:")
        if "total_return" in metrics:
            lines.append(f"  Total Return: {metrics['total_return']:.2%}")
        if "annualized_return" in metrics:
            lines.append(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        if "period_years" in metrics:
            lines.append(f"  Period: {metrics['period_years']:.2f} years")
        lines.append("")
        
        # Risk metrics
        lines.append("RISK METRICS:")
        if "sharpe_ratio" in metrics:
            lines.append(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        if "max_drawdown" in metrics:
            lines.append(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        if "calmar_ratio" in metrics:
            lines.append(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        lines.append("")
        
        # Trade metrics
        lines.append("TRADE METRICS:")
        if "total_trades" in metrics:
            lines.append(f"  Total Trades: {metrics['total_trades']}")
        if "buy_trades" in metrics:
            lines.append(f"  Buy Trades: {metrics['buy_trades']}")
        if "sell_trades" in metrics:
            lines.append(f"  Sell Trades: {metrics['sell_trades']}")
        if "avg_position_size_pct" in metrics:
            lines.append(f"  Avg Position Size: {metrics['avg_position_size_pct']:.2%}")
        lines.append("")
        
        return "\n".join(lines)
