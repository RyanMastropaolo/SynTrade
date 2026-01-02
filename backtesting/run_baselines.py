#!/usr/bin/env python3
"""
Run Baseline Strategy Backtests

This script runs both baseline strategies (buy-and-hold equal-weight and technical-only)
for comparison with SynTrade.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline.baseline_backtest_engine import BaselineBacktestEngine
from baseline.strategies import BuyAndHoldStrategy, TechnicalOnlyStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_tickers(tickers_str: str) -> List[str]:
    """Parse tickers from comma-separated string."""
    return [t.strip().upper() for t in tickers_str.split(',') if t.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline strategy backtests for comparison with SynTrade"
    )
    
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated list of tickers (e.g., 'AAPL,MSFT,GOOGL')"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["buy_and_hold", "technical_only", "both"],
        default="both",
        help="Which strategy to run (default: both)"
    )
    
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial capital (default: 100000)"
    )
    
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Commission rate (default: 0.001 = 0.1%%)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_results",
        help="Directory to save results (default: baseline_results)"
    )
    
    args = parser.parse_args()
    
    # Parse tickers
    tickers = parse_tickers(args.tickers)
    
    # Parse dates
    start_date = None
    end_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            parser.error(f"Invalid start date format: {args.start_date}. Use YYYY-MM-DD")
    
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            parser.error(f"Invalid end date format: {args.end_date}. Use YYYY-MM-DD")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("BASELINE STRATEGY BACKTESTS")
    print("=" * 70)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Initial Cash: ${args.initial_cash:,.2f}")
    print(f"Commission: {args.commission:.3%}")
    if start_date:
        print(f"Start Date: {start_date.date()}")
    if end_date:
        print(f"End Date: {end_date.date()}")
    print()
    
    results = {}
    
    # Initialize backtest engine
    engine = BaselineBacktestEngine(
        initial_cash=args.initial_cash,
        commission=args.commission,
        start_date=start_date,
        end_date=end_date
    )
    
    # Run buy-and-hold strategy
    if args.strategy in ["buy_and_hold", "both"]:
        print("Running Buy-and-Hold Equal-Weight Strategy...")
        print("-" * 70)
        
        bh_results = engine.run_portfolio_backtest(
            tickers=tickers,
            strategy_class=BuyAndHoldStrategy,
            strategy_params={'target_weight': 0.10}
        )
        results["buy_and_hold"] = bh_results
        
        if bh_results:
            print(f"\nBuy-and-Hold Results:")
            print(f"  Total Return: {bh_results['total_return']:.2%}")
            print(f"  Annual Return: {bh_results['annual_return']:.2%}")
            print(f"  Sharpe Ratio: {bh_results['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {bh_results['max_drawdown']:.2%}")
            print()
        
        # Save results
        output_file = output_dir / "buy_and_hold_results.json"
        with open(output_file, 'w') as f:
            json.dump(bh_results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")
        print()
    
    # Run technical-only strategy
    if args.strategy in ["technical_only", "both"]:
        print("Running Technical-Only Strategy...")
        print("-" * 70)
        
        tech_results = engine.run_portfolio_backtest(
            tickers=tickers,
            strategy_class=TechnicalOnlyStrategy,
            strategy_params={
                'target_weight': 0.10,
                'rebalance_frequency': 30,
                'rsi_oversold': 40,
                'rsi_overbought': 60,
                'first_rebalance_date': start_date.date() if start_date else None
            }
        )
        results["technical_only"] = tech_results
        
        if tech_results:
            print(f"\nTechnical-Only Results:")
            print(f"  Total Return: {tech_results['total_return']:.2%}")
            print(f"  Annual Return: {tech_results['annual_return']:.2%}")
            print(f"  Sharpe Ratio: {tech_results['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {tech_results['max_drawdown']:.2%}")
            print(f"  Total Trades: {tech_results['total_trades']}")
            print()
        
        # Save results
        output_file = output_dir / "technical_only_results.json"
        with open(output_file, 'w') as f:
            json.dump(tech_results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")
        print()
    
    # Print comparison if both strategies were run
    if args.strategy == "both" and results.get("buy_and_hold") and results.get("technical_only"):
        print("=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"{'Metric':<25} {'Buy-and-Hold':<20} {'Technical-Only':<20}")
        print("-" * 70)
        print(f"{'Total Return':<25} {results['buy_and_hold']['total_return']:>19.2%} {results['technical_only']['total_return']:>19.2%}")
        print(f"{'Annual Return':<25} {results['buy_and_hold']['annual_return']:>19.2%} {results['technical_only']['annual_return']:>19.2%}")
        print(f"{'Sharpe Ratio':<25} {results['buy_and_hold']['sharpe_ratio']:>19.2f} {results['technical_only']['sharpe_ratio']:>19.2f}")
        print(f"{'Max Drawdown':<25} {results['buy_and_hold']['max_drawdown']:>19.2%} {results['technical_only']['max_drawdown']:>19.2%}")
        if 'total_trades' in results['technical_only']:
            print(f"{'Total Trades':<25} {'N/A':>19} {results['technical_only']['total_trades']:>19}")
        print()
        
        # Save comparison
        comparison_file = output_dir / "comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Comparison saved to {comparison_file}")
    
    print("=" * 70)
    print("Backtest complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

