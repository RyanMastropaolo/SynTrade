#!/usr/bin/env python3
"""
Run Baseline Backtests
Runs buy-and-hold and technical-only strategies using SynTrade's exact methodology
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline.baseline_backtest_engine import BaselineBacktestEngine
from baseline.strategies import BuyAndHoldStrategy, TechnicalOnlyStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_metrics(strategy_name: str, results: dict):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 70)
    print(f"{strategy_name.upper()} STRATEGY RESULTS")
    print("=" * 70)
    print(f"Initial Cash: ${results['initial_cash']:,.2f}")
    print(f"Final Value: ${results['final_value']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annual Return: {results['annual_return']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Tickers: {len(results.get('tickers', []))}")
    if results.get('start_date') and results.get('end_date'):
        print(f"Period: {results['start_date']} to {results['end_date']}")
    print("=" * 70)


def main():
    # Exact parameters as specified
    TICKERS = ['AAPL', 'JPM', 'AMZN', 'LLY', 'NVDA', 'CVX', 'GOOGL', 'CAT', 'MSFT', 'JNJ']
    INITIAL_CASH = 10000000.0  # 10 million total (1 million per stock)
    COMMISSION = 0.001  # 0.1%
    START_DATE = datetime(2023, 12, 15)
    END_DATE = datetime(2025, 12, 15)
    
    # Create output directories
    output_dir = Path(__file__).parent.parent / "new_baseline_backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("BASELINE BACKTESTS - USING SYNTRADE METHODOLOGY")
    print("=" * 70)
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Initial Cash: ${INITIAL_CASH:,.2f}")
    print(f"Commission: {COMMISSION:.3%}")
    print(f"Date Range: {START_DATE.date()} to {END_DATE.date()}")
    print()
    
    # Initialize backtest engine
    engine = BaselineBacktestEngine(
        initial_cash=INITIAL_CASH,
        commission=COMMISSION,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # Run Buy-and-Hold Strategy
    print("Running Buy-and-Hold Strategy...")
    buy_hold_results = engine.run_portfolio_backtest(
        tickers=TICKERS,
        strategy_class=BuyAndHoldStrategy,
        strategy_params={'target_weight': 0.10}
    )
    
    if buy_hold_results:
        print_metrics("Buy-and-Hold", buy_hold_results)
        
        # Save results
        buy_hold_output = output_dir / "buy_and_hold_results.json"
        with open(buy_hold_output, 'w') as f:
            json.dump({
                "strategy": "buy_and_hold",
                "results": buy_hold_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {buy_hold_output}")
    else:
        print("❌ Buy-and-Hold backtest failed")
    
    print("\n" + "-" * 70 + "\n")
    
    # Run Technical-Only Strategy
    print("Running Technical-Only Strategy...")
    technical_results = engine.run_portfolio_backtest(
        tickers=TICKERS,
        strategy_class=TechnicalOnlyStrategy,
        strategy_params={
            'target_weight': 0.10,
            'rebalance_frequency': 30,
            'rsi_oversold': 40,
            'rsi_overbought': 60,
            'first_rebalance_date': START_DATE.date()  # Start trading on 2023-12-15
        }
    )
    
    if technical_results:
        print_metrics("Technical-Only", technical_results)
        
        # Save results
        technical_output = output_dir / "technical_only_results.json"
        with open(technical_output, 'w') as f:
            json.dump({
                "strategy": "technical_only",
                "results": technical_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, default=str)
        print(f"\n✓ Results saved to: {technical_output}")
    else:
        print("❌ Technical-Only backtest failed")
    
    # Create comparison summary
    if buy_hold_results and technical_results:
        comparison = {
            "buy_and_hold": {
                "total_return": buy_hold_results["total_return"],
                "annual_return": buy_hold_results["annual_return"],
                "sharpe_ratio": buy_hold_results["sharpe_ratio"],
                "max_drawdown": buy_hold_results["max_drawdown"],
                "total_trades": buy_hold_results["total_trades"],
            },
            "technical_only": {
                "total_return": technical_results["total_return"],
                "annual_return": technical_results["annual_return"],
                "sharpe_ratio": technical_results["sharpe_ratio"],
                "max_drawdown": technical_results["max_drawdown"],
                "total_trades": technical_results["total_trades"],
            },
            "timestamp": datetime.now().isoformat()
        }
        
        comparison_output = output_dir / "comparison.json"
        with open(comparison_output, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\n✓ Comparison saved to: {comparison_output}")
    
    print("\n" + "=" * 70)
    print("✓ All backtests complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
