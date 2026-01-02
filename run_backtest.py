#!/usr/bin/env python3
"""
Run Backtest - Backtest SynTrade decisions using Backtrader

This script runs backtests on logged SynTrade decisions and generates
performance metrics and visualizations.

Usage:
    python run_backtest.py AAPL
    python run_backtest.py --portfolio
    python run_backtest.py AAPL --start-date 2024-01-01 --end-date 2024-12-31
"""

import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtesting.backtest_engine import BacktestEngine
from backtesting.metrics import MetricsCalculator
from backtesting.visualization import BacktestVisualizer
from utils.decision_loader import DecisionLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Backtest SynTrade trading decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest single ticker
  python run_backtest.py AAPL
  
  # Backtest portfolio (all tickers)
  python run_backtest.py --portfolio
  
  # Backtest with date range
  python run_backtest.py AAPL --start-date 2024-01-01 --end-date 2024-12-31
  
  # Save results to file
  python run_backtest.py AAPL --output results.json
  
  # Generate visualizations
  python run_backtest.py AAPL --plots
        """
    )
    
    parser.add_argument(
        "ticker",
        nargs="?",
        type=str,
        help="Ticker symbol to backtest (optional if --portfolio is used)"
    )
    parser.add_argument(
        "--portfolio",
        action="store_true",
        help="Backtest all tickers in decision logs"
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
        "--pos-scale",
        type=float,
        default=1.0,
        help="Backtest-only multiplier for decision position_size_pct on BUY actions (default: 1.0)"
    )
    parser.add_argument(
        "--pos-min",
        type=float,
        default=0.01,
        help="Backtest-only minimum position size pct for BUY actions after scaling (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--pos-max",
        type=float,
        default=0.15,
        help="Backtest-only maximum position size pct for BUY actions after scaling (default: 0.15 = 15%%)"
    )
    parser.add_argument(
        "--disable-exits",
        action="store_true",
        help="Disable exit strategies (stop-loss / profit-taking / trend reversal) for this run"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to decisions.jsonl file (default: decision_logs/decisions.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backtest_results",
        help="Directory to save outputs (default: backtest_results)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.portfolio and not args.ticker:
        parser.error("Either provide a ticker or use --portfolio")
    
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
    
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_cash=args.initial_cash,
        commission=args.commission,
        log_file=args.log_file,
        position_scale=args.pos_scale,
        min_position_pct=args.pos_min,
        max_position_pct=args.pos_max,
        disable_exits=args.disable_exits,
    )
    
    print("=" * 70)
    print("SYNTRADE BACKTEST")
    print("=" * 70)
    print(f"Initial Cash: ${args.initial_cash:,.2f}")
    print(f"Commission: {args.commission:.3%}")
    if start_date:
        print(f"Start Date: {start_date.date()}")
    if end_date:
        print(f"End Date: {end_date.date()}")
    print()
    
    # Run backtest
    if args.portfolio:
        print("Running portfolio backtest...")
        results = engine.run_portfolio_backtest(
            tickers=None,
            start_date=start_date,
            end_date=end_date
        )
    else:
        print(f"Running backtest for {args.ticker}...")
        results = engine.run_backtest(
            ticker=args.ticker,
            start_date=start_date,
            end_date=end_date
        )
    
    if not results:
        print("❌ No backtest results generated. Check decision logs and price data availability.")
        return
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics_calc = MetricsCalculator()
    
    if args.portfolio:
        # Portfolio results
        portfolio_metrics = {}
        ticker_metrics = {}
        
        for ticker, ticker_result in results.get("ticker_results", {}).items():
            ticker_metrics[ticker] = metrics_calc.calculate_all_metrics(ticker_result)
        
        # Portfolio-level metrics (now computed inside BacktestEngine for portfolio runs)
        portfolio_metrics = metrics_calc.calculate_all_metrics(results)
        
        # Calculate agent metrics
        decision_loader = DecisionLoader(args.log_file)
        decisions_df = decision_loader.prepare_decisions_df()
        agent_metrics = metrics_calc.calculate_agent_metrics(decisions_df, results)
        
        # Print results
        print("\n" + "=" * 70)
        print("PORTFOLIO BACKTEST RESULTS")
        print("=" * 70)
        if "total_return" in portfolio_metrics:
            print(f"Total Return: {portfolio_metrics['total_return']:.2%}")
        if "annualized_return" in portfolio_metrics:
            print(f"Annual Return: {portfolio_metrics['annualized_return']:.2%}")
        if "sharpe_ratio" in portfolio_metrics:
            print(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
        if "max_drawdown" in portfolio_metrics:
            print(f"Max Drawdown: {portfolio_metrics['max_drawdown']:.2%}")
        if "total_trades" in portfolio_metrics:
            print(f"Total Trades: {portfolio_metrics['total_trades']}")
        print(f"Tickers: {len(results.get('ticker_results', {}))}")
        print()
        
        # Print per-ticker results
        print("Per-Ticker Results:")
        for ticker, ticker_result in results.get("ticker_results", {}).items():
            ticker_met = ticker_metrics[ticker]
            print(f"  {ticker}:")
            if "total_return" in ticker_met:
                print(f"    Return: {ticker_met['total_return']:.2%}")
            if "sharpe_ratio" in ticker_met:
                print(f"    Sharpe: {ticker_met['sharpe_ratio']:.2f}")
            print(f"    Trades: {ticker_result.get('total_trades', 0)}")
        
        # Generate visualizations for portfolio if requested
        if args.plots:
            print("\nGenerating visualizations...")
            visualizer = BacktestVisualizer()
            
            # Portfolio equity curve
            equity_path = output_dir / f"portfolio_equity_curve.png"
            visualizer.plot_equity_curve(results, save_path=equity_path, show=False)
            print(f"  ✓ Portfolio equity curve: {equity_path}")
            
            # Portfolio metrics
            metrics_path = output_dir / f"portfolio_metrics.png"
            visualizer.plot_metrics_comparison(portfolio_metrics, save_path=metrics_path, show=False)
            print(f"  ✓ Portfolio metrics: {metrics_path}")
            
            # Agent analysis
            if agent_metrics:
                agent_path = output_dir / f"portfolio_agent_analysis.png"
                visualizer.plot_agent_analysis(agent_metrics, save_path=agent_path, show=False)
                print(f"  ✓ Portfolio agent analysis: {agent_path}")
            
            # Summary report
            report_path = output_dir / f"portfolio_report.txt"
            visualizer.create_summary_report(results, portfolio_metrics, agent_metrics, output_path=report_path)
            print(f"  ✓ Portfolio summary report: {report_path}")
        
    else:
        # Single ticker results
        metrics = metrics_calc.calculate_all_metrics(results)
        
        # Calculate agent metrics
        decision_loader = DecisionLoader(args.log_file)
        decisions_df = decision_loader.prepare_decisions_df()
        agent_metrics = metrics_calc.calculate_agent_metrics(decisions_df, results)
        
        # Print results
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print(metrics_calc.format_metrics_report(metrics))
        
        # Generate visualizations
        if args.plots:
            print("\nGenerating visualizations...")
            visualizer = BacktestVisualizer()
            
            # Equity curve
            equity_path = output_dir / f"{args.ticker}_equity_curve.png"
            visualizer.plot_equity_curve(results, save_path=equity_path, show=False)
            print(f"  ✓ Equity curve: {equity_path}")
            
            # Metrics comparison
            metrics_path = output_dir / f"{args.ticker}_metrics.png"
            visualizer.plot_metrics_comparison(metrics, save_path=metrics_path, show=False)
            print(f"  ✓ Metrics comparison: {metrics_path}")
            
            # Agent analysis
            if agent_metrics:
                agent_path = output_dir / f"{args.ticker}_agent_analysis.png"
                visualizer.plot_agent_analysis(agent_metrics, save_path=agent_path, show=False)
                print(f"  ✓ Agent analysis: {agent_path}")
            
            # Summary report
            report_path = output_dir / f"{args.ticker}_report.txt"
            visualizer.create_summary_report(results, metrics, agent_metrics, output_path=report_path)
            print(f"  ✓ Summary report: {report_path}")
    
    # Save results to JSON (always save, auto-generate filename if not provided)
    if not args.output:
        # Auto-generate output filename if not provided
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.portfolio:
            args.output = str(output_dir / f"portfolio_backtest_{timestamp}.json")
        else:
            args.output = str(output_dir / f"{args.ticker}_backtest_{timestamp}.json")
    
    output_path = Path(args.output)
    if args.portfolio:
        output_metrics = portfolio_metrics
        output_agent_metrics = agent_metrics
    else:
        output_metrics = metrics
        output_agent_metrics = agent_metrics
    
    output_data = {
        "backtest_results": results,
        "metrics": output_metrics,
        "agent_metrics": output_agent_metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("✓ Backtest complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
