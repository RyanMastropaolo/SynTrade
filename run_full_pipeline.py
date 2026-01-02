#!/usr/bin/env python3
"""
Run Full Pipeline: A1 → A2 → A2b → A3 → A4

This script runs the complete SynTrade pipeline:
1. A1 Scraper: Gathers market snapshot data
2. A2 Discriminator: Extracts features and LLM signals
3. A2b Model Scorer: Scores with trained LightGBM models
4. A3 Verifier Critic: Validates outputs and provides final result
5. A4 Executor Policy: Makes trading decision and calculates position size

Usage:
    python run_full_pipeline.py AAPL
    python run_full_pipeline.py TSLA --output output.json
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root
    project_root = Path(__file__).parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loaded .env from {env_path}")
    else:
        # Try loading from current directory as fallback
        load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system env vars
    pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.A1_Scraper import gather_market_snapshot
from agents.A2_Discriminator import discriminate, build_analysis_report_from_snapshot, MarketSnapshot
from agents.A2b_Model_Scorer import score
from agents.A3_Verifier_Critic_Agent import verify
from agents.A4_Executor_Policy_Agent import decide
from utils.decision_logger import get_logger

def main():
    parser = argparse.ArgumentParser(
        description="Run full SynTrade pipeline: A1 → A2 → A2b → A3 → A4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_pipeline.py AAPL
  python run_full_pipeline.py TSLA --output results.json
  python run_full_pipeline.py MSFT --verbose
        """
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: prints to stdout)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress information"
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=None,
        help="Specific sources to fetch (news, filings, fundamentals, macro, technicals). Default: all"
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable decision logging (default: logging enabled)"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Historical date to run pipeline for (YYYY-MM-DD format). Default: current date"
    )
    
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    
    # Parse date if provided
    if args.date:
        try:
            snapshot_date = datetime.strptime(args.date, "%Y-%m-%d")
            snapshot_datetime = snapshot_date.replace(hour=16, minute=0, second=0)  # Market close time
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD format.")
            sys.exit(1)
    else:
        snapshot_datetime = datetime.now()
    
    print("=" * 70)
    print("SynTrade Full Pipeline Execution")
    print("=" * 70)
    print(f"Ticker: {ticker}")
    print(f"Timestamp: {snapshot_datetime.isoformat()}")
    if args.date:
        print(f"Historical Date: {args.date}")
    print()
    
    # Track actual execution time (not historical timestamp)
    pipeline_start_time = datetime.now()
    is_historical = args.date is not None
    
    try:
        # Step 1: A1 Scraper
        print("=" * 70)
        print("STEP 1: A1 Scraper Agent")
        print("=" * 70)
        if args.verbose:
            print(f"Gathering market snapshot for {ticker}...")
            if args.sources:
                print(f"Sources: {', '.join(args.sources)}")
            else:
                print("Sources: ALL (news, filings, fundamentals, macro, technicals)")
        
        snapshot = gather_market_snapshot(
            ticker=ticker,
            as_of=snapshot_datetime.isoformat(),
            requested_sources=args.sources
        )
        
        if args.verbose:
            print(f"✓ A1 Complete: {len(snapshot.get('sources', {}))} source types collected")
        print()
        
        # Step 2: A2 Discriminator
        print("=" * 70)
        print("STEP 2: A2 Discriminator Agent")
        print("=" * 70)
        if args.verbose:
            print("Extracting features and LLM signals...")
        
        a2_output = discriminate(snapshot)
        
        if args.verbose:
            print(f"✓ A2 Complete:")
            print(f"  - LLM Sentiment Score: {a2_output.get('llm_sentiment_score', 'N/A')}")
            print(f"  - LLM Credibility Score: {a2_output.get('llm_credibility_score', 'N/A')}")
            print(f"  - Key Factors: {len(a2_output.get('key_factors', []))} items")
            print(f"  - Extracted Signals: {len(a2_output.get('extracted_signals', {}).get('claims', []))} claims")
        print()
        
        # Step 3: A2b Model Scorer
        print("=" * 70)
        print("STEP 3: A2b Model Scorer Agent")
        print("=" * 70)
        if args.verbose:
            print("Scoring with trained LightGBM models...")
        
        # Prepare input for A2b (expects dict with a2_output and optional market_snapshot)
        a2b_input = {
            'ticker': ticker,
            'a2_output': a2_output,
            'market_snapshot': snapshot
        }
        a2b_output = score(a2b_input)
        
        if args.verbose:
            print(f"✓ A2b Complete:")
            print(f"  - Sentiment Label: {a2b_output.get('sentiment_label', 'N/A')}")
            print(f"  - Sentiment Score: {a2b_output.get('sentiment_score', 'N/A'):.3f}")
            print(f"  - Credibility Label: {a2b_output.get('credibility_label', 'N/A')}")
            print(f"  - Credibility Score: {a2b_output.get('credibility_score', 'N/A'):.3f}")
        print()
        
        # Step 4: A3 Verifier Critic
        print("=" * 70)
        print("STEP 4: A3 Verifier Critic Agent")
        print("=" * 70)
        if args.verbose:
            print("Validating outputs and checking for issues...")
        
        # Build analysis report from snapshot
        try:
            snapshot_obj = MarketSnapshot(**snapshot)
            analysis_report = build_analysis_report_from_snapshot(snapshot_obj)
        except Exception as e:
            logger.warning(f"Could not build analysis report: {e}")
            analysis_report = f"Ticker: {ticker}\nAs of: {snapshot.get('as_of', 'N/A')}\n"
        
        a3_output = verify(
            ticker=ticker,
            analysis_report=analysis_report,
            a2_output=a2_output,
            a2b_output=a2b_output,
            market_snapshot=snapshot,  # Pass snapshot for hybrid technical validation
            is_historical=is_historical  # Skip stale check in historical mode
        )
        
        if args.verbose:
            print(f"✓ A3 Complete:")
            print(f"  - Verdict: {a3_output.get('verdict', 'N/A').upper()}")
            print(f"  - Confidence: {a3_output.get('confidence_0_1', 0):.2%}")
            print(f"  - Issues Found: {len(a3_output.get('issues', []))}")
            if a3_output.get('issues'):
                for issue in a3_output['issues']:
                    print(f"    - {issue['type']} ({issue['severity']}): {issue['message']}")
        print()
        
        # Use A3's final result (which may be modified or fallback)
        final_result = a3_output.get('final_scored_result', a2b_output)
        
        # Step 5: A4 Executor Policy
        print("=" * 70)
        print("STEP 5: A4 Executor Policy Agent")
        print("=" * 70)
        if args.verbose:
            print("Making trading decision and calculating position size...")
        
        # Prepare market_state for risk adjustments
        market_state = {
            'technicals': snapshot.get('technicals', {}),
            'macro': snapshot.get('macro', {})
        }
        
        a4_output = decide(
            ticker=ticker,
            final_scored_result=final_result,
            portfolio_snapshot=None,  # Stub for now
            market_state=market_state,
            a3_verdict=a3_output.get('verdict')
        )
        
        if args.verbose:
            print(f"✓ A4 Complete:")
            print(f"  - Action: {a4_output.get('action', 'N/A').upper()}")
            print(f"  - Position Size: {a4_output.get('position_size_pct', 0):.2%}")
            print(f"  - Rationale:")
            for reason in a4_output.get('rationale', []):
                print(f"    • {reason}")
        print()
        
        # Prepare final output
        result = {
            "ticker": ticker,
            "timestamp": snapshot_datetime.isoformat(),
            "pipeline": {
                "a1_snapshot": snapshot,
                "a2_output": a2_output,
                "a2b_output": a2b_output,
                "a3_output": a3_output,
                "a4_output": a4_output
            },
            "verification": {
                "verdict": a3_output.get('verdict'),
                "confidence": a3_output.get('confidence_0_1'),
                "issues": a3_output.get('issues', [])
            },
            "summary": {
                "sentiment": {
                    "label": final_result.get('sentiment_label'),
                    "score": final_result.get('sentiment_score'),
                    "probabilities": {
                        "bearish": final_result.get('sentiment_probs', {}).get('bearish', 0),
                        "neutral": final_result.get('sentiment_probs', {}).get('neutral', 0),
                        "bullish": final_result.get('sentiment_probs', {}).get('bullish', 0)
                    }
                },
                "credibility": {
                    "label": final_result.get('credibility_label'),
                    "score": final_result.get('credibility_score'),
                    "probabilities": {
                        "low": final_result.get('credibility_probs', {}).get('low', 0),
                        "medium": final_result.get('credibility_probs', {}).get('medium', 0),
                        "high": final_result.get('credibility_probs', {}).get('high', 0)
                    }
                },
                "model_meta": final_result.get('model_meta', {})
            }
        }
        
        # Output results
        print("=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print()
        print(f"Ticker: {ticker}")
        print(f"Verification: {result['verification']['verdict'].upper()} (confidence: {result['verification']['confidence']:.2%})")
        if result['verification']['issues']:
            print(f"Issues: {len(result['verification']['issues'])} found")
        print()
        print(f"Sentiment: {result['summary']['sentiment']['label']} (score: {result['summary']['sentiment']['score']:.3f})")
        print(f"  Probabilities: Bearish={result['summary']['sentiment']['probabilities']['bearish']:.2%}, "
              f"Neutral={result['summary']['sentiment']['probabilities']['neutral']:.2%}, "
              f"Bullish={result['summary']['sentiment']['probabilities']['bullish']:.2%}")
        print(f"Credibility: {result['summary']['credibility']['label']} (score: {result['summary']['credibility']['score']:.3f})")
        print(f"  Probabilities: Low={result['summary']['credibility']['probabilities']['low']:.2%}, "
              f"Medium={result['summary']['credibility']['probabilities']['medium']:.2%}, "
              f"High={result['summary']['credibility']['probabilities']['high']:.2%}")
        print()
        print("=" * 70)
        print("TRADING DECISION (A4)")
        print("=" * 70)
        print(f"Action: {a4_output.get('action', 'N/A').upper()}")
        print(f"Position Size: {a4_output.get('position_size_pct', 0):.2%}")
        print(f"Order Preview: {a4_output.get('order_preview', {}).get('side', 'N/A').upper()} {ticker} @ Market")
        print()
        if args.verbose and a4_output.get('rationale'):
            print("Rationale:")
            for reason in a4_output['rationale']:
                print(f"  • {reason}")
            print()
        
        # Log decision (unless disabled)
        if not args.no_log:
            try:
                decision_logger = get_logger()
                # Calculate actual execution time (not time difference from historical timestamp)
                pipeline_end_time = datetime.now()
                actual_execution_time = (pipeline_end_time - pipeline_start_time).total_seconds()
                decision_id = decision_logger.log_decision(
                    ticker=ticker,
                    timestamp=result['timestamp'],
                    a1_snapshot=snapshot,
                    a2_output=a2_output,
                    a2b_output=a2b_output,
                    a3_output=a3_output,
                    a4_output=a4_output,
                    pipeline_metadata={
                        "execution_time_seconds": actual_execution_time,
                        "verbose": args.verbose,
                        "sources_requested": args.sources,
                        "is_historical": is_historical
                    }
                )
                if args.verbose:
                    print(f"✓ Decision logged: {decision_id}")
            except Exception as e:
                logger.warning(f"Failed to log decision: {e}")
        
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Full results saved to: {output_path}")
        else:
            print("=" * 70)
            print("Full JSON Output:")
            print("=" * 70)
            print(json.dumps(result, indent=2))
        
        print()
        print("=" * 70)
        print("✓ Pipeline execution complete!")
        print("=" * 70)
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ ERROR")
        print("=" * 70)
        print(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

