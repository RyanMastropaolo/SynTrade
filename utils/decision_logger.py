"""
Decision Logger - Track all pipeline outputs and decisions

This module provides structured logging for all SynTrade pipeline outputs,
including decisions from A4 Executor Policy Agent. Logs are stored in JSON
format for easy querying and analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid

logger = logging.getLogger(__name__)

# Default log directory
DEFAULT_LOG_DIR = Path("decision_logs")
DECISIONS_FILE = "decisions.jsonl"  # JSON Lines format for easy appending


class DecisionLogger:
    """
    Logger for SynTrade pipeline decisions and outputs.
    
    Stores decisions in JSON Lines format (one JSON object per line) for
    efficient appending and easy parsing.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize the decision logger.
        
        Args:
            log_dir: Directory to store log files. Defaults to 'decision_logs' in project root.
        """
        if log_dir is None:
            # Go up one level from utils/ to project root
            project_root = Path(__file__).parent.parent
            log_dir = project_root / DEFAULT_LOG_DIR
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.decisions_file = self.log_dir / DECISIONS_FILE
        
        logger.info(f"Decision logger initialized: {self.log_dir}")
    
    def log_decision(
        self,
        ticker: str,
        timestamp: str,
        a1_snapshot: Dict[str, Any],
        a2_output: Dict[str, Any],
        a2b_output: Dict[str, Any],
        a3_output: Dict[str, Any],
        a4_output: Dict[str, Any],
        pipeline_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a complete pipeline execution and decision.
        
        Args:
            ticker: Stock ticker symbol
            timestamp: ISO-8601 timestamp of execution
            a1_snapshot: Output from A1 Scraper Agent
            a2_output: Output from A2 Discriminator Agent
            a2b_output: Output from A2b Model Scorer Agent
            a3_output: Output from A3 Verifier Critic Agent
            a4_output: Output from A4 Executor Policy Agent
            pipeline_metadata: Optional metadata (e.g., execution time, errors)
        
        Returns:
            Decision ID (UUID string) for this log entry
        """
        decision_id = str(uuid.uuid4())
        
        log_entry = {
            "decision_id": decision_id,
            "ticker": ticker.upper(),
            "timestamp": timestamp,
            "pipeline": {
                "a1_snapshot": {
                    "ticker": a1_snapshot.get("ticker"),
                    "as_of": a1_snapshot.get("as_of"),
                    "sources_summary": {
                        "news_count": len(a1_snapshot.get("sources", {}).get("news", [])),
                        "filings_count": len(a1_snapshot.get("sources", {}).get("filings_links", []))
                    },
                    "fundamentals": a1_snapshot.get("fundamentals", {}),
                    "macro": a1_snapshot.get("macro", {}),
                    "technicals": a1_snapshot.get("technicals", {})
                },
                "a2_output": {
                    "ticker": a2_output.get("ticker"),
                    "llm_sentiment_score": a2_output.get("llm_sentiment_score"),
                    "llm_credibility_score": a2_output.get("llm_credibility_score"),
                    "key_factors_count": len(a2_output.get("key_factors", [])),
                    "extracted_signals_count": len(a2_output.get("extracted_signals", {}).get("claims", []))
                },
                "a2b_output": {
                    "ticker": a2b_output.get("ticker"),
                    "sentiment_label": a2b_output.get("sentiment_label"),
                    "sentiment_score": a2b_output.get("sentiment_score"),
                    "credibility_label": a2b_output.get("credibility_label"),
                    "credibility_score": a2b_output.get("credibility_score"),
                    "sentiment_probs": a2b_output.get("sentiment_probs", {}),
                    "credibility_probs": a2b_output.get("credibility_probs", {}),
                    "model_meta": a2b_output.get("model_meta", {})
                },
                "a3_output": {
                    "verdict": a3_output.get("verdict"),
                    "confidence_0_1": a3_output.get("confidence_0_1"),
                    "issues_count": len(a3_output.get("issues", [])),
                    "issues": a3_output.get("issues", [])
                },
                "a4_output": {
                    "action": a4_output.get("action"),
                    "position_size_pct": a4_output.get("position_size_pct"),
                    "rationale": a4_output.get("rationale", []),
                    "order_preview": a4_output.get("order_preview", {})
                }
            },
            "decision_summary": {
                "action": a4_output.get("action"),
                "position_size_pct": a4_output.get("position_size_pct"),
                "sentiment": a2b_output.get("sentiment_label"),
                "sentiment_score": a2b_output.get("sentiment_score"),
                "credibility": a2b_output.get("credibility_label"),
                "credibility_score": a2b_output.get("credibility_score"),
                "model_confidence": a2b_output.get("model_meta", {}).get("confidence_0_1"),
                "verification_verdict": a3_output.get("verdict"),
                "verification_confidence": a3_output.get("confidence_0_1")
            },
            "metadata": pipeline_metadata or {}
        }
        
        # Append to JSON Lines file
        try:
            with open(self.decisions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            logger.info(f"Decision logged: {decision_id} for {ticker} ({a4_output.get('action', 'N/A').upper()})")
            return decision_id
            
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
            raise
    
    def get_recent_decisions(
        self,
        limit: int = 10,
        ticker: Optional[str] = None,
        action: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent decisions from the log.
        
        Args:
            limit: Maximum number of decisions to return
            ticker: Optional filter by ticker
            action: Optional filter by action (buy/sell/hold)
        
        Returns:
            List of decision dictionaries, most recent first
        """
        if not self.decisions_file.exists():
            return []
        
        decisions = []
        
        try:
            # Read all lines (JSON Lines format)
            with open(self.decisions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        decision = json.loads(line)
                        
                        # Apply filters
                        if ticker and decision.get("ticker", "").upper() != ticker.upper():
                            continue
                        if action and decision.get("decision_summary", {}).get("action", "").lower() != action.lower():
                            continue
                        
                        decisions.append(decision)
            
            # Sort by timestamp (most recent first) and limit
            decisions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return decisions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to read decisions: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged decisions.
        
        Returns:
            Dictionary with statistics (total decisions, action counts, etc.)
        """
        if not self.decisions_file.exists():
            return {
                "total_decisions": 0,
                "by_action": {},
                "by_ticker": {},
                "by_verdict": {}
            }
        
        stats = {
            "total_decisions": 0,
            "by_action": {"buy": 0, "sell": 0, "hold": 0},
            "by_ticker": {},
            "by_verdict": {"approve": 0, "revise": 0, "reject": 0},
            "date_range": {"earliest": None, "latest": None}
        }
        
        try:
            with open(self.decisions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        decision = json.loads(line)
                        stats["total_decisions"] += 1
                        
                        # Count by action
                        action = decision.get("decision_summary", {}).get("action", "unknown")
                        if action in stats["by_action"]:
                            stats["by_action"][action] += 1
                        
                        # Count by ticker
                        ticker = decision.get("ticker", "unknown")
                        stats["by_ticker"][ticker] = stats["by_ticker"].get(ticker, 0) + 1
                        
                        # Count by verdict
                        verdict = decision.get("decision_summary", {}).get("verification_verdict", "unknown")
                        if verdict in stats["by_verdict"]:
                            stats["by_verdict"][verdict] += 1
                        
                        # Track date range
                        timestamp = decision.get("timestamp", "")
                        if timestamp:
                            if stats["date_range"]["earliest"] is None or timestamp < stats["date_range"]["earliest"]:
                                stats["date_range"]["earliest"] = timestamp
                            if stats["date_range"]["latest"] is None or timestamp > stats["date_range"]["latest"]:
                                stats["date_range"]["latest"] = timestamp
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            return stats


# Global logger instance
_default_logger: Optional[DecisionLogger] = None


def get_logger(log_dir: Optional[Path] = None) -> DecisionLogger:
    """
    Get or create the default decision logger instance.
    
    Args:
        log_dir: Optional custom log directory
    
    Returns:
        DecisionLogger instance
    """
    global _default_logger
    
    if _default_logger is None:
        _default_logger = DecisionLogger(log_dir)
    
    return _default_logger

