"""
A3 Verifier Critic Agent - Quality gate and validation for pipeline outputs.

This agent validates the outputs from A2 and A2b, checking for:
- Ticker mismatches
- Schema errors
- Stale data
- Contradictions
- Overconfidence
- Ungrounded claims
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Literal, Optional, Union
from pathlib import Path

try:
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("pydantic>=2.0.0 is required. Install with: pip install pydantic>=2.0.0")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class Issue(BaseModel):
    """Issue found during verification."""
    type: Literal["ungrounded", "contradiction", "stale", "ticker_mismatch", "schema_error", "overconfidence", "technical_contradiction"]
    severity: Literal["low", "medium", "high"]
    message: str
    affected_fields: List[str] = Field(default_factory=list)


class VerifierOutput(BaseModel):
    """A3 Verifier output schema."""
    verdict: Literal["approve", "revise", "reject"]
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    issues: List[Issue] = Field(default_factory=list)
    final_scored_result: Dict[str, Any]


# ============================================================================
# Configuration
# ============================================================================

STALE_WINDOW_HOURS = 72

CONSERVATIVE_FALLBACK = {
    "sentiment_score": 0.0,
    "sentiment_label": "neutral",
    "sentiment_probs": {
        "bearish": 0.33,
        "neutral": 0.34,
        "bullish": 0.33
    },
    "credibility_score": 0.25,
    "credibility_label": "low",
    "credibility_probs": {
        "low": 0.8,
        "medium": 0.15,
        "high": 0.05
    }
}

REJECT_RULES = [
    {"issue_type": "ticker_mismatch", "severity_at_least": "high"},
    {"issue_type": "schema_error", "severity_at_least": "high"}
]

REVISE_RULES = [
    {"issue_type": "overconfidence", "severity_at_least": "medium"},
    {"issue_type": "stale", "severity_at_least": "medium"},
    {"issue_type": "contradiction", "severity_at_least": "medium"},
    {"issue_type": "technical_contradiction", "severity_at_least": "medium"}
]


# ============================================================================
# Verification Functions
# ============================================================================

def parse_datetime_any(x: Union[str, int, float]) -> datetime:
    """Parse datetime from various formats."""
    if isinstance(x, str):
        try:
            # Try ISO-8601
            dt = datetime.fromisoformat(x.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except:
            pass
    elif isinstance(x, (int, float)):
        # Unix timestamp
        return datetime.fromtimestamp(x, tz=timezone.utc)
    return datetime.now(timezone.utc)


def check_ticker_mismatch(ticker: str, a2_output: Dict, a2b_output: Dict) -> Optional[Issue]:
    """Check if tickers match across all outputs."""
    issues = []
    
    a2_ticker = a2_output.get('ticker', '').upper()
    a2b_ticker = a2b_output.get('ticker', '').upper()
    expected_ticker = ticker.upper()
    
    if a2_ticker != expected_ticker:
        issues.append(f"A2 ticker mismatch: expected {expected_ticker}, got {a2_ticker}")
    if a2b_ticker != expected_ticker:
        issues.append(f"A2b ticker mismatch: expected {expected_ticker}, got {a2b_ticker}")
    if a2_ticker != a2b_ticker:
        issues.append(f"A2/A2b ticker mismatch: A2={a2_ticker}, A2b={a2b_ticker}")
    
    if issues:
        severity = "high" if len(issues) >= 2 else "medium"
        return Issue(
            type="ticker_mismatch",
            severity=severity,
            message="; ".join(issues),
            affected_fields=["ticker"]
        )
    return None


def check_schema_errors(a2_output: Dict, a2b_output: Dict) -> Optional[Issue]:
    """Check for schema/structural errors in outputs."""
    issues = []
    affected = []
    
    # Check A2b required fields
    required_a2b = ['sentiment_label', 'sentiment_score', 'credibility_label', 'credibility_score']
    for field in required_a2b:
        if field not in a2b_output:
            issues.append(f"Missing required A2b field: {field}")
            affected.append(field)
    
    # Check sentiment label validity
    if 'sentiment_label' in a2b_output:
        valid_sentiment = ['bearish', 'neutral', 'bullish']
        if a2b_output['sentiment_label'] not in valid_sentiment:
            issues.append(f"Invalid sentiment_label: {a2b_output['sentiment_label']}")
            affected.append('sentiment_label')
    
    # Check credibility label validity (should be medium or high only)
    if 'credibility_label' in a2b_output:
        valid_credibility = ['medium', 'high']
        if a2b_output['credibility_label'] not in valid_credibility:
            issues.append(f"Invalid credibility_label: {a2b_output['credibility_label']}")
            affected.append('credibility_label')
    
    # Check probability sums
    if 'sentiment_probs' in a2b_output:
        sent_probs = a2b_output['sentiment_probs']
        total = sum(sent_probs.values())
        if abs(total - 1.0) > 0.01:
            issues.append(f"Sentiment probabilities don't sum to 1.0: {total:.3f}")
            affected.append('sentiment_probs')
    
    if 'credibility_probs' in a2b_output:
        cred_probs = a2b_output['credibility_probs']
        total = sum(cred_probs.values())
        if abs(total - 1.0) > 0.01:
            issues.append(f"Credibility probabilities don't sum to 1.0: {total:.3f}")
            affected.append('credibility_probs')
    
    if issues:
        return Issue(
            type="schema_error",
            severity="high" if len(issues) >= 2 else "medium",
            message="; ".join(issues),
            affected_fields=affected
        )
    return None


def check_stale_data(a2_output: Dict, stale_window_hours: int = STALE_WINDOW_HOURS) -> Optional[Issue]:
    """Check if data is stale."""
    try:
        as_of_str = a2_output.get('as_of')
        if not as_of_str:
            return Issue(
                type="stale",
                severity="medium",
                message="Missing 'as_of' timestamp in A2 output",
                affected_fields=["as_of"]
            )
        
        as_of_dt = parse_datetime_any(as_of_str)
        now = datetime.now(timezone.utc)
        age_hours = (now - as_of_dt).total_seconds() / 3600
        
        if age_hours > stale_window_hours:
            return Issue(
                type="stale",
                severity="high" if age_hours > stale_window_hours * 2 else "medium",
                message=f"Data is {age_hours:.1f} hours old (threshold: {stale_window_hours}h)",
                affected_fields=["as_of"]
            )
    except Exception as e:
        return Issue(
            type="stale",
            severity="low",
            message=f"Could not parse timestamp: {e}",
            affected_fields=["as_of"]
        )
    return None


def check_overconfidence(a2b_output: Dict) -> Optional[Issue]:
    """Check for overconfident predictions (very high probabilities)."""
    issues = []
    affected = []
    
    # Check sentiment probabilities
    if 'sentiment_probs' in a2b_output:
        sent_probs = a2b_output['sentiment_probs']
        max_prob = max(sent_probs.values())
        if max_prob > 0.95:
            issues.append(f"Very high sentiment confidence: {max_prob:.2%}")
            affected.append('sentiment_probs')
    
    # Check credibility probabilities
    if 'credibility_probs' in a2b_output:
        cred_probs = a2b_output['credibility_probs']
        max_prob = max(cred_probs.values())
        if max_prob > 0.95:
            issues.append(f"Very high credibility confidence: {max_prob:.2%}")
            affected.append('credibility_probs')
    
    if issues:
        return Issue(
            type="overconfidence",
            severity="medium" if any("0.98" in msg or "0.99" in msg or "1.00" in msg for msg in issues) else "low",
            message="; ".join(issues),
            affected_fields=affected
        )
    return None


def check_contradictions(a2_output: Dict, a2b_output: Dict) -> Optional[Issue]:
    """Check for contradictions between A2 LLM scores and A2b model predictions."""
    issues = []
    affected = []
    
    # Compare sentiment
    a2_sentiment = a2_output.get('llm_sentiment_score', 0.0)
    a2b_sentiment = a2b_output.get('sentiment_score', 0.0)
    
    # Large discrepancy (more than 0.5 difference)
    if abs(a2_sentiment - a2b_sentiment) > 0.5:
        issues.append(f"Large sentiment discrepancy: LLM={a2_sentiment:.2f}, Model={a2b_sentiment:.2f}")
        affected.append('sentiment_score')
    
    # Compare credibility
    a2_credibility = a2_output.get('llm_credibility_score', 0.5)
    a2b_credibility = a2b_output.get('credibility_score', 0.5)
    
    # Large discrepancy (more than 0.4 difference)
    if abs(a2_credibility - a2b_credibility) > 0.4:
        issues.append(f"Large credibility discrepancy: LLM={a2_credibility:.2f}, Model={a2b_credibility:.2f}")
        affected.append('credibility_score')
    
    if issues:
        return Issue(
            type="contradiction",
            severity="medium" if len(issues) >= 2 else "low",
            message="; ".join(issues),
            affected_fields=affected
        )
    return None


def check_ungrounded(a2_output: Dict, analysis_report: str) -> Optional[Issue]:
    """Check for ungrounded claims (claims without supporting evidence)."""
    # Simple heuristic: if there are many claims but few news sources
    claims = a2_output.get('extracted_signals', {}).get('claims', [])
    key_factors = a2_output.get('key_factors', [])
    
    # Count news mentions in analysis report
    news_count = analysis_report.lower().count('headline:') if analysis_report else 0
    
    if len(claims) > 0 and news_count == 0:
        return Issue(
            type="ungrounded",
            severity="medium",
            message=f"{len(claims)} claims extracted but no news sources found",
            affected_fields=["extracted_signals.claims"]
        )
    
    if len(claims) > news_count * 2 and news_count > 0:
        return Issue(
            type="ungrounded",
            severity="low",
            message=f"{len(claims)} claims vs {news_count} news sources (high ratio)",
            affected_fields=["extracted_signals.claims"]
        )
    
    return None


def check_technical_contradictions(
    a2b_output: Dict,
    market_snapshot: Optional[Dict] = None
) -> Optional[Issue]:
    """
    Check for contradictions between sentiment predictions and technical indicators.
    
    This hybrid validation checks if:
    - Bullish sentiment but RSI is overbought (>70) or bearish technical signals
    - Bearish sentiment but RSI is oversold (<30) or bullish technical signals
    - MACD signals contradict sentiment direction
    - Price position relative to SMAs contradicts sentiment
    
    Args:
        a2b_output: Output from A2b Model Scorer
        market_snapshot: Optional market snapshot with technicals data
    
    Returns:
        Issue if contradiction found, None otherwise
    """
    if not market_snapshot or "technicals" not in market_snapshot:
        return None  # No technicals available, skip check
    
    technicals = market_snapshot.get("technicals", {})
    if not technicals:
        return None
    
    sentiment_score = a2b_output.get("sentiment_score", 0.0)
    sentiment_label = a2b_output.get("sentiment_label", "neutral")
    
    issues = []
    affected = []
    
    # Get technical indicators
    rsi_14 = technicals.get("rsi_14", 50.0)
    macd = technicals.get("macd", 0.0)
    macd_signal = technicals.get("macd_signal", 0.0)
    sma_20 = technicals.get("sma_20", 0.0)
    sma_50 = technicals.get("sma_50", 0.0)
    sma_200 = technicals.get("sma_200", 0.0)
    current_price = technicals.get("sma_20", 0.0)  # Use SMA20 as price proxy
    
    # RSI contradictions
    if sentiment_label == "bullish" and rsi_14 > 70:
        issues.append(f"Bullish sentiment but RSI overbought ({rsi_14:.1f} > 70)")
        affected.append("sentiment_label")
    elif sentiment_label == "bearish" and rsi_14 < 30:
        issues.append(f"Bearish sentiment but RSI oversold ({rsi_14:.1f} < 30)")
        affected.append("sentiment_label")
    
    # MACD contradictions
    # MACD > signal = bullish, MACD < signal = bearish
    macd_bullish = macd > macd_signal
    macd_bearish = macd < macd_signal
    
    if sentiment_label == "bullish" and macd_bearish and abs(macd - macd_signal) > 0.5:
        issues.append(f"Bullish sentiment but MACD bearish (MACD={macd:.2f} < Signal={macd_signal:.2f})")
        affected.append("sentiment_label")
    elif sentiment_label == "bearish" and macd_bullish and abs(macd - macd_signal) > 0.5:
        issues.append(f"Bearish sentiment but MACD bullish (MACD={macd:.2f} > Signal={macd_signal:.2f})")
        affected.append("sentiment_label")
    
    # SMA trend contradictions (only if we have valid SMAs)
    if current_price > 0 and sma_20 > 0 and sma_50 > 0:
        # Price above SMAs = bullish trend, below = bearish trend
        above_sma20 = current_price > sma_20
        above_sma50 = current_price > sma_50
        
        # Strong bullish trend: price above both SMAs
        strong_bullish_trend = above_sma20 and above_sma50
        # Strong bearish trend: price below both SMAs
        strong_bearish_trend = not above_sma20 and not above_sma50
        
        if sentiment_label == "bearish" and strong_bullish_trend:
            issues.append(f"Bearish sentiment but price above key SMAs (trend bullish)")
            affected.append("sentiment_label")
        elif sentiment_label == "bullish" and strong_bearish_trend:
            issues.append(f"Bullish sentiment but price below key SMAs (trend bearish)")
            affected.append("sentiment_label")
    
    if issues:
        # Determine severity based on number and strength of contradictions
        severity = "medium" if len(issues) >= 2 or abs(sentiment_score) > 0.6 else "low"
        return Issue(
            type="technical_contradiction",
            severity=severity,
            message="; ".join(issues),
            affected_fields=affected
        )
    
    return None


# ============================================================================
# Main Verification Logic
# ============================================================================

def determine_verdict(issues: List[Issue]) -> Literal["approve", "revise", "reject"]:
    """Determine verdict based on issues and rules."""
    # Check reject rules
    for rule in REJECT_RULES:
        issue_type = rule["issue_type"]
        min_severity = rule["severity_at_least"]
        severity_order = {"low": 0, "medium": 1, "high": 2}
        
        for issue in issues:
            if issue.type == issue_type:
                if severity_order[issue.severity] >= severity_order[min_severity]:
                    return "reject"
    
    # Check revise rules
    for rule in REVISE_RULES:
        issue_type = rule["issue_type"]
        min_severity = rule["severity_at_least"]
        severity_order = {"low": 0, "medium": 1, "high": 2}
        
        for issue in issues:
            if issue.type == issue_type:
                if severity_order[issue.severity] >= severity_order[min_severity]:
                    return "revise"
    
    # Default: approve
    return "approve"


def compute_confidence(issues: List[Issue]) -> float:
    """Compute confidence score based on issues."""
    if not issues:
        return 1.0
    
    severity_penalties = {"low": 0.1, "medium": 0.3, "high": 0.5}
    base_confidence = 1.0
    
    for issue in issues:
        base_confidence -= severity_penalties.get(issue.severity, 0.1)
    
    return max(0.0, min(1.0, base_confidence))


def create_fallback_result(a2b_output: Dict) -> Dict[str, Any]:
    """Create conservative fallback result when rejecting."""
    # Merge A2b output structure with conservative values
    fallback = a2b_output.copy()
    fallback.update(CONSERVATIVE_FALLBACK)
    return fallback


# ============================================================================
# Main Entry Point
# ============================================================================

def verify(
    ticker: str,
    analysis_report: str,
    a2_output: Dict[str, Any],
    a2b_output: Dict[str, Any],
    market_snapshot: Optional[Dict[str, Any]] = None,
    is_historical: bool = False
) -> Dict[str, Any]:
    """
    Main entry point for A3 Verifier Critic Agent.
    
    Args:
        ticker: Expected ticker symbol
        analysis_report: Analysis report string from A2
        a2_output: Output from A2 Discriminator
        a2b_output: Output from A2b Model Scorer
        market_snapshot: Optional market snapshot with technicals data for hybrid validation
        is_historical: If True, skip stale data check (for historical backtesting)
    
    Returns:
        Dict with verdict, confidence, issues, and final_scored_result
    """
    issues: List[Issue] = []
    
    # Run all checks (skip stale check in historical mode)
    checks = [
        check_ticker_mismatch(ticker, a2_output, a2b_output),
        check_schema_errors(a2_output, a2b_output),
    ]
    
    # Only check for stale data if not in historical mode
    if not is_historical:
        checks.append(check_stale_data(a2_output))
    
    # Add remaining checks
    checks.extend([
        check_overconfidence(a2b_output),
        check_contradictions(a2_output, a2b_output),
        check_ungrounded(a2_output, analysis_report),
        check_technical_contradictions(a2b_output, market_snapshot)  # Hybrid technical validation
    ])
    
    for check_result in checks:
        if check_result is not None:
            issues.append(check_result)
    
    # Determine verdict
    verdict = determine_verdict(issues)
    
    # Compute confidence
    confidence = compute_confidence(issues)
    
    # Determine final result
    if verdict == "reject":
        final_result = create_fallback_result(a2b_output)
    else:
        # Use A2b output as-is (or could apply revisions)
        final_result = a2b_output.copy()
    
    # Build output
    output = VerifierOutput(
        verdict=verdict,
        confidence_0_1=confidence,
        issues=[issue.model_dump() for issue in issues],
        final_scored_result=final_result
    )
    
    logger.info(f"A3 Verifier: {verdict.upper()} (confidence: {confidence:.2%}, {len(issues)} issues)")
    
    return output.model_dump()

