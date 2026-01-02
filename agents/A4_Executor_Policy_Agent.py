"""
A4 Executor Policy Agent - Trading Decision and Position Sizing

This agent takes the final scored result from A3 and makes a trading decision
(buy/sell/hold) with appropriate position sizing based on signal strength,
confidence, and risk adjustments.
"""

import logging
import math
from typing import Dict, List, Literal, Optional, Any, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration (from A4_Executor_Policy_Agent.json)
# ============================================================================

# Decision scoring weights (hybrid mix: sentiment + credibility + confidence + technicals)
SENTIMENT_WEIGHT = 0.40  # Increased weight for sentiment (primary signal)
CREDIBILITY_WEIGHT = 0.20  # Reduced (less impact on direction)
CONFIDENCE_WEIGHT = 0.20  # Reduced (less impact on direction)
TECHNICALS_WEIGHT = 0.20  # Increased weight for technicals (confirmation/divergence)

# Action thresholds
BUY_THRESHOLD = 0.25  # Lowered to allow more buy signals
SELL_THRESHOLD = -0.2  # Raised (less negative) to allow sell signals with moderate bearishness
HOLD_BAND = 0.08  # Reduced hold band to allow more action

# Position sizing
BASE_POSITION_PCT = 0.05
MIN_POSITION_PCT = 0.01
MAX_POSITION_PCT = 0.15

# Credibility factors
CREDIBILITY_FACTORS = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.3
}

# Safety rules
MIN_CREDIBILITY_SCORE = 0.5
MIN_CONFIDENCE = 0.4

# Risk adjustments
VOLATILITY_MULTIPLIER = 10.0
REGIME_FACTORS = {
    "bull": 1.0,
    "neutral": 0.8,
    "bear": 0.5
}
UNCERTAINTY_THRESHOLD = 0.15  # If top two probs within 15%, apply penalty

# ============================================================================
# Pydantic Models
# ============================================================================

class OrderPreview(BaseModel):
    """Order preview structure."""
    ticker: str
    side: Literal["buy", "sell", "hold"]
    qty: Optional[float] = None
    order_type: Literal["market", "limit"] = "market"
    limit_price: Optional[float] = None


class A4Output(BaseModel):
    """A4 Executor Policy output schema."""
    action: Literal["buy", "sell", "hold"]
    position_size_pct: float = Field(ge=0.0, le=1.0)
    rationale: List[str] = Field(default_factory=list)
    order_preview: OrderPreview


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_technical_signal(market_state: Optional[Dict]) -> float:
    """
    Calculate technical signal score from RSI, MACD, and SMAs.
    
    Returns a score from -1 (bearish) to 1 (bullish) based on technical indicators.
    
    Args:
        market_state: Optional dict with technicals data
    
    Returns:
        Technical signal score (-1 to 1)
    """
    if not market_state or "technicals" not in market_state:
        return 0.0  # Neutral if no technicals available
    
    technicals = market_state.get("technicals", {})
    if not technicals:
        return 0.0
    
    rsi_14 = technicals.get("rsi_14", 50.0)
    macd = technicals.get("macd", 0.0)
    macd_signal = technicals.get("macd_signal", 0.0)
    sma_20 = technicals.get("sma_20", 0.0)
    sma_50 = technicals.get("sma_50", 0.0)
    current_price = technicals.get("sma_20", 0.0)  # Use SMA20 as price proxy
    
    signals = []
    
    # RSI signal: normalize RSI (0-100) to -1 to 1
    # RSI 50 = neutral (0), RSI > 50 = bullish, RSI < 50 = bearish
    rsi_signal = (rsi_14 - 50.0) / 50.0  # Maps 0-100 to -1 to 1
    rsi_signal = max(-1.0, min(1.0, rsi_signal))  # Clamp
    signals.append(("RSI", rsi_signal, 0.4))  # 40% weight for RSI
    
    # MACD signal: MACD > signal = bullish, MACD < signal = bearish
    if macd != 0.0 and macd_signal != 0.0:
        macd_diff = macd - macd_signal
        # Normalize MACD difference (typically ranges from -5 to +5, but can be larger)
        # Use tanh to normalize to -1 to 1 range
        macd_signal_norm = math.tanh(macd_diff / 2.0)  # Scale factor of 2.0
        signals.append(("MACD", macd_signal_norm, 0.35))  # 35% weight for MACD
    else:
        signals.append(("MACD", 0.0, 0.35))
    
    # SMA trend signal: price position relative to SMAs
    if current_price > 0 and sma_20 > 0 and sma_50 > 0:
        # Calculate trend strength based on position relative to SMAs
        above_sma20 = 1.0 if current_price > sma_20 else -1.0
        above_sma50 = 1.0 if current_price > sma_50 else -1.0
        sma_signal = (above_sma20 * 0.6 + above_sma50 * 0.4)  # Weighted average
        signals.append(("SMA", sma_signal, 0.25))  # 25% weight for SMA trend
    else:
        signals.append(("SMA", 0.0, 0.25))
    
    # Weighted average of all technical signals
    total_weight = sum(weight for _, _, weight in signals)
    if total_weight == 0:
        return 0.0
    
    technical_signal = sum(signal * weight for _, signal, weight in signals) / total_weight
    return max(-1.0, min(1.0, technical_signal))  # Clamp to -1 to 1


def calculate_decision_score(
    sentiment_score: float,
    credibility_score: float,
    model_confidence: float,
    market_state: Optional[Dict] = None
) -> float:
    """
    Calculate hybrid multi-factor decision score (sentiment + credibility + confidence + technicals).
    
    Formula: decision_score = (sentiment * 0.35) + (credibility_norm * 0.25) + 
             (confidence_norm * 0.25) + (technical_signal * 0.15)
    
    Args:
        sentiment_score: Sentiment score from -1 (bearish) to 1 (bullish)
        credibility_score: Credibility score from 0 to 1
        model_confidence: Model confidence from 0 to 1
        market_state: Optional dict with technicals data for hybrid scoring
    
    Returns:
        Decision score (can be negative for sell, positive for buy)
    """
    # Normalize credibility and confidence to -1 to 1 range for decision score
    # Credibility: 0-1 -> -0.5 to 0.5 (centered around 0)
    # But reduce impact: only use deviation from neutral (0.5) as signal strength
    credibility_normalized = (credibility_score - 0.5) * 1.5  # Reduced multiplier for less impact
    
    # Confidence: 0-1 -> -0.5 to 0.5 (centered around 0)
    # Reduce impact: confidence affects position sizing more than direction
    confidence_normalized = (model_confidence - 0.5) * 1.5  # Reduced multiplier for less impact
    
    # Calculate technical signal (hybrid addition)
    technical_signal = calculate_technical_signal(market_state)
    
    decision_score = (
        sentiment_score * SENTIMENT_WEIGHT +
        credibility_normalized * CREDIBILITY_WEIGHT +
        confidence_normalized * CONFIDENCE_WEIGHT +
        technical_signal * TECHNICALS_WEIGHT
    )
    
    return decision_score


def determine_action(decision_score: float) -> Literal["buy", "sell", "hold"]:
    """
    Determine action based on decision score and thresholds.
    
    Args:
        decision_score: Calculated decision score
    
    Returns:
        Action: buy, sell, or hold
    """
    # Force hold if within hold band
    if abs(decision_score) < HOLD_BAND:
        return "hold"
    
    if decision_score >= BUY_THRESHOLD:
        return "buy"
    elif decision_score <= SELL_THRESHOLD:
        return "sell"
    else:
        return "hold"


def calculate_volatility_factor(market_state: Optional[Dict]) -> float:
    """
    Calculate volatility adjustment factor based on ATR.
    
    Formula: volatility_factor = 1.0 / (1.0 + (atr_14 / current_price) * multiplier)
    
    Args:
        market_state: Optional dict with technicals data
    
    Returns:
        Volatility factor (0.0 to 1.0)
    """
    if not market_state or "technicals" not in market_state:
        return 1.0
    
    technicals = market_state.get("technicals", {})
    atr_14 = technicals.get("atr_14", 0.0)
    current_price = technicals.get("sma_20", 0.0)  # Use SMA20 as price proxy
    
    if atr_14 <= 0 or current_price <= 0:
        return 1.0
    
    volatility_ratio = (atr_14 / current_price) * VOLATILITY_MULTIPLIER
    volatility_factor = 1.0 / (1.0 + volatility_ratio)
    
    return max(0.1, min(1.0, volatility_factor))  # Clamp between 0.1 and 1.0


def calculate_regime_factor(market_state: Optional[Dict]) -> float:
    """
    Calculate regime adjustment factor based on market regime.
    
    Args:
        market_state: Optional dict with macro data
    
    Returns:
        Regime factor (0.0 to 1.0)
    """
    if not market_state or "macro" not in market_state:
        return 0.8  # Conservative fallback
    
    macro = market_state.get("macro", {})
    risk_regime = macro.get("risk_regime", "neutral")
    
    return REGIME_FACTORS.get(risk_regime.lower(), 0.8)


def calculate_uncertainty_factor(final_result: Dict) -> float:
    """
    Calculate uncertainty penalty factor based on probability distribution.
    
    If top two sentiment probabilities are within threshold, apply 0.5x penalty.
    
    Args:
        final_result: Final scored result from A3
    
    Returns:
        Uncertainty factor (0.5 or 1.0)
    """
    sentiment_probs = final_result.get("sentiment_probs", {})
    if not sentiment_probs:
        return 1.0
    
    probs = list(sentiment_probs.values())
    if len(probs) < 2:
        return 1.0
    
    # Sort probabilities descending
    sorted_probs = sorted(probs, reverse=True)
    max_prob = sorted_probs[0]
    second_max_prob = sorted_probs[1]
    
    # If top two are close, apply penalty
    if abs(max_prob - second_max_prob) < UNCERTAINTY_THRESHOLD:
        return 0.5
    
    return 1.0


def calculate_base_position_size(
    decision_score: float,
    model_confidence: float,
    credibility_label: str
) -> float:
    """
    Calculate base position size before risk adjustments.
    
    Formula: base_position_size = base_position_pct * abs(decision_score) * model_confidence * credibility_factor
    
    Args:
        decision_score: Decision score
        model_confidence: Model confidence (0-1)
        credibility_label: Credibility label (high/medium/low)
    
    Returns:
        Base position size as percentage (0.0 to 1.0)
    """
    credibility_factor = CREDIBILITY_FACTORS.get(credibility_label.lower(), 0.5)
    
    base_size = (
        BASE_POSITION_PCT *
        abs(decision_score) *
        model_confidence *
        credibility_factor
    )
    
    return max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, base_size))


def check_safety_rules(
    final_result: Dict,
    a3_verdict: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Check safety rules before allowing trade.
    
    Args:
        final_result: Final scored result from A3
        a3_verdict: A3 verdict (approve/revise/reject)
    
    Returns:
        Tuple of (can_trade: bool, reasons: List[str])
    """
    reasons = []
    
    # A3 reject override
    if a3_verdict == "reject":
        reasons.append("A3 verdict is 'reject' - forcing hold")
        return (False, reasons)
    
    # Check credibility score
    credibility_score = final_result.get("credibility_score", 0.0)
    if credibility_score < MIN_CREDIBILITY_SCORE:
        reasons.append(f"Credibility score ({credibility_score:.2f}) below minimum ({MIN_CREDIBILITY_SCORE})")
        return (False, reasons)
    
    # Check model confidence
    model_meta = final_result.get("model_meta", {})
    model_confidence = model_meta.get("confidence_0_1", 0.0)
    if model_confidence < MIN_CONFIDENCE:
        reasons.append(f"Model confidence ({model_confidence:.2f}) below minimum ({MIN_CONFIDENCE})")
        return (False, reasons)
    
    return (True, reasons)


# ============================================================================
# Main Entry Point
# ============================================================================

def decide(
    ticker: str,
    final_scored_result: Dict[str, Any],
    portfolio_snapshot: Optional[Dict[str, Any]] = None,
    market_state: Optional[Dict[str, Any]] = None,
    a3_verdict: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for A4 Executor Policy Agent.
    
    Makes trading decision (buy/sell/hold) and calculates position size
    based on signal strength, confidence, and risk adjustments.
    
    Args:
        ticker: Stock ticker symbol
        final_scored_result: Final scored result from A3 (A2b output or fallback)
        portfolio_snapshot: Optional portfolio snapshot (stub for now)
        market_state: Optional market state with technicals/macro data for risk adjustments
        a3_verdict: Optional A3 verdict (approve/revise/reject) for safety override
    
    Returns:
        Dict with action, position_size_pct, rationale, and order_preview
    """
    rationale = []
    
    # Extract scores from final result
    sentiment_score = final_scored_result.get("sentiment_score", 0.0)
    credibility_score = final_scored_result.get("credibility_score", 0.5)
    credibility_label = final_scored_result.get("credibility_label", "medium")
    model_meta = final_scored_result.get("model_meta", {})
    model_confidence = model_meta.get("confidence_0_1", 0.5)
    
    # Check safety rules
    can_trade, safety_reasons = check_safety_rules(final_scored_result, a3_verdict)
    if not can_trade:
        rationale.extend(safety_reasons)
        return A4Output(
            action="hold",
            position_size_pct=0.0,
            rationale=rationale,
            order_preview=OrderPreview(
                ticker=ticker,
                side="hold",
                qty=None,
                order_type="market",
                limit_price=None
            )
        ).model_dump()
    
    # Calculate technical signal for hybrid scoring
    technical_signal = calculate_technical_signal(market_state)
    
    # Calculate decision score (hybrid mix)
    decision_score = calculate_decision_score(
        sentiment_score=sentiment_score,
        credibility_score=credibility_score,
        model_confidence=model_confidence,
        market_state=market_state
    )
    rationale.append(f"Decision score: {decision_score:.3f} (sentiment={sentiment_score:.3f}, credibility={credibility_score:.3f}, confidence={model_confidence:.3f}, technicals={technical_signal:.3f})")
    
    # Determine action
    action = determine_action(decision_score)
    rationale.append(f"Action: {action.upper()} (threshold: buy>{BUY_THRESHOLD}, sell<{SELL_THRESHOLD})")
    
    # Calculate base position size
    base_position_size = calculate_base_position_size(
        decision_score=decision_score,
        model_confidence=model_confidence,
        credibility_label=credibility_label
    )
    rationale.append(f"Base position size: {base_position_size:.2%}")
    
    # Apply risk adjustments
    volatility_factor = calculate_volatility_factor(market_state)
    regime_factor = calculate_regime_factor(market_state)
    uncertainty_factor = calculate_uncertainty_factor(final_scored_result)
    
    if volatility_factor < 1.0:
        rationale.append(f"Volatility adjustment: {volatility_factor:.2f}x (ATR-based)")
    if regime_factor < 1.0:
        rationale.append(f"Regime adjustment: {regime_factor:.2f}x (market regime)")
    if uncertainty_factor < 1.0:
        rationale.append(f"Uncertainty penalty: {uncertainty_factor:.2f}x (close probabilities)")
    
    # Calculate final position size
    final_position_size = base_position_size * volatility_factor * regime_factor * uncertainty_factor
    final_position_size = max(MIN_POSITION_PCT, min(MAX_POSITION_PCT, final_position_size))
    
    rationale.append(f"Final position size: {final_position_size:.2%}")
    
    # Build order preview
    order_preview = OrderPreview(
        ticker=ticker,
        side=action,
        qty=None,  # Would be calculated based on portfolio value
        order_type="market",
        limit_price=None
    )
    
    # Build output
    output = A4Output(
        action=action,
        position_size_pct=final_position_size,
        rationale=rationale,
        order_preview=order_preview.model_dump()
    )
    
    logger.info(f"A4 Policy: {action.upper()} {ticker} @ {final_position_size:.2%} (score: {decision_score:.3f})")
    
    return output.model_dump()

