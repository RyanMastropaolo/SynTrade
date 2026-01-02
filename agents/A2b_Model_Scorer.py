"""
A2b Model Scorer Agent - LightGBM-based scoring for buy/sell/hold decisions.

This module takes A2's validated output and optionally A1 market snapshot,
builds a flat feature vector, and returns model probabilities.
"""

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:
    raise ImportError("pydantic>=2.0.0 is required. Install with: pip install pydantic>=2.0.0")

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required. Install with: pip install numpy")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    logging.warning("lightgbm not installed. Will use fallback heuristic.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class Entities(BaseModel):
    """Extracted entities from signals."""
    companies: List[str] = []
    people: List[str] = []
    locations: List[str] = []
    dates: List[str] = []


class ExtractedSignals(BaseModel):
    """Structured signals extracted from content."""
    claims: List[str] = []
    catalysts: List[str] = []
    risks: List[str] = []
    entities: Entities = Entities()


class FeatureBundle(BaseModel):
    """Deterministic feature bundle for ML."""
    source_count: int = 0
    sources: List[str] = []
    recency_hours_min: Optional[float] = None
    recency_hours_max: Optional[float] = None
    conflict_count: int = 0
    rumor_language_flag: bool = False
    numbers_present_flag: bool = False
    price_mentions_count: int = 0
    event_type: Literal[
        "earnings", "guidance", "macro", "analyst_rating",
        "product", "legal", "m&a", "unknown"
    ] = "unknown"


class A2Output(BaseModel):
    """A2 output schema - must match A2_Discriminator exactly."""
    ticker: str
    as_of: str
    llm_sentiment_score: float
    llm_credibility_score: float
    key_factors: List[str] = []
    extracted_signals: ExtractedSignals = ExtractedSignals()
    feature_bundle: FeatureBundle = FeatureBundle()


class SentimentProbs(BaseModel):
    """Sentiment probabilities (bearish/neutral/bullish)."""
    bearish: float = Field(ge=0.0, le=1.0)
    neutral: float = Field(ge=0.0, le=1.0)
    bullish: float = Field(ge=0.0, le=1.0)
    
    @model_validator(mode='after')
    def validate_sum(self):
        """Ensure probabilities sum to approximately 1."""
        total = self.bearish + self.neutral + self.bullish
        if abs(total - 1.0) > 1e-3:
            if total > 0:
                self.bearish = self.bearish / total
                self.neutral = self.neutral / total
                self.bullish = self.bullish / total
            else:
                self.bearish = 1/3
                self.neutral = 1/3
                self.bullish = 1/3
        return self


class CredibilityProbs(BaseModel):
    """Credibility probabilities (low/medium/high)."""
    low: float = Field(ge=0.0, le=1.0)
    medium: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)
    
    @model_validator(mode='after')
    def validate_sum(self):
        """Ensure probabilities sum to approximately 1."""
        total = self.low + self.medium + self.high
        if abs(total - 1.0) > 1e-3:
            if total > 0:
                self.low = self.low / total
                self.medium = self.medium / total
                self.high = self.high / total
            else:
                self.low = 1/3
                self.medium = 1/3
                self.high = 1/3
        return self


class ModelMeta(BaseModel):
    """Model metadata."""
    model_version: str
    confidence_0_1: float = Field(ge=0.0, le=1.0)
    top_features: List[str] = []


class A2bInput(BaseModel):
    """A2b input schema."""
    ticker: str
    analysis_report: Optional[str] = None
    a2_output: A2Output
    market_snapshot: Optional[Dict[str, Any]] = None


class A2bOutput(BaseModel):
    """A2b output schema - validated and JSON-serializable (matches JSON contract)."""
    ticker: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: Literal["bearish", "neutral", "bullish"]
    sentiment_probs: SentimentProbs
    credibility_score: float = Field(ge=0.0, le=1.0)
    credibility_label: Literal["low", "medium", "high"]
    credibility_probs: CredibilityProbs
    model_meta: ModelMeta


# ============================================================================
# Helper Functions
# ============================================================================

def load_json_if_path(x: Union[str, Dict, Any]) -> Optional[Dict]:
    """Load JSON from file path if x is a string path, else return as-is if dict."""
    if isinstance(x, str):
        path = Path(x)
        if path.exists() and path.suffix == '.json':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load JSON from {x}: {e}")
                return None
    elif isinstance(x, dict):
        return x
    return None


def parse_datetime_any(x: Union[str, int, float]) -> datetime:
    """Parse datetime from ISO-8601 string or epoch seconds."""
    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(float(x), tz=timezone.utc)
    elif isinstance(x, str):
        try:
            if x.endswith('Z'):
                x = x[:-1] + '+00:00'
            dt = datetime.fromisoformat(x)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            try:
                dt = datetime.strptime(x, '%Y-%m-%dT%H:%M:%S')
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return datetime.now(timezone.utc)
    else:
        return datetime.now(timezone.utc)


def flatten_features(a2_output: A2Output, market_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Build a flat feature vector from A2 output and optionally market snapshot.
    Returns dict with feature names as keys and numeric values.
    """
    features = {}
    
    # A2 feature_bundle features
    fb = a2_output.feature_bundle
    features['source_count'] = float(fb.source_count)
    features['recency_hours_min'] = fb.recency_hours_min if fb.recency_hours_min is not None else 0.0
    features['recency_hours_max'] = fb.recency_hours_max if fb.recency_hours_max is not None else 0.0
    features['conflict_count'] = float(fb.conflict_count)
    features['rumor_language_flag'] = 1.0 if fb.rumor_language_flag else 0.0
    features['numbers_present_flag'] = 1.0 if fb.numbers_present_flag else 0.0
    features['price_mentions_count'] = float(fb.price_mentions_count)
    
    # One-hot encode event_type
    event_types = ["earnings", "guidance", "macro", "analyst_rating", "product", "legal", "m&a", "unknown"]
    for et in event_types:
        features[f'event_type_{et}'] = 1.0 if fb.event_type == et else 0.0
    
    # A2 LLM scores
    features['llm_sentiment_score'] = a2_output.llm_sentiment_score
    features['llm_credibility_score'] = a2_output.llm_credibility_score
    
    # Extracted signals counts
    features['claims_count'] = float(len(a2_output.extracted_signals.claims))
    features['catalysts_count'] = float(len(a2_output.extracted_signals.catalysts))
    features['risks_count'] = float(len(a2_output.extracted_signals.risks))
    features['entities_companies_count'] = float(len(a2_output.extracted_signals.entities.companies))
    features['entities_people_count'] = float(len(a2_output.extracted_signals.entities.people))
    
    # Market snapshot features (if available)
    if market_snapshot:
        # Fundamentals
        if 'fundamentals' in market_snapshot and market_snapshot['fundamentals']:
            fund = market_snapshot['fundamentals']
            features['revenue_growth_ttm'] = fund.get('revenue_growth_ttm', 0.0) or 0.0
            features['eps_growth_ttm'] = fund.get('eps_growth_ttm', 0.0) or 0.0
            features['gross_margin'] = fund.get('gross_margin', 0.0) or 0.0
            features['operating_margin'] = fund.get('operating_margin', 0.0) or 0.0
            features['debt_to_equity'] = fund.get('debt_to_equity', 0.0) or 0.0
            features['free_cash_flow_margin'] = fund.get('free_cash_flow_margin', 0.0) or 0.0
            features['dividend_yield'] = fund.get('dividend_yield', 0.0) or 0.0
        else:
            for key in ['revenue_growth_ttm', 'eps_growth_ttm', 'gross_margin', 'operating_margin',
                       'debt_to_equity', 'free_cash_flow_margin', 'dividend_yield']:
                features[key] = 0.0
        
        # Macro
        if 'macro' in market_snapshot and market_snapshot['macro']:
            macro = market_snapshot['macro']
            features['policy_rate'] = macro.get('policy_rate', 0.0) or 0.0
            features['policy_rate_change_3m'] = macro.get('policy_rate_change_3m', 0.0) or 0.0
            features['inflation_yoy'] = macro.get('inflation_yoy', 0.0) or 0.0
            features['unemployment_rate'] = macro.get('unemployment_rate', 0.0) or 0.0
        else:
            for key in ['policy_rate', 'policy_rate_change_3m', 'inflation_yoy', 'unemployment_rate']:
                features[key] = 0.0
        
        # Technicals
        if 'technicals' in market_snapshot and market_snapshot['technicals']:
            tech = market_snapshot['technicals']
            features['rsi_14'] = tech.get('rsi_14', 0.0) or 0.0
            features['macd'] = tech.get('macd', 0.0) or 0.0
            features['macd_signal'] = tech.get('macd_signal', 0.0) or 0.0
            features['sma_20'] = tech.get('sma_20', 0.0) or 0.0
            features['sma_50'] = tech.get('sma_50', 0.0) or 0.0
            features['sma_200'] = tech.get('sma_200', 0.0) or 0.0
            features['volume_zscore_20'] = tech.get('volume_zscore_20', 0.0) or 0.0
            features['atr_14'] = tech.get('atr_14', 0.0) or 0.0
            features['trend_strength'] = tech.get('trend_strength', 0.0) or 0.0
        else:
            for key in ['rsi_14', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sma_200',
                       'volume_zscore_20', 'atr_14', 'trend_strength']:
                features[key] = 0.0
    else:
        # Fill with zeros if snapshot missing
        for key in ['revenue_growth_ttm', 'eps_growth_ttm', 'gross_margin', 'operating_margin',
                   'debt_to_equity', 'free_cash_flow_margin', 'dividend_yield',
                   'policy_rate', 'policy_rate_change_3m', 'inflation_yoy', 'unemployment_rate',
                   'rsi_14', 'macd', 'macd_signal', 'sma_20', 'sma_50', 'sma_200',
                   'volume_zscore_20', 'atr_14', 'trend_strength']:
            features[key] = 0.0
    
    return features


def load_lgbm_model(model_type: str = "sentiment") -> Optional[Any]:
    """
    Load LightGBM model from file path.
    
    Args:
        model_type: "sentiment" or "credibility"
    
    Returns:
        LightGBM Booster or None
    """
    if not lgb:
        return None
    
    # Try environment variable first
    env_var = f"SYNTRADE_LGBM_{model_type.upper()}_MODEL_PATH"
    model_path = os.getenv(env_var)
    
    if model_path is None:
        # Try default repo path
        repo_root = Path(__file__).parent.parent
        default_path = repo_root / "models" / f"{model_type}_lgbm.txt"
        if default_path.exists():
            model_path = str(default_path)
        else:
            # Try .pkl extension
            default_path_pkl = default_path.with_suffix('.pkl')
            if default_path_pkl.exists():
                model_path = str(default_path_pkl)
            else:
                return None
    
    if not Path(model_path).exists():
        logger.warning(f"Model file not found: {model_path}")
        return None
    
    try:
        if model_path.endswith('.pkl'):
            import joblib
            model = joblib.load(model_path)
        else:
            model = lgb.Booster(model_file=model_path)
        return model
    except Exception as e:
        logger.error(f"Failed to load LightGBM model from {model_path}: {e}")
        return None


def get_feature_order_from_config() -> Optional[List[str]]:
    """Load feature order from feature_config.json to match training order."""
    try:
        repo_root = Path(__file__).parent.parent
        config_path = repo_root / "models" / "feature_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('feature_names', [])
    except Exception as e:
        logger.warning(f"Could not load feature_config.json: {e}")
    return None


def predict_with_model(features: Dict[str, float], model: Any, num_classes: int = 3) -> Optional[np.ndarray]:
    """
    Predict probabilities using LightGBM model.
    
    Args:
        features: Feature dict with named features
        model: LightGBM Booster (uses Column_0, Column_1, etc.)
        num_classes: Expected number of classes
    
    Returns:
        Array of probabilities or None
    """
    if model is None:
        return None
    
    try:
        # Get feature order from config (matches training order)
        feature_order = get_feature_order_from_config()
        if feature_order is None:
            # Fallback: try to get from model (but these will be Column_0, Column_1, etc.)
            model_feature_names = model.feature_name()
            # Try to map Column_X to actual feature names by position
            # This assumes features dict keys match the order in feature_config.json
            feature_order = list(features.keys())[:len(model_feature_names)]
        
        # Build feature array in the correct order
        feature_array = np.array([[features.get(name, 0.0) for name in feature_order]], dtype=np.float32)
        
        # Predict probabilities
        probs = model.predict(feature_array)[0]
        
        # Handle different output shapes
        if len(probs.shape) == 0:
            # Single value - convert to array
            probs = np.array([probs])
        
        # If we have raw scores, convert to probabilities (softmax)
        if len(probs) == num_classes:
            # Apply softmax if needed (check if they sum to ~1)
            if abs(probs.sum() - 1.0) > 0.1:
                exp_probs = np.exp(probs - np.max(probs))
                probs = exp_probs / exp_probs.sum()
            return probs
        else:
            logger.warning(f"Unexpected probability shape: {probs.shape}, expected {num_classes}")
            return None
            
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        return None


def fallback_sentiment_probs(a2_output: A2Output) -> Dict[str, float]:
    """Fallback sentiment probabilities based on LLM sentiment score."""
    sentiment = a2_output.llm_sentiment_score
    
    if sentiment < -0.3:
        return {'bearish': 0.7, 'neutral': 0.2, 'bullish': 0.1}
    elif sentiment > 0.3:
        return {'bearish': 0.1, 'neutral': 0.2, 'bullish': 0.7}
    else:
        return {'bearish': 0.2, 'neutral': 0.6, 'bullish': 0.2}


def fallback_credibility_probs(a2_output: A2Output) -> Dict[str, float]:
    """Fallback credibility probabilities based on LLM credibility score (binary: medium/high only)."""
    credibility = a2_output.llm_credibility_score
    
    if credibility > 0.67:
        return {'low': 0.0, 'medium': 0.2, 'high': 0.8}
    else:
        return {'low': 0.0, 'medium': 0.8, 'high': 0.2}


# ============================================================================
# Main Entry Point
# ============================================================================

def score(input_payload: Union[str, Dict]) -> Dict[str, Any]:
    """
    Main entry point for A2b Model Scorer.
    
    Args:
        input_payload: Can be:
            - Dict with keys: ticker, analysis_report (optional), a2_output, market_snapshot (optional)
            - Path to JSON file containing the above
    
    Returns:
        Dict conforming to A2bOutput schema (JSON-serializable)
    """
    # Normalize input
    input_dict = None
    
    if isinstance(input_payload, str):
        loaded = load_json_if_path(input_payload)
        if loaded is not None:
            input_dict = loaded
        else:
            raise ValueError(f"Could not load input from: {input_payload}")
    elif isinstance(input_payload, dict):
        input_dict = input_payload
    else:
        raise ValueError(f"Invalid input_payload type: {type(input_payload)}")
    
    # Parse input
    try:
        # Handle flexible input shapes
        if 'a2_output' in input_dict:
            a2_output_dict = input_dict['a2_output']
        elif 'ticker' in input_dict and 'llm_sentiment_score' in input_dict:
            # Input is already A2 output shape
            a2_output_dict = input_dict
        else:
            raise ValueError("Input must contain 'a2_output' or be A2 output shape")
        
        a2_output = A2Output(**a2_output_dict)
        ticker = input_dict.get('ticker', a2_output.ticker)
        market_snapshot = input_dict.get('market_snapshot')
        
    except Exception as e:
        logger.error(f"Failed to parse input: {e}")
        raise ValueError(f"Invalid input structure: {e}")
    
    # Build feature vector
    features = flatten_features(a2_output, market_snapshot)
    feature_names = list(features.keys())
    
    # Load sentiment model
    sentiment_model = load_lgbm_model("sentiment")
    sentiment_fallback = (sentiment_model is None)
    
    # Predict sentiment probabilities
    if sentiment_model:
        sentiment_probs_array = predict_with_model(features, sentiment_model, num_classes=3)
        if sentiment_probs_array is not None:
            # Assume order: [bearish, neutral, bullish]
            sentiment_probs_dict = {
                'bearish': float(sentiment_probs_array[0]),
                'neutral': float(sentiment_probs_array[1]),
                'bullish': float(sentiment_probs_array[2])
            }
        else:
            sentiment_probs_dict = fallback_sentiment_probs(a2_output)
            sentiment_fallback = True
    else:
        sentiment_probs_dict = fallback_sentiment_probs(a2_output)
    
    sentiment_probs = SentimentProbs(**sentiment_probs_dict)
    sentiment_label = max(sentiment_probs_dict.items(), key=lambda x: x[1])[0]
    
    # Load credibility model
    credibility_model = load_lgbm_model("credibility")
    credibility_fallback = (credibility_model is None)
    
    # Predict credibility probabilities (binary: medium/high only)
    if credibility_model:
        credibility_probs_array = predict_with_model(features, credibility_model, num_classes=2)
        if credibility_probs_array is not None:
            # Binary order: [medium (0), high (1)]
            credibility_probs_dict = {
                'low': 0.0,  # Always 0.0 since low is excluded
                'medium': float(credibility_probs_array[0]),
                'high': float(credibility_probs_array[1])
            }
        else:
            credibility_probs_dict = fallback_credibility_probs(a2_output)
            credibility_fallback = True
    else:
        credibility_probs_dict = fallback_credibility_probs(a2_output)
    
    credibility_probs = CredibilityProbs(**credibility_probs_dict)
    credibility_label = max(credibility_probs_dict.items(), key=lambda x: x[1])[0]
    
    # Compute final sentiment and credibility scores
    # Sentiment: weighted combination of LLM score and model probabilities
    sentiment_from_probs = (sentiment_probs.bullish - sentiment_probs.bearish)
    sentiment_score = a2_output.llm_sentiment_score * 0.5 + sentiment_from_probs * 0.5
    sentiment_score = max(-1.0, min(1.0, sentiment_score))
    
    # Credibility: weighted combination of LLM score and model probabilities
    credibility_from_probs = credibility_probs.high * 1.0 + credibility_probs.medium * 0.5
    credibility_score = a2_output.llm_credibility_score * 0.5 + credibility_from_probs * 0.5
    credibility_score = max(0.0, min(1.0, credibility_score))
    
    # Compute confidence (max probability)
    confidence = max(
        max(sentiment_probs_dict.values()),
        max(credibility_probs_dict.values())
    )
    
    # Get top features from model importance (use sentiment model if available, else credibility)
    top_features = []
    feature_order = get_feature_order_from_config()
    if feature_order and (sentiment_model or credibility_model):
        # Use sentiment model for feature importance (or credibility if sentiment not available)
        model_for_importance = sentiment_model if sentiment_model else credibility_model
        try:
            importance = model_for_importance.feature_importance(importance_type='gain')
            # Map importance to feature names using feature_order
            feature_importance = list(zip(feature_order[:len(importance)], importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_features = [name for name, _ in feature_importance[:10]]
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            # Fallback to first 10 features
            top_features = feature_order[:10] if len(feature_order) > 10 else feature_order
    else:
        # Fallback: use first 10 feature names from dict
        top_features = list(features.keys())[:10] if len(features) > 10 else list(features.keys())
    
    # Determine model version
    fallback_used = sentiment_fallback or credibility_fallback
    model_version = "1.0.0" if not fallback_used else "fallback-heuristic"
    
    # Build model metadata
    model_meta = ModelMeta(
        model_version=model_version,
        confidence_0_1=confidence,
        top_features=top_features
    )
    
    # Build output
    output = A2bOutput(
        ticker=ticker,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        sentiment_probs=sentiment_probs,
        credibility_score=credibility_score,
        credibility_label=credibility_label,
        credibility_probs=credibility_probs,
        model_meta=model_meta
    )
    
    # Return as dict (JSON-serializable)
    return output.model_dump()


# Create alias for JSON contract compatibility
model_scorer = score

