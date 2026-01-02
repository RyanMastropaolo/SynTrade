"""
A2 Discriminator Agent - LLM extraction + feature building from market snapshots.

This module processes market snapshots (from A1) or legacy analysis reports,
extracts structured signals using LLM, and computes deterministic feature bundles
for downstream ML scoring.
"""

import json
import os
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:
    raise ImportError("pydantic>=2.0.0 is required. Install with: pip install pydantic>=2.0.0")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from project root (2 levels up from this file)
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        # Try loading from current directory as fallback
        load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class NewsItem(BaseModel):
    """News article item from snapshot."""
    headline: str
    summary: Optional[str] = ""
    url: str
    source: Optional[str] = None
    datetime: Union[str, int, float]  # ISO-8601 string or epoch seconds


class SnapshotSources(BaseModel):
    """Sources section of market snapshot."""
    news: List[NewsItem] = []
    filings_links: List[Dict[str, Any]] = []


class MarketSnapshot(BaseModel):
    """Market snapshot structure from A1."""
    ticker: str
    as_of: Union[str, int, float]  # ISO-8601 or epoch seconds
    sources: SnapshotSources
    fundamentals: Optional[Dict[str, Any]] = None
    macro: Optional[Dict[str, Any]] = None
    technicals: Optional[Dict[str, Any]] = None
    freshness: Optional[Dict[str, Any]] = None


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
    """A2 output schema - validated and JSON-serializable."""
    ticker: str
    as_of: str  # ISO-8601 timezone-aware
    llm_sentiment_score: float = Field(ge=-1.0, le=1.0)
    llm_credibility_score: float = Field(ge=0.0, le=1.0)
    key_factors: List[str] = []
    extracted_signals: ExtractedSignals = ExtractedSignals()
    feature_bundle: FeatureBundle = FeatureBundle()

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ticker must be non-empty")
        return v.strip().upper()

    @model_validator(mode='after')
    def ensure_timezone_aware_as_of(self):
        """Ensure as_of is timezone-aware ISO-8601 string."""
        if isinstance(self.as_of, str):
            dt = parse_datetime_any(self.as_of)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            self.as_of = dt.isoformat()
        return self


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
    """
    Parse datetime from ISO-8601 string or epoch seconds.
    Returns timezone-aware datetime (UTC if missing tz).
    """
    if isinstance(x, (int, float)):
        # Epoch seconds
        dt = datetime.fromtimestamp(float(x), tz=timezone.utc)
        return dt
    elif isinstance(x, str):
        # Try ISO-8601
        try:
            # Handle Z suffix
            if x.endswith('Z'):
                x = x[:-1] + '+00:00'
            dt = datetime.fromisoformat(x)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            # Try other formats
            try:
                dt = datetime.strptime(x, '%Y-%m-%dT%H:%M:%S')
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                logger.warning(f"Could not parse datetime: {x}, using now()")
                return datetime.now(timezone.utc)
    else:
        return datetime.now(timezone.utc)


def build_analysis_report_from_snapshot(snapshot: MarketSnapshot) -> str:
    """Build a text analysis report from market snapshot for LLM processing."""
    lines = [f"Ticker: {snapshot.ticker}", f"As of: {snapshot.as_of}", ""]
    
    # News articles
    if snapshot.sources.news:
        lines.append("=== NEWS ARTICLES ===")
        for item in snapshot.sources.news:
            lines.append(f"Source: {item.source or 'Unknown'}")
            lines.append(f"Headline: {item.headline}")
            if item.summary:
                lines.append(f"Summary: {item.summary}")
            lines.append(f"URL: {item.url}")
            lines.append("")
    
    # Filings
    if snapshot.sources.filings_links:
        lines.append("=== SEC FILINGS ===")
        for filing in snapshot.sources.filings_links:
            lines.append(f"Filing: {filing.get('note', 'N/A')}")
            lines.append(f"URL: {filing.get('filings_url', 'N/A')}")
            lines.append("")
    
    # Fundamentals summary
    if snapshot.fundamentals:
        lines.append("=== FUNDAMENTALS ===")
        for key, value in snapshot.fundamentals.items():
            if value is not None:
                lines.append(f"{key}: {value}")
        lines.append("")
    
    # Macro summary
    if snapshot.macro:
        lines.append("=== MACRO INDICATORS ===")
        for key, value in snapshot.macro.items():
            if value is not None:
                lines.append(f"{key}: {value}")
        lines.append("")
    
    # Technicals summary
    if snapshot.technicals:
        lines.append("=== TECHNICAL INDICATORS ===")
        for key, value in snapshot.technicals.items():
            if value is not None:
                lines.append(f"{key}: {value}")
        lines.append("")
    
    return "\n".join(lines)


def compute_feature_bundle(snapshot: MarketSnapshot) -> FeatureBundle:
    """Compute deterministic feature bundle from snapshot."""
    news_items = snapshot.sources.news
    snapshot_dt = parse_datetime_any(snapshot.as_of)
    
    # Source count and unique sources
    sources_set = set()
    for item in news_items:
        if item.source:
            sources_set.add(item.source)
    source_count = len(sources_set)
    sources = sorted(list(sources_set))
    
    # Recency (hours from snapshot.as_of to each news datetime)
    recency_hours = []
    for item in news_items:
        try:
            item_dt = parse_datetime_any(item.datetime)
            delta = snapshot_dt - item_dt
            hours = delta.total_seconds() / 3600.0
            recency_hours.append(hours)
        except Exception as e:
            logger.debug(f"Could not compute recency for {item.datetime}: {e}")
    
    recency_hours_min = min(recency_hours) if recency_hours else None
    recency_hours_max = max(recency_hours) if recency_hours else None
    
    # Rumor language flag
    rumor_words = ['rumor', 'rumour', 'speculation', 'alleged', 'reportedly', 
                   'unconfirmed', 'might', 'could', 'possibly', 'perhaps',
                   'uncertain', 'unclear', 'unknown']
    rumor_language_flag = False
    all_text = " ".join([item.headline + " " + (item.summary or "") 
                         for item in news_items]).lower()
    for word in rumor_words:
        if word in all_text:
            rumor_language_flag = True
            break
    
    # Numbers present flag
    numbers_present_flag = bool(re.search(r'\d', all_text))
    
    # Price mentions count (regex for $123, 123.45, 10%, +5%, -3%, etc.)
    price_patterns = [
        r'\$\d+\.?\d*',  # $123 or $123.45
        r'\d+\.\d+%',     # 123.45%
        r'[+-]?\d+\.?\d*%',  # +5%, -3%
        r'\d+\.\d+',      # 123.45 (potential price)
    ]
    price_mentions_count = 0
    for pattern in price_patterns:
        matches = re.findall(pattern, all_text)
        price_mentions_count += len(matches)
    
    # Conflict count starts at 0 (LLM can increment if needed)
    conflict_count = 0
    
    # Event type defaults to unknown (LLM will suggest)
    event_type: Literal[
        "earnings", "guidance", "macro", "analyst_rating",
        "product", "legal", "m&a", "unknown"
    ] = "unknown"
    
    return FeatureBundle(
        source_count=source_count,
        sources=sources,
        recency_hours_min=recency_hours_min,
        recency_hours_max=recency_hours_max,
        conflict_count=conflict_count,
        rumor_language_flag=rumor_language_flag,
        numbers_present_flag=numbers_present_flag,
        price_mentions_count=price_mentions_count,
        event_type=event_type
    )


def call_llm_for_extraction(analysis_report: str, max_retries: int = 5) -> Optional[Dict[str, Any]]:
    """
    Call LLM (Gemini 2.0 Flash) to extract structured signals.
    Returns None if LLM is unavailable or fails.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No GEMINI_API_KEY found. Using fallback values.")
        return None
    
    try:
        # Try to import google.genai (new package)
        try:
            import google.genai as genai
        except ImportError:
            logger.warning("google-genai not installed. Install with: pip install google-genai")
            return None
        
        # Initialize client with API key
        client = genai.Client(api_key=api_key)
        
        # Use Gemini 2.0 Flash
        model_name = "gemini-2.0-flash-exp"
        try:
            model = client.get_model(model_name)
        except Exception:
            # Fallback to gemini-3-pro if flash not available
            try:
                model = client.get_model("gemini-3-pro")
            except Exception:
                # Final fallback to gemini-2.0-flash
                model = client.get_model("gemini-2.0-flash")
        
        prompt = f"""Analyze the following market data and extract structured signals. Output ONLY valid JSON, no markdown, no code blocks.

Market Data:
{analysis_report}

Extract and return a JSON object with this exact structure:
{{
  "llm_sentiment_score": <number between -1.0 and 1.0>,
  "llm_credibility_score": <number between 0.0 and 1.0>,
  "event_type": "<one of: earnings, guidance, macro, analyst_rating, product, legal, m&a, unknown>",
  "key_factors": ["factor1", "factor2", ...],
  "extracted_signals": {{
    "claims": ["claim1", "claim2", ...],
    "catalysts": ["catalyst1", "catalyst2", ...],
    "risks": ["risk1", "risk2", ...],
    "entities": {{
      "companies": ["company1", ...],
      "people": ["person1", ...],
      "locations": ["location1", ...],
      "dates": ["date1", ...]
    }}
  }},
  "conflict_count": <integer, 0 if no conflicts detected>
}}

Be precise and evidence-based. Only include conflicts if you detect direct contradictions between credible sources."""
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                text = response.text.strip()
                
                # Remove markdown code blocks if present
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                
                result = json.loads(text)
                
                # Validate ranges
                result['llm_sentiment_score'] = max(-1.0, min(1.0, float(result.get('llm_sentiment_score', 0.0))))
                result['llm_credibility_score'] = max(0.0, min(1.0, float(result.get('llm_credibility_score', 0.5))))
                
                # Validate event_type
                valid_types = ["earnings", "guidance", "macro", "analyst_rating", "product", "legal", "m&a", "unknown"]
                event_type = result.get('event_type', 'unknown')
                if event_type not in valid_types:
                    result['event_type'] = 'unknown'
                
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"LLM returned invalid JSON (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                return None
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                return None
        
        return None
        
    except Exception as e:
        logger.error(f"LLM setup failed: {e}")
        return None


# ============================================================================
# Main Entry Point
# ============================================================================

def discriminate(input_payload: Union[str, Dict], max_retries: int = 5) -> Dict[str, Any]:
    """
    Main entry point for A2 Discriminator.
    
    Args:
        input_payload: Can be:
            - File path to snapshot JSON (e.g., "snapshot_TSLA_*.json")
            - Dict containing market snapshot
            - Legacy analysis_report string
        max_retries: Max retries for LLM calls
    
    Returns:
        Dict conforming to A2Output schema (JSON-serializable)
    """
    # Normalize input
    snapshot_dict = None
    analysis_report = None
    ticker = None
    as_of = None
    
    if isinstance(input_payload, str):
        # Try loading as JSON file
        loaded = load_json_if_path(input_payload)
        if loaded is not None:
            snapshot_dict = loaded
        else:
            # Treat as legacy analysis_report string
            analysis_report = input_payload
            logger.info("Treating input as legacy analysis_report string")
    
    elif isinstance(input_payload, dict):
        # Check if it looks like a snapshot
        if 'ticker' in input_payload and ('sources' in input_payload or 'as_of' in input_payload):
            snapshot_dict = input_payload
        else:
            # Partial input - try to extract what we can
            snapshot_dict = input_payload
            logger.warning("Input dict may not be a complete snapshot")
    
    # Process snapshot if available
    if snapshot_dict:
        try:
            snapshot = MarketSnapshot(**snapshot_dict)
            ticker = snapshot.ticker
            as_of_dt = parse_datetime_any(snapshot.as_of)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
            as_of = as_of_dt.isoformat()
            analysis_report = build_analysis_report_from_snapshot(snapshot)
        except Exception as e:
            logger.error(f"Failed to parse snapshot: {e}")
            # Fallback: try to extract ticker and as_of manually
            ticker = snapshot_dict.get('ticker', 'UNKNOWN')
            as_of_raw = snapshot_dict.get('as_of', datetime.now(timezone.utc).isoformat())
            as_of_dt = parse_datetime_any(as_of_raw)
            if as_of_dt.tzinfo is None:
                as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
            as_of = as_of_dt.isoformat()
            analysis_report = build_analysis_report_from_snapshot(
                MarketSnapshot(**{**snapshot_dict, 'sources': SnapshotSources()})
            )
    else:
        # Legacy string - extract ticker if possible
        if not ticker:
            ticker_match = re.search(r'\b([A-Z]{1,5})\b', analysis_report or "")
            ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"
        as_of = datetime.now(timezone.utc).isoformat()
    
    # Compute deterministic feature bundle
    if snapshot_dict:
        try:
            snapshot = MarketSnapshot(**snapshot_dict)
            feature_bundle = compute_feature_bundle(snapshot)
        except Exception as e:
            logger.warning(f"Could not compute full feature bundle: {e}")
            feature_bundle = FeatureBundle()
    else:
        # Minimal feature bundle for legacy input
        feature_bundle = FeatureBundle()
    
    # Call LLM for extraction
    llm_result = call_llm_for_extraction(analysis_report or "", max_retries=max_retries)
    
    if llm_result:
        # Merge LLM results
        feature_bundle.event_type = llm_result.get('event_type', 'unknown')
        feature_bundle.conflict_count = llm_result.get('conflict_count', 0)
        
        extracted_signals = ExtractedSignals(
            claims=llm_result.get('extracted_signals', {}).get('claims', []),
            catalysts=llm_result.get('extracted_signals', {}).get('catalysts', []),
            risks=llm_result.get('extracted_signals', {}).get('risks', []),
            entities=Entities(**llm_result.get('extracted_signals', {}).get('entities', {}))
        )
        
        key_factors = llm_result.get('key_factors', [])
        llm_sentiment_score = llm_result.get('llm_sentiment_score', 0.0)
        llm_credibility_score = llm_result.get('llm_credibility_score', 0.5)
    else:
        # Fallback values
        logger.info("Using fallback values (no LLM available)")
        extracted_signals = ExtractedSignals()
        key_factors = []
        llm_sentiment_score = 0.0
        llm_credibility_score = 0.5
    
    # Build output
    output = A2Output(
        ticker=ticker or "UNKNOWN",
        as_of=as_of,
        llm_sentiment_score=llm_sentiment_score,
        llm_credibility_score=llm_credibility_score,
        key_factors=key_factors,
        extracted_signals=extracted_signals,
        feature_bundle=feature_bundle
    )
    
    # Return as dict (JSON-serializable)
    return output.model_dump()


# Create alias for JSON contract compatibility
discriminator = discriminate

