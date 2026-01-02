# SynTrade: Multi-Agent AI Trading System

SynTrade is an advanced algorithmic trading system that leverages a multi-agent architecture to make intelligent trading decisions. The system combines news sentiment analysis, fundamental data, technical indicators, and machine learning models to generate trading signals.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running the Full Pipeline](#running-the-full-pipeline)
  - [Running Backtests](#running-backtests)
  - [Running Baseline Comparisons](#running-baseline-comparisons)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Overview

SynTrade employs a sophisticated 5-agent pipeline to analyze market conditions and execute trading decisions:

1. **A1 Scraper Agent**: Gathers market data from multiple sources (news, SEC filings, fundamentals, macro indicators, technicals)
2. **A2 Discriminator Agent**: Extracts features and generates LLM-based sentiment/credibility signals
3. **A2b Model Scorer Agent**: Applies trained LightGBM models to score sentiment and credibility
4. **A3 Verifier Critic Agent**: Validates outputs, checks for issues, and provides final verification
5. **A4 Executor Policy Agent**: Makes trading decisions and calculates position sizing

The system is evaluated against two baseline strategies:
- **Buy-and-Hold**: Equal-weight portfolio with no rebalancing
- **Technical-Only**: Strategy based on RSI, MACD, and SMA indicators

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SynTrade Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  A1 Scraper → A2 Discriminator → A2b Model Scorer           │
│       ↓              ↓                    ↓                 │
│  Market Data    Features/LLM          ML Scores             │
│                                                             │
│       ↓              ↓                    ↓                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         A3 Verifier Critic Agent                    │    │
│  │  (Validates, checks issues, provides final result)  │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         A4 Executor Policy Agent                    │    │
│  │  (Trading decision + position sizing)               │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│                    Trading Decision                         │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup Steps

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd SynTrade
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys** (required for full functionality):
   Create a `.env` file in the project root with your API keys (see [Configuration](#configuration) section below)

## Configuration

### API Keys Setup

SynTrade requires API keys for various data sources and services. Create a `.env` file in the project root directory with the following keys:

```bash
# Required API Keys
GEMINI_API_KEY=your_gemini_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
FRED_API_KEY=your_fred_api_key_here
NEWSAPI_KEY=your_newsapi_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
```

**Note**: The system will still function with missing API keys, but some features may be limited or unavailable. See individual API descriptions below for details.

#### 1. Google Gemini API Key

**Purpose**: Used by the A2 Discriminator Agent for LLM-based sentiment and credibility analysis.

**Where to get it**:
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key" or "Get API Key"
4. Copy the generated API key

**Usage**: Required for sentiment analysis and credibility scoring. Without this key, the system will use fallback values.

**Free tier**: Google provides free tier access with rate limits.

---

#### 2. Finnhub API Key

**Purpose**: Used for fetching financial fundamentals, company profiles, earnings calendars, and stock metrics.

**Where to get it**:
1. Visit [Finnhub.io](https://finnhub.io/)
2. Click "Sign Up" or "Get Free API Key"
3. Create a free account
4. Navigate to your dashboard
5. Copy your API key from the dashboard

**Usage**: Used in `fundamentals_fetcher.py` and `filings_scraper.py` for:
- Financial statements and metrics
- Earnings calendar data
- Company profile information

**Free tier**: Free tier includes 60 API calls/minute. Paid plans available for higher limits.

---

#### 3. FRED API Key

**Purpose**: Used for fetching macroeconomic indicators from the Federal Reserve Economic Data (FRED) database.

**Where to get it**:
1. Visit [FRED API Registration](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Click "Request API Key" or go directly to [FRED API Key Request](https://fredaccount.stlouisfed.org/apikey)
3. Create a free account (if you don't have one)
4. Fill out the API key request form
5. Check your email for the API key
6. Copy the API key from the email or your account dashboard

**Usage**: Used in `macro_fetcher.py` for:
- Federal Funds Effective Rate (policy rate)
- Consumer Price Index (inflation data)
- Unemployment Rate

**Free tier**: FRED API is completely free with no rate limits for standard use.

---

#### 4. NewsAPI Key

**Purpose**: Used for fetching news articles from CNBC and MarketWatch via NewsAPI.ai (Event Registry).

**Where to get it**:
1. Visit [NewsAPI.ai](https://newsapi.ai/)
2. Click "Sign Up" or "Get Started"
3. Create a free account
4. Navigate to your dashboard/API settings
5. Copy your API key

**Usage**: Used in `news_scraper.py` for:
- CNBC news articles
- MarketWatch news articles
- Event Registry news aggregation

**Free tier**: Free tier includes limited API calls per month. Paid plans available for higher limits.

**⚠️ Important for Backtesting**: A **paid version of NewsAPI is required** for accessing historical news data needed for backtesting. The free tier typically only provides recent news articles and may not support historical date queries required when running the pipeline with the `--date` flag or when backtesting on historical data.

**Note**: NewsAPI.ai is different from NewsAPI.org. Make sure you're using NewsAPI.ai (Event Registry).

---

#### 5. Alpha Vantage API Key

**Purpose**: Used for fetching historical quarterly financial statements and fundamental data.

**Where to get it**:
1. Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Click "Get Free API Key" or go directly to [API Key Request](https://www.alphavantage.co/support/#api-key)
3. Fill out the form with your name and email
4. Check your email for the API key
5. Copy the API key from the email

**Usage**: Used in `fundamentals_fetcher.py` for:
- Historical quarterly financial statements
- Income statements, balance sheets, and cash flow statements
- TTM (Trailing Twelve Months) metric calculations

**Free tier**: Free tier includes 5 API calls per minute and 500 calls per day. Paid plans available for higher limits.

---

### Creating the .env File

1. In the project root directory, create a file named `.env`:

   ```bash
   touch .env
   ```

2. Add your API keys to the `.env` file:

   ```bash
   # SynTrade API Configuration
   GEMINI_API_KEY=your_actual_gemini_key_here
   FINNHUB_API_KEY=your_actual_finnhub_key_here
   FRED_API_KEY=your_actual_fred_key_here
   NEWSAPI_KEY=your_actual_newsapi_key_here
   ALPHA_VANTAGE_API_KEY=your_actual_alpha_vantage_key_here
   ```

3. **Important**: Never commit the `.env` file to version control. It should already be in `.gitignore`.

4. The system will automatically load these environment variables when you run the pipeline.

### Model Files

The system uses pre-trained LightGBM models located in the `models/` directory:
- `sentiment_lgbm.txt`: Sentiment classification model
- `credibility_lgbm.txt`: Credibility scoring model
- `feature_config.json`: Feature configuration for models

## Usage

### Running the Full Pipeline

Run the complete SynTrade pipeline for a single ticker:

```bash
# Basic usage
python run_full_pipeline.py AAPL

# With verbose output
python run_full_pipeline.py TSLA --verbose

# Save results to file
python run_full_pipeline.py MSFT --output results.json

# Run for a specific historical date
python run_full_pipeline.py AAPL --date 2024-01-15

# Specify data sources
python run_full_pipeline.py AAPL --sources news filings technicals

# Disable decision logging
python run_full_pipeline.py AAPL --no-log
```

**Output**: The pipeline generates a JSON result containing:
- Market snapshot (A1)
- Feature extraction and LLM signals (A2)
- ML model scores (A2b)
- Verification results (A3)
- Trading decision and position sizing (A4)

### Running Backtests

Backtest SynTrade decisions using logged decision data:

```bash
# Backtest single ticker
python run_backtest.py AAPL

# Backtest with date range
python run_backtest.py AAPL --start-date 2024-01-01 --end-date 2024-12-31

# Backtest entire portfolio
python run_backtest.py --portfolio

# Generate visualizations
python run_backtest.py AAPL --plots

# Custom backtest parameters
python run_backtest.py AAPL \
    --initial-cash 100000 \
    --commission 0.001 \
    --pos-scale 1.0 \
    --output-dir backtest_results

# Disable exit strategies (stop-loss/profit-taking)
python run_backtest.py AAPL --disable-exits
```

**Backtest Parameters**:
- `--initial-cash`: Starting capital (default: 100000)
- `--commission`: Commission rate per trade (default: 0.001 = 0.1%)
- `--pos-scale`: Multiplier for position sizes on BUY actions (default: 1.0). Use this to scale all position sizes up or down without regenerating decisions. For example, `--pos-scale 0.5` halves all position sizes, `--pos-scale 2.0` doubles them.
- `--pos-min`: Minimum position size percentage after scaling (default: 0.01 = 1%). Positions smaller than this will be rounded up to this minimum.
- `--pos-max`: Maximum position size percentage after scaling (default: 0.15 = 15%). Positions larger than this will be capped at this maximum.
- `--disable-exits`: Disable exit strategies (stop-loss, profit-taking, trend reversal)
- `--plots`: Generate visualization plots
- `--output-dir`: Directory to save results (default: backtest_results)

**Position Scaling Formula**: 
The effective position size is calculated as: `clamp(position_size_pct * pos_scale, pos_min, pos_max)`

**Output**: 
- Performance metrics (returns, Sharpe ratio, max drawdown, etc.)
- Equity curve plots (if `--plots` is used)
- Agent analysis visualizations
- Summary reports
- JSON results file

### Running Baseline Comparisons

Compare SynTrade against baseline strategies (Buy-and-Hold and Technical-Only):

```bash
# Run baseline backtests
python baseline/run_backtests.py
```

This script runs:
- **Buy-and-Hold Strategy**: Equal-weight portfolio (10% per stock for 10 stocks)
- **Technical-Only Strategy**: RSI/MACD/SMA-based trading with monthly rebalancing

**Parameters** (configured in `baseline/run_backtests.py`):
- Tickers: AAPL, JPM, AMZN, LLY, NVDA, CVX, GOOGL, CAT, MSFT, JNJ
- Initial Cash: $10,000,000 ($1M per stock)
- Commission: 0.1%
- Date Range: 2023-12-15 to 2025-12-15

**Output**: Results saved to `baseline_backtest_results/`:
- `buy_and_hold_results.json`
- `technical_only_results.json`
- `comparison.json`

## Project Structure

```
SynTrade/
├── agents/                          # Multi-agent system components
│   ├── A1_Scraper.py                # Market data gathering
│   ├── A2_Discriminator.py          # Feature extraction & LLM signals
│   ├── A2b_Model_Scorer.py          # ML model scoring
│   ├── A3_Verifier_Critic_Agent.py  # Verification & validation
│   └── A4_Executor_Policy_Agent.py  # Trading decisions
│
├── backtesting/                     # Backtesting engine
│   ├── backtest_engine.py           # Main backtest engine
│   ├── exit_conditions.py           # Stop-loss/profit-taking logic
│   ├── metrics.py                   # Performance metrics
│   ├── visualization.py             # Plotting functions
│   └── run_baselines.py             # Baseline comparison runner
│
├── baseline/                        # Baseline strategies
│   ├── baseline_backtest_engine.py
│   ├── run_backtests.py             # Run baseline comparisons
│   └── strategies.py                # Buy-and-Hold & Technical-Only
│
├── tools/                      # Data fetching tools
│   ├── news_scraper.py
│   ├── filings_scraper.py
│   ├── fundamentals_fetcher.py
│   ├── macro_fetcher.py
│   └── technicals_computer.py
│
├── utils/                      # Utilities
│   ├── decision_loader.py      # Load logged decisions
│   └── decision_logger.py      # Log trading decisions
│
├── models/                     # Trained ML models
│   ├── sentiment_lgbm.txt
│   ├── credibility_lgbm.txt
│   └── feature_config.json
│
├── run_full_pipeline.py        # Main pipeline runner
├── run_backtest.py             # Backtest runner
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Key Features

- **Multi-Source Data Aggregation**: News, SEC filings, fundamentals, macro indicators, and technicals
- **LLM Integration**: Google Gemini for sentiment and credibility analysis
- **Machine Learning Models**: Pre-trained LightGBM models for sentiment and credibility scoring
- **Risk Management**: Stop-loss, profit-taking, and trend reversal exit conditions
- **Comprehensive Backtesting**: Full backtesting engine with Backtrader integration
- **Decision Logging**: All trading decisions are logged for analysis and backtesting
- **Visualization**: Equity curves, metrics comparisons, and agent analysis plots

## Dependencies

Key dependencies include:
- `backtrader`: Backtesting framework
- `lightgbm`: Gradient boosting models
- `pandas`, `numpy`: Data processing
- `matplotlib`: Visualization
- `yfinance`: Market data
- `google-genai`: LLM integration
- `pydantic`: Data validation

See `requirements.txt` for the complete list.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

email: ryanmastropaolo2022@gmail.com
linkedin: https://www.linkedin.com/in/ryan-mastropaolo/

