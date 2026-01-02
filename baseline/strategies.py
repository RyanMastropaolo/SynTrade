"""
Baseline Strategy Classes
Uses the same backtesting methodology as SynTrade
"""

import backtrader as bt
from datetime import date, datetime
import logging

logger = logging.getLogger(__name__)


class BuyAndHoldStrategy(bt.Strategy):
    """
    Buy-and-hold equal-weight strategy.
    Buys equal positions (10% each for 10 stocks) at start and holds without rebalancing.
    """
    
    params = (
        ('target_weight', 0.10),  # 10% per stock (for 10 stocks)
    )
    
    def __init__(self):
        """Initialize the strategy."""
        self.target_weight = self.params.target_weight
        self.initial_positions_set = False
        
        # Track trades
        self.trades = []
        
        # Track portfolio value over time for equity curve
        self.portfolio_values = []
        self.portfolio_dates = []
    
    def next(self):
        """Called for each bar. Buy once at start, then hold."""
        current_date = self.data.datetime.date(0)
        
        # Track portfolio value at each bar for equity curve
        portfolio_value = self.broker.getvalue()
        self.portfolio_values.append(portfolio_value)
        self.portfolio_dates.append(current_date)
        
        # Set initial positions on first bar only
        if not self.initial_positions_set:
            self.set_initial_position(current_date)
            self.initial_positions_set = True
    
    def set_initial_position(self, date: date):
        """Set initial equal-weight position."""
        portfolio_value = self.broker.getvalue()
        target_value = portfolio_value * self.target_weight
        current_price = self.data.close[0]
        shares = int(target_value / current_price)
        
        if shares > 0:
            self.buy(size=shares)
            self.trades.append({
                "date": date,
                "action": "buy",
                "shares": shares,
                "price": current_price,
            })
            logger.debug(f"Initial buy on {date}: {shares} shares @ {current_price:.2f}")


class TechnicalOnlyStrategy(bt.Strategy):
    """
    Technical-only trading strategy based on RSI, MACD, and SMA indicators.
    
    Buy signals:
    - RSI < 40 (oversold)
    - MACD > MACD Signal (bullish crossover)
    - Price > SMA20 > SMA50 (uptrend)
    
    Sell signals:
    - RSI > 60 (overbought)
    - MACD < MACD Signal (bearish crossover)
    - Price < SMA20 < SMA50 (downtrend)
    
    Position sizing: 10% per position (equal weight like buy-and-hold)
    Rebalancing: Monthly (same frequency as SynTrade)
    """
    
    params = (
        ('target_weight', 0.10),  # 10% per position
        ('rebalance_frequency', 30),  # Days between rebalancing (monthly)
        ('rsi_oversold', 40),
        ('rsi_overbought', 60),
        ('first_rebalance_date', None),  # First date to start trading (e.g., 2023-12-15)
    )
    
    def __init__(self):
        """Initialize the strategy."""
        self.target_weight = self.params.target_weight
        self.rebalance_frequency = self.params.rebalance_frequency
        self.rsi_oversold = self.params.rsi_oversold
        self.rsi_overbought = self.params.rsi_overbought
        self.first_rebalance_date = self.params.first_rebalance_date
        
        self.last_rebalance_date = None
        self.initial_positions_set = False
        
        # Calculate technical indicators
        self.close = self.data.close
        
        # RSI
        self.rsi = bt.indicators.RSI(self.close, period=14)
        
        # MACD
        self.macd = bt.indicators.MACD(self.close)
        
        # SMAs
        self.sma20 = bt.indicators.SMA(self.close, period=20)
        self.sma50 = bt.indicators.SMA(self.close, period=50)
        
        # Track rebalancing dates
        self.rebalance_dates = []
        self.trades = []
        
        # Track portfolio value over time for equity curve
        self.portfolio_values = []
        self.portfolio_dates = []
    
    def next(self):
        """Called for each bar. Make trading decisions based on technicals."""
        current_date = self.data.datetime.date(0)
        
        # Track portfolio value at each bar for equity curve
        portfolio_value = self.broker.getvalue()
        self.portfolio_values.append(portfolio_value)
        self.portfolio_dates.append(current_date)
        
        # If first_rebalance_date is set, wait until that date to start trading
        if self.first_rebalance_date and current_date < self.first_rebalance_date:
            return
        
        # Set initial positions on first bar on or after first_rebalance_date
        if not self.initial_positions_set:
            self.initial_positions_set = True
            # If first_rebalance_date is set, use it; otherwise use current_date
            if self.first_rebalance_date:
                # Use the first trading day on or after first_rebalance_date
                self.last_rebalance_date = current_date
                # Trade immediately on the first available trading day
                self.evaluate_and_trade(current_date)
                return
            else:
                self.last_rebalance_date = current_date
            return
        
        # Check if it's time to rebalance (monthly)
        days_since_rebalance = (current_date - self.last_rebalance_date).days
        
        if days_since_rebalance >= self.rebalance_frequency:
            self.evaluate_and_trade(current_date)
            self.last_rebalance_date = current_date
    
    def evaluate_and_trade(self, date: date):
        """Evaluate technical signals and execute trades."""
        # Get current values
        current_price = self.close[0]
        rsi_value = self.rsi[0]
        macd_value = self.macd.macd[0]
        macd_signal_value = self.macd.signal[0]
        sma20_value = self.sma20[0]
        sma50_value = self.sma50[0]
        
        # Calculate technical signal score
        buy_signals = 0
        sell_signals = 0
        
        # RSI signals
        if rsi_value < self.rsi_oversold:
            buy_signals += 1
        elif rsi_value > self.rsi_overbought:
            sell_signals += 1
        
        # MACD signals
        if macd_value > macd_signal_value:
            buy_signals += 1
        elif macd_value < macd_signal_value:
            sell_signals += 1
        
        # SMA trend signals
        if current_price > sma20_value > sma50_value:
            buy_signals += 1
        elif current_price < sma20_value < sma50_value:
            sell_signals += 1
        
        # Make trading decision
        portfolio_value = self.broker.getvalue()
        target_value = portfolio_value * self.target_weight
        current_position_value = self.position.size * current_price
        
        # Buy if we have more buy signals than sell signals
        if buy_signals > sell_signals:
            # Calculate shares to buy
            value_diff = target_value - current_position_value
            if value_diff > portfolio_value * 0.01:  # Only trade if >1% difference
                shares_to_buy = int(value_diff / current_price)
                if shares_to_buy > 0:
                    self.buy(size=shares_to_buy)
                    self.trades.append({
                        "date": date,
                        "action": "buy",
                        "shares": shares_to_buy,
                        "price": current_price,
                        "rsi": rsi_value,
                        "macd": macd_value,
                        "macd_signal": macd_signal_value,
                        "buy_signals": buy_signals,
                        "sell_signals": sell_signals,
                    })
                    self.rebalance_dates.append(date)
                    logger.debug(f"Buy on {date}: {shares_to_buy} shares @ {current_price:.2f}, signals: {buy_signals}B/{sell_signals}S")
        
        # Sell if we have more sell signals than buy signals
        elif sell_signals > buy_signals:
            # Close position or reduce
            if self.position.size > 0:
                shares_to_sell = min(self.position.size, int(current_position_value / current_price))
                if shares_to_sell > 0:
                    self.sell(size=shares_to_sell)
                    self.trades.append({
                        "date": date,
                        "action": "sell",
                        "shares": shares_to_sell,
                        "price": current_price,
                        "rsi": rsi_value,
                        "macd": macd_value,
                        "macd_signal": macd_signal_value,
                        "buy_signals": buy_signals,
                        "sell_signals": sell_signals,
                    })
                    self.rebalance_dates.append(date)
                    logger.debug(f"Sell on {date}: {shares_to_sell} shares @ {current_price:.2f}, signals: {buy_signals}B/{sell_signals}S")
        
        # Hold if signals are balanced (do nothing)
