"""
Exit Conditions Module - Implements stop-loss, profit-taking, and trend reversal strategies
"""

import logging
from typing import Dict, Optional, Tuple, Literal
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Exit condition parameters
STOP_LOSS_PCT = 0.12  # 12% stop-loss
PROFIT_TAKING_PCT = 0.10  # 10% profit-taking threshold
PROFIT_TAKING_SELL_PCT = 0.5  # Sell 50% at profit-taking threshold
TREND_REVERSAL_SMA = 50  # Use SMA_50 for trend reversal
MAX_POSITION_PCT = 0.15  # Maximum position size (15% of portfolio)
MAX_SCALE_INS = 3  # Maximum number of scale-ins per position
REENTRY_COOLDOWN_DAYS = 30  # 1 month cooldown (30 days)


class PositionState:
    """Tracks state for a single position."""
    
    def __init__(
        self,
        ticker: str,
        entry_price: float,
        entry_date: datetime,
        initial_shares: float,
        entry_decision_score: float = 0.0
    ):
        self.ticker = ticker
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.current_shares = initial_shares
        self.initial_shares = initial_shares
        self.highest_price = entry_price
        self.entry_decision_score = entry_decision_score
        self.scale_in_count = 0
        self.profit_taking_count = 0  # Track how many times profit-taking occurred
        self.last_exit_date: Optional[datetime] = None  # For re-entry cooldown
    
    def update_highest_price(self, current_price: float):
        """Update highest price since entry."""
        if current_price > self.highest_price:
            self.highest_price = current_price
    
    def add_shares(self, shares: float, price: float, decision_score: float):
        """Add shares to position (scale in)."""
        # Calculate new average entry price (volume-weighted)
        total_value = (self.entry_price * self.current_shares) + (price * shares)
        self.current_shares += shares
        self.entry_price = total_value / self.current_shares if self.current_shares > 0 else self.entry_price
        self.scale_in_count += 1
        self.entry_decision_score = decision_score  # Update to latest decision score
    
    def reduce_shares(self, shares: float):
        """Reduce shares from position (partial sell)."""
        self.current_shares = max(0, self.current_shares - shares)
        self.profit_taking_count += 1
    
    def close_position(self, exit_date: datetime):
        """Close entire position."""
        self.current_shares = 0
        self.last_exit_date = exit_date


class ExitConditions:
    """Manages exit condition evaluation and portfolio state tracking."""
    
    def __init__(self):
        self.positions: Dict[str, PositionState] = {}  # ticker -> PositionState
        self.reentry_blocked: Dict[str, datetime] = {}  # ticker -> last exit date
    
    def check_exit_conditions(
        self,
        ticker: str,
        current_price: float,
        current_date: datetime,
        technicals: Optional[Dict] = None
    ) -> Tuple[Optional[Literal["sell_all", "sell_partial"]], Optional[str], Optional[float]]:
        """
        Check exit conditions for a position.
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            current_date: Current date
            technicals: Optional dict with technical indicators (SMA_50, etc.)
        
        Returns:
            Tuple of (action, reason, sell_pct) where:
            - action: "sell_all", "sell_partial", or None
            - reason: Reason for exit (stop_loss, profit_taking, trend_reversal)
            - sell_pct: Percentage to sell (1.0 for all, 0.5 for partial)
        """
        if ticker not in self.positions:
            return None, None, None
        
        position = self.positions[ticker]
        
        # Suppress exits on same date as entry (Rule 7)
        if current_date.date() == position.entry_date.date():
            return None, None, None
        
        # Update highest price for trailing stop
        position.update_highest_price(current_price)
        
        # Calculate current return
        current_return = (current_price - position.entry_price) / position.entry_price
        
        # Priority 1: Stop-loss (Rule 4)
        if current_return <= -STOP_LOSS_PCT:
            logger.info(f"{ticker}: Stop-loss triggered ({current_return:.2%})")
            return "sell_all", "stop_loss", 1.0
        
        # Priority 2: Profit-taking (Rule 4, Rule 1)
        if current_return >= PROFIT_TAKING_PCT:
            # Check if we've already done profit-taking
            # Rule 1: Remaining 50% is eligible for another profit-taking if it gains 10% again
            # We need to check if the remaining position has gained 10% from the last profit-taking price
            if position.profit_taking_count == 0:
                # First profit-taking: sell 50%
                logger.info(f"{ticker}: Profit-taking triggered ({current_return:.2%})")
                return "sell_partial", "profit_taking", PROFIT_TAKING_SELL_PCT
            else:
                # Subsequent profit-taking: check if remaining position gained 10% from entry
                # For simplicity, we'll use the same entry price and check if current return >= 10%
                # This allows multiple profit-taking events
                logger.info(f"{ticker}: Profit-taking triggered again ({current_return:.2%})")
                return "sell_partial", "profit_taking", PROFIT_TAKING_SELL_PCT
        
        # Priority 3: Trend reversal (Rule 4, Rule 3)
        if technicals and "sma_50" in technicals:
            sma_50 = technicals.get("sma_50")
            if sma_50 and current_price < sma_50:
                # If profit-taking already occurred, sell remaining 50%
                # Otherwise, sell entire position
                if position.profit_taking_count > 0:
                    logger.info(f"{ticker}: Trend reversal triggered (price < SMA_50), selling remaining position")
                    return "sell_all", "trend_reversal", 1.0
                else:
                    logger.info(f"{ticker}: Trend reversal triggered (price < SMA_50)")
                    return "sell_all", "trend_reversal", 1.0
        
        return None, None, None
    
    def add_position(
        self,
        ticker: str,
        entry_price: float,
        entry_date: datetime,
        shares: float,
        decision_score: float = 0.0
    ):
        """Add a new position or scale into existing position."""
        if ticker in self.positions:
            # Scale in (Rule 6)
            position = self.positions[ticker]
            if position.scale_in_count < MAX_SCALE_INS:
                position.add_shares(shares, entry_price, decision_score)
                logger.info(f"{ticker}: Scaled in {shares} shares (total: {position.current_shares:.2f})")
            else:
                logger.warning(f"{ticker}: Maximum scale-ins reached ({MAX_SCALE_INS})")
        else:
            # New position
            self.positions[ticker] = PositionState(
                ticker=ticker,
                entry_price=entry_price,
                entry_date=entry_date,
                initial_shares=shares,
                entry_decision_score=decision_score
            )
            logger.info(f"{ticker}: New position opened: {shares} shares @ ${entry_price:.2f}")
    
    def execute_sell(
        self,
        ticker: str,
        sell_pct: float,
        exit_date: datetime,
        reason: str
    ) -> float:
        """
        Execute a sell (partial or full).
        
        Args:
            ticker: Stock ticker symbol
            sell_pct: Percentage to sell (0.5 for 50%, 1.0 for 100%)
            exit_date: Exit date
            reason: Reason for exit
        
        Returns:
            Number of shares to sell
        """
        if ticker not in self.positions:
            return 0.0
        
        position = self.positions[ticker]
        shares_to_sell = position.current_shares * sell_pct
        
        if sell_pct >= 1.0:
            # Full exit
            shares_to_sell = position.current_shares
            position.close_position(exit_date)
            self.reentry_blocked[ticker] = exit_date
            del self.positions[ticker]
            logger.info(f"{ticker}: Full exit ({reason}): {shares_to_sell:.2f} shares")
        else:
            # Partial exit
            position.reduce_shares(shares_to_sell)
            logger.info(f"{ticker}: Partial exit ({reason}): {shares_to_sell:.2f} shares (remaining: {position.current_shares:.2f})")
        
        return shares_to_sell
    
    def can_reenter(
        self,
        ticker: str,
        current_date: datetime,
        current_price: float,
        technicals: Optional[Dict] = None,
        decision_score: float = 0.0
    ) -> bool:
        """
        Check if a ticker can be re-entered after exit (Rule 5).
        
        Args:
            ticker: Stock ticker symbol
            current_date: Current date
            current_price: Current stock price
            technicals: Optional dict with technical indicators
            decision_score: A4 decision score
        
        Returns:
            True if re-entry is allowed, False otherwise
        """
        if ticker not in self.reentry_blocked:
            return True  # Never exited, can enter
        
        last_exit_date = self.reentry_blocked[ticker]
        days_since_exit = (current_date - last_exit_date).days
        
        # Rule 5: Cooldown period (1 month = 30 days)
        if days_since_exit < REENTRY_COOLDOWN_DAYS:
            return False
        
        # Rule 5: Require price > SMA_50
        if technicals and "sma_50" in technicals:
            sma_50 = technicals.get("sma_50")
            if sma_50 and current_price <= sma_50:
                return False
        
        # Rule 5: Require A4 decision_score >= 0.25 (buy threshold)
        if decision_score < 0.25:
            return False
        
        # All conditions met, allow re-entry
        # Remove from blocked list
        del self.reentry_blocked[ticker]
        return True
    
    def get_position(self, ticker: str) -> Optional[PositionState]:
        """Get position state for a ticker."""
        return self.positions.get(ticker)
    
    def has_position(self, ticker: str) -> bool:
        """Check if a position exists for a ticker."""
        return ticker in self.positions and self.positions[ticker].current_shares > 0
    
    def get_entry_price(self, ticker: str) -> Optional[float]:
        """Get entry price for a position."""
        if ticker in self.positions:
            return self.positions[ticker].entry_price
        return None

