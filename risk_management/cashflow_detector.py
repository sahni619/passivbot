"""Cashflow detection module for institutional-grade deposit/withdrawal tracking.

This module provides comprehensive cashflow detection capabilities:
- Balance change monitoring with historical tracking
- Exchange API integration for deposit/withdrawal verification
- Heuristic-based detection when API data unavailable
- Balance reconciliation and discrepancy detection
- Notification generation for cashflow events

The system uses multiple detection methods in priority order:
1. Exchange API deposits/withdrawals (most reliable)
2. Balance reconciliation (expected vs actual)
3. Heuristic analysis (PnL vs balance changes)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol

from .exceptions import (
    BalanceReconciliationError,
    CashflowDetectionError,
    ExchangeAPIError,
)
from .logging_config import AuditLogger, get_logger, log_performance
from .resilience import ResilientExecutor, ResilienceConfig

logger = get_logger(__name__)
audit_logger = AuditLogger()


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class CashflowEvent:
    """Represents a detected cashflow event (deposit or withdrawal).
    
    Attributes:
        account: Account name where cashflow occurred
        flow_type: Type of cashflow ("deposit" or "withdrawal")
        amount: Absolute amount (always positive)
        currency: Currency of the cashflow
        timestamp: When the cashflow occurred
        detection_method: How the cashflow was detected
        confidence: Confidence level (0.0 to 1.0)
        exchange_tx_id: Optional transaction ID from exchange
        note: Additional notes or context
    """
    account: str
    flow_type: str  # "deposit" or "withdrawal"
    amount: float
    currency: str
    timestamp: datetime
    detection_method: str
    confidence: float = 1.0
    exchange_tx_id: Optional[str] = None
    note: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "account": self.account,
            "type": self.flow_type,
            "amount": self.amount,
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat(),
            "detection_method": self.detection_method,
            "confidence": self.confidence,
            "exchange_tx_id": self.exchange_tx_id,
            "note": self.note,
        }


@dataclass
class BalanceHistory:
    """Historical balance tracking for an account.
    
    Attributes:
        account: Account name
        current_balance: Current balance
        previous_balance: Previous balance
        balance_change: Change in balance (current - previous)
        unrealized_pnl: Current unrealized PnL
        realized_pnl: Realized PnL since last check
        last_updated: When the balance was last updated
    """
    account: str
    current_balance: float
    previous_balance: float
    balance_change: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def significant_change(self) -> bool:
        """Check if balance change is significant (> $1)."""
        return abs(self.balance_change) > 1.0


# ============================================================================
# Exchange Client Protocol
# ============================================================================

class ExchangeClientProtocol(Protocol):
    """Protocol for exchange client supporting cashflow queries."""
    
    async def fetch_deposits(
        self,
        code: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch deposit history from exchange."""
        ...
    
    async def fetch_withdrawals(
        self,
        code: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch withdrawal history from exchange."""
        ...
    
    async def fetch_transactions(
        self,
        code: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch all transactions from exchange."""
        ...


# ============================================================================
# Cashflow Detector Configuration
# ============================================================================

@dataclass
class CashflowDetectorConfig:
    """Configuration for cashflow detection behavior.
    
    Attributes:
        enabled: Enable/disable cashflow detection
        detection_threshold: Minimum balance change to investigate (in base currency)
        lookback_minutes: How far back to check for transactions
        use_exchange_api: Try to use exchange API for verification
        use_heuristic: Use heuristic detection if API unavailable
        reconciliation_enabled: Enable balance reconciliation
        reconciliation_tolerance: Acceptable discrepancy (absolute value)
        min_confidence: Minimum confidence to report event
    """
    enabled: bool = True
    detection_threshold: float = 10.0  # $10 minimum
    lookback_minutes: int = 10  # Check last 10 minutes
    use_exchange_api: bool = True
    use_heuristic: bool = True
    reconciliation_enabled: bool = True
    reconciliation_tolerance: float = 1.0  # $1 tolerance
    min_confidence: float = 0.7


# ============================================================================
# Cashflow Detector
# ============================================================================

class CashflowDetector:
    """Detects and categorizes cashflow events (deposits/withdrawals).
    
    This class provides the core logic for:
    1. Tracking balance history per account
    2. Detecting significant balance changes
    3. Querying exchange APIs for transaction history
    4. Using heuristics when API data unavailable
    5. Generating cashflow events with confidence scores
    
    Example:
        >>> detector = CashflowDetector(config)
        >>> events = await detector.detect_cashflows(
        ...     account_name="Binance",
        ...     current_balance=10500.0,
        ...     previous_balance=10000.0,
        ...     exchange_client=binance_client,
        ... )
    """
    
    def __init__(
        self,
        config: Optional[CashflowDetectorConfig] = None,
        resilience_config: Optional[ResilienceConfig] = None,
    ) -> None:
        """Initialize cashflow detector.
        
        Args:
            config: Detection configuration
            resilience_config: Configuration for resilient API calls
        """
        self.config = config or CashflowDetectorConfig()
        self._balance_history: Dict[str, BalanceHistory] = {}
        self._executor = ResilientExecutor(
            "cashflow_detector",
            resilience_config,
        )
        self._logger = get_logger(__name__)
    
    def update_balance(
        self,
        account: str,
        current_balance: float,
        *,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
    ) -> BalanceHistory:
        """Update balance history for an account.
        
        Args:
            account: Account name
            current_balance: Current balance
            unrealized_pnl: Current unrealized PnL
            realized_pnl: Realized PnL since last update
        
        Returns:
            Updated balance history
        """
        previous = self._balance_history.get(account)
        previous_balance = previous.current_balance if previous else current_balance
        
        history = BalanceHistory(
            account=account,
            current_balance=current_balance,
            previous_balance=previous_balance,
            balance_change=current_balance - previous_balance,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            last_updated=datetime.now(timezone.utc),
        )
        
        self._balance_history[account] = history
        return history
    
    def get_balance_history(self, account: str) -> Optional[BalanceHistory]:
        """Get balance history for an account."""
        return self._balance_history.get(account)
    
    async def detect_cashflows(
        self,
        account: str,
        current_balance: float,
        previous_balance: float,
        *,
        exchange_client: Optional[Any] = None,
        currency: str = "USDT",
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        account_data: Optional[Dict[str, Any]] = None,
    ) -> List[CashflowEvent]:
        """Detect cashflow events for an account.
        
        Args:
            account: Account name
            current_balance: Current balance
            previous_balance: Previous balance
            exchange_client: Optional exchange client for API queries
            currency: Currency of the balance
            unrealized_pnl: Current unrealized PnL
            realized_pnl: Realized PnL since last check
            account_data: Additional account data for context
        
        Returns:
            List of detected cashflow events
        """
        if not self.config.enabled:
            return []
        
        with log_performance(
            "detect_cashflows",
            self._logger,
            account=account,
        ):
            # Calculate balance change
            balance_change = current_balance - previous_balance
            
            # Check if change is significant
            if abs(balance_change) < self.config.detection_threshold:
                return []
            
            self._logger.info(
                "Significant balance change detected on %s: %.2f %s",
                account,
                balance_change,
                currency,
                extra={
                    "account": account,
                    "balance_change": balance_change,
                    "current_balance": current_balance,
                    "previous_balance": previous_balance,
                }
            )
            
            # Try multiple detection methods
            events: List[CashflowEvent] = []
            
            # Method 1: Exchange API (most reliable)
            if self.config.use_exchange_api and exchange_client:
                api_events = await self._detect_via_exchange_api(
                    account=account,
                    balance_change=balance_change,
                    client=exchange_client,
                    currency=currency,
                )
                events.extend(api_events)
            
            # Method 2: Heuristic (fallback)
            if not events and self.config.use_heuristic:
                heuristic_event = self._detect_via_heuristic(
                    account=account,
                    balance_change=balance_change,
                    currency=currency,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=unrealized_pnl,
                    account_data=account_data,
                )
                if heuristic_event:
                    events.append(heuristic_event)
            
            # Filter by confidence threshold
            events = [e for e in events if e.confidence >= self.config.min_confidence]
            
            # Log audit trail for detected events
            for event in events:
                audit_logger.log_cashflow(
                    flow_type=event.flow_type,
                    amount=event.amount,
                    account=event.account,
                    currency=event.currency,
                    detection_method=event.detection_method,
                )
            
            return events
    
    async def _detect_via_exchange_api(
        self,
        account: str,
        balance_change: float,
        client: Any,
        currency: str,
    ) -> List[CashflowEvent]:
        """Detect cashflows by querying exchange API.
        
        Args:
            account: Account name
            balance_change: Balance change amount
            client: Exchange client
            currency: Currency code
        
        Returns:
            List of detected cashflow events
        """
        events: List[CashflowEvent] = []
        
        # Calculate time window
        now = datetime.now(timezone.utc)
        since_ms = int((now - timedelta(minutes=self.config.lookback_minutes)).timestamp() * 1000)
        
        try:
            # Determine which API to call based on balance change direction
            if balance_change > 0:
                # Check for deposits
                transactions = await self._fetch_deposits_resilient(
                    client, currency, since_ms
                )
                flow_type = "deposit"
            else:
                # Check for withdrawals
                transactions = await self._fetch_withdrawals_resilient(
                    client, currency, since_ms
                )
                flow_type = "withdrawal"
            
            # Match transactions to balance change
            if transactions:
                self._logger.info(
                    "Found %d %s transaction(s) for %s in last %d minutes",
                    len(transactions),
                    flow_type,
                    account,
                    self.config.lookback_minutes,
                )
                
                # Sum up all transactions in the window
                total_amount = sum(
                    abs(float(tx.get("amount", 0))) for tx in transactions
                )
                
                # Check if total matches balance change (within tolerance)
                expected_change = total_amount if flow_type == "deposit" else -total_amount
                discrepancy = abs(balance_change - expected_change)
                
                if discrepancy <= self.config.reconciliation_tolerance:
                    # Perfect match - high confidence
                    for tx in transactions:
                        events.append(CashflowEvent(
                            account=account,
                            flow_type=flow_type,
                            amount=abs(float(tx.get("amount", 0))),
                            currency=currency,
                            timestamp=self._parse_timestamp(tx.get("timestamp")),
                            detection_method="exchange_api",
                            confidence=1.0,
                            exchange_tx_id=tx.get("id"),
                            note=f"Verified via {client.__class__.__name__} API",
                        ))
                else:
                    # Mismatch - still report but with lower confidence
                    self._logger.warning(
                        "Transaction amount mismatch on %s: "
                        "expected %.2f, found %.2f (discrepancy: %.2f)",
                        account,
                        balance_change,
                        expected_change,
                        discrepancy,
                    )
                    
                    # Report aggregate event with reduced confidence
                    events.append(CashflowEvent(
                        account=account,
                        flow_type=flow_type,
                        amount=abs(balance_change),
                        currency=currency,
                        timestamp=now,
                        detection_method="exchange_api_partial",
                        confidence=0.8,
                        note=f"Partial match: {len(transactions)} transaction(s), discrepancy ${discrepancy:.2f}",
                    ))
        
        except Exception as exc:
            self._logger.warning(
                "Failed to fetch %s history from exchange API for %s: %s",
                flow_type if balance_change > 0 else "withdrawal",
                account,
                exc,
                exc_info=True,
            )
            # Don't raise - fall back to heuristic
        
        return events
    
    async def _fetch_deposits_resilient(
        self,
        client: Any,
        currency: str,
        since_ms: int,
    ) -> List[Dict[str, Any]]:
        """Fetch deposits with resilience."""
        return await self._executor.execute(
            lambda: client.fetch_deposits(
                code=currency,
                since=since_ms,
                params={},
            ),
            fallback=lambda: []  # Return empty list on failure
        )
    
    async def _fetch_withdrawals_resilient(
        self,
        client: Any,
        currency: str,
        since_ms: int,
    ) -> List[Dict[str, Any]]:
        """Fetch withdrawals with resilience."""
        return await self._executor.execute(
            lambda: client.fetch_withdrawals(
                code=currency,
                since=since_ms,
                params={},
            ),
            fallback=lambda: []  # Return empty list on failure
        )
    
    def _detect_via_heuristic(
        self,
        account: str,
        balance_change: float,
        currency: str,
        realized_pnl: float,
        unrealized_pnl: float,
        account_data: Optional[Dict[str, Any]],
    ) -> Optional[CashflowEvent]:
        """Detect cashflow using heuristic analysis.
        
        Heuristic logic:
        - If balance change is much larger than realized PnL, likely cashflow
        - If no open positions and significant balance change, likely cashflow
        - Confidence is lower than API-based detection
        
        Args:
            account: Account name
            balance_change: Balance change amount
            currency: Currency code
            realized_pnl: Realized PnL
            unrealized_pnl: Unrealized PnL
            account_data: Additional account data
        
        Returns:
            Detected cashflow event or None
        """
        # Calculate expected balance change from PnL
        expected_change = realized_pnl
        
        # Calculate discrepancy
        discrepancy = abs(balance_change - expected_change)
        
        # If discrepancy is small, likely just PnL
        if discrepancy < self.config.detection_threshold:
            return None
        
        # Check if there are open positions
        has_open_positions = False
        if account_data:
            positions = account_data.get("positions", [])
            has_open_positions = any(
                abs(float(pos.get("contracts", 0))) > 0
                for pos in positions
                if isinstance(pos, dict)
            )
        
        # Calculate confidence score
        confidence = 0.7  # Base confidence for heuristic
        
        # Increase confidence if no open positions
        if not has_open_positions:
            confidence = 0.85
        
        # Increase confidence if discrepancy is very large
        if discrepancy > abs(balance_change) * 0.9:
            confidence = min(0.95, confidence + 0.1)
        
        # Determine flow type
        flow_type = "deposit" if balance_change > 0 else "withdrawal"
        
        self._logger.info(
            "Heuristic detection: %s of %.2f %s on %s (confidence: %.2f)",
            flow_type,
            abs(balance_change),
            currency,
            account,
            confidence,
            extra={
                "account": account,
                "flow_type": flow_type,
                "amount": abs(balance_change),
                "confidence": confidence,
                "discrepancy": discrepancy,
                "realized_pnl": realized_pnl,
            }
        )
        
        return CashflowEvent(
            account=account,
            flow_type=flow_type,
            amount=abs(balance_change),
            currency=currency,
            timestamp=datetime.now(timezone.utc),
            detection_method="heuristic",
            confidence=confidence,
            note=f"Heuristic: discrepancy ${discrepancy:.2f} vs realized PnL ${realized_pnl:.2f}",
        )
    
    def reconcile_balance(
        self,
        account: str,
        expected_balance: float,
        actual_balance: float,
        *,
        starting_balance: float,
        realized_pnl: float,
        unrealized_pnl: float,
        cashflows: List[CashflowEvent],
    ) -> Optional[BalanceReconciliationError]:
        """Reconcile expected vs actual balance.
        
        Formula:
            expected_balance = starting_balance + realized_pnl + net_cashflows
            (unrealized PnL not included as it's mark-to-market)
        
        Args:
            account: Account name
            expected_balance: Calculated expected balance
            actual_balance: Actual balance from exchange
            starting_balance: Balance at start of period
            realized_pnl: Realized PnL during period
            unrealized_pnl: Current unrealized PnL
            cashflows: List of cashflow events
        
        Returns:
            BalanceReconciliationError if discrepancy found, None otherwise
        """
        if not self.config.reconciliation_enabled:
            return None
        
        # Calculate net cashflows
        net_cashflows = sum(
            event.amount if event.flow_type == "deposit" else -event.amount
            for event in cashflows
        )
        
        # Calculate expected balance
        calculated_expected = starting_balance + realized_pnl + net_cashflows
        
        # Calculate discrepancy
        discrepancy = actual_balance - calculated_expected
        
        if abs(discrepancy) <= self.config.reconciliation_tolerance:
            self._logger.debug(
                "Balance reconciliation PASSED for %s: "
                "expected=%.2f, actual=%.2f, discrepancy=%.2f",
                account,
                calculated_expected,
                actual_balance,
                discrepancy,
            )
            return None
        
        # Reconciliation failed
        self._logger.error(
            "Balance reconciliation FAILED for %s: "
            "expected=%.2f, actual=%.2f, discrepancy=%.2f",
            account,
            calculated_expected,
            actual_balance,
            discrepancy,
            extra={
                "account": account,
                "expected_balance": calculated_expected,
                "actual_balance": actual_balance,
                "discrepancy": discrepancy,
                "starting_balance": starting_balance,
                "realized_pnl": realized_pnl,
                "net_cashflows": net_cashflows,
            }
        )
        
        # Log to audit trail
        audit_logger.log_event(
            event_type="BALANCE_RECONCILIATION_FAILED",
            description=f"Balance mismatch on {account}: ${discrepancy:.2f}",
            severity="ERROR",
            account=account,
            expected_balance=calculated_expected,
            actual_balance=actual_balance,
            discrepancy=discrepancy,
        )
        
        return BalanceReconciliationError(
            f"Balance reconciliation failed for {account}",
            account=account,
            expected_balance=calculated_expected,
            actual_balance=actual_balance,
            discrepancy=discrepancy,
        )
    
    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse timestamp from various formats."""
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, (int, float)):
            # Assume milliseconds
            return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                pass
        
        # Default to now
        return datetime.now(timezone.utc)


__all__ = [
    "CashflowEvent",
    "BalanceHistory",
    "CashflowDetectorConfig",
    "CashflowDetector",
    "ExchangeClientProtocol",
]
