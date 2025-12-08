"""Backwards-compatible import for portfolio history persistence."""

from services.persistence.history import PortfolioHistoryStore

__all__ = ["PortfolioHistoryStore"]
