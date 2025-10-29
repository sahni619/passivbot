"""Compatibility wrapper for :mod:`risk_management.io.account_clients`."""

from .io.account_clients import AccountClientProtocol, CCXTAccountClient

__all__ = ["AccountClientProtocol", "CCXTAccountClient"]
