from risk_management import history
from services.persistence.history import PortfolioHistoryStore


def test_history_alias_exposes_store():
    assert history.PortfolioHistoryStore is PortfolioHistoryStore
