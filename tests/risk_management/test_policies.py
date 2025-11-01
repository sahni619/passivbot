import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_management import policies as module
from risk_management.configuration import PolicyConfig, PolicyTriggerConfig


def test_render_message_handles_missing_values() -> None:
    policy = PolicyConfig(
        name="Portfolio drawdown escalation",
        trigger=PolicyTriggerConfig(
            type="account",
            metric="accounts.Test.drawdown_pct",
            operator=">",
            value=0.0,
        ),
        actions=[],
    )
    evaluator = module.PolicyEvaluator([policy])

    template = "Value {value:.2f}, threshold {threshold:.1f}"
    message = evaluator._render_message(
        template,
        policy=policy,
        value=None,
        threshold=None,
        snapshot={},
        status="triggered",
    )

    assert message == "Value n/a, threshold n/a"


def test_render_message_formats_values_when_present() -> None:
    policy = PolicyConfig(
        name="Balance warning",
        trigger=PolicyTriggerConfig(
            type="account",
            metric="accounts.Test.balance",
            operator="<",
            value=1.0,
        ),
        actions=[],
    )
    evaluator = module.PolicyEvaluator([policy])

    template = "Value {value:.1f}, threshold {threshold:.1f}"
    message = evaluator._render_message(
        template,
        policy=policy,
        value=123.456,
        threshold=100.0,
        snapshot={},
        status="triggered",
    )

    assert message == "Value 123.5, threshold 100.0"
