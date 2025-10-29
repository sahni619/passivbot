# Test suite guidelines

The risk management notification tests require 100% coverage for the primary alert
modules. All CI and local runs should execute:

```bash
pytest tests/risk_management --cov=risk_management.email_notifications \
    --cov=risk_management._notifications --cov=risk_management.snapshot_utils \
    --cov-report=term-missing --cov-fail-under=100
```

The `tests/risk_management` package relies exclusively on fakes and dependency
injection, so running the suite does **not** contact external services such as
SMTP or Telegram. If you add new alerting code paths, extend the mocks in the
unit tests so the coverage threshold remains satisfied.
