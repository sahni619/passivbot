import io
import logging

import logging_setup


def _configure_logging_to_stream() -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    logging_setup.configure_logging(debug=2, stream_target=stream)
    logger = logging.getLogger("test_logging")
    logger.setLevel(logging.DEBUG)
    return logger, stream


def test_sensitive_data_is_redacted_from_logs() -> None:
    logger, stream = _configure_logging_to_stream()

    logger.debug(
        "Request headers: %s",
        {
            "X-MBX-APIKEY": "2jlFsTPzvm8Y4X66LFvR28IPypdakaZJYjynu2dL5ZZ8ZxyZW3Jq7lFAExLVQBua",
            "signature": "deadbeefcafebabe",
            "apiKey": "should_not_leak",
        },
    )
    logger.debug(
        "Signed URL: https://fapi.binance.com/fapi/v1/openOrders?signature=%s",
        "abcdef1234567890",
    )

    output = stream.getvalue()

    assert "should_not_leak" not in output
    assert "abcdef1234567890" not in output
    assert "2jlFsTPzvm8Y4X66LFvR28IPypdakaZJYjynu2dL5ZZ8ZxyZW3Jq7lFAExLVQBua" not in output
    assert output.count("***REDACTED***") >= 3


def test_non_sensitive_messages_remain_intact() -> None:
    logger, stream = _configure_logging_to_stream()

    message = "Order book depth fetch succeeded"
    logger.info(message)

    output = stream.getvalue()

    assert message in output


def test_redaction_applies_to_external_handlers() -> None:
    stream = io.StringIO()
    logging_setup.configure_logging(debug=2, stream_target=stream)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    external_logger = logging.getLogger("external.http")
    external_logger.handlers = [handler]
    external_logger.propagate = False
    external_logger.setLevel(logging.DEBUG)

    external_logger.debug("payload %s", {"X-MBX-APIKEY": "leaky", "signature": "should-hide"})

    output = stream.getvalue()

    assert "leaky" not in output
    assert "should-hide" not in output
    assert "***REDACTED***" in output
