from types import SimpleNamespace
from unittest.mock import patch

from aragora.integrations.webhooks import WebhookConfig, WebhookDispatcher


def test_webhook_delivery_retries_with_exponential_backoff():
    cfg = WebhookConfig(
        name="retry-hook",
        url="https://example.com/webhook",
        max_retries=3,
        backoff_base_s=1.5,
    )
    dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
    event = {"type": "debate_end", "loop_id": "loop-1"}
    sleeps: list[float] = []
    responses = [
        SimpleNamespace(status_code=500, headers={}),
        SimpleNamespace(status_code=502, headers={}),
        SimpleNamespace(status_code=200, headers={}),
    ]

    with (
        patch("aragora.security.safe_http.safe_post", side_effect=responses),
        patch("aragora.integrations.webhooks.random.uniform", return_value=0.0),
        patch("aragora.integrations.webhooks.time.sleep", side_effect=sleeps.append),
    ):
        assert dispatcher._deliver(cfg, event) is True

    assert sleeps == [1.5, 3.0]


def test_webhook_delivery_honors_retry_after_before_exponential_backoff():
    cfg = WebhookConfig(name="retry-hook", url="https://example.com/webhook", max_retries=3)
    dispatcher = WebhookDispatcher([cfg], allow_localhost=True)
    event = {"type": "debate_end", "loop_id": "loop-1"}
    sleeps: list[float] = []
    responses = [
        SimpleNamespace(status_code=429, headers={"Retry-After": "7"}),
        SimpleNamespace(status_code=500, headers={}),
        SimpleNamespace(status_code=200, headers={}),
    ]

    with (
        patch("aragora.security.safe_http.safe_post", side_effect=responses),
        patch("aragora.integrations.webhooks.random.uniform", return_value=0.0),
        patch("aragora.integrations.webhooks.time.sleep", side_effect=sleeps.append),
    ):
        assert dispatcher._deliver(cfg, event) is True

    assert sleeps == [7.0, 2.0]
