import sys
from types import ModuleType

from tests.fixtures.shared.api_mocks import (
    MockAnthropicClient,
    MockAsyncAnthropicClient,
    apply_api_mocks,
)


def test_apply_api_mocks_supports_lightweight_anthropic_stub(monkeypatch):
    anthropic_stub = ModuleType("anthropic")
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_stub)

    apply_api_mocks(monkeypatch)

    assert anthropic_stub.Anthropic is MockAnthropicClient
    assert anthropic_stub.AsyncAnthropic is MockAsyncAnthropicClient
