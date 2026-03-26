"""Shared fixtures for Aragora SDK tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Allow running SDK tests from the repo root without requiring an editable install.
_SDK_ROOT = Path(__file__).resolve().parents[1]
_sdk_root = str(_SDK_ROOT)
if _sdk_root not in sys.path:
    sys.path.insert(0, _sdk_root)

from aragora_sdk.client import AragoraAsyncClient, AragoraClient  # noqa: E402


@pytest.fixture
def client() -> AragoraClient:
    """Create a synchronous Aragora client for testing."""
    c = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
    yield c
    c.close()


@pytest.fixture
def unauthenticated_client() -> AragoraClient:
    """Create a synchronous client without API key."""
    c = AragoraClient(base_url="https://api.aragora.ai")
    yield c
    c.close()


@pytest.fixture
def mock_request():
    """Patch AragoraClient.request and yield the mock."""
    with patch.object(AragoraClient, "request") as mock:
        yield mock


@pytest.fixture
def mock_async_request():
    """Patch AragoraAsyncClient.request and yield the mock."""
    with patch.object(AragoraAsyncClient, "request") as mock:
        yield mock
