"""Shared fixtures for Aragora SDK tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


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
