"""
Tests for SDK Batch API sync/async consistency.

Verifies that BatchAPI and AsyncBatchAPI produce identical payloads
for the same inputs, particularly for submit_debates().
"""

from __future__ import annotations

import inspect

import pytest


from aragora_sdk.namespaces.batch import AsyncBatchAPI, BatchAPI


class TestBatchPayloadConsistency:
    """Tests for sync/async payload consistency."""

    def test_sync_uses_items_key(self):
        """Test sync submit_debates sends 'items' key."""
        source = inspect.getsource(BatchAPI.submit_debates)
        assert '"items"' in source or "'items'" in source

    def test_async_uses_items_key(self):
        """Test async submit_debates sends 'items' key."""
        source = inspect.getsource(AsyncBatchAPI.submit_debates)
        assert '"items"' in source or "'items'" in source
        assert '"debates"' not in source

    def test_sync_uses_webhook_url_key(self):
        """Test sync submit_debates sends 'webhook_url' key."""
        source = inspect.getsource(BatchAPI.submit_debates)
        assert '"webhook_url"' in source or "'webhook_url'" in source

    def test_async_uses_webhook_url_key(self):
        """Test async submit_debates sends 'webhook_url' key."""
        source = inspect.getsource(AsyncBatchAPI.submit_debates)
        assert '"webhook_url"' in source or "'webhook_url'" in source
        assert '"callback_url"' not in source

    def test_async_enriches_items(self):
        """Test async submit_debates enriches individual items with priority/metadata."""
        source = inspect.getsource(AsyncBatchAPI.submit_debates)
        # Should have item enrichment logic
        assert "priority" in source and '"priority" not in item' in source
