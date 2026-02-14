"""Tests for OAuth Wizard namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

class TestOAuthWizardConfiguration:
    """Tests for wizard configuration methods."""

    def test_get_config(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "available_providers": 12,
                "configured_providers": 3,
                "completion_percent": 25,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.get_config()
            mock_request.assert_called_once_with("GET", "/api/v2/integrations/wizard")
            assert result["available_providers"] == 12
            assert result["completion_percent"] == 25
            client.close()

class TestAsyncOAuthWizard:
    """Tests for async OAuth wizard methods."""

    @pytest.mark.asyncio
    async def test_get_config(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"available_providers": 12, "completion_percent": 25}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.oauth_wizard.get_config()
            mock_request.assert_called_once_with("GET", "/api/v2/integrations/wizard")
            assert result["available_providers"] == 12
            await client.close()

