"""Tests for Partner namespace API.

Tests both PartnerAPI (sync) and AsyncPartnerAPI (async) classes for:
- Partner registration and profile
- API key management
- Usage statistics
- Webhook configuration
- Rate limit information
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Registration
# =========================================================================

