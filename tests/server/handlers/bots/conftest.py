"""Test fixtures for bot handler tests.

Resets module-level constants that are captured at import time,
preventing test pollution across different ordering.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _reset_whatsapp_constants():
    """Reset WhatsApp module-level constants between tests.

    The WhatsApp handler evaluates os.environ.get() at import time for
    WHATSAPP_VERIFY_TOKEN, WHATSAPP_ACCESS_TOKEN, etc. Without reset,
    tests that patch these values leak into subsequent tests.
    """
    try:
        import aragora.server.handlers.bots.whatsapp as wa_mod

        # Save originals
        orig = {
            "WHATSAPP_VERIFY_TOKEN": wa_mod.WHATSAPP_VERIFY_TOKEN,
            "WHATSAPP_ACCESS_TOKEN": wa_mod.WHATSAPP_ACCESS_TOKEN,
            "WHATSAPP_PHONE_NUMBER_ID": wa_mod.WHATSAPP_PHONE_NUMBER_ID,
            "WHATSAPP_APP_SECRET": wa_mod.WHATSAPP_APP_SECRET,
        }
    except (ImportError, AttributeError):
        yield
        return

    yield

    # Restore originals after test
    for attr, value in orig.items():
        setattr(wa_mod, attr, value)
