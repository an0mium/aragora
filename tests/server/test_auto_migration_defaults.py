"""Tests for auto-migration development defaults (Gap 4).

Verifies that _run_migrations() auto-migrates in development environments
and skips in production, with explicit env var overrides.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_auto_migrations():
    """Patch the auto-migration function."""
    with patch(
        "aragora.server.auto_migrations.run_auto_migrations",
        new_callable=AsyncMock,
        return_value={"success": True, "skipped": False},
    ) as mock:
        yield mock


class TestAutoMigrationDefaults:
    """Test the 3-way migration logic in _run_migrations."""

    @pytest.mark.asyncio
    async def test_explicit_true_always_migrates(self, mock_auto_migrations):
        """ARAGORA_AUTO_MIGRATE_ON_STARTUP=true always runs migrations."""
        import os as real_os
        from aragora.server.startup import _run_migrations

        env = {"ARAGORA_AUTO_MIGRATE_ON_STARTUP": "true", "ARAGORA_ENV": "production"}
        with patch.dict("os.environ", env, clear=True):
            result = await _run_migrations(real_os)
            assert not result.get("skipped", False)

    @pytest.mark.asyncio
    async def test_explicit_false_always_skips(self, mock_auto_migrations):
        """ARAGORA_AUTO_MIGRATE_ON_STARTUP=false always skips migrations."""
        import os as real_os
        from aragora.server.startup import _run_migrations

        env = {"ARAGORA_AUTO_MIGRATE_ON_STARTUP": "false", "ARAGORA_ENV": "development"}
        with patch.dict("os.environ", env, clear=True):
            result = await _run_migrations(real_os)
            assert result.get("skipped", False) is True

    @pytest.mark.asyncio
    async def test_unset_dev_env_auto_migrates(self, mock_auto_migrations):
        """Unset var + ARAGORA_ENV=development auto-migrates."""
        import os as real_os
        from aragora.server.startup import _run_migrations

        env = {"ARAGORA_ENV": "development"}
        with patch.dict("os.environ", env, clear=True):
            result = await _run_migrations(real_os)
            assert not result.get("skipped", False)

    @pytest.mark.asyncio
    async def test_unset_prod_env_skips(self, mock_auto_migrations):
        """Unset var + ARAGORA_ENV=production skips migrations."""
        import os as real_os
        from aragora.server.startup import _run_migrations

        env = {"ARAGORA_ENV": "production"}
        with patch.dict("os.environ", env, clear=True):
            result = await _run_migrations(real_os)
            assert result.get("skipped", False) is True

    @pytest.mark.asyncio
    async def test_unset_no_env_defaults_to_dev(self, mock_auto_migrations):
        """Both vars unset → defaults to development → auto-migrates."""
        import os as real_os
        from aragora.server.startup import _run_migrations

        with patch.dict("os.environ", {}, clear=True):
            result = await _run_migrations(real_os)
            assert not result.get("skipped", False)

    @pytest.mark.asyncio
    async def test_test_env_auto_migrates(self, mock_auto_migrations):
        """ARAGORA_ENV=test auto-migrates."""
        import os as real_os
        from aragora.server.startup import _run_migrations

        env = {"ARAGORA_ENV": "test"}
        with patch.dict("os.environ", env, clear=True):
            result = await _run_migrations(real_os)
            assert not result.get("skipped", False)

    @pytest.mark.asyncio
    async def test_local_env_auto_migrates(self, mock_auto_migrations):
        """ARAGORA_ENV=local auto-migrates."""
        import os as real_os
        from aragora.server.startup import _run_migrations

        env = {"ARAGORA_ENV": "local"}
        with patch.dict("os.environ", env, clear=True):
            result = await _run_migrations(real_os)
            assert not result.get("skipped", False)
