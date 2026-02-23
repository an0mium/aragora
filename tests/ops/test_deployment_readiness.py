"""Tests for deployment readiness probe."""

import json
import time

import pytest

from aragora.ops.deployment_validator import (
    DeploymentValidator,
    Severity,
    ValidationResult,
    readiness_check,
)


@pytest.fixture
def validator():
    return DeploymentValidator()


class TestDeploymentReadiness:
    @pytest.mark.asyncio
    async def test_readiness_passes_with_api_key(self, validator, monkeypatch):
        """Returns ready=True when ANTHROPIC_API_KEY set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        result = await validator.readiness_check()
        assert isinstance(result, ValidationResult)
        assert result.ready is True

    @pytest.mark.asyncio
    async def test_readiness_reports_missing_api_keys(self, validator, monkeypatch):
        """Returns issue when no provider keys."""
        for key in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        result = await validator.readiness_check()
        assert result.ready is False
        api_issues = [i for i in result.issues if i.component == "api_keys"]
        assert len(api_issues) >= 1

    @pytest.mark.asyncio
    async def test_readiness_checks_fewer_components(self, validator, monkeypatch):
        """Runs 3 checks vs full validate's 11."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        result = await validator.readiness_check()
        # Should have at most 3 component types (storage, database, api_keys)
        component_names = {c.name for c in result.components}
        assert component_names <= {"storage", "database", "api_keys"}

    @pytest.mark.asyncio
    async def test_readiness_result_serializable(self, validator, monkeypatch):
        """to_dict() produces valid JSON-ready dict."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        result = await validator.readiness_check()
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "ready" in d
        assert "issues" in d
        assert "components" in d
        json.dumps(d)  # Should not raise

    @pytest.mark.asyncio
    async def test_readiness_fast_under_500ms(self, validator, monkeypatch):
        """Completes in <500ms (no network calls in dev)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        start = time.time()
        await validator.readiness_check()
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 500  # Generous for CI, but much faster than full validate

    @pytest.mark.asyncio
    async def test_readiness_database_failure(self, validator, monkeypatch):
        """Reports CRITICAL when DB unreachable."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.setenv("ARAGORA_DB_BACKEND", "postgres")
        # No DSN set -> should report critical
        monkeypatch.delenv("ARAGORA_POSTGRES_DSN", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        result = await validator.readiness_check()
        db_issues = [
            i
            for i in result.issues
            if i.component == "database" and i.severity == Severity.CRITICAL
        ]
        assert len(db_issues) >= 1

    @pytest.mark.asyncio
    async def test_readiness_module_level_function(self, monkeypatch):
        """Module-level readiness_check() function works."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        result = await readiness_check()
        assert isinstance(result, ValidationResult)
        assert result.ready is True
        assert result.live is True
