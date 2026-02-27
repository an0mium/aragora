"""Shared fixtures for gauntlet tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _bypass_rbac_for_gauntlet_tests(monkeypatch):
    """Bypass RBAC permission checks for gauntlet handler tests."""
    try:
        from aragora.rbac import decorators
        from aragora.rbac.models import AuthorizationContext

        mock_auth_ctx = AuthorizationContext(
            user_id="test-user-001",
            org_id="test-org-001",
            roles={"admin", "owner"},
            permissions={"*"},
        )

        original_get_context = decorators._get_context_from_args

        def patched_get_context(args, kwargs, context_param):
            result = original_get_context(args, kwargs, context_param)
            if result is None:
                return mock_auth_ctx
            return result

        monkeypatch.setattr(decorators, "_get_context_from_args", patched_get_context)
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _mock_gauntlet_runner_external_calls(monkeypatch):
    """Prevent GauntletRunner from making real API calls.

    The GauntletRunner._run_red_team, _run_probes, and _run_scenarios methods
    import RedTeamMode, CapabilityProber, MatrixDebateRunner respectively and
    attempt to create real LLM agents. This causes tests to hang waiting for
    network responses.

    This fixture patches these internal methods to return empty summaries,
    ensuring tests exercise the runner's orchestration logic without making
    any real API calls.
    """
    from aragora.gauntlet.result import AttackSummary, ProbeSummary, ScenarioSummary
    from aragora.gauntlet.runner import GauntletRunner

    async def _mock_run_red_team(self, input_content, context, result, report_progress):
        return AttackSummary()

    async def _mock_run_probes(self, input_content, context, result, report_progress):
        return ProbeSummary()

    async def _mock_run_scenarios(self, input_content, context, result, report_progress):
        return ScenarioSummary()

    monkeypatch.setattr(GauntletRunner, "_run_red_team", _mock_run_red_team)
    monkeypatch.setattr(GauntletRunner, "_run_probes", _mock_run_probes)
    monkeypatch.setattr(GauntletRunner, "_run_scenarios", _mock_run_scenarios)


@pytest.fixture(autouse=True)
def _isolate_signing_key(monkeypatch):
    """Remove ARAGORA_RECEIPT_SIGNING_KEY to prevent invalid hex errors."""
    monkeypatch.delenv("ARAGORA_RECEIPT_SIGNING_KEY", raising=False)
