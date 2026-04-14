"""Tests for aragora/swarm/dispatch_contract_gate.py.

Covers:
  1. _target_agent fallback chain (requested → runner_type → config default → "codex")
  2. dispatch_contract_gate: passing preflight returns None
  3. dispatch_contract_gate: missing slices returns a terminal blocked dict
  4. dispatch_contract_gate: failed contract returns blocked dict
  5. dispatch_contract_gate: preflight receipt failure returns terminal outcome dict
  6. dispatch_contract_gate: claimed_runner_id triggers release on gate block
  7. dispatch_contract_gate: exception during preflight receipt returns synthetic summary
  8. _gate_reason: builds reasons and next_actions for contract_valid=False + missing slices
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.swarm.dispatch_contract_gate import (
    _gate_reason,
    _target_agent,
    dispatch_contract_gate,
)
from aragora.swarm.boss_feed import GitHubIssue
from aragora.swarm.preflight import PreflightReceipt
from aragora.swarm.terminal_truth import TerminalClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(number: int = 42) -> GitHubIssue:
    return GitHubIssue(
        number=number,
        title="Test issue",
        body="Test body",
        labels=[],
        url=f"https://github.com/test/repo/issues/{number}",
        state="open",
        created_at="2026-01-01T00:00:00Z",
    )


def _make_loop(
    *,
    default_target_agent: str = "",
    target_branch: str = "main",
    execution_mode: str = "autonomous",
    auto_publish_deliverables: bool = False,
    auto_close_already_done_issues: bool = False,
    allow_claude_dangerously_skip_permissions: bool = False,
    allow_codex_full_auto: bool = True,
) -> MagicMock:
    loop = MagicMock()
    loop._env = {}
    loop.config.default_target_agent = default_target_agent
    loop.config.target_branch = target_branch
    loop.config.execution_mode = execution_mode
    loop.config.auto_publish_deliverables = auto_publish_deliverables
    loop.config.auto_close_already_done_issues = auto_close_already_done_issues
    loop.config.allow_claude_dangerously_skip_permissions = (
        allow_claude_dangerously_skip_permissions
    )
    loop.config.allow_codex_full_auto = allow_codex_full_auto
    return loop


def _make_passing_receipt(check_type: str = "scratch") -> PreflightReceipt:
    return PreflightReceipt(
        receipt_id="receipt-abc",
        envelope_seal="seal-xyz",
        repo_root="/tmp/repo",
        check_type=check_type,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        passed=True,
        checks=[{"name": "worktree", "passed": True, "detail": "ok"}],
        expires_at="2026-01-02T00:00:00Z",
    )


def _make_failing_receipt(
    check_type: str = "scratch",
    terminal_class: TerminalClass = TerminalClass.BLOCKED_NOT_DISPATCH_BOUNDED,
) -> PreflightReceipt:
    receipt = PreflightReceipt(
        receipt_id="receipt-fail",
        envelope_seal="seal-fail",
        repo_root="/tmp/repo",
        check_type=check_type,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        passed=False,
        checks=[{"name": "worktree", "passed": False, "detail": "checkout conflict"}],
        expires_at="",
    )
    return receipt


# ---------------------------------------------------------------------------
# _target_agent fallback chain
# ---------------------------------------------------------------------------


class TestTargetAgent:
    def test_returns_requested_target_agent_when_provided(self) -> None:
        loop = _make_loop(default_target_agent="claude")
        result = _target_agent(
            loop,
            selected_runner={"runner_type": "codex"},
            requested_target_agent="gemini",
        )
        assert result == "gemini"

    def test_falls_back_to_runner_type_when_no_requested_agent(self) -> None:
        loop = _make_loop(default_target_agent="claude")
        result = _target_agent(
            loop,
            selected_runner={"runner_type": "codex"},
            requested_target_agent=None,
        )
        assert result == "codex"

    def test_falls_back_to_config_default_when_runner_type_empty(self) -> None:
        loop = _make_loop(default_target_agent="claude")
        result = _target_agent(
            loop,
            selected_runner={"runner_type": ""},
            requested_target_agent=None,
        )
        assert result == "claude"

    def test_falls_back_to_codex_as_ultimate_default(self) -> None:
        loop = _make_loop(default_target_agent="")
        result = _target_agent(
            loop,
            selected_runner=None,
            requested_target_agent=None,
        )
        assert result == "codex"

    def test_strips_and_lowercases_requested_agent(self) -> None:
        loop = _make_loop()
        result = _target_agent(
            loop,
            selected_runner=None,
            requested_target_agent="  CLAUDE  ",
        )
        assert result == "claude"

    def test_runner_type_takes_precedence_over_config_default(self) -> None:
        loop = _make_loop(default_target_agent="claude")
        result = _target_agent(
            loop,
            selected_runner={"runner_type": "codex"},
            requested_target_agent=None,
        )
        assert result == "codex"

    def test_selected_runner_none_falls_through_to_config(self) -> None:
        loop = _make_loop(default_target_agent="myagent")
        result = _target_agent(
            loop,
            selected_runner=None,
            requested_target_agent=None,
        )
        assert result == "myagent"


# ---------------------------------------------------------------------------
# _gate_reason
# ---------------------------------------------------------------------------


class TestGateReason:
    def test_contract_invalid_adds_reason_and_action(self) -> None:
        reasons, actions = _gate_reason(
            issue_number=99,
            target_agent="codex",
            missing_slices=[],
            contract_valid=False,
        )
        assert any("99" in r for r in reasons)
        assert len(actions) >= 1

    def test_missing_slice_runner_adds_targeted_message(self) -> None:
        reasons, actions = _gate_reason(
            issue_number=7,
            target_agent="claude",
            missing_slices=["runner"],
            contract_valid=True,
        )
        assert any("runner" in r.lower() for r in reasons)
        assert any("runner" in a.lower() for a in actions)

    def test_missing_slice_git_adds_git_message(self) -> None:
        reasons, actions = _gate_reason(
            issue_number=7,
            target_agent="codex",
            missing_slices=["git"],
            contract_valid=True,
        )
        assert any("git" in r.lower() for r in reasons)

    def test_empty_missing_slices_and_valid_contract_produces_empty_output(self) -> None:
        reasons, actions = _gate_reason(
            issue_number=1,
            target_agent="codex",
            missing_slices=[],
            contract_valid=True,
        )
        assert reasons == []
        assert actions == []

    def test_next_actions_are_deduplicated(self) -> None:
        _, actions = _gate_reason(
            issue_number=1,
            target_agent="codex",
            missing_slices=["runner", "runner"],  # type: ignore[arg-type]
            contract_valid=True,
        )
        # Even with duplicate slices, the action should appear only once
        assert len(actions) == len(set(actions))


# ---------------------------------------------------------------------------
# dispatch_contract_gate — full integration via mocks
# ---------------------------------------------------------------------------

_BASE_PATCHES = [
    "aragora.swarm.dispatch_contract_gate.build_worker_contract",
    "aragora.swarm.dispatch_contract_gate.run_contract_preflight_receipt",
    "aragora.swarm.dispatch_contract_gate._persist_preview_contract",
    "aragora.swarm.dispatch_contract_gate.CredentialEnvelope.from_environment",
]


def _make_valid_contract_mock() -> MagicMock:
    contract = MagicMock()
    contract.validate.return_value = None
    contract.admission_check.return_value = True
    contract.checksum.return_value = "abc123deadbeef"
    contract.to_dict.return_value = {"agent": "codex"}
    return contract


class TestDispatchContractGate:
    def test_passing_preflight_returns_none(self, tmp_path) -> None:
        """All checks pass → gate opens, returns None."""
        loop = _make_loop()
        issue = _make_issue(10)
        spec = MagicMock()
        spec.work_orders = None
        spec.file_scope_hints = ["aragora/swarm/preflight.py"]
        spec.mission_context_policies = {}

        passing_receipt = _make_passing_receipt()

        with (
            patch(
                "aragora.swarm.dispatch_contract_gate.build_worker_contract",
                return_value=_make_valid_contract_mock(),
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate._persist_preview_contract",
                return_value=tmp_path / "contract.json",
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate.run_contract_preflight_receipt",
                return_value=passing_receipt,
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate.CredentialEnvelope.from_environment"
            ) as mock_env,
        ):
            mock_env.return_value.missing_slices.return_value = []
            mock_env.return_value.preflight_cache_payload.return_value = {}

            result = dispatch_contract_gate(
                loop,
                issue,
                spec,
                selected_runner={"runner_type": "codex"},
                requested_target_agent=None,
                worker_env=None,
                claimed_runner_id=None,
            )

        assert result is None

    def test_missing_slices_returns_blocked_outcome(self, tmp_path) -> None:
        """Missing credential slices block dispatch before preflight."""
        loop = _make_loop()
        issue = _make_issue(20)
        spec = MagicMock()
        spec.work_orders = None
        spec.file_scope_hints = []
        spec.mission_context_policies = {}

        with (
            patch(
                "aragora.swarm.dispatch_contract_gate.build_worker_contract",
                return_value=_make_valid_contract_mock(),
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate.CredentialEnvelope.from_environment"
            ) as mock_env,
        ):
            mock_env.return_value.missing_slices.return_value = ["runner"]
            mock_env.return_value.preflight_cache_payload.return_value = {}

            result = dispatch_contract_gate(
                loop,
                issue,
                spec,
                selected_runner=None,
                requested_target_agent=None,
                worker_env=None,
                claimed_runner_id=None,
            )

        assert result is not None
        assert result["status"] == "needs_human"
        assert "runner" in result["dispatch_contract"]["missing_slices"]
        assert result["outcome"] in {"blocked_auth_failure", "blocked", "blocked_no_runner"}

    def test_invalid_contract_returns_blocked_dict(self, tmp_path) -> None:
        """Contract admission_check=False → blocked gate dict."""
        loop = _make_loop()
        issue = _make_issue(30)
        spec = MagicMock()
        spec.work_orders = None
        spec.file_scope_hints = []
        spec.mission_context_policies = {}

        bad_contract = MagicMock()
        bad_contract.validate.return_value = None
        bad_contract.admission_check.return_value = False
        bad_contract.checksum.return_value = "badcheck"
        bad_contract.to_dict.return_value = {}

        with (
            patch(
                "aragora.swarm.dispatch_contract_gate.build_worker_contract",
                return_value=bad_contract,
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate.CredentialEnvelope.from_environment"
            ) as mock_env,
        ):
            mock_env.return_value.missing_slices.return_value = []
            mock_env.return_value.preflight_cache_payload.return_value = {}

            result = dispatch_contract_gate(
                loop,
                issue,
                spec,
                selected_runner=None,
                requested_target_agent=None,
                worker_env=None,
                claimed_runner_id=None,
            )

        assert result is not None
        assert result["status"] == "needs_human"
        assert result["dispatch_contract"]["contract_valid"] is False
        assert any("30" in r for r in result["reasons"])

    def test_failed_preflight_receipt_returns_blocked_dict(self, tmp_path) -> None:
        """Failing preflight receipt → returns outcome dict (not None)."""
        loop = _make_loop()
        issue = _make_issue(50)
        spec = MagicMock()
        spec.work_orders = None
        spec.file_scope_hints = ["aragora/swarm/preflight.py"]
        spec.mission_context_policies = {}

        failing_receipt = _make_failing_receipt()

        with (
            patch(
                "aragora.swarm.dispatch_contract_gate.build_worker_contract",
                return_value=_make_valid_contract_mock(),
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate._persist_preview_contract",
                return_value=tmp_path / "contract.json",
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate.run_contract_preflight_receipt",
                return_value=failing_receipt,
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate.CredentialEnvelope.from_environment"
            ) as mock_env,
        ):
            mock_env.return_value.missing_slices.return_value = []
            mock_env.return_value.preflight_cache_payload.return_value = {}

            result = dispatch_contract_gate(
                loop,
                issue,
                spec,
                selected_runner=None,
                requested_target_agent="codex",
                worker_env=None,
                claimed_runner_id=None,
            )

        assert result is not None
        assert result["status"] == "needs_human"
        assert len(result["dispatch_contract"]["preflight_receipts"]) >= 1
        failed = result["dispatch_contract"]["preflight_receipts"][0]
        assert failed["passed"] is False

    def test_claimed_runner_id_released_on_gate_block(self, tmp_path) -> None:
        """When gate blocks, claimed_runner_id must be released."""
        loop = _make_loop()
        issue = _make_issue(60)
        spec = MagicMock()
        spec.work_orders = None
        spec.file_scope_hints = []
        spec.mission_context_policies = {}

        with (
            patch(
                "aragora.swarm.dispatch_contract_gate.build_worker_contract",
                return_value=_make_valid_contract_mock(),
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate.CredentialEnvelope.from_environment"
            ) as mock_env,
        ):
            mock_env.return_value.missing_slices.return_value = ["runner"]
            mock_env.return_value.preflight_cache_payload.return_value = {}

            result = dispatch_contract_gate(
                loop,
                issue,
                spec,
                selected_runner=None,
                requested_target_agent=None,
                worker_env=None,
                claimed_runner_id="runner-lease-xyz",
            )

        assert result is not None
        loop._release_runner_claim.assert_called_once_with("runner-lease-xyz")

    def test_preflight_exception_produces_synthetic_summary(self, tmp_path) -> None:
        """Exception during preflight receipt generation → synthetic failure summary, not crash."""
        loop = _make_loop()
        issue = _make_issue(70)
        spec = MagicMock()
        spec.work_orders = None
        spec.file_scope_hints = ["aragora/swarm/worker_contract.py"]
        spec.mission_context_policies = {}

        with (
            patch(
                "aragora.swarm.dispatch_contract_gate.build_worker_contract",
                return_value=_make_valid_contract_mock(),
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate._persist_preview_contract",
                side_effect=RuntimeError("disk full"),
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate.CredentialEnvelope.from_environment"
            ) as mock_env,
        ):
            mock_env.return_value.missing_slices.return_value = []
            mock_env.return_value.preflight_cache_payload.return_value = {}

            result = dispatch_contract_gate(
                loop,
                issue,
                spec,
                selected_runner=None,
                requested_target_agent="codex",
                worker_env=None,
                claimed_runner_id=None,
            )

        assert result is not None
        assert result["status"] == "needs_human"
        receipts = result["dispatch_contract"]["preflight_receipts"]
        assert len(receipts) == 1
        assert receipts[0]["passed"] is False

    def test_result_structure_keys_always_present(self, tmp_path) -> None:
        """Blocked result always has the required top-level and nested keys."""
        loop = _make_loop()
        issue = _make_issue(80)
        spec = MagicMock()
        spec.work_orders = None
        spec.file_scope_hints = []
        spec.mission_context_policies = {}

        with (
            patch(
                "aragora.swarm.dispatch_contract_gate.build_worker_contract",
                return_value=_make_valid_contract_mock(),
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate.CredentialEnvelope.from_environment"
            ) as mock_env,
        ):
            mock_env.return_value.missing_slices.return_value = ["github_api"]
            mock_env.return_value.preflight_cache_payload.return_value = {}

            result = dispatch_contract_gate(
                loop,
                issue,
                spec,
                selected_runner=None,
                requested_target_agent=None,
                worker_env=None,
                claimed_runner_id=None,
            )

        assert result is not None
        for key in ("status", "outcome", "reasons", "next_actions", "dispatch_contract"):
            assert key in result, f"missing key: {key}"
        dc = result["dispatch_contract"]
        for key in (
            "target_agent",
            "contract_valid",
            "missing_slices",
            "credential_envelope",
            "required_receipts",
            "preflight_receipts",
        ):
            assert key in dc, f"dispatch_contract missing key: {key}"

    def test_requested_target_agent_propagated_into_dispatch_contract(self, tmp_path) -> None:
        """target_agent chosen from requested_target_agent appears in the gate result."""
        loop = _make_loop()
        issue = _make_issue(90)
        spec = MagicMock()
        spec.work_orders = None
        spec.file_scope_hints = []
        spec.mission_context_policies = {}

        with (
            patch(
                "aragora.swarm.dispatch_contract_gate.build_worker_contract",
                return_value=_make_valid_contract_mock(),
            ),
            patch(
                "aragora.swarm.dispatch_contract_gate.CredentialEnvelope.from_environment"
            ) as mock_env,
        ):
            mock_env.return_value.missing_slices.return_value = ["runner"]
            mock_env.return_value.preflight_cache_payload.return_value = {}

            result = dispatch_contract_gate(
                loop,
                issue,
                spec,
                selected_runner=None,
                requested_target_agent="claude",
                worker_env=None,
                claimed_runner_id=None,
            )

        assert result is not None
        assert result["dispatch_contract"]["target_agent"] == "claude"
