from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.pipeline.execution_mode import ExecutionMode
from aragora.swarm.worker_contract import (
    WorkerContract,
    build_worker_contract,
    checksum_contract_payload,
)
from aragora.swarm.worker_process import LaunchConfig


def test_build_worker_contract_includes_mission_lineage_and_context_policy(tmp_path) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    config = LaunchConfig(
        allow_codex_full_auto=True,
        execution_mode=ExecutionMode.AUTONOMOUS,
    )

    contract = build_worker_contract(
        agent="codex",
        config=config,
        worktree_path=str(worktree),
        env={},
        work_order={
            "mission_id": "mission-rs-credential-envelope",
            "stage_id": "stage-contract-aware-preflight",
            "assertion_ids": ["RS-04-ASSERT-1"],
            "file_scope": ["aragora/swarm/preflight.py"],
            "evidence_expectations": ["validation_command", "worker_contract", "receipt"],
        },
    )

    payload = contract.to_dict()

    assert payload["mission_id"] == "mission-rs-credential-envelope"
    assert payload["stage_id"] == "stage-contract-aware-preflight"
    assert payload["assertion_ids"] == ["RS-04-ASSERT-1"]
    assert payload["mission_context_policy"]["role"] == "worker"
    assert payload["mission_context_policy"]["transcript_allowance"] == "none"
    contract.validate()


def _make_valid_contract() -> WorkerContract:
    """Create a valid WorkerContract with all required fields populated."""
    return WorkerContract(
        runner_type="claude-cli",
        agent="claude",
        model="opus",
        profile="default",
        permissions={"allow_dangerous_permissions": False},
        execution_mode="auto",
        git_auth_mode="https",
        gh_api_auth_mode="user",
        budget={
            "max_wall_time_seconds": 600.0,
            "no_progress_timeout_seconds": 300.0,
            "max_turns": None,
            "max_tokens": None,
            "max_retries": None,
        },
        env_checksum="abc123",
        mission_id="mission-rs04-worker-contract",
        stage_id="stage-admission-check",
        assertion_ids=["RS-04-ASSERT-1"],
        evidence_expectations=["worker_contract", "worker_contract_checksum"],
        mission_context_policy={
            "role": "worker",
            "allowed_artifact_classes": ["worker_contract", "receipt"],
            "max_source_count": 4,
            "max_chars": 2000,
            "transcript_allowance": "none",
        },
    )


_REQUIRED_STRING_FIELDS = (
    "runner_type",
    "agent",
    "model",
    "profile",
    "execution_mode",
    "git_auth_mode",
    "gh_api_auth_mode",
    "env_checksum",
    "contract_version",
)


class TestAdmissionCheckExists:
    def test_method_exists(self) -> None:
        assert hasattr(WorkerContract, "admission_check")

    def test_method_is_callable(self) -> None:
        contract = _make_valid_contract()
        assert callable(getattr(contract, "admission_check"))


class TestAdmissionCheckValidContract:
    def test_valid_contract_passes(self) -> None:
        contract = _make_valid_contract()
        assert contract.admission_check() is True

    def test_valid_contract_checksum_matches(self) -> None:
        contract = _make_valid_contract()
        expected = checksum_contract_payload(contract.to_dict())
        assert contract._expected_checksum == expected


class TestAdmissionCheckMissingFieldsParametrized:
    @pytest.mark.parametrize("field_name", _REQUIRED_STRING_FIELDS)
    def test_empty_string_field_rejected(self, field_name: str) -> None:
        contract = _make_valid_contract()
        object.__setattr__(contract, field_name, "")
        assert contract.admission_check() is False

    @pytest.mark.parametrize("field_name", _REQUIRED_STRING_FIELDS)
    def test_none_field_rejected(self, field_name: str) -> None:
        contract = _make_valid_contract()
        object.__setattr__(contract, field_name, None)
        assert contract.admission_check() is False

    def test_none_budget_rejected(self) -> None:
        contract = _make_valid_contract()
        contract.budget = None  # type: ignore[assignment]
        assert contract.admission_check() is False

    def test_empty_budget_rejected(self) -> None:
        contract = _make_valid_contract()
        contract.budget = ""  # type: ignore[assignment]
        assert contract.admission_check() is False

    def test_none_mission_context_policy_rejected(self) -> None:
        contract = _make_valid_contract()
        contract.mission_context_policy = None  # type: ignore[assignment]
        assert contract.admission_check() is False

    def test_empty_mission_context_policy_rejected(self) -> None:
        contract = _make_valid_contract()
        contract.mission_context_policy = {}
        assert contract.admission_check() is False


class TestAdmissionCheckDrift:
    def test_modified_agent_detected(self) -> None:
        contract = _make_valid_contract()
        assert contract.admission_check() is True
        contract.agent = "codex"
        assert contract.admission_check() is False

    def test_modified_model_detected(self) -> None:
        contract = _make_valid_contract()
        contract.model = "gpt-4"
        assert contract.admission_check() is False

    def test_modified_budget_detected(self) -> None:
        contract = _make_valid_contract()
        contract.budget = {"max_wall_time_seconds": 9999.0}
        assert contract.admission_check() is False

    def test_modified_permissions_detected(self) -> None:
        contract = _make_valid_contract()
        contract.permissions = {"allow_dangerous_permissions": True}
        assert contract.admission_check() is False

    def test_modified_runner_type_detected(self) -> None:
        contract = _make_valid_contract()
        contract.runner_type = "codex-cli"
        assert contract.admission_check() is False

    def test_modified_execution_mode_detected(self) -> None:
        contract = _make_valid_contract()
        contract.execution_mode = "supervised"
        assert contract.admission_check() is False


class TestAdmissionCheckFailClosed:
    def test_none_permissions_no_raise(self) -> None:
        contract = _make_valid_contract()
        contract.permissions = None  # type: ignore[assignment]
        result = contract.admission_check()
        assert result is False

    def test_pathological_input_no_raise(self) -> None:
        contract = _make_valid_contract()
        contract.budget = "not_a_dict"  # type: ignore[assignment]
        result = contract.admission_check()
        assert result is False

    def test_integer_field_no_raise(self) -> None:
        contract = _make_valid_contract()
        contract.agent = 12345  # type: ignore[assignment]
        result = contract.admission_check()
        assert isinstance(result, bool)

    def test_corrupted_expected_checksum_no_raise(self) -> None:
        contract = _make_valid_contract()
        object.__setattr__(contract, "_expected_checksum", None)
        result = contract.admission_check()
        assert result is False


class TestAdmissionCheckDistinct:
    def test_both_methods_exist(self) -> None:
        contract = _make_valid_contract()
        assert hasattr(contract, "validate")
        assert hasattr(contract, "admission_check")

    def test_validate_raises_on_invalid(self) -> None:
        contract = _make_valid_contract()
        contract.runner_type = ""
        with pytest.raises(ValueError):
            contract.validate()

    def test_admission_check_returns_bool_on_invalid(self) -> None:
        contract = _make_valid_contract()
        contract.runner_type = ""
        result = contract.admission_check()
        assert result is False
        assert isinstance(result, bool)

    def test_admission_check_includes_drift_detection(self) -> None:
        contract = _make_valid_contract()
        contract.agent = "codex"
        contract.validate()
        assert contract.admission_check() is False

    def test_validate_and_admission_check_agree_on_valid(self) -> None:
        contract = _make_valid_contract()
        contract.validate()
        assert contract.admission_check() is True


class TestBuildWorkerContractAdmission:
    @patch("aragora.swarm.worker_contract._git_auth_mode", return_value="https")
    def test_claude_contract_passes_admission(self, _mock_git: object) -> None:
        config = LaunchConfig(timeout_seconds=600, no_progress_timeout_seconds=300)
        contract = build_worker_contract(
            agent="claude",
            config=config,
            worktree_path="/tmp/fake-worktree",
            env={"GH_TOKEN": "ghp_test123", "ARAGORA_API_TOKEN": "tok"},
        )
        assert contract.admission_check() is True

    @patch("aragora.swarm.worker_contract._git_auth_mode", return_value="ssh")
    def test_codex_contract_passes_admission(self, _mock_git: object) -> None:
        config = LaunchConfig(timeout_seconds=1200, no_progress_timeout_seconds=600)
        contract = build_worker_contract(
            agent="codex",
            config=config,
            worktree_path="/tmp/fake-worktree",
            env={"GITHUB_TOKEN": "gho_abc"},
        )
        assert contract.admission_check() is True

    @patch("aragora.swarm.worker_contract._git_auth_mode", return_value="unknown")
    def test_unknown_agent_contract_passes_admission(self, _mock_git: object) -> None:
        config = LaunchConfig(timeout_seconds=600, no_progress_timeout_seconds=300)
        contract = build_worker_contract(
            agent="custom-agent",
            config=config,
            worktree_path="/tmp/fake-worktree",
            env={},
        )
        assert contract.admission_check() is True

    @patch("aragora.swarm.worker_contract._git_auth_mode", return_value="https")
    def test_build_output_has_correct_checksum(self, _mock_git: object) -> None:
        config = LaunchConfig(timeout_seconds=600, no_progress_timeout_seconds=300)
        contract = build_worker_contract(
            agent="claude",
            config=config,
            worktree_path="/tmp/fake-worktree",
            env={"GH_TOKEN": "ghp_test"},
        )
        assert contract._expected_checksum == checksum_contract_payload(contract.to_dict())

    @patch("aragora.swarm.worker_contract._git_auth_mode", return_value="https")
    def test_build_output_not_drifted_after_construction(self, _mock_git: object) -> None:
        config = LaunchConfig(timeout_seconds=600, no_progress_timeout_seconds=300)
        contract = build_worker_contract(
            agent="claude",
            config=config,
            worktree_path="/tmp/fake-worktree",
            env={"GH_TOKEN": "ghp_test"},
        )
        assert contract.admission_check() is True
        contract.model = "tampered"
        assert contract.admission_check() is False
