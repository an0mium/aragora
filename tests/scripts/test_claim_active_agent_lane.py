"""Tests for ``scripts/claim_active_agent_lane.py``.

The helper is a thin write-side complement to the lane registry shipped
in ``scripts/agent_bridge.py``. These tests cover:

- A fresh claim creates the registry file + a single row.
- A claim refresh from the same owner overwrites in place.
- A claim from a different owner is rejected with exit code 2.
- ``--force`` overrides the owner-mismatch rejection.
- Released / conflict statuses round-trip.
- The persisted row matches the ``LaneRecord`` schema so the existing
  agent_bridge reader picks it up unmodified.
- The script imports no aragora package (pure stdlib).
"""

from __future__ import annotations

import importlib
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "claim_active_agent_lane.py"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
claim_module = importlib.import_module("claim_active_agent_lane")


@pytest.fixture()
def tmp_registry(tmp_path: Path) -> Path:
    return tmp_path / "lanes.json"


def test_fresh_claim_writes_single_row(tmp_registry: Path) -> None:
    result = claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="droid/phase-x",
        owner_session="claude-20260517",
        goal="phase X goal",
        source="plan",
        status="active",
        branch="droid/phase-x-20260517",
        worktree="/tmp/wt-x",
    )
    assert result["lane_id"] == "droid/phase-x"
    assert result["owner_session"] == "claude-20260517"
    assert tmp_registry.exists()
    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 1
    assert payload[0]["lane_id"] == "droid/phase-x"
    assert payload[0]["status"] == "active"
    assert payload[0]["updated_at"].endswith("Z")
    assert payload[0]["last_heartbeat_at"] == payload[0]["updated_at"]
    assert payload[0]["last_steering_outcome"] == "unknown"
    assert payload[0]["next_action"] == "unspecified active lane action"


def test_lifecycle_state_machine_statuses_round_trip(tmp_registry: Path) -> None:
    result = claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="control-plane-lane",
        owner_session="codex-control-plane",
        status="working",
        next_action="finish current mailbox steering",
        last_heartbeat_at="2026-05-21T20:00:00Z",
        last_steering_outcome="obeyed",
    )

    assert result["status"] == "working"
    assert result["last_heartbeat_at"] == "2026-05-21T20:00:00Z"
    assert result["last_steering_outcome"] == "obeyed"


def test_same_owner_refresh_overwrites_in_place(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="droid/phase-x",
        owner_session="claude-20260517",
        status="active",
    )
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="droid/phase-x",
        owner_session="claude-20260517",
        status="completed",
        next_action="PR opened",
    )
    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["status"] == "completed"
    assert payload[0]["next_action"] == "PR opened"


def test_different_owner_is_rejected(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="shared-lane",
        owner_session="claude-A",
    )
    with pytest.raises(claim_module.ClaimError) as excinfo:
        claim_module.claim_lane(
            registry_path=tmp_registry,
            lane_id="shared-lane",
            owner_session="claude-B",
        )
    assert "already claimed" in str(excinfo.value)
    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert payload[0]["owner_session"] == "claude-A"


def test_force_overrides_owner_mismatch(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="shared-lane",
        owner_session="claude-A",
    )
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="shared-lane",
        owner_session="claude-B",
        force=True,
    )
    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert payload[0]["owner_session"] == "claude-B"


def test_different_lane_same_pr_is_rejected_by_default(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-a",
        owner_session="codex-A",
        pr_number=7245,
    )

    with pytest.raises(claim_module.ClaimError) as excinfo:
        claim_module.claim_lane(
            registry_path=tmp_registry,
            lane_id="lane-b",
            owner_session="codex-B",
            pr_number=7245,
        )

    assert "pr_number='7245' already claimed" in str(excinfo.value)


def test_different_lane_same_pr_can_be_allowed_when_requested(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-a",
        owner_session="codex-A",
        pr_number=7245,
    )

    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-b",
        owner_session="codex-B",
        pr_number=7245,
        allow_resource_conflicts=True,
    )

    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert sorted(row["lane_id"] for row in payload) == ["lane-a", "lane-b"]


def test_different_lane_same_branch_is_rejected_by_default(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-a",
        owner_session="codex-A",
        branch="worktree-codex-insights",
    )

    with pytest.raises(claim_module.ClaimError) as excinfo:
        claim_module.claim_lane(
            registry_path=tmp_registry,
            lane_id="lane-b",
            owner_session="codex-B",
            branch="worktree-codex-insights",
        )

    assert "branch='worktree-codex-insights' already claimed" in str(excinfo.value)


def test_different_lane_same_worktree_is_rejected_by_default(
    tmp_registry: Path,
) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-a",
        owner_session="codex-A",
        worktree="/tmp/aragora-pr7245",
    )

    with pytest.raises(claim_module.ClaimError) as excinfo:
        claim_module.claim_lane(
            registry_path=tmp_registry,
            lane_id="lane-b",
            owner_session="codex-B",
            worktree="/tmp/aragora-pr7245",
        )

    canonical = claim_module._normalize_worktree_token("/tmp/aragora-pr7245")
    assert f"worktree={canonical!r} already claimed" in str(excinfo.value)


def test_same_owner_can_refresh_same_pr_identity(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-a",
        owner_session="codex-A",
        pr_number=7245,
    )
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-b",
        owner_session="codex-A",
        pr_number=7245,
    )

    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert sorted(row["lane_id"] for row in payload) == ["lane-a", "lane-b"]


def test_released_lane_identity_does_not_block_new_owner(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-a",
        owner_session="codex-A",
        status="released",
        pr_number=7245,
    )
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-b",
        owner_session="codex-B",
        pr_number=7245,
    )

    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert len(payload) == 2


def test_other_lanes_are_preserved_during_claim(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-a",
        owner_session="claude-A",
    )
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-b",
        owner_session="claude-B",
    )
    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    lane_ids = sorted(row["lane_id"] for row in payload)
    assert lane_ids == ["lane-a", "lane-b"]


def _spawn_cli_claim(
    registry: Path,
    *,
    lane_id: str,
    owner_session: str,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--lane-id",
            lane_id,
            "--owner-session",
            owner_session,
            "--registry-path",
            str(registry),
            "--json",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _collect_claim_processes(
    processes: list[subprocess.Popen[str]],
) -> list[tuple[int, str, str]]:
    results: list[tuple[int, str, str]] = []
    for proc in processes:
        stdout, stderr = proc.communicate(timeout=30)
        results.append((proc.returncode, stdout, stderr))
    return results


def test_concurrent_different_lane_claims_are_preserved(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    expected_lane_ids = [f"lane-{idx:02d}" for idx in range(16)]

    results = _collect_claim_processes(
        [
            _spawn_cli_claim(
                registry,
                lane_id=lane_id,
                owner_session=f"owner-{idx:02d}",
            )
            for idx, lane_id in enumerate(expected_lane_ids)
        ]
    )

    assert all(returncode == 0 for returncode, _stdout, _stderr in results), results
    payload = json.loads(registry.read_text(encoding="utf-8"))
    lane_ids = sorted(row["lane_id"] for row in payload)
    assert lane_ids == expected_lane_ids


def test_concurrent_same_lane_claims_do_not_both_succeed(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"

    results = _collect_claim_processes(
        [
            _spawn_cli_claim(
                registry,
                lane_id="shared-lane",
                owner_session=f"owner-{idx:02d}",
            )
            for idx in range(12)
        ]
    )

    successes = [stdout for returncode, stdout, _stderr in results if returncode == 0]
    conflicts = [stderr for returncode, _stdout, stderr in results if returncode == 2]
    assert len(successes) == 1, results
    assert len(conflicts) == 11, results
    assert all("already claimed" in stderr for stderr in conflicts)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["owner_session"] == json.loads(successes[0])["owner_session"]


def test_conflict_status_round_trips(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="contested-lane",
        owner_session="claude-A",
        status="conflict",
        conflict_session="claude-B",
        conflict_reason="same branch",
    )
    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert payload[0]["status"] == "conflict"
    assert payload[0]["conflict_session"] == "claude-B"
    assert payload[0]["conflict_reason"] == "same branch"


def test_released_status_round_trips(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="phase-x",
        owner_session="claude-A",
        status="active",
    )
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="phase-x",
        owner_session="claude-A",
        status="released",
    )
    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert payload[0]["status"] == "released"


def test_release_stale_claims_only_releases_old_owner_rows(tmp_registry: Path) -> None:
    old = "2026-05-17T10:00:00Z"
    fresh = "2026-05-17T11:59:00Z"
    now = dt.datetime(2026, 5, 17, 12, 0, tzinfo=dt.UTC)
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="old-owned",
        owner_session="codex-A",
        status="active",
        updated_at=old,
    )
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="fresh-owned",
        owner_session="codex-A",
        status="active",
        updated_at=fresh,
    )
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="old-other",
        owner_session="codex-B",
        status="active",
        updated_at=old,
    )

    result = claim_module.release_stale_claims(
        registry_path=tmp_registry,
        owner_session="codex-A",
        ttl_minutes=30,
        updated_at="2026-05-17T12:00:00Z",
        now=now,
    )

    payload = {row["lane_id"]: row for row in json.loads(tmp_registry.read_text())}
    assert result["released_lane_ids"] == ["old-owned"]
    assert payload["old-owned"]["status"] == "released"
    assert payload["fresh-owned"]["status"] == "active"
    assert payload["old-other"]["status"] == "active"


def test_invalid_status_is_rejected(tmp_registry: Path) -> None:
    with pytest.raises(ValueError):
        claim_module.claim_lane(
            registry_path=tmp_registry,
            lane_id="phase-x",
            owner_session="claude-A",
            status="bogus",
        )


def test_empty_lane_id_is_rejected(tmp_registry: Path) -> None:
    with pytest.raises(ValueError):
        claim_module.claim_lane(
            registry_path=tmp_registry,
            lane_id="",
            owner_session="claude-A",
        )


def test_empty_owner_is_rejected(tmp_registry: Path) -> None:
    with pytest.raises(ValueError):
        claim_module.claim_lane(
            registry_path=tmp_registry,
            lane_id="phase-x",
            owner_session="",
        )


def test_persisted_schema_matches_lane_record_keys(tmp_registry: Path) -> None:
    """The persisted row must use only the LaneRecord-known keys so that
    ``scripts/agent_bridge.py`` ``_load_lane_registry`` accepts it without
    surfacing as malformed."""
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="phase-y",
        owner_session="claude-X",
        goal="g",
        source="s",
        status="active",
        next_action="n",
        branch="b",
        worktree="w",
        pr_number=1234,
        conflict_session="",
        conflict_reason="",
    )
    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    keys = set(payload[0].keys())
    allowed = set(claim_module.LANE_RECORD_KEYS)
    assert keys.issubset(allowed)


def test_desktop_identity_metadata_round_trips(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="codex-b-review",
        owner_session="codex-B",
        desktop_label="Codex B",
        codex_thread_id="019e-test-thread",
        codex_rollout_path="/Users/armand/.codex/sessions/rollout.jsonl",
        session_title="Review #7286",
    )

    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    assert payload[0]["desktop_label"] == "Codex B"
    assert payload[0]["codex_thread_id"] == "019e-test-thread"
    assert payload[0]["codex_rollout_path"].endswith("rollout.jsonl")
    assert payload[0]["session_title"] == "Review #7286"


def test_contact_metadata_round_trips(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="codex-steer",
        owner_session="codex-owner",
        contact_method="codex-exec-resume:019e-thread",
        contact_payload={"thread_id": "019e-thread", "socket": "/tmp/codex.sock"},
        last_mailbox_check_at="2026-05-20T01:00:00Z",
        last_delivery_at="2026-05-20T01:01:00Z",
        last_ack_at="2026-05-20T01:02:00Z",
    )

    payload = json.loads(tmp_registry.read_text(encoding="utf-8"))
    row = payload[0]
    assert row["contact_method"] == "codex-exec-resume:019e-thread"
    assert row["contact_payload"]["thread_id"] == "019e-thread"
    assert row["last_mailbox_check_at"] == "2026-05-20T01:00:00Z"
    assert row["last_delivery_at"] == "2026-05-20T01:01:00Z"
    assert row["last_ack_at"] == "2026-05-20T01:02:00Z"


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0


def test_cli_writes_registry_via_subprocess(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--lane-id",
            "phase-cli",
            "--owner-session",
            "claude-cli",
            "--branch",
            "droid/phase-cli",
            "--desktop-label",
            "Codex C",
            "--codex-thread-id",
            "019e-cli-thread",
            "--codex-rollout-path",
            "/Users/armand/.codex/sessions/cli.jsonl",
            "--session-title",
            "CLI lane claim",
            "--contact-method",
            "tmux:aragora:1",
            "--contact-payload",
            '{"target": "aragora:1"}',
            "--status",
            "active",
            "--registry-path",
            str(registry),
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    payload_out = json.loads(result.stdout)
    assert payload_out["lane_id"] == "phase-cli"
    assert payload_out["desktop_label"] == "Codex C"
    assert registry.exists()
    file_payload = json.loads(registry.read_text(encoding="utf-8"))
    assert len(file_payload) == 1
    assert file_payload[0]["lane_id"] == "phase-cli"
    assert file_payload[0]["codex_thread_id"] == "019e-cli-thread"
    assert file_payload[0]["codex_rollout_path"].endswith("cli.jsonl")
    assert file_payload[0]["session_title"] == "CLI lane claim"
    assert file_payload[0]["contact_method"] == "tmux:aragora:1"
    assert file_payload[0]["contact_payload"]["target"] == "aragora:1"


def test_cli_conflict_returns_exit_code_2(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--lane-id",
            "shared",
            "--owner-session",
            "claude-A",
            "--registry-path",
            str(registry),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--lane-id",
            "shared",
            "--owner-session",
            "claude-B",
            "--registry-path",
            str(registry),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "already claimed" in result.stderr


def test_module_imports_no_aragora_package() -> None:
    """Pure stdlib invariant: the module must not import ``aragora``
    so it works during partial bootstraps and on freshly cloned checkouts."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "import aragora" not in source
    assert "from aragora" not in source


def test_resolve_registry_path_prefers_repo_local(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / ".aragora" / "agent-bridge").mkdir(parents=True)
    expected = repo_root / ".aragora" / "agent-bridge" / "lanes.json"
    actual = claim_module.resolve_registry_path(repo_root=repo_root)
    assert actual == expected


def test_resolve_registry_path_falls_back_to_user_home(tmp_path: Path) -> None:
    repo_root = tmp_path / "no_aragora_dir"
    repo_root.mkdir()
    actual = claim_module.resolve_registry_path(repo_root=repo_root)
    assert actual == claim_module.USER_LANE_PATH


def test_explicit_registry_path_wins(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / ".aragora" / "agent-bridge").mkdir(parents=True)
    override = tmp_path / "override.json"
    actual = claim_module.resolve_registry_path(repo_root=repo_root, explicit=override)
    assert actual == override


def test_normalize_branch_token_strips_refs_heads_prefix() -> None:
    assert claim_module._normalize_branch_token("refs/heads/feat/x") == "feat/x"
    assert claim_module._normalize_branch_token("refs/remotes/origin/feat/y") == "feat/y"
    assert claim_module._normalize_branch_token("origin/feat/z") == "feat/z"


def test_normalize_branch_token_strips_whitespace_and_trailing_slash() -> None:
    assert claim_module._normalize_branch_token("  feat/x/ ") == "feat/x"
    assert claim_module._normalize_branch_token("feat/x//") == "feat/x"
    assert claim_module._normalize_branch_token("") == ""


def test_normalize_worktree_token_expands_and_strips_trailing_slash(tmp_path: Path) -> None:
    target = tmp_path / "worktrees" / "x"
    target.mkdir(parents=True)
    assert claim_module._normalize_worktree_token(f"{target}/") == str(target.resolve())
    assert claim_module._normalize_worktree_token("") == ""


def test_branch_collision_detected_across_refs_heads_prefix(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-a",
        owner_session="codex-A",
        branch="refs/heads/feat/x",
    )

    with pytest.raises(claim_module.ClaimError) as excinfo:
        claim_module.claim_lane(
            registry_path=tmp_registry,
            lane_id="lane-b",
            owner_session="codex-B",
            branch="feat/x",
        )

    assert "branch='feat/x' already claimed" in str(excinfo.value)


def test_branch_collision_detected_across_trailing_slash(tmp_registry: Path) -> None:
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-a",
        owner_session="codex-A",
        branch="feat/x",
    )

    with pytest.raises(claim_module.ClaimError) as excinfo:
        claim_module.claim_lane(
            registry_path=tmp_registry,
            lane_id="lane-b",
            owner_session="codex-B",
            branch="feat/x/",
        )

    assert "branch='feat/x' already claimed" in str(excinfo.value)


def test_worktree_collision_detected_across_trailing_slash(
    tmp_path: Path, tmp_registry: Path
) -> None:
    worktree = tmp_path / "wt-a"
    worktree.mkdir()
    claim_module.claim_lane(
        registry_path=tmp_registry,
        lane_id="lane-a",
        owner_session="codex-A",
        worktree=str(worktree),
    )

    with pytest.raises(claim_module.ClaimError) as excinfo:
        claim_module.claim_lane(
            registry_path=tmp_registry,
            lane_id="lane-b",
            owner_session="codex-B",
            worktree=f"{worktree}/",
        )

    assert "worktree=" in str(excinfo.value)
    assert str(worktree.resolve()) in str(excinfo.value)
