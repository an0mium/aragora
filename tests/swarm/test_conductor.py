from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from aragora.swarm.conductor import Conductor, ConductorStep
from aragora.swarm.terminal_truth import TerminalClass


def _result(
    *,
    session_id: str | None = "sess-1",
    status: str = "needs_human",
    outcome: str = "",
    terminal_class: TerminalClass | str | None = None,
    worker_output: str = "",
    error: str = "",
    reasons: list[str] | None = None,
    changed_files: list[str] | None = None,
    work_order_changed: list[str] | None = None,
    worker_outcome: str = "",
    agent: str = "codex",
    deliverable: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": status,
        "outcome": outcome,
        "agent": agent,
        "run": {
            "work_orders": [
                {
                    "worker_outcome": worker_outcome,
                    "stdout_tail": worker_output,
                    "stderr_tail": error,
                    "changed_paths": list(work_order_changed or []),
                }
            ]
        },
    }
    if session_id is not None:
        result["session_id"] = session_id
    if terminal_class is not None:
        result["terminal_class"] = (
            terminal_class.value if isinstance(terminal_class, TerminalClass) else terminal_class
        )
    if worker_output:
        result["worker_output"] = worker_output
    if error:
        result["error"] = error
    if reasons is not None:
        result["reasons"] = list(reasons)
    if changed_files is not None:
        result["changed_files"] = list(changed_files)
    if deliverable is not None:
        result["deliverable"] = dict(deliverable)
    return result


def _install_tmux_scripts(repo_root: Path) -> None:
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "tmux_send_prompt.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (scripts_dir / "tmux_harvest.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")


def test_done_on_deliverable_created(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        101,
        _result(
            status="completed",
            outcome="deliverable_created",
            changed_files=["aragora/swarm/conductor.py"],
        ),
    )

    assert step.next_action == "done"
    assert step.terminal_class == TerminalClass.DELIVERABLE_BRANCH_PUSHED
    assert step.next_prompt == ""


def test_done_on_pr_adopted(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        102,
        _result(
            status="completed",
            outcome="pr_adopted",
            deliverable={"type": "pr", "pr_url": "https://github.com/org/repo/pull/12"},
        ),
    )

    assert step.next_action == "done"
    assert step.terminal_class == TerminalClass.DELIVERABLE_PR_CREATED


def test_done_on_already_resolved_marker_without_changes(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        103,
        _result(
            worker_output="Already implemented; nothing to commit.",
            worker_outcome="clean_exit_no_effect",
        ),
    )

    assert step.next_action == "done"
    assert step.terminal_class == TerminalClass.ISSUE_ALREADY_RESOLVED


def test_timeout_retries_same_agent(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        104,
        _result(terminal_class=TerminalClass.RESCUE_TIMEOUT, worker_output="Worker timed out"),
    )

    assert step.next_action == "retry_same"
    assert "## What was tried" in step.next_prompt
    assert "## What to try differently" in step.next_prompt


def test_verification_failure_retries_different_agent(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        105,
        _result(
            terminal_class=TerminalClass.RESCUE_VERIFICATION_FAILED,
            worker_output="pytest tests/swarm/test_x.py -q\nAssertionError: boom",
            changed_files=["aragora/swarm/x.py"],
        ),
    )

    assert step.next_action == "retry_different_agent"
    assert "## Context" in step.next_prompt
    assert "different_agent" in step.next_prompt
    assert "aragora/swarm/x.py" in step.next_prompt


def test_worker_crash_retries_same_agent(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        106,
        _result(
            terminal_class=TerminalClass.RESCUE_WORKER_CRASH,
            worker_output="Traceback: ValueError",
            error="worker crashed while parsing diff",
        ),
    )

    assert step.next_action == "retry_same"
    assert "ValueError" in step.worker_output


def test_worker_crash_escalates_on_auth_failure(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        107,
        _result(
            terminal_class=TerminalClass.RESCUE_WORKER_CRASH,
            worker_output="gh auth failed: permission denied",
        ),
    )

    assert step.next_action == "escalate"
    assert "permission denied" in step.next_prompt


def test_blocked_scope_broadness_decomposes(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        108,
        _result(
            terminal_class=TerminalClass.BLOCKED_NOT_DISPATCH_BOUNDED,
            worker_output=(
                "Task is too broad and should be split. "
                "It touches aragora/swarm/a.py, aragora/swarm/b.py, "
                "tests/swarm/test_a.py, tests/swarm/test_b.py."
            ),
        ),
    )

    assert step.next_action == "decompose"
    assert "Split the issue into 2-5 bounded subtasks" in step.next_prompt


def test_blocked_auth_failure_escalates(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        109,
        _result(terminal_class="blocked_auth_failure", worker_output="401 unauthorized"),
    )

    assert step.next_action == "escalate"
    assert step.terminal_class == TerminalClass.BLOCKED_AUTH_FAILURE


def test_blocked_decomposition_limit_escalates(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        110,
        _result(terminal_class=TerminalClass.BLOCKED_DECOMPOSITION_LIMIT),
    )

    assert step.next_action == "escalate"


def test_no_deliverable_decomposes_when_output_requests_split(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        111,
        _result(
            terminal_class=TerminalClass.RESCUE_NO_DELIVERABLE,
            worker_output="Please decompose this; it spans auth, billing, and CLI setup.",
        ),
    )

    assert step.next_action == "decompose"


def test_no_deliverable_retries_different_agent_after_prior_history(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)
    first = conductor.evaluate_worker_output(
        112,
        _result(
            terminal_class=TerminalClass.RESCUE_NO_DELIVERABLE,
            worker_output="Tried one approach but produced no deliverable.",
        ),
    )
    second = conductor.evaluate_worker_output(
        112,
        _result(
            session_id=None,
            terminal_class=TerminalClass.RESCUE_NO_DELIVERABLE,
            worker_output="Still no deliverable after retry.",
        ),
    )

    assert first.next_action == "retry_same"
    assert second.next_action == "retry_different_agent"


def test_generate_retry_prompt_includes_required_sections(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    prompt = conductor.generate_retry_prompt(
        113,
        "pytest tests/swarm/test_conductor.py -q\nAssertionError: boom",
        "verification_failed",
    )

    assert "## What was tried" in prompt
    assert "## Why it failed" in prompt
    assert "## What to try differently" in prompt
    assert "verification_failed" in prompt


def test_conductor_step_from_dict_coerces_unknown_next_action() -> None:
    step = ConductorStep.from_dict(
        {
            "issue_number": 114,
            "session_id": "sess-114",
            "worker_output": "needs a manual retry",
            "terminal_class": TerminalClass.RESCUE_NO_DELIVERABLE.value,
            "changed_files": ["aragora/swarm/conductor.py"],
            "next_action": "retry_later",
            "next_prompt": "Escalate to a human.",
        }
    )

    assert step.next_action == "escalate"


def test_generate_retry_prompt_references_prior_attempts_from_store(tmp_path: Path) -> None:
    Conductor(tmp_path).evaluate_worker_output(
        114,
        _result(terminal_class=TerminalClass.RESCUE_TIMEOUT, worker_output="timed out"),
    )

    prompt = Conductor(tmp_path).generate_retry_prompt(
        114,
        "timed out again",
        "worker_timeout",
    )

    assert "Attempt 1" in prompt
    assert "rescue_timeout" in prompt


def test_session_persistence_round_trips_across_instances(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)
    conductor.evaluate_worker_output(
        115,
        _result(
            terminal_class=TerminalClass.RESCUE_TIMEOUT,
            worker_output="worker timed out after narrow diff",
        ),
    )

    restored = Conductor(tmp_path)
    steps = restored._session_store.load_steps(115)

    assert len(steps) == 1
    assert steps[0].terminal_class == TerminalClass.RESCUE_TIMEOUT
    assert restored._session_store.latest_session_id(115) == "sess-1"


def test_explicit_terminal_class_string_is_respected(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        116,
        _result(terminal_class="blocked_no_runner", worker_output="No runner available"),
    )

    assert step.terminal_class == TerminalClass.BLOCKED_NO_RUNNER
    assert step.next_action == "escalate"


def test_invalid_terminal_class_logs_and_falls_back(caplog, tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    with caplog.at_level("WARNING"):
        step = conductor.evaluate_worker_output(
            116,
            _result(terminal_class="not-a-real-terminal-class", worker_outcome="timeout"),
        )

    assert step.terminal_class == TerminalClass.RESCUE_TIMEOUT
    assert "invalid terminal_class" in caplog.text
    assert "not-a-real-terminal-class" in caplog.text


def test_extracts_changed_files_from_run_work_orders(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        117,
        _result(
            terminal_class=TerminalClass.RESCUE_VERIFICATION_FAILED,
            changed_files=["aragora/swarm/conductor.py"],
            work_order_changed=["aragora/swarm/conductor.py", "tests/swarm/test_conductor.py"],
        ),
    )

    assert step.changed_files == [
        "aragora/swarm/conductor.py",
        "tests/swarm/test_conductor.py",
    ]


def test_session_id_falls_back_to_existing_issue_session(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)
    conductor.evaluate_worker_output(
        118,
        _result(session_id="session-keep", terminal_class=TerminalClass.RESCUE_TIMEOUT),
    )

    step = conductor.evaluate_worker_output(
        118,
        _result(session_id=None, terminal_class=TerminalClass.RESCUE_TIMEOUT),
    )

    assert step.session_id == "session-keep"


def test_retry_prompt_uses_validation_specific_guidance(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    prompt = conductor.generate_retry_prompt(
        119,
        "pytest -q tests/swarm/test_conductor.py\nAssertionError: expected 1 == 2",
        "verification_failed",
    )

    assert "Reproduce the failing verification first" in prompt


def test_publish_deferred_escalates(tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)

    step = conductor.evaluate_worker_output(
        120,
        _result(
            terminal_class=TerminalClass.RESCUE_PUBLISH_DEFERRED,
            worker_output="Deliverable exists but publish was deferred by gate.",
        ),
    )

    assert step.next_action == "escalate"


def test_dispatch_step_to_tmux_sends_prompt_and_harvests_output(
    monkeypatch, tmp_path: Path
) -> None:
    _install_tmux_scripts(tmp_path)
    conductor = Conductor(tmp_path)
    step = conductor.evaluate_worker_output(
        121,
        _result(
            terminal_class=TerminalClass.RESCUE_VERIFICATION_FAILED,
            worker_output="pytest tests/swarm/test_conductor.py -q\nAssertionError: boom",
        ),
    )
    seen_prompt_path: Path | None = None

    def fake_run(cmd: list[str], **kwargs) -> SimpleNamespace:  # noqa: ANN003
        nonlocal seen_prompt_path
        if cmd[0].endswith("tmux_send_prompt.sh"):
            prompt_index = cmd.index("--prompt-file") + 1
            seen_prompt_path = Path(cmd[prompt_index])
            assert seen_prompt_path.exists()
            assert seen_prompt_path.read_text(encoding="utf-8") == step.next_prompt
            return SimpleNamespace(returncode=0, stdout="Prompt sent", stderr="")
        if cmd[0].endswith("tmux_harvest.sh"):
            return SimpleNamespace(returncode=0, stdout="--- session ---\nagent output", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("aragora.swarm.conductor.subprocess.run", fake_run)

    result = conductor.dispatch_step_to_tmux(
        step,
        session_name="codex-conductor",
        wait_seconds=5,
        harvest_lines=80,
    )

    assert result["sent"] is True
    assert result["session_name"] == "codex-conductor"
    assert result["next_action"] == "retry_different_agent"
    assert result["prompt_chars"] == len(step.next_prompt)
    assert result["wait_seconds"] == 5
    assert result["harvest_lines"] == 80
    assert result["harvest_output"] == "--- session ---\nagent output"
    assert seen_prompt_path is not None
    assert not seen_prompt_path.exists()


def test_dispatch_step_to_tmux_skips_done_steps(monkeypatch, tmp_path: Path) -> None:
    conductor = Conductor(tmp_path)
    step = conductor.evaluate_worker_output(
        122,
        _result(
            status="completed",
            outcome="deliverable_created",
            changed_files=["aragora/swarm/conductor.py"],
        ),
    )

    def fail_run(cmd: list[str], **kwargs) -> SimpleNamespace:  # noqa: ANN003
        raise AssertionError(f"subprocess should not be invoked: {cmd}")

    monkeypatch.setattr("aragora.swarm.conductor.subprocess.run", fail_run)

    result = conductor.dispatch_step_to_tmux(step, session_name="codex-conductor")

    assert result == {
        "sent": False,
        "reason": "no_prompt",
        "session_name": "codex-conductor",
        "next_action": "done",
    }


def test_dispatch_step_to_tmux_raises_and_cleans_prompt_file(monkeypatch, tmp_path: Path) -> None:
    _install_tmux_scripts(tmp_path)
    conductor = Conductor(tmp_path)
    step = conductor.evaluate_worker_output(
        123,
        _result(
            terminal_class=TerminalClass.RESCUE_TIMEOUT,
            worker_output="Worker timed out before finishing.",
        ),
    )
    seen_prompt_path: Path | None = None

    def fake_run(cmd: list[str], **kwargs) -> SimpleNamespace:  # noqa: ANN003
        nonlocal seen_prompt_path
        if cmd[0].endswith("tmux_send_prompt.sh"):
            prompt_index = cmd.index("--prompt-file") + 1
            seen_prompt_path = Path(cmd[prompt_index])
            assert seen_prompt_path.exists()
            return SimpleNamespace(returncode=1, stdout="", stderr="Window not found")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("aragora.swarm.conductor.subprocess.run", fake_run)

    try:
        conductor.dispatch_step_to_tmux(step, session_name="missing-session")
    except RuntimeError as exc:
        assert str(exc) == "Window not found"
    else:
        raise AssertionError("Expected dispatch_step_to_tmux to raise on send failure")

    assert seen_prompt_path is not None
    assert not seen_prompt_path.exists()
