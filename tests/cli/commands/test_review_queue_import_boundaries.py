from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_isolated_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


def test_review_queue_module_import_does_not_eager_load_heavy_helpers() -> None:
    proc = _run_isolated_python(
        """
        import importlib.abc
        import sys

        BLOCKED = (
            "aragora.review.reviewer_output",
            "aragora.swarm.pr_review_protocol",
            "aragora.triage.auto_handle_calibration",
            "aragora.worktree",
            "scripts.post_merge_lane_audit",
        )

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(BLOCKED):
                    raise RuntimeError(f"blocked eager import: {fullname}")
                return None

        sys.meta_path.insert(0, Blocker())

        import aragora.cli.commands.review_queue as review_queue

        assert review_queue.ReviewPacket.__name__ == "ReviewPacket"
        """
    )

    assert proc.returncode == 0, proc.stderr


def test_pr_review_protocol_metadata_packet_does_not_eager_load_agent_factory() -> None:
    proc = _run_isolated_python(
        """
        import importlib.abc
        import sys

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {"aragora.agents.base", "asyncio"}:
                    raise RuntimeError("live-review dependencies should load only for live reviewers")
                return None

        sys.meta_path.insert(0, Blocker())

        from aragora.swarm.pr_review_protocol import default_pr_review_protocol

        packet = default_pr_review_protocol().build_packet(
            repo="synaptent/aragora",
            pr_number=7841,
            title="fix helper startup",
            base_sha="base",
            head_sha="head",
            mergeable="MERGEABLE",
            review_decision="",
            checks_summary="all checks passed",
            has_failures=False,
            has_pending=False,
            additions=1,
            deletions=1,
            changed_files=1,
            labels=[],
            high_risk_paths=[],
            validation_commands=[],
            machine_recommendation="approve_candidate",
            machine_recommendation_reason="test",
            reviewer_outputs=[],
            execution_failures=[],
        ).to_dict()

        assert packet["protocol_version"] == "pr_review_protocol.v1"
        assert packet["availability_summary"]["total_slots"] == 5
        """
    )

    assert proc.returncode == 0, proc.stderr


def test_review_queue_merge_packet_help_uses_lightweight_cli_path() -> None:
    proc = _run_isolated_python(
        """
        import importlib.abc
        import sys

        BLOCKED = (
            "aragora.cli.parser",
            "aragora.config.secrets",
            "aragora.modes",
        )

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(BLOCKED):
                    raise RuntimeError(f"blocked full CLI startup import: {fullname}")
                return None

        sys.meta_path.insert(0, Blocker())

        import aragora.cli.main as main

        sys.argv = ["aragora", "review-queue", "merge-packet", "--help"]
        raise SystemExit(main.main())
        """
    )

    assert proc.returncode == 0, proc.stderr
    assert "merge-packet" in proc.stdout


def test_review_queue_record_settlement_help_uses_lightweight_cli_path() -> None:
    proc = _run_isolated_python(
        """
        import importlib.abc
        import sys

        BLOCKED = (
            "aragora.cli.parser",
            "aragora.config.secrets",
            "aragora.modes",
            "aragora.review.reviewer_output",
            "aragora.triage.auto_handle_calibration",
        )

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(BLOCKED):
                    raise RuntimeError(f"blocked eager startup import: {fullname}")
                return None

        sys.meta_path.insert(0, Blocker())

        import aragora.cli.main as main

        sys.argv = ["aragora", "review-queue", "record-settlement", "--help"]
        raise SystemExit(main.main())
        """
    )

    assert proc.returncode == 0, proc.stderr
    assert "record-settlement" in proc.stdout
