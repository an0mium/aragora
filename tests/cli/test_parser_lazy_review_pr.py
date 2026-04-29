"""Tests for keeping review-pr parser wiring lazy in the main CLI parser."""

from __future__ import annotations

import builtins
import sys

from aragora.cli.parser import build_parser


def test_build_parser_keeps_review_pr_runtime_lazy(monkeypatch):
    sys.modules.pop("aragora.cli.commands.review_pr", None)
    imported: list[str] = []
    real_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "aragora.cli.commands.review_pr":
            imported.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    parser = build_parser()
    args = parser.parse_args(
        [
            "review-pr",
            "123",
            "--fixer",
            "codex",
            "--auto-rerun",
            "--json",
        ]
    )

    assert imported == []
    assert args.pr == "123"
    assert args.fixer == "codex"
    assert args.auto_rerun is True
    assert args.json_output is True
    assert args.func.__name__ == "cmd_review_pr"


def test_build_parser_keeps_triage_status_free_of_heavy_review_imports(monkeypatch):
    for module_name in (
        "aragora.cli.commands.review_pr",
        "aragora.worktree",
        "aragora.worktree.fleet",
    ):
        sys.modules.pop(module_name, None)

    imported: list[str] = []
    real_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {
            "aragora.cli.commands.review_pr",
            "aragora.worktree",
            "aragora.worktree.fleet",
        }:
            imported.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    parser = build_parser()
    args = parser.parse_args(["triage", "status"])

    assert imported == []
    assert args.command == "triage"
    assert args.triage_command == "status"


def test_build_parser_keeps_review_queue_runtime_lazy(monkeypatch):
    sys.modules.pop("aragora.cli.commands.review_queue", None)
    imported: list[str] = []
    real_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "aragora.cli.commands.review_queue":
            imported.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    parser = build_parser()
    args = parser.parse_args(
        [
            "review-queue",
            "act",
            "6280",
            "--request-changes",
            "--reason",
            "needs a test",
            "--json",
        ]
    )

    assert imported == []
    assert args.command == "review-queue"
    assert args.review_queue_command == "act"
    assert args.pr == "6280"
    assert args.request_changes is True
    assert args.reason == "needs a test"
    assert args.json_output is True
    assert args.func.__name__ == "cmd_review_queue"


def test_build_parser_registers_review_queue_baseline_lazily(monkeypatch):
    sys.modules.pop("aragora.cli.commands.review_queue", None)
    imported: list[str] = []
    real_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "aragora.cli.commands.review_queue":
            imported.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    parser = build_parser()
    args = parser.parse_args(
        [
            "review-queue",
            "baseline",
            "--window-days",
            "14",
            "--min-samples",
            "75",
            "--safety-margin",
            "0.4",
            "--minimum-meaningful-rate",
            "0.02",
            "--placeholder-value",
            "0.06",
            "--calibration-db",
            "/tmp/calibration.sqlite3",
            "--review-queue-root",
            "/tmp/review-queue",
            "--json",
        ]
    )

    assert imported == []
    assert args.command == "review-queue"
    assert args.review_queue_command == "baseline"
    assert args.window_days == 14
    assert args.min_samples == 75
    assert args.safety_margin == 0.4
    assert args.minimum_meaningful_rate == 0.02
    assert args.placeholder_value == 0.06
    assert args.calibration_db == "/tmp/calibration.sqlite3"
    assert args.review_queue_root == "/tmp/review-queue"
    assert args.json is True
    assert args.func.__name__ == "cmd_review_queue"


def test_build_parser_registers_review_queue_merge_packet_lazily(monkeypatch):
    sys.modules.pop("aragora.cli.commands.review_queue", None)
    imported: list[str] = []
    real_import = builtins.__import__

    def tracking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "aragora.cli.commands.review_queue":
            imported.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracking_import)

    parser = build_parser()
    args = parser.parse_args(
        [
            "review-queue",
            "merge-packet",
            "--pr",
            "6779",
            "--execute-reviewers",
            "--json",
        ]
    )

    assert imported == []
    assert args.command == "review-queue"
    assert args.review_queue_command == "merge-packet"
    assert args.pr == ["6779"]
    assert args.execute_reviewers is True
    assert args.json_output is True
    assert args.func.__name__ == "cmd_review_queue"
