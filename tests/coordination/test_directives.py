"""Tests for the durable coordination directive board."""

from __future__ import annotations

from aragora.coordination.directives import DirectiveBoard, SessionDirective


class TestSessionDirective:
    def test_roundtrip(self):
        directive = SessionDirective(
            target="codex-a",
            task="SDK parity consolidation",
            scope=["#2684"],
            constraints=["no queue drain"],
            assigned_by="boss-codex",
            status="active",
            created_at=1000.0,
            updated_at=1001.0,
        )
        restored = SessionDirective.from_dict(directive.to_dict())
        assert restored.target == "codex-a"
        assert restored.task == "SDK parity consolidation"
        assert restored.scope == ["#2684"]


class TestDirectiveBoard:
    def test_assign_and_get(self, tmp_path):
        board = DirectiveBoard(repo_path=tmp_path)

        directive = board.assign(
            "codex-a",
            "SDK parity consolidation",
            scope=["#2684"],
            constraints=["no queue drain"],
            assigned_by="boss-codex",
        )

        assert directive.target == "codex-a"
        stored = board.get("codex-a")
        assert stored is not None
        assert stored.task == "SDK parity consolidation"
        assert stored.scope == ["#2684"]
        assert stored.constraints == ["no queue drain"]

    def test_assign_overwrites_and_preserves_created_at(self, tmp_path):
        board = DirectiveBoard(repo_path=tmp_path)
        first = board.assign("codex-a", "old task", assigned_by="boss-codex")
        second = board.assign("codex-a", "new task", assigned_by="boss-codex")

        assert second.created_at == first.created_at
        assert second.updated_at >= first.updated_at
        assert board.get("codex-a") is not None
        assert board.get("codex-a").task == "new task"  # type: ignore[union-attr]

    def test_list_returns_sorted_directives(self, tmp_path):
        board = DirectiveBoard(repo_path=tmp_path)
        board.assign("codex-b", "task b")
        board.assign("codex-a", "task a")

        directives = board.list()
        assert [item.target for item in directives] == ["codex-a", "codex-b"]

    def test_clear_removes_directive(self, tmp_path):
        board = DirectiveBoard(repo_path=tmp_path)
        board.assign("codex-a", "task")

        assert board.clear("codex-a") is True
        assert board.get("codex-a") is None

    def test_clear_missing_returns_false(self, tmp_path):
        board = DirectiveBoard(repo_path=tmp_path)
        assert board.clear("missing") is False

    def test_corrupt_payload_falls_back_to_empty(self, tmp_path):
        path = tmp_path / ".aragora_coordination" / "directives.json"
        path.parent.mkdir(parents=True)
        path.write_text("{broken json")

        board = DirectiveBoard(repo_path=tmp_path)
        assert board.list() == []
        directive = board.assign("codex-a", "task")
        assert directive.target == "codex-a"
