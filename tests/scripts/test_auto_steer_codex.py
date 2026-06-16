from __future__ import annotations

from types import SimpleNamespace

import scripts.auto_steer_codex as mod


def test_gather_preserves_recent_ledger_order_before_truncating(monkeypatch) -> None:
    ledger_rows = [
        SimpleNamespace(pr=9002),
        SimpleNamespace(pr=9002),
        SimpleNamespace(pr=9001),
        SimpleNamespace(pr=8000),
    ]
    checked_prs: list[int] = []

    monkeypatch.setattr(mod, "default_codex_home", lambda: None)
    monkeypatch.setattr(mod, "read_ledgers", lambda *_args, **_kwargs: ledger_rows)

    def fake_gh_json(args: list[str], *, timeout: float = 30.0) -> object | None:
        if args[:2] == ["pr", "list"]:
            return []
        if args[:2] == ["pr", "view"]:
            checked_prs.append(int(args[2]))
            return {"state": "MERGED"}
        raise AssertionError(f"unexpected gh args: {args}")

    monkeypatch.setattr(mod, "_gh_json", fake_gh_json)

    signals, warnings = mod._gather("synaptent/aragora", since_hours=24.0, max_ledger_check=2)

    assert warnings == []
    assert checked_prs == [9002, 9001]
    assert signals.stale_ledger_prs == (9002, 9001)
