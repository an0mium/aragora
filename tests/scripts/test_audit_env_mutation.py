from __future__ import annotations

from pathlib import Path

import scripts.audit_env_mutation as audit


def test_default_targets_include_epistemic_surface() -> None:
    assert audit.REPO_ROOT / "aragora" / "epistemic" in audit.DEFAULT_TARGETS


def test_context_budget_handler_is_no_longer_allowlisted() -> None:
    path = audit.REPO_ROOT / "aragora" / "server" / "handlers" / "context_budget.py"
    rel = audit.relative(path)

    assert rel not in audit._ALLOWLIST  # noqa: SLF001
    assert audit.scan_file(path) == []


def test_detects_mutating_environ_methods(tmp_path: Path) -> None:
    path = tmp_path / "module.py"
    path.write_text(
        "import os\nos.environ['FLAG'] = '1'\nos.environ.setdefault('OTHER_FLAG', '1')\n"
    )

    assert audit.scan_file(path) == [
        (2, "os.environ[...] = ..."),
        (3, "os.environ.setdefault(...)"),
    ]


def test_default_targets_are_clean_after_allowlist() -> None:
    findings: list[tuple[str, int, str]] = []

    for path in audit.iter_targets(audit.DEFAULT_TARGETS):
        rel = audit.relative(path)
        if rel in audit._ALLOWLIST:  # noqa: SLF001
            continue
        for line, pattern in audit.scan_file(path):
            findings.append((rel, line, pattern))

    assert findings == []
