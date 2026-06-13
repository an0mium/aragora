"""Unit tests for scripts/ci/check_root_allowlist.py.

Exercises the pure parse/offender logic and the ``main`` CLI wiring against a
fake checkout (``REPO_ROOT`` and ``list_tracked_root_files`` are monkeypatched)
so the tests stay fast and never depend on the live repo root. The end-to-end
behavior on real origin/main (green on a clean tree, a tamper file trips it,
restore greens again) is covered by the VAL-P1-005 acceptance check.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER_PATH = REPO_ROOT / "scripts" / "ci" / "check_root_allowlist.py"

_spec = importlib.util.spec_from_file_location("check_root_allowlist", _CHECKER_PATH)
cra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cra)


SAMPLE_DOC = """# Root allowlist

Some intro prose mentioning docs/reference paths that must be ignored.

<!-- ROOT_ALLOWLIST_BEGIN -->
```text
README.md
aragora_logo.png
favicon.png
.gitignore
# a comment inside the block is ignored
docs/should_be_ignored.md
```
<!-- ROOT_ALLOWLIST_END -->

zz_outside_block.txt
"""


# --- pure logic -------------------------------------------------------------


def test_parse_allowlist_extracts_block_entries():
    assert cra.parse_allowlist(SAMPLE_DOC) == {
        "README.md",
        "aragora_logo.png",
        "favicon.png",
        ".gitignore",
    }


def test_parse_allowlist_ignores_tokens_outside_block_and_paths():
    parsed = cra.parse_allowlist(SAMPLE_DOC)
    assert "zz_outside_block.txt" not in parsed  # outside the markers
    assert "docs/should_be_ignored.md" not in parsed  # contains '/'
    assert "text" not in parsed  # the ```text fence info string


def test_find_offenders_flags_unlisted():
    tracked = ["README.md", "aragora_logo.png", "zz_val_tamper.xyz"]
    allowlist = {"README.md", "aragora_logo.png"}
    assert cra.find_offenders(tracked, allowlist) == ["zz_val_tamper.xyz"]


def test_find_offenders_clean_returns_empty():
    tracked = ["README.md", "favicon.png"]
    allowlist = {"README.md", "favicon.png", "extra"}
    assert cra.find_offenders(tracked, allowlist) == []


# --- main CLI wiring --------------------------------------------------------


def _write_doc(root: Path, body: str = SAMPLE_DOC) -> None:
    doc = root / "docs" / "reference" / "ROOT_ALLOWLIST.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(body, encoding="utf-8")


def test_main_green_on_clean_tree(tmp_path, monkeypatch):
    _write_doc(tmp_path)
    monkeypatch.setattr(cra, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        cra,
        "list_tracked_root_files",
        lambda: ["README.md", "aragora_logo.png", "favicon.png", ".gitignore"],
    )
    assert cra.main() == 0


def test_main_flags_tamper_and_names_offender(tmp_path, monkeypatch, capsys):
    _write_doc(tmp_path)
    monkeypatch.setattr(cra, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        cra,
        "list_tracked_root_files",
        lambda: ["README.md", "zz_val_tamper.xyz"],
    )
    assert cra.main() == 1
    assert "zz_val_tamper.xyz" in capsys.readouterr().out


def test_main_missing_doc_returns_2(tmp_path, monkeypatch):
    monkeypatch.setattr(cra, "REPO_ROOT", tmp_path)  # no doc written
    assert cra.main() == 2


def test_main_empty_allowlist_returns_2(tmp_path, monkeypatch):
    _write_doc(
        tmp_path,
        "# empty\n<!-- ROOT_ALLOWLIST_BEGIN -->\n<!-- ROOT_ALLOWLIST_END -->\n",
    )
    monkeypatch.setattr(cra, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cra, "list_tracked_root_files", lambda: ["README.md"])
    assert cra.main() == 2
