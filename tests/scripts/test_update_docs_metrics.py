from __future__ import annotations

import sys
from pathlib import Path

_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import update_docs_metrics as mod  # noqa: E402


def test_replace_adapter_count_updates_readme_placeholder() -> None:
    content = "Knowledge Mound (<!-- adpt-count -->0<!-- /adpt-count --> adapters)"

    updated = mod._replace_adapter_count(content, 42)

    assert updated == "Knowledge Mound (<!-- adpt-count -->42<!-- /adpt-count --> adapters)"


def test_update_adapter_count_uses_registry_count_not_directory_noise(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("KM: <!-- adpt-count -->999<!-- /adpt-count --> adapters\n", encoding="utf-8")

    noisy_adapters = tmp_path / "aragora" / "knowledge" / "mound" / "adapters"
    noisy_adapters.mkdir(parents=True)
    (noisy_adapters / "CLAUDE.md").write_text("not an adapter registry entry\n", encoding="utf-8")
    (noisy_adapters / "_base.py").write_text("not counted directly\n", encoding="utf-8")
    (noisy_adapters / "performance").mkdir()

    monkeypatch.setattr(mod, "get_adapter_count", lambda: 42)

    mod.update_adapter_count(project_root=tmp_path)

    assert readme.read_text(encoding="utf-8") == (
        "KM: <!-- adpt-count -->42<!-- /adpt-count --> adapters\n"
    )
    captured = capsys.readouterr()
    assert "Successfully updated README.md with adapter count: 42" in captured.out
