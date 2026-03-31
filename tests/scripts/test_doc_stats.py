from __future__ import annotations

from scripts import doc_stats


def test_patch_docs_uses_coarse_repo_test_rounding(monkeypatch, tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "**Scale:** 3,000+ Python modules | 212,000+ tests across 5,000+ test files\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(doc_stats, "ROOT", tmp_path)
    changed = doc_stats.patch_docs(
        doc_stats.Stats(
            python_modules=3869,
            test_count=212_650,
            test_files=5219,
            api_paths=2603,
            api_operations=3084,
            ws_event_types=272,
            km_adapters_registered=0,
            workflow_templates=62,
            ts_namespaces=186,
            agent_types_allowlisted=34,
        ),
        write=True,
    )

    assert changed == 1
    assert "210,000+ tests" in readme.read_text(encoding="utf-8")
