"""Regression coverage for repository-wide pre-commit policy."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _hook(hook_id: str) -> dict[str, object]:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    for repository in config["repos"]:
        for hook in repository["hooks"]:
            if hook["id"] == hook_id:
                return hook
    raise AssertionError(f"missing pre-commit hook: {hook_id}")


@pytest.mark.parametrize("hook_id", ["trailing-whitespace", "end-of-file-fixer"])
def test_text_normalizers_exclude_nested_markdown(hook_id: str) -> None:
    pattern = re.compile(str(_hook(hook_id)["exclude"]))
    assert pattern.search("tests/server/handlers/CLAUDE.md")
    assert pattern.search("aragora/server/handlers/oracle_essay.md")
    assert not pattern.search("aragora/live/e2e/export.spec.ts")


@pytest.mark.parametrize("path", [".grok/settings.json", "replays/demo.json", "uv.lock"])
def test_eof_normalizer_excludes_structured_artifacts(path: str) -> None:
    pattern = re.compile(str(_hook("end-of-file-fixer")["exclude"]))
    assert pattern.search(path)


def test_yaml_hook_excludes_only_helm_template_trees() -> None:
    pattern = re.compile(str(_hook("check-yaml")["exclude"]))
    assert pattern.search("deploy/kubernetes/helm/aragora/templates/deployment-backend.yaml")
    assert pattern.search("deploy/helm/aragora/templates/deployment.yaml")
    assert not pattern.search("deploy/kubernetes/ordinary-config.yaml")


def test_documentation_does_not_embed_pem_markers() -> None:
    guide = (REPO_ROOT / "docs/guides/GITHUB_APP_SETUP.md").read_text(encoding="utf-8")
    assert "BEGIN RSA PRIVATE KEY" not in guide
    assert "GITHUB_APP_PRIVATE_KEY_PATH=/run/secrets/github-app.pem" in guide
    assert "$(cat " not in guide


def test_ruff_exceptions_are_narrow_and_explicit() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ignores = config["tool"]["ruff"]["lint"]["per-file-ignores"]
    assert ignores["benchmarks/*.py"] == ["S101"]
    assert ignores["deploy/liftmode/briefing.py"] == ["T201", "BLE001"]
