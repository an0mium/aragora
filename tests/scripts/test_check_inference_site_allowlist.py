"""Tests for ``scripts/check_inference_site_allowlist.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "check_inference_site_allowlist.py"
    spec = importlib.util.spec_from_file_location("inference_site_allowlist_under_test", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_module()


def test_repository_manifest_matches_current_tree() -> None:
    result = checker.check_allowlist()
    payload = json.loads(checker.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    eligible = [site for site in payload["sites"] if site["classification"] == "proxy-eligible"]

    assert result.ok is True, result.to_dict()
    assert result.policy_consumers == ("scripts/consult_claude.py",)
    assert [(site["path"], site["anchor"]) for site in eligible] == [
        ("scripts/consult_claude.py", "_run_vibeproxy")
    ]


def _write_source(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _write_manifest(root: Path, payload: dict[str, Any]) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _template(root: Path) -> dict[str, Any]:
    return checker.template_manifest(checker.discover(root))


def test_discovery_groups_by_stable_symbol_anchor(tmp_path: Path) -> None:
    source = """from openai import AsyncOpenAI

class Runner:
    async def generate(self):
        client = AsyncOpenAI()
        return await client.chat.completions.create(model="gpt", messages=[])
"""
    path = _write_source(tmp_path, "aragora/runner.py", source)
    first = checker.discover(tmp_path)
    path.write_text("\n\n\n" + source, encoding="utf-8")
    second = checker.discover(tmp_path)

    assert first.sites == second.sites
    assert {site.anchor for site in first.sites} == {"Runner.generate"}
    assert first.raw_detections == 2


def test_discovery_finds_urls_methods_and_transport_policy_calls(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        "scripts/consult_claude.py",
        """from somewhere import ModelTransportPolicy
ANTHROPIC = "https://api.anthropic.com/v1/messages"
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
def consult(policy: ModelTransportPolicy):
    return policy.generate_anthropic(model="claude", messages=[])
""",
    )

    discovery = checker.discover(tmp_path)

    assert discovery.policy_consumers == ("scripts/consult_claude.py",)
    assert {(site.provider, site.protocol) for site in discovery.sites} == {
        ("anthropic", "messages"),
        ("openrouter", "chat"),
    }
    assert any("transport-policy-call" in site.detectors for site in discovery.sites)


def test_exact_manifest_passes_and_template_defaults_to_direct_only(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        "scripts/consult_claude.py",
        """from somewhere import ModelTransportPolicy
def consult(policy: ModelTransportPolicy):
    return policy.generate_anthropic(model="claude", messages=[])
""",
    )
    manifest = _write_manifest(tmp_path, _template(tmp_path))

    result = checker.check_allowlist(tmp_path, manifest)

    assert result.ok is True
    entries = json.loads(manifest.read_text())["sites"]
    assert [entry["classification"] for entry in entries] == ["direct-only"]


def test_unclassified_stale_and_count_changes_fail(tmp_path: Path) -> None:
    source = """from openai import OpenAI
def run():
    return OpenAI()
"""
    path = _write_source(tmp_path, "aragora/run.py", source)
    payload = _template(tmp_path)
    manifest = _write_manifest(tmp_path, payload)

    path.write_text(
        source.replace("return OpenAI()", "OpenAI()\n    return OpenAI()"), encoding="utf-8"
    )
    changed = checker.check_allowlist(tmp_path, manifest)
    assert changed.ok is False
    assert len(changed.changed) == 1

    _write_source(
        tmp_path, "aragora/new.py", "from anthropic import Anthropic\nclient = Anthropic()\n"
    )
    missing = checker.check_allowlist(tmp_path, manifest)
    assert missing.unclassified

    path.unlink()
    stale = checker.check_allowlist(tmp_path, manifest)
    assert stale.stale


def test_direct_only_requires_rationale(tmp_path: Path) -> None:
    _write_source(tmp_path, "aragora/run.py", "from openai import OpenAI\nclient = OpenAI()\n")
    payload = _template(tmp_path)
    payload["sites"][0]["rationale"] = ""
    manifest = _write_manifest(tmp_path, payload)

    result = checker.check_allowlist(tmp_path, manifest)

    assert result.ok is False
    assert any("needs a rationale" in error for error in result.manifest_errors)


@pytest.mark.parametrize(
    "relative",
    [
        ".github/scripts/check.py",
        "scripts/ci/check.py",
        "aragora/server/handler.py",
        "scripts/rotate_keys.py",
        "aragora/gateway/proxy.py",
        "aragora/verification/formal.py",
    ],
)
def test_protected_paths_cannot_be_proxy_eligible(tmp_path: Path, relative: str) -> None:
    _write_source(
        tmp_path,
        relative,
        "from anthropic import Anthropic\nclient = Anthropic()\n",
    )
    payload = _template(tmp_path)
    payload["sites"][0]["classification"] = "proxy-eligible"
    manifest = _write_manifest(tmp_path, payload)

    result = checker.check_allowlist(tmp_path, manifest)

    assert result.ok is False
    assert any("must be direct-only" in error for error in result.manifest_errors)


def test_forbidden_port_fails_even_without_inference_site(tmp_path: Path) -> None:
    _write_source(tmp_path, "aragora/config.py", "# never use localhost:8317\nVALUE = 1\n")
    manifest = _write_manifest(
        tmp_path,
        {"schema_version": 1, "transport_policy_consumers": [], "sites": []},
    )

    result = checker.check_allowlist(tmp_path, manifest)

    assert result.ok is False
    assert result.forbidden_ports == ("aragora/config.py:1",)


def test_forbidden_port_in_manifest_fails(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path,
        {
            "schema_version": 1,
            "transport_policy_consumers": [],
            "sites": [],
            "endpoint": "http://localhost:8317",
        },
    )

    result = checker.check_allowlist(tmp_path, manifest)

    assert result.ok is False
    assert any("forbidden port" in error for error in result.manifest_errors)


def test_central_port_prohibition_is_allowed_but_other_uses_fail(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        "aragora/agents/transports/vibeproxy.py",
        "PROHIBITED_PORTS = {8317}\nOTHER_PORT = 8317\n",
    )
    manifest = _write_manifest(
        tmp_path,
        {"schema_version": 1, "transport_policy_consumers": [], "sites": []},
    )

    result = checker.check_allowlist(tmp_path, manifest)

    assert result.ok is False
    assert result.forbidden_ports == ("aragora/agents/transports/vibeproxy.py:2",)


def test_transport_policy_consumer_inventory_is_exact(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        "scripts/consult_claude.py",
        "from somewhere import ModelTransportPolicy as Policy\npolicy = Policy.from_env()\n",
    )
    payload = _template(tmp_path)
    payload["transport_policy_consumers"] = []
    manifest = _write_manifest(tmp_path, payload)

    result = checker.check_allowlist(tmp_path, manifest)

    assert result.ok is False
    assert result.policy_errors


def test_malformed_and_duplicate_manifest_entries_fail(tmp_path: Path) -> None:
    _write_source(tmp_path, "aragora/run.py", "from openai import OpenAI\nclient = OpenAI()\n")
    payload = _template(tmp_path)
    payload["sites"].append(dict(payload["sites"][0]))
    payload["sites"][0]["classification"] = "sometimes"
    manifest = _write_manifest(tmp_path, payload)

    result = checker.check_allowlist(tmp_path, manifest)

    assert result.ok is False
    assert any("invalid classification" in error for error in result.manifest_errors)
    assert any("duplicate site" in error for error in result.manifest_errors)


def test_json_cli_reports_machine_readable_failure(tmp_path: Path, capsys: Any) -> None:
    _write_source(tmp_path, "aragora/run.py", "from openai import OpenAI\nclient = OpenAI()\n")
    manifest = _write_manifest(
        tmp_path,
        {"schema_version": 1, "transport_policy_consumers": [], "sites": []},
    )

    exit_code = checker.main(["--root", str(tmp_path), "--manifest", str(manifest), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["unclassified"]
