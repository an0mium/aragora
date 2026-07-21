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
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_module()


def test_repository_manifest_matches_current_tree() -> None:
    result = checker.check_allowlist()
    payload = json.loads(checker.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert result.ok is True and result.policy_consumers == ("scripts/consult_claude.py",) and [(site["path"], site["anchor"]) for site in payload["sites"] if site["classification"] == "proxy-eligible"] == [("scripts/consult_claude.py", "_run_vibeproxy")], result  # fmt: skip


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


def _empty_manifest(root: Path) -> Path:
    return _write_manifest(root, {"schema_version": 1, "transport_policy_consumers": [], "sites": []})  # fmt: skip


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
    assert first.sites == second.sites and first.raw_detections == 2
    assert {site.anchor for site in first.sites} == {"Runner.generate"}


@pytest.mark.parametrize("module,name", [("openai", "OpenAI"), ("anthropic", "Anthropic")])
def test_discovery_finds_aliased_constructors(tmp_path: Path, module: str, name: str) -> None:
    _write_source(tmp_path, "aragora/run.py", f"from {module} import {name} as Client\nClient()\n")
    assert checker.discover(tmp_path).sites[0].provider in {"openai-compatible", "anthropic"}


def test_method_detection_uses_sdk_provenance(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        "aragora/run.py",
        """import asyncio, openai, openai as oai, google.generativeai as genai; from google import genai as modern_genai; from google.genai import Client as GeminiClient; from anthropic import Anthropic
async def run(api: openai.OpenAI):
    await asyncio.to_thread(api.responses.create); api.responses.stream()
client_store.responses.create(); openai.chat.completions.create(); oai.responses.create(); modern = modern_genai.Client(); modern.models.generate_content("hi"); modern.models.generate_content_stream("hi"); direct = GeminiClient(); direct.models.generate_content("hi"); Other.Client().models.generate_content("ignore"); anthropic = Anthropic(); anthropic.messages.stream(model="claude", messages=[])
openai.OpenAI().responses.create()
genai.GenerativeModel().generate_content("hello")
""",
    )
    discovery = checker.discover(tmp_path)
    protocols = {site.protocol for site in discovery.sites}
    expected = {"client", "chat", "responses", "messages", "generate-content"}
    assert discovery.raw_detections == 15 and protocols == expected


def test_discovery_finds_urls_methods_and_transport_policy_calls(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        "scripts/consult_claude.py",
        """from somewhere import ModelTransportPolicy as MTP
ANTHROPIC = "https://api.anthropic.com/v1/messages"
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
def consult(policy: MTP):
    other.anthropic_message(model="fake"); assigned = MTP.from_env(); assigned.client.anthropic_message(model="claude"); return policy.generate_anthropic(model="claude", messages=[])
""",
    )
    # fmt: off
    _write_source(tmp_path, "aragora/live/src/run.tsx", 'urls = ["https://api.openai.com/v1/responses",\n "https://api.mistral.ai/v1",\n "https://api.deepseek.com/v1",\n "https://api.moonshot.cn/v1",\n "https://api.thinkingmachines.ai/v1/*quoted*/"]\nthis.#endpoint = "https://api.x.ai/v1"\nconst ai = new OpenAI()\nai.chat\n .completions.create({})\nnew OpenAI({ apiKey }).responses.create({})\nunproven.responses.create({})\n// https://api.x.ai/v1/commented\n/*\nhttps://generativelanguage.googleapis.com/v1\n*/\n')
    discovery = checker.discover(tmp_path)
    assert discovery.policy_consumers == ("scripts/consult_claude.py",)
    assert {(site.provider, site.protocol) for site in discovery.sites} == {("anthropic", "messages"), ("deepseek", "base"), ("kimi", "base"), ("mistral", "base"), ("openai", "responses"), ("openai-compatible", "chat"), ("openai-compatible", "responses"), ("openrouter", "chat"), ("tinker", "base"), ("xai", "base")}
    assert next(site for site in discovery.sites if site.provider == "openai-compatible" and site.protocol == "responses").detectors == {"inference-method": 1}
    # fmt: on
    assert sum(site.detectors.get("transport-policy-call", 0) for site in discovery.sites) == 2
    payload = _template(tmp_path)
    assert {site["classification"] for site in payload["sites"]} == {"direct-only"}
    payload["sites"][0]["classification"] = "proxy-eligible"
    payload["transport_policy_consumers"] = []
    result = checker.check_allowlist(tmp_path, _write_manifest(tmp_path, payload))
    assert result.policy_errors and len(result.manifest_errors) >= 2


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
    assert len(changed.changed) == 1
    _write_source(
        tmp_path, "aragora/new.py", "from anthropic import Anthropic\nclient = Anthropic()\n"
    )
    missing = checker.check_allowlist(tmp_path, manifest)
    assert missing.unclassified
    path.unlink()
    stale = checker.check_allowlist(tmp_path, manifest)
    assert stale.stale


# fmt: off
@pytest.mark.parametrize("relative", [".github/scripts/check.py", "scripts/ci/check.py", "aragora/server/handler.py", "aragora/live/src/lib/provider-keys.ts", "aragora/gateway/proxy.py", "aragora/compat/openclaw/skills/pr-reviewer/policy.yaml"])
# fmt: on
def test_protected_paths_cannot_be_proxy_eligible(tmp_path: Path, relative: str) -> None:
    _write_source(tmp_path, relative, 'URL = "https://api.anthropic.com/v1/messages"\n')
    payload = _template(tmp_path)
    payload["sites"][0]["classification"] = "proxy-eligible"
    result = checker.check_allowlist(tmp_path, _write_manifest(tmp_path, payload))
    assert any("must be direct-only" in error for error in result.manifest_errors)


@pytest.mark.parametrize("source", ['URL: "http://localhost:08317"\n', 'PORT: "8317"\n'])
def test_forbidden_port_fails_even_without_inference_site(tmp_path: Path, source: str) -> None:
    _write_source(tmp_path, ".github/workflows/config.yml", source)
    result = checker.check_allowlist(tmp_path, _empty_manifest(tmp_path))
    assert result.forbidden_ports == (".github/workflows/config.yml:1",)


def test_unparseable_scanned_source_fails_closed(tmp_path: Path) -> None:
    _write_source(tmp_path, "aragora/broken.py", "OpenAI(\n")
    result = checker.check_allowlist(tmp_path, _empty_manifest(tmp_path))
    assert result.scan_errors and "aragora/broken.py: SyntaxError" in result.scan_errors[0]


def test_central_port_prohibition_is_allowed_but_other_uses_fail(tmp_path: Path) -> None:
    _write_source(
        tmp_path,
        "aragora/agents/transports/vibeproxy.py",
        'PROHIBITED_PORTS = {8317}\nOTHER_PORT = 8317\nPORTS = ["8317"]\n',
    )
    result = checker.check_allowlist(tmp_path, _empty_manifest(tmp_path))
    assert result.forbidden_ports == ("aragora/agents/transports/vibeproxy.py:2", "aragora/agents/transports/vibeproxy.py:3")  # fmt: skip


def test_malformed_and_duplicate_manifest_entries_fail(tmp_path: Path) -> None:
    _write_source(tmp_path, "aragora/run.py", "from openai import OpenAI\nclient = OpenAI()\n")
    payload = _template(tmp_path)
    payload["sites"].append(dict(payload["sites"][0]))
    payload["sites"][0]["classification"] = "sometimes"
    payload["sites"][1]["rationale"] = ""
    payload["port"] = "8317"
    result = checker.check_allowlist(tmp_path, _write_manifest(tmp_path, payload))
    errors = "\n".join(result.manifest_errors)
    assert all(message in errors for message in ("invalid classification", "duplicate site", "forbidden port"))
