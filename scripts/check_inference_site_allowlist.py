#!/usr/bin/env python3
"""Check the stable, reviewed inventory of production inference call sites."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path(__file__).with_name("inference_site_allowlist.json")
SCAN_ROOTS = ("aragora", "scripts", ".github")
SELF_PATH = "scripts/check_inference_site_allowlist.py"
CLASSIFICATIONS = frozenset({"proxy-eligible", "direct-only"})
FORBIDDEN_PORT_RE = re.compile(r":0*8317\b")
PORT_ENFORCEMENT_PATH = "aragora/agents/transports/vibeproxy.py"
PORT_ENFORCEMENT_LINE = "PROHIBITED_PORTS = {8317}"
# fmt: off
PROVIDER_HOSTS = {"api.openai.com": "openai", "api.anthropic.com": "anthropic", "openrouter.ai": "openrouter", "api.x.ai": "xai", "generativelanguage.googleapis.com": "gemini", "api.moonshot.ai": "kimi"}
PROTOCOL_PATHS = (("/audio/transcriptions", "audio"), ("/chat/completions", "chat"), ("/responses", "responses"), ("/messages", "messages"), ("/embeddings", "embeddings"), ("/completions", "completions"))
CONSTRUCTORS = {"OpenAI": ("openai-compatible", "client"), "AsyncOpenAI": ("openai-compatible", "client"), "Anthropic": ("anthropic", "client"), "AsyncAnthropic": ("anthropic", "client"), "GenerativeModel": ("gemini", "client")}
METHOD_SUFFIXES = (("audio.transcriptions.create", "openai-compatible", "audio"), ("chat.completions.create", "openai-compatible", "chat"), ("responses.create", "openai-compatible", "responses"), ("messages.create", "anthropic", "messages"), ("embeddings.create", "openai-compatible", "embeddings"), ("completions.create", "openai-compatible", "completions"), ("models.generate_content", "gemini", "generate-content"), ("models.generate_content_async", "gemini", "generate-content"), ("generate_content", "gemini", "generate-content"), ("generate_content_async", "gemini", "generate-content"))
PROTECTED_PATHS = {
    "ci": (".github/**", "scripts/ci/**"),
    "production-server": ("aragora/server/**",),
    "credential-validation": ("**/*key*.py", "**/*secret*.py", "scripts/rotate_keys.py", "scripts/migrate_secrets_to_aws.py", "aragora/cli/setup.py"),
    "public-gateway": ("aragora/gateway/**", "aragora/security/api_key_proxy.py"),
    "evidence-or-settlement": ("**/*evidence*.py", "**/*quorum*.py", "**/*review*.py", "**/*settle*.py", "aragora/verification/**"),
    "production-preflight": ("**/*preflight*.py", "**/*live_fire*.py"),
}
# fmt: on


@dataclass(frozen=True, order=True)
class SiteKey:
    path: str
    anchor: str
    provider: str
    protocol: str


@dataclass(frozen=True)
class Site:
    path: str
    anchor: str
    provider: str
    protocol: str
    detectors: dict[str, int]

    @property
    def key(self) -> SiteKey:
        return SiteKey(self.path, self.anchor, self.provider, self.protocol)


@dataclass(frozen=True)
class Discovery:
    sites: tuple[Site, ...]
    scanned_files: int
    raw_detections: int
    policy_consumers: tuple[str, ...]
    forbidden_ports: tuple[str, ...]
    scan_errors: tuple[str, ...]


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    scanned_files: int
    site_count: int
    raw_detections: int
    policy_consumers: tuple[str, ...]
    unclassified: tuple[str, ...]
    stale: tuple[str, ...]
    changed: tuple[str, ...]
    manifest_errors: tuple[str, ...]
    policy_errors: tuple[str, ...]
    forbidden_ports: tuple[str, ...]
    scan_errors: tuple[str, ...]


def _attr_chain(node: ast.AST, *, unwrap_calls: bool = False) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute) or (unwrap_calls and isinstance(node, ast.Call)):
        if isinstance(node, ast.Call):
            node = node.func
            continue
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _protocol_for_url(value: str) -> str:
    return next((name for fragment, name in PROTOCOL_PATHS if fragment in value.lower()), "base")


def _provider_for_url(value: str) -> str | None:
    return next((name for host, name in PROVIDER_HOSTS.items() if host in value.lower()), None)


def _contains_forbidden_port(value: Any, key: str = "") -> bool:
    if type(value) is int:
        return value == 8317
    if isinstance(value, str):
        return bool(FORBIDDEN_PORT_RE.search(value)) or (
            "port" in key.lower() and bool(re.fullmatch(r"0*8317", value))
        )
    if isinstance(value, list):
        return any(_contains_forbidden_port(item, key) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_forbidden_port(item, str(name)) or _contains_forbidden_port(name)
            for name, item in value.items()
        )
    return False


def _targets(node: ast.AST) -> tuple[str, ...]:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return ()
    raw = node.targets if isinstance(node, ast.Assign) else [node.target]
    return tuple(chain for target in raw if (chain := _attr_chain(target)))


def _constructor_aliases(tree: ast.AST) -> dict[str, tuple[str, str]]:
    aliases = dict(CONSTRUCTORS)
    nodes = tuple(ast.walk(tree))
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.module in {"openai", "anthropic"}:
            for name in node.names:
                if name.name in CONSTRUCTORS:
                    aliases[name.asname or name.name] = CONSTRUCTORS[name.name]
    for _ in range(3):
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                if node.value is None:
                    continue
                source = _attr_chain(node.value).rsplit(".", 1)[-1]
                if source in aliases:
                    aliases.update(
                        {target.rsplit(".", 1)[-1]: aliases[source] for target in _targets(node)}
                    )
    return aliases


def _typed_client(annotation: ast.AST | None, aliases: dict[str, tuple[str, str]]) -> bool:
    return annotation is not None and any(
        _attr_chain(part).rsplit(".", 1)[-1] in aliases for part in ast.walk(annotation)
    )


def _client_provenance(
    tree: ast.AST, aliases: dict[str, tuple[str, str]]
) -> tuple[set[str], set[str]]:
    nodes = tuple(ast.walk(tree))
    functions = (
        node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    factories = {
        node.name
        for node in functions
        if (_typed_client(node.returns, aliases))
        or any(
            isinstance(part, ast.Call) and _attr_chain(part.func).rsplit(".", 1)[-1] in aliases
            for part in ast.walk(node)
        )
    }
    clients = {
        node.arg
        for node in nodes
        if isinstance(node, ast.arg) and _typed_client(node.annotation, aliases)
    }

    def backed(value: ast.AST) -> bool:
        if isinstance(value, ast.IfExp):
            return backed(value.body) or backed(value.orelse)
        value = value.value if isinstance(value, ast.Await) else value
        chain = _attr_chain(value.func) if isinstance(value, ast.Call) else _attr_chain(value)
        terminal = chain.rsplit(".", 1)[-1]
        return chain in clients or terminal in aliases or terminal in factories

    for _ in range(3):
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                if value is None:
                    continue
                typed = isinstance(node, ast.AnnAssign) and _typed_client(node.annotation, aliases)
                if typed or backed(value):
                    clients.update(_targets(node))
    return clients, factories


class _InferenceVisitor(ast.NodeVisitor):
    def __init__(self, path: str, tree: ast.AST) -> None:
        self.path = path
        self.scope: list[tuple[str, str]] = []
        self.detections: list[tuple[SiteKey, str]] = []
        self.mentions_policy = False
        self.constructor_aliases = _constructor_aliases(tree)
        self.client_receivers, self.client_factories = _client_provenance(
            tree, self.constructor_aliases
        )

    def _anchor(self) -> str:
        if not self.scope:
            return "<module>"
        kind, name = self.scope[0]
        if kind == "class" and len(self.scope) > 1:
            return f"{name}.{self.scope[1][1]}"
        return name

    def _record(self, provider: str, protocol: str, detector: str) -> None:
        self.detections.append((SiteKey(self.path, self._anchor(), provider, protocol), detector))

    def _has_client(self, receiver: str) -> bool:
        prefixes = tuple(f"{client}." for client in self.client_receivers)
        rooted = receiver in self.client_receivers or receiver.startswith(prefixes)
        terminal = receiver.rsplit(".", 1)[-1]
        known = self.client_factories | self.constructor_aliases.keys()
        return rooted or terminal in known

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(("class", node.name))
        self.generic_visit(node)
        self.scope.pop()

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(("function", node.name))
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "ModelTransportPolicy":
            self.mentions_policy = True
        chain = _attr_chain(node, unwrap_calls=True)
        for suffix, provider, protocol in METHOD_SUFFIXES:
            receiver = chain[: -len(suffix)].rstrip(".")
            if chain.endswith(suffix) and self._has_client(receiver):
                self._record(provider, protocol, "inference-method")
                break
        if chain.endswith(("generate_anthropic", "anthropic_message")):
            self._record("anthropic", "messages", "transport-policy-call")
        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str):
            return
        provider = _provider_for_url(node.value)
        if provider is not None:
            self._record(provider, _protocol_for_url(node.value), "endpoint-literal")

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        value = "".join(
            str(part.value) if isinstance(part, ast.Constant) else "{}" for part in node.values
        )
        provider = _provider_for_url(value)
        if provider is not None:
            self._record(provider, _protocol_for_url(value), "endpoint-literal")
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                self.visit(part.value)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(alias.name == "ModelTransportPolicy" for alias in node.names):
            self.mentions_policy = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        chain = _attr_chain(node.func, unwrap_calls=True)
        terminal = chain.rsplit(".", 1)[-1]
        constructor = self.constructor_aliases.get(terminal)
        if constructor is not None:
            self._record(*constructor, "client-constructor")
        self.generic_visit(node)


def iter_python_files(root: Path) -> tuple[Path, ...]:
    files = {path for relative in SCAN_ROOTS for path in (root / relative).rglob("*.py")}
    return tuple(
        sorted(
            path
            for path in files
            if path.is_file() and path.relative_to(root).as_posix() != SELF_PATH
        )
    )


def discover(root: Path = REPO_ROOT) -> Discovery:
    grouped: dict[SiteKey, Counter[str]] = {}
    policy_consumers: set[str] = set()
    forbidden_ports: list[str] = []
    scan_errors: list[str] = []
    files = iter_python_files(root)
    for path in files:
        relative = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            scan_errors.append(f"{relative}: {type(exc).__name__}: {exc}")
            continue
        source_lines = source.splitlines()
        for line_no, line in enumerate(source_lines, 1):
            if FORBIDDEN_PORT_RE.search(line):
                forbidden_ports.append(f"{relative}:{line_no}")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and re.fullmatch(r"0*8317", node.value)
            ):
                forbidden_ports.append(f"{relative}:{node.lineno}")
            if not (
                isinstance(node, ast.Constant)
                and node.value == 8317
                and not isinstance(node.value, bool)
            ):
                continue
            line = source_lines[node.lineno - 1].strip()
            if relative == PORT_ENFORCEMENT_PATH and line == PORT_ENFORCEMENT_LINE:
                continue
            forbidden_ports.append(f"{relative}:{node.lineno}")
        visitor = _InferenceVisitor(relative, tree)
        visitor.visit(tree)
        if visitor.mentions_policy and relative != "aragora/agents/transports/vibeproxy.py":
            policy_consumers.add(relative)
        for key, detector in visitor.detections:
            grouped.setdefault(key, Counter())[detector] += 1
    sites = tuple(
        Site(
            path=key.path,
            anchor=key.anchor,
            provider=key.provider,
            protocol=key.protocol,
            detectors=dict(sorted(detectors.items())),
        )
        for key, detectors in sorted(grouped.items())
    )
    return Discovery(
        sites=sites,
        scanned_files=len(files),
        raw_detections=sum(sum(site.detectors.values()) for site in sites),
        policy_consumers=tuple(sorted(policy_consumers)),
        forbidden_ports=tuple(forbidden_ports),
        scan_errors=tuple(scan_errors),
    )


def protected_reasons(path: str) -> tuple[str, ...]:
    return tuple(
        reason
        for reason, patterns in PROTECTED_PATHS.items()
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
    )


def _key_text(key: SiteKey) -> str:
    return f"{key.path}::{key.anchor}::{key.provider}::{key.protocol}"


def _site_from_manifest(raw: Any, index: int, errors: list[str]) -> tuple[Site, str] | None:
    if not isinstance(raw, dict):
        errors.append(f"sites[{index}] must be an object")
        return None
    required = ("path", "anchor", "provider", "protocol", "detectors", "classification")
    if any(name not in raw for name in required):
        errors.append(f"sites[{index}] is missing required fields")
        return None
    classification = raw["classification"]
    if classification not in CLASSIFICATIONS:
        errors.append(f"sites[{index}] has invalid classification {classification!r}")
    rationale = raw.get("rationale", "")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append(f"sites[{index}] {classification} entry needs a rationale")
    detectors = raw["detectors"]
    if not isinstance(detectors, dict) or not detectors:
        errors.append(f"sites[{index}] detectors must be a non-empty object")
        return None
    normalized: dict[str, int] = {}
    for name, count in detectors.items():
        if not isinstance(name, str) or not isinstance(count, int) or count < 1:
            errors.append(f"sites[{index}] detector counts must be positive integers")
            continue
        normalized[name] = count
    if classification == "proxy-eligible" and not normalized.get("transport-policy-call"):
        errors.append(f"sites[{index}] proxy-eligible needs transport-policy-call")
    values = [raw[name] for name in required[:4]]
    if not all(isinstance(value, str) and value for value in values):
        errors.append(f"sites[{index}] identity fields must be non-empty strings")
        return None
    return (
        Site(
            path=str(raw["path"]),
            anchor=str(raw["anchor"]),
            provider=str(raw["provider"]),
            protocol=str(raw["protocol"]),
            detectors=normalized,
        ),
        classification,
    )


def load_manifest(path: Path) -> tuple[dict[SiteKey, tuple[Site, str]], tuple[str, ...], list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, (), [f"cannot read manifest: {exc}"]
    if _contains_forbidden_port(payload):
        errors.append("manifest must not contain forbidden port 8317")
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return {}, (), ["manifest schema_version must be 1"]
    raw_consumers = payload.get("transport_policy_consumers")
    if not isinstance(raw_consumers, list) or not all(
        isinstance(item, str) and item for item in raw_consumers
    ):
        errors.append("transport_policy_consumers must be a list of paths")
        consumers: tuple[str, ...] = ()
    else:
        consumers = tuple(sorted(set(raw_consumers)))
    entries: dict[SiteKey, tuple[Site, str]] = {}
    raw_sites = payload.get("sites")
    if not isinstance(raw_sites, list):
        return {}, consumers, errors + ["sites must be a list"]
    for index, raw in enumerate(raw_sites):
        parsed = _site_from_manifest(raw, index, errors)
        if parsed is None:
            continue
        site, classification = parsed
        if site.key in entries:
            errors.append(f"duplicate site {_key_text(site.key)}")
        entries[site.key] = (site, classification)
        reasons = protected_reasons(site.path)
        if reasons and classification != "direct-only":
            errors.append(
                f"protected site {_key_text(site.key)} must be direct-only ({', '.join(reasons)})"
            )
        if classification == "proxy-eligible" and site.path not in consumers:
            errors.append(f"proxy-eligible site {_key_text(site.key)} is not a policy consumer")
    return entries, consumers, errors


def check_allowlist(root: Path = REPO_ROOT, manifest_path: Path = DEFAULT_MANIFEST) -> CheckResult:
    discovery = discover(root)
    expected, expected_consumers, manifest_errors = load_manifest(manifest_path)
    actual = {site.key: site for site in discovery.sites}
    unclassified = tuple(_key_text(key) for key in sorted(actual.keys() - expected.keys()))
    stale = tuple(_key_text(key) for key in sorted(expected.keys() - actual.keys()))
    changed = tuple(
        f"{_key_text(key)} expected={expected[key][0].detectors} actual={actual[key].detectors}"
        for key in sorted(actual.keys() & expected.keys())
        if actual[key].detectors != expected[key][0].detectors
    )
    policy_errors: list[str] = []
    if discovery.policy_consumers != expected_consumers:
        policy_errors.append(
            "transport policy consumers differ: "
            f"expected={list(expected_consumers)} actual={list(discovery.policy_consumers)}"
        )
    violations = (
        unclassified,
        stale,
        changed,
        manifest_errors,
        policy_errors,
        discovery.forbidden_ports,
        discovery.scan_errors,
    )
    ok = not any(violations)
    return CheckResult(
        ok=ok,
        scanned_files=discovery.scanned_files,
        site_count=len(discovery.sites),
        raw_detections=discovery.raw_detections,
        policy_consumers=discovery.policy_consumers,
        unclassified=unclassified,
        stale=stale,
        changed=changed,
        manifest_errors=tuple(manifest_errors),
        policy_errors=tuple(policy_errors),
        forbidden_ports=discovery.forbidden_ports,
        scan_errors=discovery.scan_errors,
    )


def template_manifest(discovery: Discovery) -> dict[str, Any]:
    sites: list[dict[str, Any]] = []
    for site in discovery.sites:
        reasons = protected_reasons(site.path)
        rationale = (
            f"protected {', '.join(reasons)} surface remains direct by issue #9409"
            if reasons
            else "not yet routed through the central exact-match transport policy"
        )
        sites.append(
            {
                **asdict(site),
                "classification": "direct-only",
                "rationale": rationale,
            }
        )
    return {
        "generated_by": "scripts/check_inference_site_allowlist.py --emit-template",
        "policy_note": "Template output defaults to direct-only; preserve reviewed classifications and rationales.",
        "schema_version": 1,
        "transport_policy_consumers": list(discovery.policy_consumers),
        "sites": sites,
    }


def _print_human(result: CheckResult) -> None:
    if result.ok:
        print(
            "inference-site allowlist: ok "
            f"({result.site_count} sites, {result.raw_detections} detections, "
            f"{result.scanned_files} files)"
        )
        return
    print("inference-site allowlist: violations found")
    for label in (
        "manifest_errors",
        "policy_errors",
        "unclassified",
        "stale",
        "changed",
        "forbidden_ports",
        "scan_errors",
    ):
        for value in getattr(result, label):
            print(f"  {label}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable results.")
    parser.add_argument(
        "--emit-template",
        action="store_true",
        help="Print a candidate manifest; never writes the reviewed file.",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    if args.emit_template:
        print(json.dumps(template_manifest(discover(args.root)), indent=2, sort_keys=True))
        return 0
    result = check_allowlist(args.root, args.manifest)
    if args.json:
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        _print_human(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
