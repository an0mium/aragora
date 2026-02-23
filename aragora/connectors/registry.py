"""Connector registry discovery and loading utilities."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable


_CLASS_PATTERN = re.compile(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\(|:)", re.MULTILINE)

_EXCLUDED_CLASS_NAMES = {
    "BaseConnector",
    "Connector",
    "ConnectorProtocol",
    "ConnectorCapabilities",
    "ConnectorHealth",
    "ConnectorDataclass",
    "ConnectorConfig",
    "ConnectorError",
    "ConnectorAuthError",
    "ConnectorRateLimitError",
    "ConnectorAPIError",
    "ConnectorNotFoundError",
    "ConnectorPermissionError",
    "DeviceConnector",
    "DeviceConnectorConfig",
    "ChatPlatformConnector",
    "AutomationConnector",
    "EnterpriseConnector",
    "ConnectorRegistry",
    "ConnectorRecord",
    "SyncService",
}

_EXTRA_CLASS_NAMES = {
    "RepositoryCrawler",
}

_CLASS_SUFFIXES = ("Connector", "SyncService")

_EVIDENCE_STEMS = {
    "arxiv",
    "gaap",
    "fasb",
    "clinical_tables",
    "courtlistener",
    "crossref",
    "govinfo",
    "github",
    "hackernews",
    "irs",
    "local_docs",
    "newsapi",
    "nice_guidance",
    "pubmed",
    "reddit",
    "repository_crawler",
    "sec",
    "westlaw",
    "lexis",
    "semantic_scholar",
    "sql",
    "twitter",
    "web",
    "whisper",
    "wikipedia",
    "youtube_uploader",
    "rxnav",
}

_OPERATIONAL_SEGMENTS = {
    "accounting",
    "advertising",
    "analytics",
    "calendar",
    "crm",
    "devops",
    "devices",
    "ecommerce",
    "email",
    "legal",
    "marketing",
    "marketplace",
    "metrics",
    "payments",
    "support",
    "browser",
}


@dataclass(frozen=True)
class ConnectorRecord:
    name: str
    module: str
    path: str
    kind: str
    category: str
    status: str = "implemented"


@dataclass(frozen=True)
class ConnectorRegistry:
    generated_at: str
    total: int
    by_kind: dict[str, int]
    by_category: dict[str, int]
    connectors: list[ConnectorRecord]

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "total": self.total,
            "by_kind": self.by_kind,
            "by_category": self.by_category,
            "connectors": [asdict(connector) for connector in self.connectors],
        }


def _infer_kind_category(connectors_root: Path, path: Path) -> tuple[str, str]:
    rel = path.relative_to(connectors_root)
    parts = rel.parts
    if not parts:
        return "operational", "misc"

    top = parts[0]
    if top == "enterprise":
        category = f"enterprise.{parts[1]}" if len(parts) > 1 else "enterprise"
        return "enterprise", category
    if top == "automation":
        return "operational", "automation"
    if top == "knowledge":
        return "evidence", "knowledge"
    if top == "documents":
        return "evidence", "documents"
    if top == "chat":
        return "operational", "chat"
    if top in _OPERATIONAL_SEGMENTS:
        return "operational", top

    if len(parts) == 1:
        stem = path.stem
        if stem in _EVIDENCE_STEMS:
            return "evidence", "evidence"
        return "operational", "misc"

    return "operational", top


def _iter_python_files(connectors_root: Path) -> Iterable[Path]:
    for path in connectors_root.rglob("*.py"):
        yield path


def _iter_class_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [match.group(1) for match in _CLASS_PATTERN.finditer(source)]
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def discover_connectors(connectors_root: Path) -> list[ConnectorRecord]:
    repo_root = connectors_root.parents[1]
    records: list[ConnectorRecord] = []
    seen: set[tuple[str, str]] = set()

    for path in _iter_python_files(connectors_root):
        source = path.read_text(encoding="utf-8")
        class_names = _iter_class_names(source)
        if not class_names:
            continue
        module = "aragora.connectors." + ".".join(
            path.relative_to(connectors_root).with_suffix("").parts
        )
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        kind, category = _infer_kind_category(connectors_root, path)
        for name in class_names:
            if name in _EXCLUDED_CLASS_NAMES:
                continue
            if not (name.endswith(_CLASS_SUFFIXES) or name in _EXTRA_CLASS_NAMES):
                continue
            key = (module, name)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                ConnectorRecord(
                    name=name,
                    module=module,
                    path=str(path.relative_to(repo_root).as_posix()),
                    kind=kind,
                    category=category,
                )
            )

    return sorted(
        records, key=lambda record: (record.kind, record.category, record.name, record.module)
    )


def build_registry(connectors_root: Path) -> ConnectorRegistry:
    connectors = discover_connectors(connectors_root)
    by_kind: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for connector in connectors:
        by_kind[connector.kind] = by_kind.get(connector.kind, 0) + 1
        by_category[connector.category] = by_category.get(connector.category, 0) + 1
    return ConnectorRegistry(
        generated_at=datetime.now(timezone.utc).isoformat(),
        total=len(connectors),
        by_kind=dict(sorted(by_kind.items())),
        by_category=dict(sorted(by_category.items())),
        connectors=connectors,
    )


def load_registry(path: Path) -> ConnectorRegistry:
    payload = json.loads(path.read_text(encoding="utf-8"))
    connectors = [ConnectorRecord(**entry) for entry in payload.get("connectors", [])]
    return ConnectorRegistry(
        generated_at=payload.get("generated_at", ""),
        total=payload.get("total", len(connectors)),
        by_kind=payload.get("by_kind", {}),
        by_category=payload.get("by_category", {}),
        connectors=connectors,
    )
