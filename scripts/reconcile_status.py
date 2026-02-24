#!/usr/bin/env python3
"""Reconcile feature status across Aragora source-of-truth documents.

Parses four documents and cross-references features to find discrepancies:
  1. docs/CAPABILITY_MATRIX.md  -- feature vs surface coverage
  2. docs/GA_CHECKLIST.md       -- GA readiness items
  3. docs/STATUS.md             -- feature implementation status
  4. ROADMAP.md                 -- roadmap phase items

Exit codes:
  0 -- no critical discrepancies
  1 -- critical discrepancies found (GA blockers)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FeatureEntry:
    """A feature mentioned in a document with its status."""

    name: str
    raw_name: str  # original text before normalisation
    status: str  # normalised: DONE, PENDING, ACTIVE, UNMAPPED, etc.
    source: str  # which document
    category: str = ""  # optional grouping (e.g. GA checklist section)
    detail: str = ""  # extra info


@dataclass
class Discrepancy:
    """A contradiction between two documents."""

    feature: str
    doc_a: str
    status_a: str
    doc_b: str
    status_b: str
    severity: str  # CRITICAL, WARNING, INFO
    message: str = ""


@dataclass
class ReconciliationReport:
    features_by_source: dict[str, list[FeatureEntry]] = field(default_factory=dict)
    discrepancies: list[Discrepancy] = field(default_factory=list)
    missing_from: dict[str, set[str]] = field(
        default_factory=dict
    )  # doc -> set of feature names missing
    summary: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def normalise_feature_name(name: str) -> str:
    """Produce a canonical key from a feature name for fuzzy matching."""
    s = name.strip().lower()
    # Remove markdown bold/italic markers (but keep underscores -- they're word separators)
    s = re.sub(r"[*`~]", "", s)
    # Convert underscores to spaces so they normalise uniformly with other separators
    s = s.replace("_", " ")
    # Remove leading checkbox markers
    s = re.sub(r"^\[[ x]\]\s*", "", s)
    # Remove trailing status annotations like "(STABLE)", "(Active)", etc.
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    # Remove common prefixes
    for prefix in ("complete ", "implement ", "deploy ", "finalize "):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    # Collapse whitespace, hyphens to single underscore
    s = re.sub(r"[\s\-]+", "_", s)
    # Strip non-alphanumeric (except underscore)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def normalise_status(raw: str) -> str:
    """Map various status strings to a canonical set."""
    s = raw.strip().lower()
    # Remove markdown
    s = re.sub(r"[*_`~]", "", s)

    if s in (
        "x",
        "done",
        "complete",
        "completed",
        "shipped",
        "active",
        "stable",
        "production",
        "production ready",
        "production-ready",
        "exported",
        "resolved",
        "fixed",
    ):
        return "DONE"
    if s in (
        "",
        " ",
        "pending",
        "not started",
        "planned",
        "todo",
        "under consideration",
        "researching",
    ):
        return "PENDING"
    if s in ("in progress", "in-progress", "wip", "experimental", "beta", "alpha", "partial"):
        return "IN_PROGRESS"
    if s in ("unmapped",):
        return "UNMAPPED"
    if s in ("deprecated",):
        return "DEPRECATED"
    # Checkbox status
    if s == "checked":
        return "DONE"
    if s == "unchecked":
        return "PENDING"
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_capability_matrix(text: str) -> list[FeatureEntry]:
    """Extract features from CAPABILITY_MATRIX.md.

    Handles multiple formats:
    - Bullet lists under "Missing UI/CHANNELS" or "Unmapped Capabilities" sections
    - Tables under "Additional Mapped Capabilities" or "UI Coverage"
    - Capability names in backtick-quoted table cells
    """
    entries: list[FeatureEntry] = []
    source = "CAPABILITY_MATRIX"
    seen_keys: set[str] = set()

    section = ""
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()

        # Detect section headers
        if stripped.startswith("### Missing UI"):
            section = "missing_ui"
            in_table = False
            continue
        elif stripped.startswith("### Missing CHANNELS"):
            section = "missing_channels"
            in_table = False
            continue
        elif stripped.startswith("## Unmapped Capabilities"):
            section = "unmapped"
            in_table = False
            continue
        elif stripped.startswith("## Additional Mapped Capabilities"):
            section = "mapped"
            in_table = False
            continue
        elif "UI Coverage" in stripped and stripped.startswith("**"):
            section = "ui_coverage"
            in_table = False
            continue
        elif stripped.startswith("## ") or (
            stripped.startswith("### ") and section not in ("missing_ui", "missing_channels")
        ):
            section = ""
            in_table = False
            continue

        # Skip table header separators
        if re.match(r"^\|[-\s|]+\|$", stripped):
            in_table = True
            continue
        # Skip table header rows
        if stripped.startswith("| Capability") or stripped.startswith("| Surface"):
            in_table = True
            continue

        # Bullet-list items with backtick names (Missing/Unmapped sections)
        if section in ("missing_ui", "missing_channels", "unmapped") and stripped.startswith("- `"):
            m = re.match(r"- `([^`]+)`", stripped)
            if m:
                raw = m.group(1)
                key = normalise_feature_name(raw)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                if section == "unmapped":
                    entries.append(
                        FeatureEntry(
                            name=key,
                            raw_name=raw,
                            status="UNMAPPED",
                            source=source,
                            category="unmapped",
                            detail="Listed as unmapped capability",
                        )
                    )
                else:
                    entries.append(
                        FeatureEntry(
                            name=key,
                            raw_name=raw,
                            status="DONE",
                            source=source,
                            category=section,
                            detail=f"Missing from {section.replace('_', ' ')}",
                        )
                    )

        # Table rows with backtick capability names (mapped/ui_coverage sections)
        if section in ("mapped", "ui_coverage") and stripped.startswith("| `"):
            m = re.match(r"\|\s*`([^`]+)`", stripped)
            if m:
                raw = m.group(1)
                key = normalise_feature_name(raw)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                # Determine coverage from Y/N columns for mapped table
                detail = ""
                if section == "mapped":
                    cols = [c.strip() for c in stripped.strip("|").split("|")]
                    if len(cols) >= 5:
                        surfaces = {"API": cols[1], "CLI": cols[2], "SDK": cols[3], "UI": cols[4]}
                        missing = [s for s, v in surfaces.items() if v.strip() == "-"]
                        detail = f"Missing: {', '.join(missing)}" if missing else "Full coverage"
                entries.append(
                    FeatureEntry(
                        name=key,
                        raw_name=raw,
                        status="DONE",
                        source=source,
                        category=section,
                        detail=detail or "Mapped capability",
                    )
                )

    return entries


def parse_ga_checklist(text: str) -> list[FeatureEntry]:
    """Extract items from GA_CHECKLIST.md checkbox lists."""
    entries: list[FeatureEntry] = []
    source = "GA_CHECKLIST"
    current_section = ""

    for line in text.splitlines():
        line_stripped = line.strip()

        # Section headers
        section_match = re.match(r"^##\s+(.+)", line_stripped)
        if section_match:
            current_section = section_match.group(1).strip()
            continue

        # Checkbox items: - [x] **Name** - description  OR  - [ ] **Name** - description
        cb_match = re.match(r"^-\s+\[([ xX])\]\s+\*\*(.+?)\*\*(.*)$", line_stripped)
        if cb_match:
            checked = cb_match.group(1).strip().lower() == "x"
            raw_name = cb_match.group(2).strip()
            detail = cb_match.group(3).strip().lstrip("-").strip()
            key = normalise_feature_name(raw_name)
            status = "DONE" if checked else "PENDING"
            entries.append(
                FeatureEntry(
                    name=key,
                    raw_name=raw_name,
                    status=status,
                    source=source,
                    category=current_section,
                    detail=detail,
                )
            )

    # Also parse the GA Blockers table
    in_blocker_table = False
    for line in text.splitlines():
        if "## GA Blockers" in line:
            in_blocker_table = True
            continue
        if in_blocker_table:
            if line.strip().startswith("| ") and "---" not in line and "Blocker" not in line:
                cols = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cols) >= 3:
                    raw_name = cols[0].strip()
                    status_raw = cols[2].strip()
                    key = normalise_feature_name(raw_name)
                    status = normalise_status(status_raw)
                    if status == "UNKNOWN" and "not started" in status_raw.lower():
                        status = "PENDING"
                    entries.append(
                        FeatureEntry(
                            name=key,
                            raw_name=raw_name,
                            status=status,
                            source=source,
                            category="GA Blockers",
                            detail=status_raw,
                        )
                    )
            elif line.strip().startswith("## "):
                in_blocker_table = False

    return entries


def parse_status_md(text: str) -> list[FeatureEntry]:
    """Extract features from STATUS.md Feature Integration Status tables."""
    entries: list[FeatureEntry] = []
    source = "STATUS"

    # Parse the Feature Integration Status tables
    in_table = False
    table_name = ""
    for line in text.splitlines():
        stripped = line.strip()

        # Detect table section headers
        if re.match(
            r"^###\s+(Fully Integrated|Recently Surfaced|Handler .+ Status|Handler Test Coverage)",
            stripped,
        ):
            in_table = True
            m = re.match(r"^###\s+(.+?)(?:\s+\(\d+\))?$", stripped)
            table_name = m.group(1) if m else stripped
            continue

        if in_table:
            # End of table
            if stripped.startswith("## ") or stripped.startswith("### "):
                in_table = False
                table_name = ""
                # Check if new table
                if re.match(
                    r"^###\s+(Fully Integrated|Recently Surfaced|Handler .+ Status|Handler Test Coverage)",
                    stripped,
                ):
                    in_table = True
                    m = re.match(r"^###\s+(.+?)(?:\s+\(\d+\))?$", stripped)
                    table_name = m.group(1) if m else stripped
                continue

            # Skip header separator
            if re.match(r"^\|[-\s|]+\|$", stripped):
                continue
            # Skip header row
            if stripped.startswith("| Feature") or stripped.startswith("| Handler"):
                continue

            # Table row
            if stripped.startswith("| "):
                cols = [c.strip() for c in stripped.strip("|").split("|")]
                if len(cols) >= 2:
                    raw_name = cols[0].strip()
                    status_raw = cols[1].strip() if len(cols) >= 2 else ""
                    key = normalise_feature_name(raw_name)
                    if not key:
                        continue
                    status = normalise_status(status_raw)
                    entries.append(
                        FeatureEntry(
                            name=key,
                            raw_name=raw_name,
                            status=status,
                            source=source,
                            category=table_name,
                            detail=status_raw,
                        )
                    )

    # Also extract named features from prose that declare STABLE/PRODUCTION status
    for line in text.splitlines():
        m = re.match(
            r"^-\s+\*\*(.+?)\*\*\s*[-:]?\s*(STABLE|PRODUCTION|BETA|ALPHA|DEPRECATED)", line.strip()
        )
        if m:
            raw_name = m.group(1)
            status_raw = m.group(2)
            key = normalise_feature_name(raw_name)
            status = normalise_status(status_raw)
            entries.append(
                FeatureEntry(
                    name=key,
                    raw_name=raw_name,
                    status=status,
                    source=source,
                    category="inline_status",
                    detail=f"Declared {status_raw} in prose",
                )
            )

    # Extract from Channel Production Readiness table
    in_channel_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if "Channel Production Readiness" in stripped or "Channel | Readiness" in stripped:
            in_channel_table = True
            continue
        if in_channel_table:
            if stripped.startswith("| **"):
                cols = [c.strip() for c in stripped.strip("|").split("|")]
                if len(cols) >= 2:
                    raw_name = re.sub(r"[*]", "", cols[0]).strip()
                    readiness = cols[1].strip()
                    key = normalise_feature_name(raw_name + "_integration")
                    pct_match = re.search(r"(\d+)%", readiness)
                    if pct_match:
                        pct = int(pct_match.group(1))
                        status = "DONE" if pct >= 90 else "IN_PROGRESS" if pct >= 50 else "PENDING"
                    else:
                        status = normalise_status(readiness)
                    entries.append(
                        FeatureEntry(
                            name=key,
                            raw_name=raw_name,
                            status=status,
                            source=source,
                            category="Channel Readiness",
                            detail=readiness,
                        )
                    )
            elif stripped.startswith("### ") and in_channel_table:
                in_channel_table = False

    return entries


def parse_roadmap(text: str) -> list[FeatureEntry]:
    """Extract features from ROADMAP.md checkbox lists and tables."""
    entries: list[FeatureEntry] = []
    source = "ROADMAP"
    current_section = ""

    for line in text.splitlines():
        stripped = line.strip()

        # Section headers
        section_match = re.match(r"^###\s+(.+)", stripped)
        if section_match:
            current_section = section_match.group(1).strip()
            continue
        h2_match = re.match(r"^##\s+(.+)", stripped)
        if h2_match:
            current_section = h2_match.group(1).strip()
            continue

        # Checkbox items
        cb_match = re.match(r"^-\s+\[([ xX])\]\s+(.+)$", stripped)
        if cb_match:
            checked = cb_match.group(1).strip().lower() == "x"
            raw_name = cb_match.group(2).strip()
            # Strip trailing detail after first dash or parenthesis for name
            name_part = re.split(r"\s+[-\(]", raw_name)[0].strip()
            key = normalise_feature_name(name_part)
            status = "DONE" if checked else "PENDING"
            entries.append(
                FeatureEntry(
                    name=key,
                    raw_name=name_part,
                    status=status,
                    source=source,
                    category=current_section,
                    detail=raw_name,
                )
            )

    # Feature Requests table
    in_feature_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if "## Feature Requests" in stripped:
            in_feature_table = True
            continue
        if in_feature_table:
            if stripped.startswith("| ") and "---" not in stripped and "Feature" not in stripped:
                cols = [c.strip() for c in stripped.strip("|").split("|")]
                if len(cols) >= 3:
                    raw_name = cols[0].strip()
                    status_raw = cols[2].strip()
                    key = normalise_feature_name(raw_name)
                    # Detect "Shipped vX.Y"
                    if "shipped" in status_raw.lower():
                        status = "DONE"
                    elif "planned" in status_raw.lower():
                        status = "PENDING"
                    else:
                        status = normalise_status(status_raw)
                    entries.append(
                        FeatureEntry(
                            name=key,
                            raw_name=raw_name,
                            status=status,
                            source=source,
                            category="Feature Requests",
                            detail=status_raw,
                        )
                    )
            elif stripped.startswith("## "):
                in_feature_table = False

    return entries


# ---------------------------------------------------------------------------
# Cross-reference engine
# ---------------------------------------------------------------------------

# Feature aliases: map alternative names that refer to the same thing
FEATURE_ALIASES: dict[str, str] = {
    # GA Checklist <-> Roadmap: Penetration testing
    "external_penetration_test": "external_pen_test",
    "third_party_penetration_testing": "external_pen_test",
    "penetration_testing": "external_pen_test",
    "external_pen_test": "external_pen_test",
    # Docker
    "docker_compose": "docker_compose",
    "docker_compose_production_stack": "docker_compose",
    # MFA
    "mfa": "mfa",
    "mfa_enforcement_for_admin_access": "mfa",
    # RBAC
    "rbac_v2": "rbac_v2",
    # Backup / DR
    "backup_restore": "backup_dr",
    "backuprestore": "backup_dr",
    "backup_disaster_recovery": "backup_dr",
    "backup__restore_cli": "backup_dr",
    # SSO
    "oidc_saml_sso": "sso_authentication",
    "sso_authentication": "sso_authentication",
    # Multi-tenancy
    "multi_tenancy": "multi_tenancy",
    # Redis
    "redis_ha": "redis_ha",
    "redis_sentinel_cluster_support": "redis_ha",
    "redis_cluster_mode_support": "redis_ha",
    # Circuit breaker
    "circuit_breaker": "circuit_breaker",
    "enhanced_circuit_breaker_coverage_for_all_connectors": "circuit_breaker",
    # Decision receipts
    "decision_receipts": "decision_receipts",
    "decision_receipts_v1": "decision_receipts",
    # Slack
    "slack_integration": "slack_integration",
    # Teams
    "teams_integration": "teams_integration",
    "microsoft_teams_integration": "teams_integration",
    # Knowledge Mound
    "knowledge_mound": "knowledge_mound",
    # Workflow engine
    "workflow_engine": "workflow_engine",
    # Test suite
    "test_suite": "test_suite",
    # Prometheus
    "prometheus_metrics": "prometheus_metrics",
    # SLO
    "slo_alerting": "slo_alerting",
    # Structured logging
    "structured_logging": "structured_logging",
    # Helm / Kubernetes
    "helm_chart_for_kubernetes": "kubernetes_deployment",
    "kubernetes_manifests": "kubernetes_deployment",
    # Distributed tracing / OpenTelemetry
    "distributed_tracing": "distributed_tracing",
    "opentelemetry_tracing": "distributed_tracing",
    # Control plane
    "control_plane": "control_plane",
    # Supermemory
    "supermemory": "supermemory",
    # Telegram
    "telegram_connector": "telegram_connector",
    # WhatsApp
    "whatsapp_connector": "whatsapp_connector",
    # Webhook
    "webhook_integrations": "webhook_integrations",
    # Kafka / RabbitMQ
    "kafka_streaming": "kafka_streaming",
    "rabbitmq_streaming": "rabbitmq_streaming",
    # Prompt evolution
    "prompt_evolution": "prompt_evolution",
    # RLM
    "rlm": "rlm",
    # Pulse
    "pulse_trending": "pulse_trending",
    # Extended debates
    "extended_debates": "extended_debates",
    # Marketplace
    "marketplace": "marketplace",
    # Compliance
    "compliance_framework": "compliance_framework",
    # OpenAPI
    "openapi_spec": "openapi_31_specification",
    "openapi_31_specification": "openapi_31_specification",
}


def resolve_alias(name: str) -> str:
    """Resolve a feature name to its canonical form."""
    return FEATURE_ALIASES.get(name, name)


def build_feature_index(
    all_entries: list[FeatureEntry],
) -> dict[str, dict[str, FeatureEntry]]:
    """Build {canonical_name: {source: entry}} index."""
    index: dict[str, dict[str, FeatureEntry]] = {}
    for entry in all_entries:
        canonical = resolve_alias(entry.name)
        if canonical not in index:
            index[canonical] = {}
        # If same source appears multiple times, keep the more informative one
        if entry.source not in index[canonical]:
            index[canonical][entry.source] = entry
        else:
            # Prefer DONE/PENDING over UNKNOWN
            existing = index[canonical][entry.source]
            if existing.status == "UNKNOWN" and entry.status != "UNKNOWN":
                index[canonical][entry.source] = entry
    return index


def is_ga_blocking(entry: FeatureEntry) -> bool:
    """Check whether a feature is a GA blocker."""
    if entry.category == "GA Blockers":
        return True
    # Items in the GA checklist that are unchecked
    if entry.source == "GA_CHECKLIST" and entry.status == "PENDING":
        return True
    return False


def detect_discrepancies(
    index: dict[str, dict[str, FeatureEntry]],
) -> list[Discrepancy]:
    """Find status contradictions across documents."""
    discrepancies = []

    for canonical, sources in index.items():
        source_list = list(sources.items())
        for i in range(len(source_list)):
            for j in range(i + 1, len(source_list)):
                src_a, entry_a = source_list[i]
                src_b, entry_b = source_list[j]

                # Skip if statuses agree
                if entry_a.status == entry_b.status:
                    continue

                # Skip if one is UNKNOWN or UNMAPPED (not a real contradiction)
                if entry_a.status in ("UNKNOWN", "UNMAPPED") or entry_b.status in (
                    "UNKNOWN",
                    "UNMAPPED",
                ):
                    continue

                # Determine severity
                severity = "INFO"

                # CRITICAL: GA checklist says PENDING but another doc says DONE
                # (or vice versa -- indicates either stale checklist or false completion)
                if entry_a.source == "GA_CHECKLIST" or entry_b.source == "GA_CHECKLIST":
                    ga_entry = entry_a if entry_a.source == "GA_CHECKLIST" else entry_b
                    other_entry = entry_b if entry_a.source == "GA_CHECKLIST" else entry_a
                    if ga_entry.status == "PENDING" and other_entry.status == "DONE":
                        severity = "WARNING"
                        if is_ga_blocking(ga_entry):
                            severity = "CRITICAL"
                    elif ga_entry.status == "DONE" and other_entry.status == "PENDING":
                        severity = "WARNING"

                # WARNING: Roadmap says PENDING but STATUS says DONE (stale roadmap)
                if (entry_a.source == "ROADMAP" and entry_b.source == "STATUS") or (
                    entry_b.source == "ROADMAP" and entry_a.source == "STATUS"
                ):
                    road_entry = entry_a if entry_a.source == "ROADMAP" else entry_b
                    stat_entry = entry_b if entry_a.source == "ROADMAP" else entry_a
                    if road_entry.status == "PENDING" and stat_entry.status == "DONE":
                        severity = "WARNING"

                # DONE vs IN_PROGRESS is a mild discrepancy
                if {entry_a.status, entry_b.status} == {"DONE", "IN_PROGRESS"}:
                    if severity == "INFO":
                        severity = "INFO"

                message = (
                    f"'{entry_a.raw_name}' is {entry_a.status} in {src_a} "
                    f"but '{entry_b.raw_name}' is {entry_b.status} in {src_b}"
                )

                discrepancies.append(
                    Discrepancy(
                        feature=canonical,
                        doc_a=src_a,
                        status_a=entry_a.status,
                        doc_b=src_b,
                        status_b=entry_b.status,
                        severity=severity,
                        message=message,
                    )
                )

    return discrepancies


def detect_cross_doc_gaps(
    index: dict[str, dict[str, FeatureEntry]],
    all_sources: list[str],
) -> dict[str, list[tuple[str, str]]]:
    """Find features that appear in one doc but are absent from others.

    Returns {source: [(canonical_name, raw_name)]} of features unique to that source.
    """
    unique: dict[str, list[tuple[str, str]]] = {s: [] for s in all_sources}

    for canonical, sources in index.items():
        if len(sources) == 1:
            src, entry = next(iter(sources.items()))
            unique[src].append((canonical, entry.raw_name))

    return unique


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_report(
    all_entries: list[FeatureEntry],
    discrepancies: list[Discrepancy],
    unique_features: dict[str, list[tuple[str, str]]],
    index: dict[str, dict[str, FeatureEntry]],
) -> str:
    """Produce a plain-text report."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  ARAGORA STATUS RECONCILIATION REPORT")
    lines.append("=" * 72)
    lines.append("")

    # -- Summary table --
    sources = ["CAPABILITY_MATRIX", "GA_CHECKLIST", "STATUS", "ROADMAP"]
    lines.append("DOCUMENT SUMMARY")
    lines.append("-" * 50)
    lines.append(f"  {'Document':<25} {'Features':>10}")
    lines.append(f"  {'-' * 25} {'-' * 10}")
    for src in sources:
        count = sum(1 for e in all_entries if e.source == src)
        lines.append(f"  {src:<25} {count:>10}")
    lines.append(f"  {'TOTAL (unique keys)':<25} {len(index):>10}")
    lines.append("")

    # -- Discrepancies --
    critical = [d for d in discrepancies if d.severity == "CRITICAL"]
    warnings = [d for d in discrepancies if d.severity == "WARNING"]
    infos = [d for d in discrepancies if d.severity == "INFO"]

    lines.append("DISCREPANCY SUMMARY")
    lines.append("-" * 50)
    lines.append(f"  CRITICAL : {len(critical)}")
    lines.append(f"  WARNING  : {len(warnings)}")
    lines.append(f"  INFO     : {len(infos)}")
    lines.append(f"  TOTAL    : {len(discrepancies)}")
    lines.append("")

    if critical:
        lines.append("CRITICAL DISCREPANCIES (GA blockers)")
        lines.append("=" * 72)
        for d in critical:
            lines.append(f"  [{d.severity}] {d.feature}")
            lines.append(f"    {d.message}")
            lines.append("")

    if warnings:
        lines.append("WARNINGS (stale data or status drift)")
        lines.append("-" * 72)
        for d in warnings:
            lines.append(f"  [{d.severity}] {d.feature}")
            lines.append(f"    {d.message}")
            lines.append("")

    if infos:
        lines.append("INFO (minor differences)")
        lines.append("-" * 72)
        for d in infos:
            lines.append(f"  [{d.severity}] {d.feature}")
            lines.append(f"    {d.message}")
            lines.append("")

    # -- GA Blockers check --
    lines.append("GA BLOCKER STATUS")
    lines.append("-" * 50)
    ga_entries = [e for e in all_entries if e.source == "GA_CHECKLIST"]
    pending_ga = [e for e in ga_entries if e.status == "PENDING"]
    done_ga = [e for e in ga_entries if e.status == "DONE"]
    lines.append(f"  GA items done    : {len(done_ga)}")
    lines.append(f"  GA items pending : {len(pending_ga)}")
    if pending_ga:
        lines.append("")
        lines.append("  Pending GA items:")
        for e in pending_ga:
            cat = f" [{e.category}]" if e.category else ""
            lines.append(f"    - {e.raw_name}{cat}")
            if e.detail:
                lines.append(f"      Detail: {e.detail}")
    lines.append("")

    # -- Roadmap vs implementation drift --
    roadmap_pending = {
        resolve_alias(e.name): e
        for e in all_entries
        if e.source == "ROADMAP" and e.status == "PENDING"
    }
    already_done = []
    for name, road_entry in roadmap_pending.items():
        if name in index:
            for src, entry in index[name].items():
                if src != "ROADMAP" and entry.status == "DONE":
                    already_done.append((road_entry.raw_name, src, entry.raw_name))
                    break

    if already_done:
        lines.append("ROADMAP ITEMS ALREADY COMPLETED (roadmap may be stale)")
        lines.append("-" * 72)
        for road_name, done_src, done_name in already_done:
            lines.append(
                f"  - '{road_name}' (PENDING in ROADMAP) -> DONE in {done_src} as '{done_name}'"
            )
        lines.append("")

    # -- Features unique to one document --
    lines.append("FEATURES UNIQUE TO A SINGLE DOCUMENT")
    lines.append("-" * 72)
    for src in sources:
        unique_list = unique_features.get(src, [])
        if unique_list:
            lines.append(f"  {src} only ({len(unique_list)}):")
            # Show first 15
            for canonical, raw in sorted(unique_list)[:15]:
                lines.append(f"    - {raw} ({canonical})")
            if len(unique_list) > 15:
                lines.append(f"    ... and {len(unique_list) - 15} more")
            lines.append("")

    # -- Capability Matrix coverage gaps --
    cap_entries = [e for e in all_entries if e.source == "CAPABILITY_MATRIX"]
    unmapped = [e for e in cap_entries if e.status == "UNMAPPED"]
    if unmapped:
        lines.append("UNMAPPED CAPABILITIES (in Capability Matrix)")
        lines.append("-" * 50)
        lines.append(f"  {len(unmapped)} capabilities not yet mapped to surfaces:")
        for e in sorted(unmapped, key=lambda x: x.name):
            lines.append(f"    - {e.raw_name}")
        lines.append("")

    # -- Final verdict --
    lines.append("=" * 72)
    if critical:
        lines.append(f"RESULT: FAIL -- {len(critical)} critical discrepancy(ies) found")
        lines.append("These must be resolved before GA.")
    elif warnings:
        lines.append(f"RESULT: PASS (with {len(warnings)} warning(s))")
        lines.append("No critical discrepancies. Warnings indicate stale documentation.")
    else:
        lines.append("RESULT: PASS -- All documents are consistent")
    lines.append("=" * 72)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    doc_paths = {
        "CAPABILITY_MATRIX": repo_root / "docs" / "CAPABILITY_MATRIX.md",
        "GA_CHECKLIST": repo_root / "docs" / "GA_CHECKLIST.md",
        "STATUS": repo_root / "docs" / "STATUS.md",
        "ROADMAP": repo_root / "ROADMAP.md",
    }

    # Read all documents
    texts: dict[str, str] = {}
    missing_docs = []
    for name, path in doc_paths.items():
        if path.exists():
            texts[name] = path.read_text(encoding="utf-8")
        else:
            missing_docs.append(str(path))

    if missing_docs:
        print("ERROR: Missing documents:\n  " + "\n  ".join(missing_docs), file=sys.stderr)
        return 1

    # Parse each document
    all_entries: list[FeatureEntry] = []
    all_entries.extend(parse_capability_matrix(texts["CAPABILITY_MATRIX"]))
    all_entries.extend(parse_ga_checklist(texts["GA_CHECKLIST"]))
    all_entries.extend(parse_status_md(texts["STATUS"]))
    all_entries.extend(parse_roadmap(texts["ROADMAP"]))

    # Build index and cross-reference
    index = build_feature_index(all_entries)
    discrepancies = detect_discrepancies(index)
    sources = list(doc_paths.keys())
    unique_features = detect_cross_doc_gaps(index, sources)

    # Generate and print report
    report = format_report(all_entries, discrepancies, unique_features, index)
    print(report)

    # Exit code
    critical_count = sum(1 for d in discrepancies if d.severity == "CRITICAL")
    return 1 if critical_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
