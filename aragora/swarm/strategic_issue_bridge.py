"""Strategic issue bridge for boss-ready candidate generation.

Transforms roadmap/docs + scanner/planner signals into bounded issue candidates
aligned with active roadmap milestones. Provides deterministic heuristics with
optional LLM-assisted planning that safely falls back when unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from aragora.nomic.strategic_scanner import StrategicAssessment, StrategicFinding, StrategicScanner
from aragora.swarm.spec import SwarmSpec

logger = logging.getLogger(__name__)

_PRIORITY_PREFIX_RE = re.compile(r"^\d+\.\s+\*\*(?P<name>.+?)\*\*")
_EPIC_RE = re.compile(r"^##\s+Epic\s+\d+\s+[-\u2014]\s+(?P<name>.+)$")
_MILESTONE_RE = re.compile(r"^###\s+Milestone\s+\d+\.\d+\s+[-\u2014]\s+(?P<name>.+)$")
_TASK_RE = re.compile(r"^-\s+\[[ xX]\]\s+\*\*(?P<code>[A-Z]+-\d+)\*\*\s+(?P<title>.+)$")

EPIC_PREFIX_MAP = {
    "RS": "Reliability Substrate",
    "BC": "Bounded Autonomy Control Plane",
    "TW": "Trust-Wedge Product Loops",
    "UDW": "Unified DAG Workbench",
    "MCF": "Memory & Context Fabric",
    "DIC": "Decision Integrity Core",
}

THEME_FILE_HINTS = {
    "RS": [
        "aragora/swarm/worker_launcher.py",
        "aragora/swarm/tranche_integrate.py",
        "aragora/swarm/boss_validation.py",
        "aragora/swarm/runner_registry.py",
        "aragora/swarm/spec.py",
    ],
    "BC": [
        "aragora/swarm/session.py",
        "aragora/swarm/runner_registry.py",
        "aragora/nomic/dev_coordination.py",
        "aragora/swarm/spec.py",
    ],
    "TW": [
        "aragora/cli/commands/idea.py",
        "aragora/cli/commands/build.py",
        "aragora/swarm/spec.py",
    ],
    "UDW": [
        "aragora/pipeline/universal_node.py",
        "aragora/pipeline/graph_state.py",
        "aragora/pipeline/execution_mode.py",
    ],
    "MCF": [
        "aragora/knowledge/mound/store.py",
        "aragora/knowledge/mound/adapters/base.py",
        "aragora/memory/store.py",
    ],
    "DIC": [
        "aragora/debate/output_quality.py",
        "aragora/agents/grounded.py",
        "aragora/nomic/feedback_orchestrator.py",
    ],
}

THEME_VALIDATION = {
    "RS": "python3 -m pytest tests/swarm -q",
    "BC": "python3 -m pytest tests/swarm -q",
    "TW": "python3 -m pytest tests/cli -q",
    "UDW": "python3 -m pytest tests/pipeline -q",
    "MCF": "python3 -m pytest tests/knowledge -q",
    "DIC": "python3 -m pytest tests/debate -q",
}

FALLBACK_VALIDATION = "python3 -m pytest tests/ -q"


@dataclass(frozen=True)
class RoadmapSignalHint:
    keywords: tuple[str, ...]
    file_hints: tuple[str, ...]
    validation: str


ROADMAP_SIGNAL_HINTS = (
    RoadmapSignalHint(
        keywords=(
            "terminal-truth",
            "canonical failure taxonomy",
            "task-shape failures",
        ),
        file_hints=(
            "aragora/swarm/worker_launcher.py",
            "aragora/swarm/tranche_integrate.py",
            "aragora/swarm/boss_validation.py",
        ),
        validation=(
            "python3 -m pytest tests/swarm/test_worker_launcher.py "
            "tests/swarm/test_tranche_integrate.py tests/swarm/test_boss_validation.py -q"
        ),
    ),
    RoadmapSignalHint(
        keywords=(
            "benchmark scoring",
            "scoring lane",
            "regression guardrails",
            "guardrails in ci",
        ),
        file_hints=(
            ".github/workflows/benchmarks.yml",
            "scripts/check_benchmark_regression.py",
            "scripts/dogfood_score.py",
        ),
        validation=(
            "python3 -m pytest tests/scripts/test_run_dogfood_benchmark.py "
            "tests/scripts/test_phase0b_role_benchmark.py -q"
        ),
    ),
    RoadmapSignalHint(
        keywords=(
            "benchmark fixtures",
            "benchmark corpus",
            "rescue receipts",
            "publication-failure receipts",
        ),
        file_hints=(
            "scripts/run_dogfood_benchmark.py",
            "scripts/dogfood_score.py",
            "aragora/swarm/worker_launcher.py",
        ),
        validation=(
            "python3 -m pytest tests/scripts/test_run_dogfood_benchmark.py "
            "tests/swarm/test_worker_launcher.py -q"
        ),
    ),
    RoadmapSignalHint(
        keywords=(
            "workercontract",
            "credentialenvelope",
            "admission rules",
            "complete contracts",
        ),
        file_hints=(
            "aragora/swarm/worker_contract.py",
            "aragora/swarm/worker_launcher.py",
            "aragora/swarm/runner_registry.py",
        ),
        validation=(
            "python3 -m pytest tests/swarm/test_worker_launcher.py "
            "tests/swarm/test_runner_registry.py -q"
        ),
    ),
    RoadmapSignalHint(
        keywords=(
            "sanitizer outcomes",
            "rewritten",
            "dropped",
            "quarantined",
        ),
        file_hints=(
            "aragora/swarm/spec.py",
            "aragora/swarm/boss_validation.py",
            "aragora/swarm/prompt_refiner.py",
        ),
        validation=(
            "python3 -m pytest tests/swarm/test_spec.py tests/swarm/test_boss_validation.py -q"
        ),
    ),
    RoadmapSignalHint(
        keywords=(
            "preflight run --contract",
            "production code path",
            "receipt-backed preflight",
            "shell-only host checks",
        ),
        file_hints=(
            "aragora/swarm/preflight.py",
            "scripts/swarm_host_preflight.sh",
            "scripts/nomic/preflight.py",
        ),
        validation="python3 -m pytest tests/scripts/test_swarm_host_preflight.py -q",
    ),
    RoadmapSignalHint(
        keywords=(
            "quarantine",
            "publication failures",
            "permission mismatch",
            "rate limits",
        ),
        file_hints=(
            "aragora/swarm/tranche_integrate.py",
            "aragora/swarm/runner_registry.py",
            "aragora/swarm/boss_validation.py",
        ),
        validation=(
            "python3 -m pytest tests/swarm/test_tranche_integrate.py "
            "tests/swarm/test_runner_registry.py tests/swarm/test_boss_validation.py -q"
        ),
    ),
    RoadmapSignalHint(
        keywords=(
            "reviewable specs",
            "explicit constraints",
            "manual rewrite",
            "missing context",
        ),
        file_hints=(
            "aragora/swarm/spec.py",
            "aragora/cli/commands/spec.py",
            "aragora/prompt_engine/spec_builder.py",
        ),
        validation=("python3 -m pytest tests/swarm/test_spec.py tests/cli/test_spec_command.py -q"),
    ),
)


@dataclass(frozen=True)
class RoadmapItem:
    code: str
    title: str
    epic: str
    milestone: str
    priority_rank: int


@dataclass
class StrategicIssueCandidate:
    title: str
    description: str
    file_scope: list[str]
    validation_command: str
    acceptance_criteria: list[str]
    complexity: str
    fingerprint: str
    priority: int
    success_estimate: float
    source: str
    roadmap_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "file_scope": list(self.file_scope),
            "validation_command": self.validation_command,
            "acceptance_criteria": list(self.acceptance_criteria),
            "complexity": self.complexity,
            "fingerprint": self.fingerprint,
            "priority": self.priority,
            "success_estimate": self.success_estimate,
            "source": self.source,
            "roadmap_refs": list(self.roadmap_refs),
            "metadata": dict(self.metadata),
        }


@dataclass
class StrategicIssueBridgeConfig:
    max_issues: int = 10
    max_per_theme: int = 4
    heuristic_only: bool = False
    enable_scanner: bool = True
    enable_llm: bool = False
    context_max_lines: int = 80
    context_max_chars: int = 8000


class StrategicIssueBridge:
    """Bridge for generating strategic boss-ready issue candidates."""

    def __init__(self, repo_root: Path | str, config: StrategicIssueBridgeConfig | None = None):
        self.repo_root = Path(repo_root)
        self.config = config or StrategicIssueBridgeConfig()

    def load_context(self) -> dict[str, str]:
        canonical = _read_text(self.repo_root / "docs" / "CANONICAL_GOALS.md")
        active = _read_text(self.repo_root / "docs" / "status" / "ACTIVE_EXECUTION_ISSUES.md")
        roadmap = _read_text(self.repo_root / "ROADMAP.md")
        return {
            "canonical_goals": canonical,
            "active_execution": active,
            "roadmap": roadmap,
        }

    def build_context_summary(self, context: dict[str, str]) -> dict[str, list[str]]:
        summaries: dict[str, list[str]] = {}
        for key, text in context.items():
            summaries[key] = _summarize_text(
                text,
                max_lines=self.config.context_max_lines,
                max_chars=self.config.context_max_chars,
            )
        return summaries

    def parse_roadmap_items(self, active_text: str) -> tuple[list[RoadmapItem], list[str]]:
        priority_order: list[str] = []
        items: list[RoadmapItem] = []
        current_epic = ""
        current_milestone = ""

        for raw_line in active_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            priority_match = _PRIORITY_PREFIX_RE.match(line)
            if priority_match:
                priority_order.append(priority_match.group("name").strip())
                continue
            epic_match = _EPIC_RE.match(line)
            if epic_match:
                current_epic = epic_match.group("name").strip()
                current_milestone = ""
                continue
            milestone_match = _MILESTONE_RE.match(line)
            if milestone_match:
                current_milestone = milestone_match.group("name").strip()
                continue
            task_match = _TASK_RE.match(line)
            if task_match:
                code = task_match.group("code").strip()
                title = task_match.group("title").strip()
                epic_name = current_epic or EPIC_PREFIX_MAP.get(code.split("-")[0], "")
                priority_rank = _priority_rank(epic_name, priority_order)
                items.append(
                    RoadmapItem(
                        code=code,
                        title=title,
                        epic=epic_name,
                        milestone=current_milestone,
                        priority_rank=priority_rank,
                    )
                )
        return items, priority_order

    def generate_candidates(self) -> list[StrategicIssueCandidate]:
        context = self.load_context()
        active_text = context.get("active_execution", "")
        roadmap_items, priority_order = self.parse_roadmap_items(active_text)

        llm_candidates: list[StrategicIssueCandidate] = []
        if self.config.enable_llm and not self.config.heuristic_only:
            llm_candidates = self._generate_llm_candidates(roadmap_items, context)

        if llm_candidates:
            ranked = self._rank_candidates(llm_candidates, priority_order)
            return ranked[: self.config.max_issues]

        heuristic_candidates = self._generate_heuristic_candidates(
            roadmap_items,
            context,
            priority_order,
        )
        ranked = self._rank_candidates(heuristic_candidates, priority_order)
        limited = _limit_per_theme(ranked, self.config.max_per_theme)
        return limited[: self.config.max_issues]

    def _generate_heuristic_candidates(
        self,
        roadmap_items: list[RoadmapItem],
        context: dict[str, str],
        priority_order: list[str],
    ) -> list[StrategicIssueCandidate]:
        candidates: list[StrategicIssueCandidate] = []

        for item in roadmap_items:
            if len(candidates) >= self.config.max_issues:
                break
            candidate = self._candidate_from_roadmap_item(item, context)
            if candidate:
                candidates.append(candidate)

        if self.config.enable_scanner:
            scan_candidates = self._candidates_from_scanner(priority_order)
            candidates.extend(scan_candidates)

        return self._dedupe_candidates(candidates)

    def _candidate_from_roadmap_item(
        self, item: RoadmapItem, context: dict[str, str]
    ) -> StrategicIssueCandidate | None:
        prefix = item.code.split("-")[0]
        hint = _roadmap_signal_hint(item)
        theme_files: list[str] = []
        validation = THEME_VALIDATION.get(prefix, FALLBACK_VALIDATION)
        if hint is not None:
            theme_files = _existing_paths(self.repo_root, hint.file_hints)
            if not theme_files:
                theme_files = list(hint.file_hints[:3])
            validation = hint.validation
        else:
            theme_files = _existing_paths(self.repo_root, THEME_FILE_HINTS.get(prefix, []))
        if not theme_files:
            theme_files = list(THEME_FILE_HINTS.get(prefix, [])[:2])
        file_scope = _sanitize_file_scope(theme_files)
        if not file_scope:
            return None

        acceptance = _default_acceptance_criteria(item, validation, file_scope)

        description = _compose_description(item, context)
        priority = _priority_from_rank(item.priority_rank)
        success_estimate = _success_estimate(item, file_scope)
        fingerprint = _fingerprint("roadmap", item.code, item.title, file_scope)

        return StrategicIssueCandidate(
            title=f"{item.code}: {item.title}",
            description=description,
            file_scope=file_scope,
            validation_command=validation,
            acceptance_criteria=acceptance,
            complexity=_complexity_from_title(item.title),
            fingerprint=fingerprint,
            priority=priority,
            success_estimate=success_estimate,
            source="roadmap",
            roadmap_refs=[item.code],
            metadata={
                "epic": item.epic,
                "milestone": item.milestone,
                "priority_rank": item.priority_rank,
                "theme": prefix,
            },
        )

    def _candidates_from_scanner(self, priority_order: list[str]) -> list[StrategicIssueCandidate]:
        assessment = self._scan_repo()
        if assessment is None:
            return []

        candidates: list[StrategicIssueCandidate] = []
        for finding in assessment.findings[: self.config.max_issues]:
            candidate = self._candidate_from_finding(finding, priority_order)
            if candidate:
                candidates.append(candidate)
        return candidates

    def _scan_repo(self) -> StrategicAssessment | None:
        try:
            scanner = StrategicScanner(repo_path=self.repo_root)
            objective = "reliability autonomy trust receipts"
            return scanner.scan(objective=objective)
        except Exception as exc:
            logger.warning("strategic_scan_failed: %s", exc)
            return None

    def _candidate_from_finding(
        self, finding: StrategicFinding, priority_order: list[str]
    ) -> StrategicIssueCandidate | None:
        if not finding.file_path:
            return None
        file_scope = _sanitize_file_scope([finding.file_path])
        if not file_scope:
            return None
        prefix = _prefix_for_path(finding.file_path)
        validation = THEME_VALIDATION.get(prefix, FALLBACK_VALIDATION)
        acceptance = [
            finding.suggested_action,
            f"Evidence captured: {finding.evidence}",
            f"Run and satisfy: {validation}",
        ]
        title = f"{_theme_label(prefix)}: {finding.suggested_action}"
        description = (
            f"Strategic scanner flagged {finding.category} in {finding.file_path}. "
            f"{finding.description}.\n\nEvidence: {finding.evidence}\n"
        )
        fingerprint = _fingerprint("scanner", finding.file_path, finding.description, file_scope)
        priority = _priority_from_rank(_priority_rank(_theme_label(prefix), priority_order))
        success_estimate = 0.55 if finding.severity in {"high", "critical"} else 0.7

        return StrategicIssueCandidate(
            title=title,
            description=description,
            file_scope=file_scope,
            validation_command=validation,
            acceptance_criteria=acceptance,
            complexity=_complexity_from_finding(finding),
            fingerprint=fingerprint,
            priority=priority,
            success_estimate=success_estimate,
            source="scanner",
            roadmap_refs=[prefix] if prefix else [],
            metadata={
                "category": finding.category,
                "severity": finding.severity,
                "track": finding.track,
                "theme": prefix,
            },
        )

    def _generate_llm_candidates(
        self,
        roadmap_items: list[RoadmapItem],
        context: dict[str, str],
    ) -> list[StrategicIssueCandidate]:
        if not _llm_keys_present():
            return []
        try:
            from aragora.nomic.meta_planner import MetaPlanner, MetaPlannerConfig, Track
        except Exception:
            return []

        objective = _objective_from_context(context)
        config = MetaPlannerConfig(scan_mode=False, quick_mode=False)
        planner = MetaPlanner(config=config)

        async def _run() -> list[Any]:
            return await planner.prioritize_work(
                objective=objective,
                available_tracks=list(Track),
            )

        try:
            goals = asyncio.run(_run())
        except RuntimeError:
            return []
        except Exception as exc:
            logger.warning("llm_planner_failed: %s", exc)
            return []

        candidates: list[StrategicIssueCandidate] = []
        for goal in goals:
            candidate = _candidate_from_goal(goal, self.repo_root, roadmap_items)
            if candidate:
                candidates.append(candidate)
        return candidates

    def _rank_candidates(
        self, candidates: list[StrategicIssueCandidate], priority_order: list[str]
    ) -> list[StrategicIssueCandidate]:
        def score(candidate: StrategicIssueCandidate) -> tuple[int, int, float, str]:
            is_roadmap = 0 if candidate.source == "roadmap" else 1
            rank = _priority_rank(candidate.metadata.get("epic", ""), priority_order)
            return (is_roadmap, rank, -candidate.success_estimate, candidate.title)

        return sorted(candidates, key=score)

    def _dedupe_candidates(
        self, candidates: Iterable[StrategicIssueCandidate]
    ) -> list[StrategicIssueCandidate]:
        seen: set[str] = set()
        result: list[StrategicIssueCandidate] = []
        for candidate in candidates:
            if candidate.fingerprint in seen:
                continue
            seen.add(candidate.fingerprint)
            result.append(candidate)
        return result


def _read_text(path: Path, max_chars: int = 200000) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _summarize_text(text: str, max_lines: int, max_chars: int) -> list[str]:
    if not text:
        return []
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    summary: list[str] = []
    total_chars = 0
    for line in lines:
        if len(summary) >= max_lines:
            break
        if total_chars + len(line) > max_chars:
            break
        summary.append(line)
        total_chars += len(line)
    return summary


def _priority_rank(epic_name: str, priority_order: list[str]) -> int:
    if not epic_name:
        return len(priority_order) + 1
    try:
        return priority_order.index(epic_name) + 1
    except ValueError:
        return len(priority_order) + 1


def _priority_from_rank(rank: int) -> int:
    if rank <= 2:
        return 1
    if rank <= 4:
        return 2
    if rank <= 6:
        return 3
    return 4


def _complexity_from_title(title: str) -> str:
    lowered = title.lower()
    if any(word in lowered for word in ("define", "introduce", "build", "replace")):
        return "medium"
    if any(word in lowered for word in ("add", "capture", "surface")):
        return "low"
    return "medium"


def _complexity_from_finding(finding: StrategicFinding) -> str:
    if finding.severity == "critical":
        return "high"
    if finding.severity == "high":
        return "medium"
    return "low"


def _success_estimate(item: RoadmapItem, file_scope: list[str]) -> float:
    base = 0.6
    if item.priority_rank <= 2:
        base += 0.1
    if len(file_scope) <= 2:
        base += 0.05
    return min(base, 0.85)


def _fingerprint(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts if str(part).strip())
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _existing_paths(repo_root: Path, hints: Iterable[str]) -> list[str]:
    existing: list[str] = []
    for hint in hints:
        if not hint:
            continue
        path = repo_root / hint
        if path.exists():
            existing.append(hint)
    return existing


def _sanitize_file_scope(paths: Iterable[str]) -> list[str]:
    result: list[str] = []
    for path in paths:
        normalized = SwarmSpec.sanitize_file_scope_entry(path)
        if normalized:
            result.append(normalized)
    return list(dict.fromkeys(result))


def _default_acceptance_criteria(
    item: RoadmapItem, validation: str, file_scope: list[str]
) -> list[str]:
    scoped_files = ", ".join(file_scope[:3])
    criteria = [f"Deliver milestone {item.code} scope: {item.title}"]
    if scoped_files:
        criteria.append(f"Keep the diff focused to: {scoped_files}")
    criteria.append(f"Run and satisfy: {validation}")
    return criteria


def _roadmap_signal_hint(item: RoadmapItem) -> RoadmapSignalHint | None:
    haystack = " ".join(
        part.strip().lower()
        for part in (item.code, item.title, item.milestone, item.epic)
        if part and part.strip()
    )
    for hint in ROADMAP_SIGNAL_HINTS:
        if any(keyword in haystack for keyword in hint.keywords):
            return hint
    return None


def _compose_description(item: RoadmapItem, context: dict[str, str]) -> str:
    roadmap_excerpt = _extract_excerpt(context.get("roadmap", ""), item.code)
    active_excerpt = _extract_excerpt(context.get("active_execution", ""), item.code)
    pieces = [f"Roadmap target: {item.title}."]
    if item.epic:
        pieces.append(f"Epic: {item.epic}.")
    if item.milestone:
        pieces.append(f"Milestone: {item.milestone}.")
    if roadmap_excerpt:
        pieces.append(f"Roadmap context: {roadmap_excerpt}")
    if active_excerpt:
        pieces.append(f"Execution context: {active_excerpt}")
    return " ".join(pieces)


def _extract_excerpt(text: str, code: str) -> str:
    if not text or not code:
        return ""
    for line in text.splitlines():
        if code in line:
            return " ".join(line.strip().split())
    return ""


def _prefix_for_path(path: str) -> str:
    lower = path.lower()
    if "/swarm/" in lower:
        return "RS"
    if "/pipeline/" in lower:
        return "UDW"
    if "/knowledge/" in lower or "/memory/" in lower:
        return "MCF"
    if "/debate/" in lower or "/agents/" in lower or "/nomic/" in lower:
        return "DIC"
    if "/cli/" in lower:
        return "TW"
    return "RS"


def _theme_label(prefix: str) -> str:
    return EPIC_PREFIX_MAP.get(prefix, "Reliability Substrate")


def _objective_from_context(context: dict[str, str]) -> str:
    canonical = context.get("canonical_goals", "")
    roadmap = context.get("roadmap", "")
    lines: list[str] = []
    for text in (canonical, roadmap):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## ") or stripped.startswith("- "):
                lines.append(stripped.lstrip("- "))
            if len(lines) >= 15:
                break
        if len(lines) >= 15:
            break
    return " ".join(lines) or "Improve reliability and autonomy for bounded execution"


def _candidate_from_goal(
    goal: Any, repo_root: Path, roadmap_items: list[RoadmapItem]
) -> StrategicIssueCandidate | None:
    description = str(getattr(goal, "description", "")).strip()
    if not description:
        return None
    file_scope = _sanitize_file_scope(getattr(goal, "file_hints", []))
    if not file_scope:
        file_scope = _sanitize_file_scope(
            _existing_paths(repo_root, THEME_FILE_HINTS.get("RS", []))
        )
    validation = FALLBACK_VALIDATION
    acceptance = ["Deliver the goal outcome", f"Run and satisfy: {validation}"]
    title = str(getattr(goal, "description", ""))[:80]
    fingerprint = _fingerprint("goal", title, file_scope)

    return StrategicIssueCandidate(
        title=title,
        description=description,
        file_scope=file_scope,
        validation_command=validation,
        acceptance_criteria=acceptance,
        complexity="medium",
        fingerprint=fingerprint,
        priority=int(getattr(goal, "priority", 3)),
        success_estimate=0.6,
        source="planner",
        roadmap_refs=_match_goal_to_roadmap(title, roadmap_items),
        metadata={"track": getattr(goal, "track", "")},
    )


def _match_goal_to_roadmap(title: str, roadmap_items: list[RoadmapItem]) -> list[str]:
    lowered = title.lower()
    for item in roadmap_items:
        if item.code.lower() in lowered or item.title.lower() in lowered:
            return [item.code]
    return []


def _llm_keys_present() -> bool:
    keys = (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "XAI_API_KEY",
        "GROK_API_KEY",
    )
    return any(os.environ.get(key) for key in keys)


def _limit_per_theme(
    candidates: Iterable[StrategicIssueCandidate], max_per_theme: int
) -> list[StrategicIssueCandidate]:
    if max_per_theme <= 0:
        return list(candidates)
    counts: dict[str, int] = {}
    result: list[StrategicIssueCandidate] = []
    for candidate in candidates:
        theme = str(candidate.metadata.get("theme") or "")
        if theme:
            counts.setdefault(theme, 0)
            if counts[theme] >= max_per_theme:
                continue
            counts[theme] += 1
        result.append(candidate)
    return result


def dump_candidates_json(candidates: list[StrategicIssueCandidate]) -> str:
    payload = [candidate.to_dict() for candidate in candidates]
    return json.dumps(payload, indent=2)


def format_candidates_markdown(candidates: list[StrategicIssueCandidate]) -> str:
    lines: list[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        lines.append(f"{idx}. {candidate.title}")
        lines.append(
            f"   Priority: {candidate.priority}  Success: {candidate.success_estimate:.2f}"
        )
        lines.append(f"   File scope: {', '.join(candidate.file_scope)}")
        lines.append(f"   Validation: {candidate.validation_command}")
        lines.append("   Acceptance:")
        for criterion in candidate.acceptance_criteria:
            lines.append(f"   - {criterion}")
        lines.append("")
    return "\n".join(lines).strip()


__all__ = [
    "StrategicIssueBridge",
    "StrategicIssueBridgeConfig",
    "StrategicIssueCandidate",
    "RoadmapItem",
    "dump_candidates_json",
    "format_candidates_markdown",
]
