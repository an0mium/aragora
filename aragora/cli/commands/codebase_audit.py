"""Staged codebase audit command.

Runs a deterministic multi-pass pipeline before invoking LLM audit modes:

1. Triage: strip vendored/generated noise and map bespoke code.
2. Surface: identify trust boundaries and rank risky files.
3. Interrogate: run Deep Audit on the highest-risk slice.
4. Blast radius: translate technical risks into plain-English failure modes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".aragora",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    "coverage",
    ".next",
    ".turbo",
    ".parcel-cache",
    "vendor",
    "third_party",
    "sdk",
    "docs",
    "examples",
    "benchmarks",
}

IGNORED_SUFFIXES = {
    ".lock",
    ".pyc",
    ".pyo",
    ".so",
    ".dylib",
    ".dll",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".mp3",
    ".mp4",
    ".mov",
    ".zip",
    ".tar",
    ".gz",
}

TEXT_FILE_SUFFIXES = {
    ".py",
    ".pyi",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".swift",
    ".rb",
    ".php",
    ".sol",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".sql",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".md",
    ".txt",
    ".ini",
    ".cfg",
    ".conf",
    ".html",
    ".css",
    ".scss",
}

BOUNDARY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "llm_system": (
        "create_agent",
        "Arena(",
        "DebateProtocol",
        "subprocess",
        "os.system",
        "asyncio.create_subprocess",
        "webhook",
        "launch",
        "dispatch",
        "worker",
        "apply_patch",
    ),
    "crypto_fiat": (
        "web3",
        "erc20",
        "erc721",
        "erc8004",
        "sign_transaction",
        "transfer(",
        "wallet",
        "private_key",
        "solidity",
        "ethers",
        "contract",
    ),
    "user_perimeter": (
        "BaseHandler",
        "can_handle(",
        "parse_qs",
        "request",
        "headers",
        "Authorization",
        "oauth",
        "slack",
        "gmail",
        "callback",
        "json_response",
        "error_response",
    ),
}

PATH_HINTS: dict[str, tuple[str, ...]] = {
    "llm_system": ("aragora/swarm/", "aragora/debate/", "aragora/agents/", "scripts/"),
    "crypto_fiat": ("contracts/", "aragora/settlement/", "aragora/payments/"),
    "user_perimeter": ("aragora/server/", "aragora/connectors/", "aragora/cli/"),
}

SURFACE_IGNORED_PREFIXES = (
    "tests/",
    "sdk/",
    "docs/",
    "examples/",
    "benchmarks/",
)


@dataclass
class FileRecord:
    path: str
    loc: int
    boundary_scores: dict[str, int] = field(default_factory=dict)
    dominant_boundary: str = "general"


@dataclass
class DirectoryRecord:
    path: str
    loc: int
    files: int
    description: str


def _should_ignore(path: Path, repo_root: Path) -> bool:
    parts = path.relative_to(repo_root).parts
    if any(part in IGNORED_DIRS for part in parts):
        return True
    if path.name.startswith(".") and path.name not in {".env.example", ".env.template"}:
        return True
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return True
    if path.name in {"uv.lock", "yarn.lock", "package-lock.json", "pnpm-lock.yaml"}:
        return True
    if path.stat().st_size > 1_500_000:
        return True
    return False


def _count_loc(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _directory_description(path: str) -> str:
    normalized = path.replace("\\", "/")
    mapping = (
        ("aragora/agents", "Agent implementations and provider adapters."),
        ("aragora/debate", "Core multi-agent debate orchestration and consensus logic."),
        ("aragora/swarm", "Autonomous worker orchestration, dispatch, and control-plane logic."),
        ("aragora/server", "HTTP/API handlers, auth boundaries, and external request ingress."),
        ("aragora/nomic", "Self-improvement, planning, and campaign loop infrastructure."),
        ("aragora/cli", "Human and automation entry points for running Aragora workflows."),
        ("contracts", "Deterministic smart-contract logic and chain-facing execution surface."),
        ("deploy", "Deployment, infrastructure, and runtime rollout logic."),
        ("scripts", "Operational scripts and one-off automation entry points."),
        ("tests", "Regression and behavior coverage for the codebase."),
        ("aragora/live", "Frontend/runtime UI surface."),
    )
    for prefix, description in mapping:
        if normalized == prefix or normalized.startswith(prefix + "/"):
            return description
    return "Hand-written application logic without a more specific domain hint."


def collect_repo_triage(repo_root: Path, *, max_dirs: int = 25) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    file_records: list[FileRecord] = []
    loc_by_dir: defaultdict[str, int] = defaultdict(int)
    files_by_dir: defaultdict[str, int] = defaultdict(int)
    ext_counter: Counter[str] = Counter()

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if _should_ignore(path, repo_root):
            continue
        if path.suffix.lower() not in TEXT_FILE_SUFFIXES and path.suffix:
            continue
        rel_path = path.relative_to(repo_root).as_posix()
        text = _read_text(path)
        loc = _count_loc(text)
        if loc == 0:
            continue
        file_records.append(FileRecord(path=rel_path, loc=loc))
        parent = str(Path(rel_path).parent).replace("\\", "/")
        loc_by_dir[parent] += loc
        files_by_dir[parent] += 1
        ext_counter[path.suffix.lower() or "[no-ext]"] += 1

    directories = [
        DirectoryRecord(
            path=directory,
            loc=loc,
            files=files_by_dir[directory],
            description=_directory_description(directory),
        )
        for directory, loc in loc_by_dir.items()
    ]
    directories.sort(key=lambda item: (-item.loc, item.path))
    file_records.sort(key=lambda item: (-item.loc, item.path))

    return {
        "repo_root": str(repo_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bespoke_loc": sum(item.loc for item in file_records),
        "scanned_files": len(file_records),
        "top_extensions": ext_counter.most_common(12),
        "directories": [asdict(item) for item in directories[:max_dirs]],
        "largest_files": [asdict(item) for item in file_records[:15]],
    }


def _score_boundary(path: str, text: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    lowered = text.lower()
    for boundary, keywords in BOUNDARY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            score += lowered.count(keyword.lower())
        for path_hint in PATH_HINTS[boundary]:
            if path.startswith(path_hint):
                score += 3
        if boundary == "crypto_fiat" and path.endswith(".sol"):
            score += 5
        scores[boundary] = score
    return scores


def _should_ignore_surface_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized.startswith(SURFACE_IGNORED_PREFIXES):
        return True
    if "/generated/" in normalized or normalized.endswith(("/openapi.py", "/openapi.ts")):
        return True
    return False


def identify_threat_surface(
    repo_root: Path,
    *,
    top_files: int = 12,
    max_preview_chars: int = 4000,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    ranked: list[FileRecord] = []
    boundaries: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if _should_ignore(path, repo_root):
            continue
        rel_path = path.relative_to(repo_root).as_posix()
        if _should_ignore_surface_path(rel_path):
            continue
        if path.suffix.lower() not in TEXT_FILE_SUFFIXES and path.suffix:
            continue
        text = _read_text(path)
        if not text.strip():
            continue
        scores = _score_boundary(rel_path, text)
        dominant_boundary = max(scores, key=scores.get)
        dominant_score = scores[dominant_boundary]
        if dominant_score <= 0:
            continue
        record = FileRecord(
            path=rel_path,
            loc=_count_loc(text),
            boundary_scores=scores,
            dominant_boundary=dominant_boundary,
        )
        ranked.append(record)
        boundaries[dominant_boundary].append(
            {
                "path": rel_path,
                "score": dominant_score,
                "loc": record.loc,
                "preview": text[:max_preview_chars],
            }
        )

    ranked.sort(
        key=lambda item: (
            -max(item.boundary_scores.values(), default=0),
            -item.loc,
            item.path,
        )
    )
    for items in boundaries.values():
        items.sort(key=lambda item: (-int(item["score"]), -int(item["loc"]), str(item["path"])))

    top = ranked[:top_files]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_risk_files": [
            {
                "path": item.path,
                "loc": item.loc,
                "dominant_boundary": item.dominant_boundary,
                "score": max(item.boundary_scores.values(), default=0),
                "boundary_scores": item.boundary_scores,
            }
            for item in top
        ],
        "boundaries": {
            boundary: items[: min(len(items), top_files)] for boundary, items in boundaries.items()
        },
    }


def _build_deep_audit_inputs(
    *,
    repo_root: Path,
    triage: dict[str, Any],
    surface: dict[str, Any],
    top_files: int,
    max_file_chars: int,
) -> tuple[str, str]:
    top_targets = list(surface.get("top_risk_files", []))[:top_files]
    target_lines = []
    context_parts = [
        "Repository triage summary:",
        json.dumps(
            {
                "repo_root": triage.get("repo_root"),
                "bespoke_loc": triage.get("bespoke_loc"),
                "scanned_files": triage.get("scanned_files"),
                "top_directories": triage.get("directories", [])[:10],
            },
            indent=2,
        ),
        "\nThreat surface ranking:",
        json.dumps(top_targets, indent=2),
    ]

    for item in top_targets:
        path = str(item.get("path", "")).strip()
        if not path:
            continue
        boundary = str(item.get("dominant_boundary", "general"))
        score = int(item.get("score", 0))
        target_lines.append(f"- {path} [{boundary}] score={score}")
        abs_path = repo_root / path
        try:
            snippet = _read_text(abs_path)[:max_file_chars]
        except OSError:
            snippet = ""
        if snippet:
            context_parts.append(f"\nFile: {path}\n```text\n{snippet}\n```")

    task = (
        "Perform a staged architectural and security audit of this repository slice. "
        "Attack the weakest premise first. Separate descriptive reality from README-level claims. "
        "Focus on trust boundaries, prompt-injection risk, arbitrary code execution, state drift, "
        "and smart-contract or external action blast radius.\n\n"
        "Highest-risk files:\n"
        + ("\n".join(target_lines) if target_lines else "- No high-risk files identified")
    )
    return task, "\n".join(context_parts)


def _blast_radius_for_boundary(boundary: str) -> tuple[str, str]:
    if boundary == "crypto_fiat":
        return (
            "A compromised or hallucinating agent can trigger irreversible financial or on-chain actions.",
            "Strict execution gating, deterministic transaction policies, and offline review before signing.",
        )
    if boundary == "user_perimeter":
        return (
            "External input can cross the trust boundary into privileged handlers or LLM prompts and drive exfiltration or auth mistakes.",
            "Input normalization, authn/authz checks, and prompt-isolation before model exposure.",
        )
    return (
        "An agent or orchestration bug can execute tools, mutate state, or wedge long-running control-plane loops.",
        "Sandboxed execution, bounded tool contracts, and explicit state reconciliation.",
    )


def build_blast_radius(surface: dict[str, Any]) -> dict[str, Any]:
    items = []
    for entry in surface.get("top_risk_files", []):
        boundary = str(entry.get("dominant_boundary", "llm_system"))
        worst_case, crux = _blast_radius_for_boundary(boundary)
        items.append(
            {
                "path": entry.get("path"),
                "boundary": boundary,
                "score": entry.get("score"),
                "worst_case": worst_case,
                "main_crux": crux,
            }
        )
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "items": items}


def _choose_verdict(surface: dict[str, Any], deep_audit_result: dict[str, Any] | None) -> str:
    top_files = list(surface.get("top_risk_files", []))
    if not top_files:
        return "sound"
    if any(str(item.get("dominant_boundary")) == "crypto_fiat" for item in top_files):
        return "unsound"
    if deep_audit_result and deep_audit_result.get("status") == "completed":
        findings = deep_audit_result.get("verdict", {}).get("findings", [])
        if len(findings) >= 3:
            return "partly_sound"
    return "partly_sound"


def _summary_markdown(
    *,
    triage: dict[str, Any],
    surface: dict[str, Any],
    deep_audit_result: dict[str, Any] | None,
    blast_radius: dict[str, Any],
) -> str:
    verdict = _choose_verdict(surface, deep_audit_result)
    lines = [
        "# Staged Codebase Audit",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"- Bespoke LOC: `{triage.get('bespoke_loc', 0)}`",
        f"- Scanned files: `{triage.get('scanned_files', 0)}`",
        f"- Top risk files: `{len(surface.get('top_risk_files', []))}`",
    ]
    if deep_audit_result:
        lines.append(f"- Deep audit stage: `{deep_audit_result.get('status', 'unknown')}`")
    lines.extend(["", "## Top Risk Files"])
    for item in surface.get("top_risk_files", [])[:8]:
        lines.append(f"- `{item['path']}` [{item['dominant_boundary']}] score={item['score']}")
    lines.extend(["", "## Blast Radius"])
    for item in blast_radius.get("items", [])[:6]:
        lines.append(f"- `{item['path']}`: {item['worst_case']}")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _make_artifact_dir(repo_root: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return repo_root / ".aragora" / "codebase-audit" / timestamp


def _build_agents(agents_str: str) -> tuple[list[Any], list[str]]:
    from aragora.agents.base import create_agent

    created = []
    errors: list[str] = []
    for idx, spec in enumerate([item.strip() for item in agents_str.split(",") if item.strip()]):
        role = ["analyst", "critic", "synthesizer"][idx % 3]
        try:
            created.append(
                create_agent(
                    spec,
                    name=f"{spec}-{role}",
                    role=role,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{spec}: {exc}")
    return created, errors


async def _run_interrogate_stage(
    *,
    repo_root: Path,
    triage: dict[str, Any],
    surface: dict[str, Any],
    agents_str: str,
    top_files: int,
    max_file_chars: int,
    dry_run: bool,
) -> dict[str, Any]:
    from aragora.modes.deep_audit import CODE_ARCHITECTURE_AUDIT, run_deep_audit

    task, context = _build_deep_audit_inputs(
        repo_root=repo_root,
        triage=triage,
        surface=surface,
        top_files=top_files,
        max_file_chars=max_file_chars,
    )
    result: dict[str, Any] = {"task": task, "context_preview": context[:4000]}
    if dry_run:
        result["status"] = "skipped_dry_run"
        return result

    agents, errors = _build_agents(agents_str)
    if len(agents) < 1:
        result["status"] = "skipped_no_agents"
        result["errors"] = errors
        return result

    verdict = await run_deep_audit(
        task=task,
        agents=agents,
        context=context,
        config=CODE_ARCHITECTURE_AUDIT,
    )
    result["status"] = "completed"
    result["errors"] = errors
    result["verdict"] = {
        "recommendation": verdict.recommendation,
        "confidence": verdict.confidence,
        "findings": [asdict(item) for item in verdict.findings],
        "unanimous_issues": verdict.unanimous_issues,
        "split_opinions": verdict.split_opinions,
        "risk_areas": verdict.risk_areas,
        "citations": verdict.citations,
        "cross_examination_notes": verdict.cross_examination_notes,
    }
    return result


def cmd_codebase_audit(args: argparse.Namespace) -> int:
    repo_root = Path(getattr(args, "repo", ".")).expanduser().resolve()
    artifact_dir = _make_artifact_dir(repo_root, getattr(args, "artifact_dir", None))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    triage = collect_repo_triage(repo_root, max_dirs=getattr(args, "max_dirs", 25))
    surface = identify_threat_surface(
        repo_root,
        top_files=getattr(args, "top_files", 12),
        max_preview_chars=getattr(args, "max_preview_chars", 4000),
    )
    deep_audit_result = asyncio.run(
        _run_interrogate_stage(
            repo_root=repo_root,
            triage=triage,
            surface=surface,
            agents_str=getattr(args, "agents", ""),
            top_files=getattr(args, "top_files", 12),
            max_file_chars=getattr(args, "max_file_chars", 12000),
            dry_run=getattr(args, "dry_run", False),
        )
    )
    blast_radius = build_blast_radius(surface)

    summary = {
        "repo_root": str(repo_root),
        "artifact_dir": str(artifact_dir),
        "verdict": _choose_verdict(surface, deep_audit_result),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "triage": {
            "bespoke_loc": triage.get("bespoke_loc"),
            "scanned_files": triage.get("scanned_files"),
        },
        "surface": {
            "top_risk_files": surface.get("top_risk_files", [])[:8],
        },
        "deep_audit": {
            "status": deep_audit_result.get("status"),
            "errors": deep_audit_result.get("errors", []),
        },
    }

    _write_json(artifact_dir / "triage.json", triage)
    _write_json(artifact_dir / "surface.json", surface)
    _write_json(artifact_dir / "interrogate.json", deep_audit_result)
    _write_json(artifact_dir / "blast_radius.json", blast_radius)
    _write_json(artifact_dir / "run.json", summary)
    (artifact_dir / "summary.md").write_text(
        _summary_markdown(
            triage=triage,
            surface=surface,
            deep_audit_result=deep_audit_result,
            blast_radius=blast_radius,
        )
    )

    if getattr(args, "json", False):
        print(json.dumps(summary, indent=2))
    else:
        print(f"artifact_dir={artifact_dir}")
        print(
            "verdict={verdict} bespoke_loc={loc} scanned_files={files} deep_audit={status}".format(
                verdict=summary["verdict"],
                loc=triage.get("bespoke_loc", 0),
                files=triage.get("scanned_files", 0),
                status=deep_audit_result.get("status", "unknown"),
            )
        )
        for item in surface.get("top_risk_files", [])[:5]:
            print(
                f"- {item['path']} [{item['dominant_boundary']}] score={item['score']} loc={item['loc']}"
            )
    return 0
