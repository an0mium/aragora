#!/usr/bin/env python3
"""Mine recurring harness weaknesses from gate receipts and conductor traces."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence


SEVERITY_WEIGHTS = {"P0": 13, "P1": 8, "P2": 5, "P3": 2, "INFO": 1}
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET))\s*[:=]\s*\S+"),
]
SEVERITY_RE = re.compile(r"\[(P[0-3])\]|\b(P[0-3])\b")
TAXONOMY_HEADING_RE = re.compile(r"^#{2,4}\s+(\d+)\.\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}")


@dataclass(frozen=True)
class WeaknessExample:
    id: str
    source: str
    target: str
    created_at: str
    severity: str
    text: str
    url: str | None = None


@dataclass(frozen=True)
class ClassifiedExample:
    example: WeaknessExample
    taxonomy_id: str
    finding_class: str
    causal_mechanism: str
    harness_surface: str
    emergent_cluster: str
    evidence_summary: str

    @property
    def id(self) -> str:
        return self.example.id

    @property
    def severity(self) -> str:
        return self.example.severity

    @property
    def target(self) -> str:
        return self.example.target

    @property
    def url(self) -> str | None:
        return self.example.url


@dataclass(frozen=True)
class WeaknessCluster:
    pass_name: str
    cluster_key: str
    title: str
    finding_class: str
    causal_mechanism: str
    harness_surfaces: list[str]
    rank_score: int
    examples: list[ClassifiedExample]

    @property
    def example_count(self) -> int:
        return len(self.examples)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["example_count"] = self.example_count
        return payload


@dataclass(frozen=True)
class MiningResult:
    ok: bool
    generated_at: str
    input_count: int
    classified_count: int
    clusters: list[WeaknessCluster]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "generated_at": self.generated_at,
            "input_count": self.input_count,
            "classified_count": self.classified_count,
            "cluster_count": len(self.clusters),
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "warnings": self.warnings,
        }


def parse_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _redact(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def _severity_from_text(text: str, explicit: str | None = None) -> str:
    if explicit:
        normalized = explicit.strip().upper()
        if normalized in SEVERITY_WEIGHTS:
            return normalized
    match = SEVERITY_RE.search(text)
    if not match:
        return "INFO"
    return next(group for group in match.groups() if group)


def _target_label(value: Any) -> str:
    if isinstance(value, dict):
        if value.get("pr") is not None:
            return f"PR #{value['pr']}"
        if value.get("issue") is not None:
            return f"issue #{value['issue']}"
        if value.get("branch"):
            return f"branch {value['branch']}"
        return json.dumps(value, sort_keys=True)
    if value is None:
        return "unknown"
    return str(value)


def _within_window(created_at: str, *, since_days: int, now: datetime) -> bool:
    try:
        parsed = parse_timestamp(created_at)
    except ValueError:
        return True
    return parsed >= now - timedelta(days=since_days)


def _ledger_text(record: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("blocker_class", "result", "progress_kind", "action"):
        value = record.get(key)
        if value:
            pieces.append(f"{key}: {value}")
    blockers = record.get("blockers")
    if isinstance(blockers, list):
        pieces.extend(str(item) for item in blockers)
    elif blockers:
        pieces.append(str(blockers))
    return "\n".join(pieces).strip()


def _comment_is_relevant(body: str) -> bool:
    markers = (
        "CHANGES-REQUESTED",
        "[P0]",
        "[P1]",
        "[P2]",
        "[P3]",
        "blocker",
        "dissent",
        "park record",
    )
    upper_body = body.upper()
    return any(marker.upper() in upper_body for marker in markers)


def _load_input_examples(path: Path) -> list[WeaknessExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of examples")
    examples: list[WeaknessExample] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path} example {index} is not an object")
        text = _redact(str(item.get("text") or item.get("body") or ""))
        if not text.strip():
            continue
        examples.append(
            WeaknessExample(
                id=str(item.get("id") or f"input:{index}"),
                source=str(item.get("source") or "input"),
                target=_target_label(item.get("target") or item.get("pr") or item.get("issue")),
                created_at=str(item.get("created_at") or item.get("timestamp") or ""),
                severity=_severity_from_text(text, str(item.get("severity") or "")),
                text=text,
                url=str(item["url"]) if item.get("url") else None,
            )
        )
    return examples


def _load_ledger_examples(path: Path, *, since_days: int, now: datetime) -> list[WeaknessExample]:
    examples: list[WeaknessExample] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        created_at = str(record.get("timestamp") or record.get("generated_at") or "")
        if created_at and not _within_window(created_at, since_days=since_days, now=now):
            continue
        text = _redact(_ledger_text(record))
        if not text:
            continue
        examples.append(
            WeaknessExample(
                id=f"ledger:{path.name}:{line_no}",
                source="ledger",
                target=_target_label(record.get("target") or record.get("pr")),
                created_at=created_at,
                severity=_severity_from_text(text, str(record.get("severity") or "")),
                text=text,
                url=None,
            )
        )
    return examples


def _load_comment_examples(path: Path, *, since_days: int, now: datetime) -> list[WeaknessExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        comments = payload.get("comments") or payload.get("items") or []
    else:
        comments = payload
    if not isinstance(comments, list):
        raise ValueError(f"{path} must contain a list or an object with comments/items")
    examples: list[WeaknessExample] = []
    for index, item in enumerate(comments, start=1):
        if not isinstance(item, dict):
            continue
        body = str(item.get("body") or item.get("text") or "")
        if not _comment_is_relevant(body):
            continue
        created_at = str(item.get("created_at") or item.get("createdAt") or "")
        if created_at and not _within_window(created_at, since_days=since_days, now=now):
            continue
        text = _redact(body)
        examples.append(
            WeaknessExample(
                id=str(item.get("id") or f"comment:{item.get('pr') or 'unknown'}:{index}"),
                source="github_comment",
                target=_target_label(item.get("target") or item.get("pr") or item.get("issue")),
                created_at=created_at,
                severity=_severity_from_text(text, str(item.get("severity") or "")),
                text=text,
                url=str(item["url"]) if item.get("url") else None,
            )
        )
    return examples


def collect_examples(
    *,
    input_json: Path | None = None,
    ledger_paths: Sequence[Path] = (),
    comment_json_paths: Sequence[Path] = (),
    since_days: int = 30,
    now: datetime | None = None,
) -> list[WeaknessExample]:
    now = now or datetime.now(UTC)
    examples: list[WeaknessExample] = []
    if input_json is not None:
        examples.extend(_load_input_examples(input_json))
    for path in ledger_paths:
        if path.exists():
            examples.extend(_load_ledger_examples(path, since_days=since_days, now=now))
    for path in comment_json_paths:
        if path.exists():
            examples.extend(_load_comment_examples(path, since_days=since_days, now=now))
    return examples


def load_taxonomy(path: Path) -> dict[str, str]:
    taxonomy: dict[str, str] = {}
    if not path.exists():
        return taxonomy
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TAXONOMY_HEADING_RE.match(line)
        if match:
            taxonomy[match.group(1)] = match.group(2).strip()
    return taxonomy


def _normalize_key(value: str) -> str:
    tokens = TOKEN_RE.findall(value.lower().replace("_", "-"))
    return "-".join(tokens) or "unknown"


def _load_classification_fixture(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {
            str(item["id"]): item for item in payload if isinstance(item, dict) and item.get("id")
        }
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items() if isinstance(value, dict)}
    raise ValueError(f"{path} must contain a JSON object or list")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("classifier output did not contain a JSON object")
    return json.loads(stripped[start : end + 1])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _consult_classifier(
    examples: Sequence[WeaknessExample],
    taxonomy: dict[str, str],
    *,
    timeout: int,
) -> dict[str, dict[str, Any]]:
    consult = _repo_root() / "scripts" / "consult_claude.py"
    if not consult.exists():
        consult = Path.home() / ".codex" / "skills" / "consult-fable" / "consult_claude.py"
    if not consult.exists():
        raise RuntimeError("no consult_claude.py helper found for LLM classification")
    compact_examples = [
        {
            "id": example.id,
            "source": example.source,
            "target": example.target,
            "severity": example.severity,
            "text": example.text[:1200],
        }
        for example in examples
    ]
    prompt = {
        "task": "Classify recurring Aragora harness weakness examples for issue #8973.",
        "requirements": [
            "Use taxonomy_id/finding_class when the seed taxonomy applies.",
            "Name the causal mechanism, not only the surface symptom.",
            "Name the implicated harness surface.",
            "Assign an emergent_cluster key suitable for bottom-up grouping.",
            "Return only a JSON object keyed by example id.",
        ],
        "taxonomy": taxonomy,
        "examples": compact_examples,
    }
    proc = subprocess.run(
        [sys.executable, str(consult), "--json", json.dumps(prompt, sort_keys=True)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"LLM classifier failed: {proc.stderr.strip() or proc.stdout.strip()}")
    envelope = json.loads(proc.stdout)
    response_text = str(envelope.get("response") or envelope.get("text") or envelope)
    payload = _extract_json_object(response_text)
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def _classified_examples(
    examples: Sequence[WeaknessExample],
    taxonomy: dict[str, str],
    *,
    classification_json: Path | None,
    classifier: str,
    classifier_timeout: int,
) -> tuple[list[ClassifiedExample], list[str]]:
    warnings: list[str] = []
    if classification_json is not None:
        classifications = _load_classification_fixture(classification_json)
    elif classifier == "llm":
        classifications = _consult_classifier(examples, taxonomy, timeout=classifier_timeout)
    else:
        raise RuntimeError(
            "classification requires --classification-json or --classifier llm; "
            "issue #8973 intentionally avoids regex-only clustering"
        )
    classified: list[ClassifiedExample] = []
    for example in examples:
        item = classifications.get(example.id)
        if not item:
            warnings.append(f"missing classification for {example.id}")
            continue
        taxonomy_id = str(item.get("taxonomy_id") or "unclassified")
        finding_class = str(
            item.get("finding_class") or taxonomy.get(taxonomy_id) or "Unclassified"
        )
        causal_mechanism = str(item.get("causal_mechanism") or item.get("mechanism") or "").strip()
        harness_surface = str(item.get("harness_surface") or item.get("surface") or "").strip()
        emergent_cluster = str(item.get("emergent_cluster") or "").strip()
        evidence_summary = str(item.get("evidence_summary") or example.text[:500]).strip()
        if not causal_mechanism or not harness_surface:
            warnings.append(f"incomplete classification for {example.id}")
            continue
        classified.append(
            ClassifiedExample(
                example=example,
                taxonomy_id=taxonomy_id,
                finding_class=finding_class,
                causal_mechanism=causal_mechanism,
                harness_surface=harness_surface,
                emergent_cluster=emergent_cluster,
                evidence_summary=evidence_summary,
            )
        )
    return classified, warnings


def _rank_score(examples: Iterable[ClassifiedExample]) -> int:
    score = 0
    for item in examples:
        score += SEVERITY_WEIGHTS.get(item.severity, 1)
    return score


def _cluster_from_group(
    *,
    pass_name: str,
    cluster_key: str,
    examples: list[ClassifiedExample],
    title: str,
    finding_class: str,
    causal_mechanism: str,
) -> WeaknessCluster:
    return WeaknessCluster(
        pass_name=pass_name,
        cluster_key=cluster_key,
        title=title,
        finding_class=finding_class,
        causal_mechanism=causal_mechanism,
        harness_surfaces=sorted({item.harness_surface for item in examples}),
        rank_score=_rank_score(examples),
        examples=sorted(examples, key=lambda item: (item.target, item.id)),
    )


def _taxonomy_seeded_clusters(
    classified: Sequence[ClassifiedExample],
    *,
    min_cluster_size: int,
) -> list[WeaknessCluster]:
    groups: dict[tuple[str, str], list[ClassifiedExample]] = defaultdict(list)
    for item in classified:
        groups[(item.taxonomy_id, _normalize_key(item.causal_mechanism))].append(item)
    clusters: list[WeaknessCluster] = []
    for (taxonomy_id, mechanism_key), examples in groups.items():
        if len({item.id for item in examples}) < min_cluster_size:
            continue
        first = examples[0]
        clusters.append(
            _cluster_from_group(
                pass_name="taxonomy_seeded",
                cluster_key=f"taxonomy:{taxonomy_id}:{mechanism_key}",
                examples=examples,
                title=f"{first.finding_class}: {first.causal_mechanism}",
                finding_class=first.finding_class,
                causal_mechanism=first.causal_mechanism,
            )
        )
    return sorted(clusters, key=lambda cluster: (-cluster.rank_score, cluster.cluster_key))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in {"a", "an", "the", "and", "for", "is", "to"}
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _density_components(
    classified: Sequence[ClassifiedExample],
    *,
    threshold: float,
) -> list[list[ClassifiedExample]]:
    remaining = set(range(len(classified)))
    token_sets = [
        _tokens(f"{item.evidence_summary} {item.causal_mechanism} {item.harness_surface}")
        for item in classified
    ]
    components: list[list[ClassifiedExample]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component_indexes = {seed}
        while stack:
            current = stack.pop()
            for candidate in list(remaining):
                if _jaccard(token_sets[current], token_sets[candidate]) >= threshold:
                    remaining.remove(candidate)
                    component_indexes.add(candidate)
                    stack.append(candidate)
        components.append([classified[index] for index in sorted(component_indexes)])
    return components


def _emergent_clusters(
    classified: Sequence[ClassifiedExample],
    *,
    min_cluster_size: int,
    similarity_threshold: float,
) -> list[WeaknessCluster]:
    explicit_groups: dict[str, list[ClassifiedExample]] = defaultdict(list)
    unlabelled: list[ClassifiedExample] = []
    for item in classified:
        if item.emergent_cluster:
            explicit_groups[item.emergent_cluster].append(item)
        else:
            unlabelled.append(item)
    groups = list(explicit_groups.items())
    for component in _density_components(unlabelled, threshold=similarity_threshold):
        if not component:
            continue
        groups.append((_normalize_key(component[0].causal_mechanism), component))
    clusters: list[WeaknessCluster] = []
    for label, examples in groups:
        if len({item.id for item in examples}) < min_cluster_size:
            continue
        mechanism = _summarize_mechanisms(examples)
        finding_class = _summarize_finding_classes(examples)
        clusters.append(
            _cluster_from_group(
                pass_name="emergent_bottom_up",
                cluster_key=f"emergent:{label}",
                examples=examples,
                title=f"{finding_class}: {mechanism}",
                finding_class=finding_class,
                causal_mechanism=mechanism,
            )
        )
    return sorted(clusters, key=lambda cluster: (-cluster.rank_score, cluster.cluster_key))


def _summarize_mechanisms(examples: Sequence[ClassifiedExample]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for item in examples:
        counts[item.causal_mechanism] += 1
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]


def _summarize_finding_classes(examples: Sequence[ClassifiedExample]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for item in examples:
        counts[item.finding_class] += 1
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]


def build_clusters(
    classified: Sequence[ClassifiedExample],
    *,
    min_cluster_size: int,
    similarity_threshold: float,
) -> list[WeaknessCluster]:
    return [
        *_taxonomy_seeded_clusters(classified, min_cluster_size=min_cluster_size),
        *_emergent_clusters(
            classified,
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold,
        ),
    ]


def run_miner(
    *,
    input_json: Path | None = None,
    taxonomy_path: Path,
    classification_json: Path | None = None,
    ledger_paths: Sequence[Path] = (),
    comment_json_paths: Sequence[Path] = (),
    since_days: int = 30,
    min_cluster_size: int = 2,
    similarity_threshold: float = 0.42,
    classifier: str = "llm",
    classifier_timeout: int = 600,
) -> MiningResult:
    taxonomy = load_taxonomy(taxonomy_path)
    examples = collect_examples(
        input_json=input_json,
        ledger_paths=ledger_paths,
        comment_json_paths=comment_json_paths,
        since_days=since_days,
    )
    classified, warnings = _classified_examples(
        examples,
        taxonomy,
        classification_json=classification_json,
        classifier=classifier,
        classifier_timeout=classifier_timeout,
    )
    clusters = build_clusters(
        classified,
        min_cluster_size=min_cluster_size,
        similarity_threshold=similarity_threshold,
    )
    return MiningResult(
        ok=bool(clusters),
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        input_count=len(examples),
        classified_count=len(classified),
        clusters=clusters,
        warnings=warnings,
    )


def render_markdown(result: MiningResult) -> str:
    lines = [
        "# Harness Weakness Report",
        "",
        f"Generated: `{result.generated_at}`",
        f"Examples classified: `{result.classified_count}/{result.input_count}`",
        f"Clusters: `{len(result.clusters)}`",
        "",
        "This report is advisory input for the harness-edit loop. It does not trigger edits, evidence, settlement, or merges.",
        "",
    ]
    if result.warnings:
        lines.extend(["## Warnings", ""])
        for warning in result.warnings:
            lines.append(f"- {warning}")
        lines.append("")
    for index, cluster in enumerate(result.clusters, start=1):
        lines.extend(
            [
                f"## {index}. {cluster.title}",
                "",
                f"- Pass: `{cluster.pass_name}`",
                f"- Key: `{cluster.cluster_key}`",
                f"- Rank score: `{cluster.rank_score}`",
                f"- Distinct examples: `{cluster.example_count}`",
                f"- Harness surfaces: {', '.join(cluster.harness_surfaces)}",
                "",
                "| Target | Severity | Source | Evidence |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in cluster.examples:
            evidence = item.evidence_summary.replace("\n", " ").strip()
            if len(evidence) > 180:
                evidence = f"{evidence[:177]}..."
            target = item.target
            if item.url:
                target = f"[{target}]({item.url})"
            lines.append(f"| {target} | `{item.severity}` | `{item.example.source}` | {evidence} |")
        lines.append("")
    return "\n".join(lines)


def _default_ledger_paths(root: Path) -> list[Path]:
    paths = [root / ".aragora" / "conductor_cycles" / "long_run_ledger.jsonl"]
    return [path for path in paths if path.exists()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, help="Normalized fixture/input examples JSON")
    parser.add_argument(
        "--ledger",
        action="append",
        type=Path,
        default=[],
        help="Conductor ledger JSONL input. Defaults to .aragora/conductor_cycles/long_run_ledger.jsonl when present.",
    )
    parser.add_argument(
        "--comments-json",
        action="append",
        type=Path,
        default=[],
        help="Exported PR/issue comments JSON fixture. Network fetching is intentionally out of scope.",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("docs/artifacts/2026-07-reviewer-failure-taxonomy.md"),
    )
    parser.add_argument(
        "--classification-json",
        type=Path,
        help="Offline classification fixture keyed by example id",
    )
    parser.add_argument("--classifier", choices=["llm"], default="llm")
    parser.add_argument("--classifier-timeout", type=int, default=600)
    parser.add_argument("--since-days", type=int, default=30)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--similarity-threshold", type=float, default=0.42)
    parser.add_argument("--output", type=Path, help="Write Markdown report to this path")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = _repo_root()
    taxonomy = args.taxonomy
    if not taxonomy.is_absolute():
        taxonomy = root / taxonomy
    ledger_paths = list(args.ledger)
    if not ledger_paths and args.input_json is None:
        ledger_paths = _default_ledger_paths(root)
    try:
        result = run_miner(
            input_json=args.input_json,
            taxonomy_path=taxonomy,
            classification_json=args.classification_json,
            ledger_paths=ledger_paths,
            comment_json_paths=args.comments_json,
            since_days=args.since_days,
            min_cluster_size=args.min_cluster_size,
            similarity_threshold=args.similarity_threshold,
            classifier=args.classifier,
            classifier_timeout=args.classifier_timeout,
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_markdown(result), encoding="utf-8")
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
