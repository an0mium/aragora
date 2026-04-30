"""Prompt loading for the Round 30f heterogeneity contamination probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_PILOT_CLASS_QUOTAS: dict[str, int] = {
    # The planning doc's 20-prompt table uses one null_negative prompt,
    # but the locked acceptance gate requires at least two prompts in
    # every class. The correlated-priming gate also needs at least six
    # perfect prompts for the 95% Wilson upper bound to fall below 0.40.
    # A 23-prompt default satisfies both gates while staying within the
    # 20-30 pilot budget.
    "clean_neutral": 4,
    "single_seeded_error": 6,
    "multi_seeded_error": 3,
    "correlated_priming": 6,
    "red_team_paraphrase": 2,
    "null_negative": 2,
}


@dataclass(frozen=True)
class SeededError:
    """Ground truth for a seeded-error prompt."""

    description: str
    verification_ref: str | None = None


@dataclass(frozen=True)
class ProbePrompt:
    """One authored prompt plus front-matter metadata."""

    prompt_id: str
    prompt_class: str
    body: str
    path: Path
    seeded_error: SeededError | None = None
    seeded_errors: tuple[SeededError, ...] = ()
    expected_flags: int | None = None
    expected_independent_flag_rate: float | None = None
    priming_framing: str | None = None
    paraphrase_of: str | None = None
    verification_refs: tuple[str, ...] = ()


def _split_front_matter(text: str, path: Path) -> tuple[Mapping[str, Any], str]:
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing YAML front matter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path}: unterminated YAML front matter")
    raw_meta = yaml.safe_load(parts[1]) or {}
    if not isinstance(raw_meta, dict):
        raise ValueError(f"{path}: YAML front matter must be a mapping")
    return raw_meta, parts[2].lstrip("\n")


def _coerce_seeded_error(raw: object, path: Path) -> SeededError | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: seeded_error must be null or a mapping")
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{path}: seeded_error.description must be a non-empty string")
    verification_ref = raw.get("verification_ref")
    if verification_ref is not None and not isinstance(verification_ref, str):
        raise ValueError(f"{path}: seeded_error.verification_ref must be a string")
    return SeededError(description=description.strip(), verification_ref=verification_ref)


def _coerce_seeded_errors(raw: object, path: Path) -> tuple[SeededError, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list | tuple):
        raise ValueError(f"{path}: seeded_errors must be a list")
    errors: list[SeededError] = []
    for item in raw:
        error = _coerce_seeded_error(item, path)
        if error is not None:
            errors.append(error)
    return tuple(errors)


def _coerce_float(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, int | float):
        return float(raw)
    raise ValueError(f"expected float-compatible value, got {raw!r}")


def load_prompt_file(path: str | Path) -> ProbePrompt:
    """Load one prompt markdown file with YAML front matter."""
    prompt_path = Path(path)
    meta, body = _split_front_matter(prompt_path.read_text(encoding="utf-8"), prompt_path)
    prompt_id = meta.get("prompt_id")
    prompt_class = meta.get("class")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError(f"{prompt_path}: prompt_id must be a non-empty string")
    if not isinstance(prompt_class, str) or not prompt_class.strip():
        raise ValueError(f"{prompt_path}: class must be a non-empty string")
    verification_refs = meta.get("verification_refs") or ()
    if not isinstance(verification_refs, list | tuple):
        raise ValueError(f"{prompt_path}: verification_refs must be a list")
    if not all(isinstance(item, str) for item in verification_refs):
        raise ValueError(f"{prompt_path}: verification_refs entries must be strings")
    expected_flags = meta.get("expected_flags")
    if expected_flags is not None and not isinstance(expected_flags, int):
        raise ValueError(f"{prompt_path}: expected_flags must be an int or null")
    seeded_error = _coerce_seeded_error(meta.get("seeded_error"), prompt_path)
    seeded_errors = _coerce_seeded_errors(meta.get("seeded_errors"), prompt_path)
    if seeded_error is not None and seeded_errors:
        raise ValueError(f"{prompt_path}: use either seeded_error or seeded_errors, not both")
    all_seeded_errors = seeded_errors or ((seeded_error,) if seeded_error is not None else ())
    return ProbePrompt(
        prompt_id=prompt_id.strip(),
        prompt_class=prompt_class.strip(),
        body=body.rstrip() + "\n",
        path=prompt_path,
        seeded_error=all_seeded_errors[0] if all_seeded_errors else None,
        seeded_errors=all_seeded_errors,
        expected_flags=expected_flags,
        expected_independent_flag_rate=_coerce_float(meta.get("expected_independent_flag_rate")),
        priming_framing=meta.get("priming_framing"),
        paraphrase_of=meta.get("paraphrase_of"),
        verification_refs=tuple(verification_refs),
    )


def load_prompt_set(root: str | Path) -> list[ProbePrompt]:
    """Load all prompt markdown files below ``root`` except README files."""
    prompt_root = Path(root)
    prompts = [
        load_prompt_file(path)
        for path in sorted(prompt_root.glob("*/*.md"))
        if path.name.lower() != "readme.md"
    ]
    ids = [prompt.prompt_id for prompt in prompts]
    duplicates = sorted({prompt_id for prompt_id in ids if ids.count(prompt_id) > 1})
    if duplicates:
        raise ValueError("duplicate prompt_id values: " + ", ".join(duplicates))
    return prompts


def select_pilot_prompts(
    prompts: list[ProbePrompt],
    quotas: Mapping[str, int] | None = None,
) -> list[ProbePrompt]:
    """Select the deterministic pilot subset by class quotas."""
    selected: list[ProbePrompt] = []
    effective_quotas = quotas or DEFAULT_PILOT_CLASS_QUOTAS
    for prompt_class, needed in effective_quotas.items():
        candidates = sorted(
            [prompt for prompt in prompts if prompt.prompt_class == prompt_class],
            key=lambda prompt: (prompt.path.name, prompt.prompt_id),
        )
        if len(candidates) < needed:
            raise ValueError(
                f"not enough {prompt_class} prompts: need {needed}, found {len(candidates)}"
            )
        selected.extend(candidates[:needed])
    return selected
