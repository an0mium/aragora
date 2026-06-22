from __future__ import annotations

import hashlib

import pytest

from aragora.heterogeneity.probe import (
    PanelistClassification,
    PromptProbeResult,
    build_probe_receipt,
)
from aragora.heterogeneity.prompts import load_prompt_file
from aragora.heterogeneity.receipt import (
    MissingProvenanceError,
    build_source_artifact,
    compute_receipt_id,
    require_source_artifacts,
    sha256_file,
    source_artifact_status,
    write_receipt,
)


def test_receipt_id_excludes_produced_at(tmp_path) -> None:
    result = PromptProbeResult(
        prompt_id="p1",
        prompt_class="single_seeded_error",
        seeded_error="seed",
        classifications=(
            PanelistClassification(agent="a", verdict="flagged_correctly"),
            PanelistClassification(agent="b", verdict="missed"),
        ),
    )
    first = build_probe_receipt(
        run_id="run",
        results=[result],
        panel_models=["a", "b"],
        judge_model="fixture",
        produced_at="2026-04-30T00:00:00+00:00",
    )
    second = {**first, "produced_at": "2026-05-01T00:00:00+00:00"}
    assert compute_receipt_id(first) == compute_receipt_id(second)

    path = write_receipt(first, tmp_path)
    assert path.exists()
    assert path.name == f"{first['receipt_id']}.json"
    assert first["metrics"]["independent_flag_successes"] == 1
    assert first["metrics"]["independent_flag_trials"] == 2
    assert first["metrics"]["partial_multi_seeded_successes"] == 0
    assert first["metrics"]["partial_multi_seeded_trials"] == 0


def test_receipt_preserves_plural_seeded_errors() -> None:
    prompt = load_prompt_file(
        "tests/heterogeneity/probe_prompts/multi_seeded_error/01_thresholds_and_window.md"
    )
    result = PromptProbeResult.from_prompt(
        prompt,
        (
            PanelistClassification(agent="a", verdict="flagged_correctly"),
            PanelistClassification(agent="b", verdict="partial_multi_seeded"),
        ),
    )
    receipt = build_probe_receipt(
        run_id="run",
        results=[result],
        panel_models=["a", "b"],
        judge_model="fixture",
        produced_at="2026-04-30T00:00:00+00:00",
    )
    breakdown = receipt["per_prompt_breakdown"][0]
    assert len(breakdown["seeded_errors"]) == 2
    assert breakdown["seeded_error"] == breakdown["seeded_errors"][0]
    assert receipt["metrics"]["independent_flag_successes"] == 1
    assert receipt["metrics"]["partial_multi_seeded_successes"] == 1
    assert receipt["metrics"]["partial_multi_seeded_trials"] == 2


def test_source_artifact_hashes_exact_file_bytes(tmp_path) -> None:
    transcript = tmp_path / "transcript.json"
    transcript.write_bytes(b'{"transcript": "one"}\n')

    artifact = build_source_artifact(
        transcript,
        format="test_transcript.v1",
        root=tmp_path,
        required_for_rejudge=True,
        text_capture="full",
    )

    assert artifact["path"] == "transcript.json"
    assert artifact["bytes"] == transcript.stat().st_size
    assert artifact["sha256"] == hashlib.sha256(transcript.read_bytes()).hexdigest()
    assert sha256_file(transcript) == artifact["sha256"]


def test_source_artifacts_are_part_of_receipt_id(tmp_path) -> None:
    transcript = tmp_path / "transcript.json"
    transcript.write_text('{"transcript": "one"}\n', encoding="utf-8")
    artifact = build_source_artifact(
        transcript,
        format="test_transcript.v1",
        root=tmp_path,
    )
    result = PromptProbeResult(
        prompt_id="p1",
        prompt_class="single_seeded_error",
        seeded_error="seed",
        classifications=(PanelistClassification(agent="a", verdict="missed"),),
    )

    bound = build_probe_receipt(
        run_id="run",
        results=[result],
        panel_models=["a"],
        judge_model="fixture",
        source_artifacts=[artifact],
        produced_at="2026-04-30T00:00:00+00:00",
    )
    unbound = build_probe_receipt(
        run_id="run",
        results=[result],
        panel_models=["a"],
        judge_model="fixture",
        produced_at="2026-04-30T00:00:00+00:00",
    )

    assert bound["receipt_id"] != unbound["receipt_id"]
    assert bound["source_artifacts"] == [artifact]


def test_source_artifact_status_distinguishes_legacy_bound_and_mutated(tmp_path) -> None:
    transcript = tmp_path / "transcript.json"
    transcript.write_text('{"transcript": "one"}\n', encoding="utf-8")
    artifact = build_source_artifact(
        transcript,
        format="test_transcript.v1",
        root=tmp_path,
    )
    receipt = {"source_artifacts": [artifact]}

    legacy = source_artifact_status({})
    assert legacy["canonical"] is False
    assert legacy["status"] == "legacy_unbound"

    bound = source_artifact_status(receipt, base_dir=tmp_path)
    assert bound["canonical"] is True
    assert bound["status"] == "bound"

    transcript.write_text('{"transcript": "mutated"}\n', encoding="utf-8")
    mutated = source_artifact_status(receipt, base_dir=tmp_path)
    assert mutated["canonical"] is False
    assert mutated["status"] == "hash_mismatch"


def test_write_receipt_can_fail_closed_on_missing_source_artifacts(tmp_path) -> None:
    receipt = {
        "schema_version": "heterogeneity_probe_receipt.v1",
        "receipt_id": "abc123",
        "metrics": {},
    }

    with pytest.raises(MissingProvenanceError, match="legacy_unbound"):
        write_receipt(receipt, tmp_path, require_bound_source_artifacts=True)

    with pytest.raises(MissingProvenanceError, match="legacy_unbound"):
        require_source_artifacts(receipt)
