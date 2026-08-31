from __future__ import annotations

import json
from pathlib import Path

import pytest

from aragora.evaluation.outcome_backed_batch import (
    DevelopmentBatchPlanError,
    build_development_plan,
    load_packet_set_manifest,
    validate_development_plan,
)
from aragora.evaluation.outcome_backed_corpus import BENCHMARK_ID, canonical_json_sha256
from aragora.evaluation.outcome_backed_packets import PACKET_SET_SCHEMA


def _cases() -> list[dict[str, str]]:
    return [{"case_id": f"dev-{index:02d}", "split": "development"} for index in range(16)] + [
        {"case_id": f"hold-{index:02d}", "split": "holdout"} for index in range(8)
    ]


def _packet_set(case_ids: list[str] | None = None) -> dict[str, object]:
    ids = case_ids or [f"dev-{index:02d}" for index in range(16)]
    manifest: dict[str, object] = {
        "schema_version": PACKET_SET_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "split": "development",
        "packet_count": len(ids),
        "source_count": 12,
        "packets": [
            {"case_id": case_id, "packet_sha256": f"{index + 1:064x}"}
            for index, case_id in enumerate(ids)
        ],
    }
    manifest["packet_set_sha256"] = canonical_json_sha256(manifest)
    return manifest


def test_build_development_plan_is_deterministic_and_outcome_blind() -> None:
    forward = build_development_plan(_cases(), _packet_set())
    reverse = build_development_plan(list(reversed(_cases())), _packet_set())

    assert forward == reverse
    assert forward["case_count"] == 16
    assert forward["batch_count"] == 4
    assert [batch["batch_id"] for batch in forward["batches"]] == [
        "development-01",
        "development-02",
        "development-03",
        "development-04",
    ]
    assert all(
        str(case_id).startswith("dev-")
        for batch in forward["batches"]
        for case_id in batch["case_ids"]
    )
    assert validate_development_plan(forward) == forward["plan_sha256"]


def test_build_development_plan_supports_bounded_final_batch() -> None:
    plan = build_development_plan(_cases(), _packet_set(), batch_size=6)

    assert [len(batch["case_ids"]) for batch in plan["batches"]] == [6, 6, 4]
    assert plan["batch_count"] == 3


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda manifest: manifest.update(split="holdout"), "split must be development"),
        (lambda manifest: manifest.update(packet_set_sha256="0" * 64), "hash mismatch"),
        (
            lambda manifest: manifest["packets"].reverse(),
            "case IDs must be unique and sorted",
        ),
    ],
)
def test_packet_set_validation_fails_closed(mutate, match: str) -> None:
    manifest = _packet_set()
    mutate(manifest)

    with pytest.raises(DevelopmentBatchPlanError, match=match):
        build_development_plan(_cases(), manifest)


def test_build_rejects_packet_case_set_drift_even_with_valid_manifest_hash() -> None:
    ids = [f"dev-{index:02d}" for index in range(15)] + ["unknown-case"]
    manifest = _packet_set(sorted(ids))

    with pytest.raises(DevelopmentBatchPlanError, match="do not match"):
        build_development_plan(_cases(), manifest)


def test_build_rejects_holdout_case_in_development_packets() -> None:
    ids = [f"dev-{index:02d}" for index in range(15)] + ["hold-00"]
    manifest = _packet_set(sorted(ids))

    with pytest.raises(DevelopmentBatchPlanError, match="do not match"):
        build_development_plan(_cases(), manifest)


def test_validate_development_plan_rejects_tampering() -> None:
    plan = build_development_plan(_cases(), _packet_set())
    plan["batches"][0]["case_ids"][0] = "dev-tampered"

    with pytest.raises(DevelopmentBatchPlanError):
        validate_development_plan(plan)


def test_independent_validation_rebinds_cases_and_packet_set() -> None:
    packet_set = _packet_set()
    plan = build_development_plan(_cases(), packet_set)
    plan["batches"][0]["case_ids"][0] = "dev-00-replaced"
    plan["plan_sha256"] = canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )

    with pytest.raises(DevelopmentBatchPlanError, match="case binding mismatch"):
        validate_development_plan(plan, cases=_cases(), packet_set=packet_set)


def test_independent_validation_requires_both_source_artifacts() -> None:
    plan = build_development_plan(_cases(), _packet_set())

    with pytest.raises(DevelopmentBatchPlanError, match="requires both"):
        validate_development_plan(plan, cases=_cases())


def test_manifest_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "packet-set.json"
    path.write_text('{"schema_version":"a","schema_version":"b"}\n', encoding="utf-8")

    with pytest.raises(DevelopmentBatchPlanError, match="duplicate JSON key"):
        load_packet_set_manifest(path)
