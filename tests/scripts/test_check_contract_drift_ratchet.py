"""Tests for scripts/check_contract_drift_ratchet.py."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import scripts.check_contract_drift_ratchet as ratchet
import scripts.generate_contract_drift_inventory as gen

PROGRAM_REL = "scripts/baselines/contract_drift_program.json"
_DISCOVER_CANONICAL_ARTIFACT = ratchet._discover_canonical_artifact


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _cohort_items(docs: dict[str, dict]) -> list[dict]:
    return [
        {
            "id": item_id,
            "source": list_key,
            "class": "start_cohort",
            "discovered_on": gen.COHORT_DATE,
            "provenance": gen.COHORT_PROVENANCE,
            "status": "open",
        }
        for item_id, list_key in sorted(gen.collect_ids(docs).items())
    ]


def _write_inventory(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gen.render_inventory(sorted(items, key=lambda i: i["id"]), "test"))


def _commit(repo: Path, msg: str = "snap") -> str:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            msg,
            "--allow-empty",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_docs(repo: Path, docs: dict[str, dict]) -> None:
    for alias, (rel_path, _keys) in gen.BASELINE_SPECS.items():
        if alias in docs:
            _write_json(repo / rel_path, docs[alias])


def _seed(
    tmp_path: Path,
    *,
    verify: dict | None = None,
    routes: dict | None = None,
    parity: dict | None = None,
    program: dict | None = None,
    inventory_items: list[dict] | None = None,
    commit: bool = True,
) -> tuple[dict[str, Path], Path, str | None]:
    """Create a git repo with baselines at canonical paths; the initial commit
    is both the test's cohort commit and (for pr-mode tests) the base ref."""
    verify = (
        verify
        if verify is not None
        else {
            "python_sdk_drift": ["a", "b"],
            "typescript_sdk_drift": ["x", "y", "z"],
            "missing_stable": [],
        }
    )
    routes = (
        routes
        if routes is not None
        else {"missing_in_spec": ["m1", "m2"], "orphaned_in_spec": ["o1"]}
    )
    parity = parity if parity is not None else {"missing_from_both_sdks": ["p1", "p2"]}
    docs = {"verify": verify, "routes": routes, "parity": parity}

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    _write_docs(repo, docs)
    paths = {alias: repo / rel_path for alias, (rel_path, _k) in gen.BASELINE_SPECS.items()}
    paths["program"] = repo / PROGRAM_REL
    paths["inventory"] = repo / gen.DEFAULT_INVENTORY
    if program is not None:
        _write_json(paths["program"], program)
    items = inventory_items if inventory_items is not None else _cohort_items(docs)
    _write_inventory(paths["inventory"], items)

    sha = _commit(repo, "cohort") if commit else None
    return paths, repo, sha


def _argv(paths: dict[str, Path], repo: Path, cohort: str, *extra: str) -> list[str]:
    return [
        "check_contract_drift_ratchet.py",
        "--repo-root",
        str(repo),
        "--cohort-commit",
        cohort,
        "--program-baseline",
        str(paths["program"]),
        "--verify-baseline",
        str(paths["verify"]),
        "--routes-baseline",
        str(paths["routes"]),
        "--parity-baseline",
        str(paths["parity"]),
        "--inventory",
        str(paths["inventory"]),
        *extra,
    ]


def _result(paths: dict[str, Path], as_of: str, *, repo: Path, cohort: str, **kwargs) -> dict:
    return ratchet.build_ratchet_result(
        mode=kwargs.pop("mode", "program"),
        program_baseline=paths["program"],
        verify_baseline=paths["verify"],
        routes_baseline=paths["routes"],
        parity_baseline=paths["parity"],
        inventory_path=paths["inventory"],
        repo_root=repo,
        as_of=date.fromisoformat(as_of),
        cohort_commit=cohort,
        **kwargs,
    )


def _edit_inventory(paths: dict[str, Path], mutate) -> None:
    inventory = json.loads(paths["inventory"].read_text())
    mutate(inventory)
    paths["inventory"].write_text(json.dumps(inventory))


# ---------------------------------------------------------------- program mode


def test_strict_passes_on_program_start(monkeypatch, tmp_path: Path):
    today = date.today().isoformat()
    paths, repo, cohort = _seed(
        tmp_path,
        program={
            "start_date": today,
            "start_total_items": 10,
            "weekly_reduction": 0.1,
            "grace_weeks": 0,
        },
    )
    monkeypatch.setattr(sys, "argv", _argv(paths, repo, cohort, "--strict", "--as-of", today))
    assert ratchet.main() == 0


def test_strict_fails_when_above_target(monkeypatch, tmp_path: Path):
    today = date.today()
    paths, repo, cohort = _seed(
        tmp_path,
        program={
            "start_date": today.isoformat(),
            "start_total_items": 10,
            "weekly_reduction": 0.1,
            "grace_weeks": 0,
        },
    )
    as_of = (today + timedelta(days=8)).isoformat()
    monkeypatch.setattr(sys, "argv", _argv(paths, repo, cohort, "--strict", "--as-of", as_of))
    assert ratchet.main() == 1


def test_program_numbers_read_only_from_program_baseline(tmp_path: Path):
    """Changing contract_drift_program.json (and nothing else) moves the target."""
    program = {
        "start_date": "2026-06-01",
        "start_total_items": 40,
        "weekly_reduction": 0.5,
        "grace_weeks": 0,
    }
    paths, repo, cohort = _seed(tmp_path, program=program)
    result = _result(paths, "2026-06-08", repo=repo, cohort=cohort)
    assert result["program"]["start_total_items"] == 40
    assert result["target"]["max_open_items"] == 20  # 40 * 0.5 after one week

    _write_json(paths["program"], dict(program, start_total_items=80))
    later = _result(paths, "2026-06-08", repo=repo, cohort=cohort)
    assert later["target"]["max_open_items"] == 40


def test_program_schedule_math_per_class_and_batch_clocks(tmp_path: Path):
    cohort_verify = {"python_sdk_drift": ["a", "b"], "typescript_sdk_drift": []}
    routes = {"missing_in_spec": [], "orphaned_in_spec": []}
    parity = {"missing_from_both_sdks": []}
    paths, repo, cohort = _seed(
        tmp_path,
        verify=cohort_verify,
        routes=routes,
        parity=parity,
        program={
            "start_date": "2026-06-01",
            "start_total_items": 30,
            "weekly_reduction": 0.1,
            "grace_weeks": 0,
        },
    )

    # Post-cohort: a discovered batch of 10 (8 still open) lands on 2026-06-01.
    _write_json(
        paths["verify"],
        dict(cohort_verify, typescript_sdk_drift=[f"d{i}" for i in range(1, 9)]),
    )
    discovered = [
        {
            "id": f"typescript_sdk_drift:d{i}",
            "source": "typescript_sdk_drift",
            "class": "discovered",
            "discovered_on": "2026-06-01",
            "provenance": "batch from #1234",
            "status": "open" if i <= 8 else "resolved",
            **({} if i <= 8 else {"resolved_on": "2026-06-10"}),
        }
        for i in range(1, 11)
    ]
    docs = {"verify": cohort_verify, "routes": routes, "parity": parity}
    _write_inventory(paths["inventory"], _cohort_items(docs) + discovered)

    result = _result(paths, "2026-06-15", repo=repo, cohort=cohort)
    assert result["integrity"]["passing"], result["integrity"]["issues"]
    classes = {cls["name"]: cls for cls in result["classes"]}

    cohort_cls = classes["start_cohort"]
    assert cohort_cls["batch_size"] == 30
    assert cohort_cls["target_max"] == ratchet._target_after_weeks(30, 0.1, 2)
    assert cohort_cls["open_items"] == 2
    assert cohort_cls["passing"]

    batch = classes["discovered:2026-06-01"]
    assert batch["batch_size"] == 10  # open + resolved; clock starts at its own date
    assert batch["weeks_elapsed"] == 2
    assert batch["target_max"] == 8  # 10 -> 9 -> 8
    assert batch["open_items"] == 8  # resolved items excluded from open count
    assert batch["passing"]
    assert result["passing"]

    # One week later the batch target drops to 7 while 8 remain open -> FAIL.
    later = _result(paths, "2026-06-22", repo=repo, cohort=cohort)
    batch_later = {c["name"]: c for c in later["classes"]}["discovered:2026-06-01"]
    assert batch_later["target_max"] == 7
    assert not batch_later["passing"]
    assert not later["passing"]


def test_fail_closed_missing_inventory(monkeypatch, tmp_path: Path):
    today = date.today().isoformat()
    paths, repo, cohort = _seed(
        tmp_path,
        program={"start_date": today, "start_total_items": 10, "weekly_reduction": 0.1},
    )
    paths["inventory"].unlink()
    # Fails even without --strict: integrity violations always fail closed.
    monkeypatch.setattr(sys, "argv", _argv(paths, repo, cohort, "--as-of", today))
    assert ratchet.main() == 1


def test_fail_closed_missing_program_baseline(monkeypatch, tmp_path: Path):
    today = date.today().isoformat()
    paths, repo, cohort = _seed(tmp_path)  # no program file written
    monkeypatch.setattr(sys, "argv", _argv(paths, repo, cohort, "--as-of", today))
    assert ratchet.main() == 1


def test_fail_closed_unexplained_baseline_entry(monkeypatch, tmp_path: Path):
    today = date.today().isoformat()
    paths, repo, cohort = _seed(
        tmp_path,
        program={"start_date": today, "start_total_items": 10, "weekly_reduction": 0.1},
    )
    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].append("sneaky-new-item")  # baseline grows, no inventory
    _write_json(paths["verify"], verify)
    monkeypatch.setattr(sys, "argv", _argv(paths, repo, cohort, "--as-of", today))
    assert ratchet.main() == 1


def test_fail_closed_unknown_class(monkeypatch, tmp_path: Path):
    today = date.today().isoformat()
    paths, repo, cohort = _seed(
        tmp_path,
        program={"start_date": today, "start_total_items": 10, "weekly_reduction": 0.1},
    )
    _edit_inventory(paths, lambda inv: inv["items"][0].update(**{"class": "grandfathered"}))
    monkeypatch.setattr(sys, "argv", _argv(paths, repo, cohort, "--as-of", today))
    assert ratchet.main() == 1


def test_fail_closed_unknown_status(tmp_path: Path):
    today = date.today().isoformat()
    paths, repo, cohort = _seed(
        tmp_path,
        program={"start_date": today, "start_total_items": 10, "weekly_reduction": 0.1},
    )
    _edit_inventory(paths, lambda inv: inv["items"][0].update(status="wip"))
    result = _result(paths, today, repo=repo, cohort=cohort)
    assert not result["integrity"]["passing"]
    assert any("Unknown status" in issue for issue in result["integrity"]["issues"])


def test_resolved_items_excluded_but_retained(tmp_path: Path):
    today = date.today().isoformat()
    cohort_verify = {"python_sdk_drift": ["a", "gone"], "typescript_sdk_drift": []}
    routes = {"missing_in_spec": [], "orphaned_in_spec": []}
    parity = {"missing_from_both_sdks": []}
    paths, repo, cohort = _seed(
        tmp_path,
        verify=cohort_verify,
        routes=routes,
        parity=parity,
        program={"start_date": today, "start_total_items": 5, "weekly_reduction": 0.1},
    )
    # "gone" was fixed: pruned from the baseline, resolved in the inventory.
    _write_json(paths["verify"], dict(cohort_verify, python_sdk_drift=["a"]))

    def resolve_gone(inv):
        for item in inv["items"]:
            if item["id"] == "python_sdk_drift:gone":
                item["status"] = "resolved"
                item["resolved_on"] = "2026-05-01"

    _edit_inventory(paths, resolve_gone)

    result = _result(paths, today, repo=repo, cohort=cohort)
    assert result["integrity"]["passing"], result["integrity"]["issues"]
    cohort_cls = {c["name"]: c for c in result["classes"]}["start_cohort"]
    assert cohort_cls["open_items"] == 1  # resolved item not counted
    assert len(json.loads(paths["inventory"].read_text())["items"]) == 2  # retained


def test_program_mode_future_discovered_on_fails(tmp_path: Path):
    paths, repo, cohort = _seed(
        tmp_path,
        program={
            "start_date": "2026-04-17",
            "start_total_items": 10,
            "weekly_reduction": 0.1,
        },
    )
    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].append("future1")
    _write_json(paths["verify"], verify)
    _edit_inventory(
        paths,
        lambda inv: inv["items"].append(
            {
                "id": "python_sdk_drift:future1",
                "source": "python_sdk_drift",
                "class": "discovered",
                "discovered_on": "2026-08-01",  # after as_of below
                "provenance": "claimed in #9",
                "status": "open",
            }
        ),
    )
    result = _result(paths, "2026-07-16", repo=repo, cohort=cohort)
    assert not result["integrity"]["passing"]
    assert any("out of bounds" in issue for issue in result["integrity"]["issues"])


def test_program_mode_pre_cohort_discovered_on_fails(tmp_path: Path):
    paths, repo, cohort = _seed(
        tmp_path,
        program={
            "start_date": "2026-04-17",
            "start_total_items": 10,
            "weekly_reduction": 0.1,
        },
    )
    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].append("ancient1")
    _write_json(paths["verify"], verify)
    _edit_inventory(
        paths,
        lambda inv: inv["items"].append(
            {
                "id": "python_sdk_drift:ancient1",
                "source": "python_sdk_drift",
                "class": "discovered",
                "discovered_on": "2026-01-01",  # before the program start
                "provenance": "claimed in #9",
                "status": "open",
            }
        ),
    )
    result = _result(paths, "2026-07-16", repo=repo, cohort=cohort)
    assert not result["integrity"]["passing"]
    assert any("out of bounds" in issue for issue in result["integrity"]["issues"])


def test_cohort_reclassification_fails_both_modes(monkeypatch, tmp_path: Path):
    """Forging class=discovered with a fresh date on a cohort item must fail
    integrity in BOTH modes (derivable-metadata invariant)."""
    paths, repo, cohort = _seed(
        tmp_path,
        program={
            "start_date": "2026-04-17",
            "start_total_items": 10,
            "weekly_reduction": 0.1,
        },
    )
    _edit_inventory(
        paths,
        lambda inv: inv["items"][0].update(
            **{
                "class": "discovered",
                "discovered_on": "2026-07-01",
                "provenance": "forged reset #1",
            }
        ),
    )

    program_result = _result(paths, "2026-07-16", repo=repo, cohort=cohort)
    assert not program_result["integrity"]["passing"]
    assert any("reclassified" in i for i in program_result["integrity"]["issues"])

    pr_result = _result(paths, "2026-07-16", repo=repo, cohort=cohort, mode="pr", base_ref=cohort)
    assert not pr_result["integrity"]["passing"]
    assert not pr_result["passing"]

    monkeypatch.setattr(sys, "argv", _argv(paths, repo, cohort, "--as-of", "2026-07-16"))
    assert ratchet.main() == 1  # exit 1 even without --strict


# --------------------------------------------------------------------- pr mode

# 10 items @ -10%/week from 2026-04-17: by 2026-07-16 the target is well below
# the 10 seeded open items, so the program schedule is red at that as-of date.
RED_PROGRAM = {
    "start_date": "2026-04-17",
    "start_total_items": 10,
    "weekly_reduction": 0.1,
    "grace_weeks": 0,
}


def test_pr_mode_passes_on_equal_counts_while_program_red(tmp_path: Path):
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)
    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["passing"]
    assert not result["program_passing"]  # program schedule still honestly red
    assert result["pr_delta"]["increased"] == []


def test_pr_mode_passes_on_decrease_via_legitimate_resolution(tmp_path: Path):
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].remove("b")
    _write_json(paths["verify"], verify)

    def resolve_b(inv):
        for item in inv["items"]:
            if item["id"] == "python_sdk_drift:b":
                item["status"] = "resolved"
                item["resolved_on"] = "2026-07-16"

    _edit_inventory(paths, resolve_b)

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["integrity"]["passing"]  # open -> resolved is a legal transition
    assert result["pr_delta"]["counts"]["verify_python_sdk_drift"]["delta"] == -1
    assert result["passing"]


# NOTE: the original test_pr_mode_fails_on_any_single_list_increase asserted
# that even a fully inventoried discovered entry fails pr mode. That design
# had no intake path for legitimately discovered debt (first contact: #9332)
# and was amended: its exact scenario is now the designed PASS path, covered
# by test_pr_mode_increase_with_inventoried_discovered_intake_passes below.
# Increases that are NOT explained intake still fail — see the
# "pr mode: discovered intake" section.


def test_duplicate_baseline_entry_fails_integrity_program_mode(tmp_path: Path):
    """A duplicated baseline entry inflates the count-based ratchet while the
    id-deduped inventory holds one item — fail closed rather than let the
    duplicate sit as a count-decrease freebie."""
    verify = {"python_sdk_drift": ["a", "a", "b"], "typescript_sdk_drift": [], "missing_stable": []}
    paths, repo, cohort = _seed(tmp_path, verify=verify, program=RED_PROGRAM)
    result = _result(paths, "2026-07-16", repo=repo, cohort=cohort)
    assert not result["integrity"]["passing"]
    assert any(
        "Duplicate baseline entry: python_sdk_drift:a" in i for i in result["integrity"]["issues"]
    )
    assert not result["passing"]


def test_pr_mode_inherited_duplicate_fails_despite_equal_counts(tmp_path: Path):
    """A duplicate present at base AND head leaves every count delta at zero —
    only the duplicate integrity check catches it."""
    verify = {"python_sdk_drift": ["a", "a", "b"], "typescript_sdk_drift": [], "missing_stable": []}
    paths, repo, base = _seed(tmp_path, verify=verify, program=RED_PROGRAM)
    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["pr_delta"]["increased"] == []
    assert not result["integrity"]["passing"]
    assert not result["passing"]


def test_pr_mode_passes_on_duplicate_removal(tmp_path: Path):
    """Removing a duplicated baseline entry is a pure dedup: count decreases by
    one, the inventory (already deduped by id) needs no change, and pr mode
    passes."""
    verify = {"python_sdk_drift": ["a", "a", "b"], "typescript_sdk_drift": [], "missing_stable": []}
    paths, repo, base = _seed(tmp_path, verify=verify, program=RED_PROGRAM)

    deduped = json.loads(paths["verify"].read_text())
    deduped["python_sdk_drift"] = ["a", "b"]
    _write_json(paths["verify"], deduped)

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["integrity"]["passing"]
    assert result["pr_delta"]["counts"]["verify_python_sdk_drift"]["delta"] == -1
    assert result["passing"]


def test_pr_mode_fails_on_integrity_violation(monkeypatch, tmp_path: Path):
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].append("unexplained")  # not added to inventory
    _write_json(paths["verify"], verify)

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert not result["integrity"]["passing"]
    assert not result["passing"]

    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            paths,
            repo,
            base,
            "--mode",
            "pr",
            "--base-ref",
            base,
            "--as-of",
            "2026-07-16",
        ),
    )
    assert ratchet.main() == 1  # fails closed even without --strict


def test_pr_mode_immutable_field_mutation_fails(tmp_path: Path):
    """A PR may not rewrite class/discovered_on/provenance of an existing item."""
    paths, repo, cohort = _seed(tmp_path, program=RED_PROGRAM)

    # Base (post-cohort) commit adds a legitimate discovered item x1.
    verify = json.loads(paths["verify"].read_text())
    verify["typescript_sdk_drift"].append("x1")
    _write_json(paths["verify"], verify)
    _edit_inventory(
        paths,
        lambda inv: inv["items"].append(
            {
                "id": "typescript_sdk_drift:x1",
                "source": "typescript_sdk_drift",
                "class": "discovered",
                "discovered_on": "2026-06-01",
                "provenance": "tracked in #77",
                "status": "open",
            }
        ),
    )
    base = _commit(repo, "base with x1")

    # Head attempts to reset x1's burn-down clock. Counts are unchanged.
    def reset_clock(inv):
        for item in inv["items"]:
            if item["id"] == "typescript_sdk_drift:x1":
                item["discovered_on"] = "2026-07-01"

    _edit_inventory(paths, reset_clock)

    result = _result(paths, "2026-07-16", repo=repo, cohort=cohort, mode="pr", base_ref=base)
    assert result["pr_delta"]["increased"] == []  # only metadata was forged
    assert not result["integrity"]["passing"]
    assert any(
        "Immutable inventory field 'discovered_on'" in i for i in result["integrity"]["issues"]
    )
    assert not result["passing"]


def test_pr_mode_reopen_with_new_date_fails(tmp_path: Path):
    """Reopening a resolved item must preserve its original clock."""
    paths, repo, cohort = _seed(tmp_path, program=RED_PROGRAM)

    # Base: x1 was discovered 2026-06-01 and already resolved.
    _edit_inventory(
        paths,
        lambda inv: inv["items"].append(
            {
                "id": "typescript_sdk_drift:x1",
                "source": "typescript_sdk_drift",
                "class": "discovered",
                "discovered_on": "2026-06-01",
                "provenance": "tracked in #77",
                "status": "resolved",
                "resolved_on": "2026-06-10",
            }
        ),
    )
    base = _commit(repo, "base with resolved x1")

    # Head: x1 regresses back into the baseline, reopened with a reset clock.
    verify = json.loads(paths["verify"].read_text())
    verify["typescript_sdk_drift"].append("x1")
    _write_json(paths["verify"], verify)

    def reopen_reset(inv):
        for item in inv["items"]:
            if item["id"] == "typescript_sdk_drift:x1":
                item["status"] = "open"
                item.pop("resolved_on", None)
                item["discovered_on"] = "2026-07-01"  # forged clock reset

    _edit_inventory(paths, reopen_reset)
    result = _result(paths, "2026-07-16", repo=repo, cohort=cohort, mode="pr", base_ref=base)
    assert not result["integrity"]["passing"]
    assert any(
        "Immutable inventory field 'discovered_on'" in i for i in result["integrity"]["issues"]
    )

    # Reopening with the ORIGINAL date keeps integrity clean; the PR still
    # fails via the increase gate — a reopen is a regression, never intake
    # (see test_pr_mode_reopen_is_regression_not_intake).
    def reopen_honest(inv):
        for item in inv["items"]:
            if item["id"] == "typescript_sdk_drift:x1":
                item["discovered_on"] = "2026-06-01"

    _edit_inventory(paths, reopen_honest)
    honest = _result(paths, "2026-07-16", repo=repo, cohort=cohort, mode="pr", base_ref=base)
    assert honest["integrity"]["passing"], honest["integrity"]["issues"]
    assert honest["pr_delta"]["increased"] == ["verify_typescript_sdk_drift"]
    assert not honest["passing"]


def test_pr_mode_inventory_deletion_fails(tmp_path: Path):
    """Deleting an item (instead of resolving it) violates append-only."""
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].remove("b")
    _write_json(paths["verify"], verify)
    _edit_inventory(
        paths,
        lambda inv: inv.update(items=[i for i in inv["items"] if i["id"] != "python_sdk_drift:b"]),
    )

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    # Counts decreased, but the audit trail was destroyed -> fail closed.
    assert result["pr_delta"]["counts"]["verify_python_sdk_drift"]["delta"] == -1
    assert not result["integrity"]["passing"]
    assert any("append-only" in i for i in result["integrity"]["issues"])
    assert not result["passing"]


def test_pr_mode_missing_file_at_base_treated_as_empty(tmp_path: Path):
    paths, repo, _ = _seed(
        tmp_path,
        parity={"missing_from_both_sdks": []},
        program=RED_PROGRAM,
        commit=False,
    )
    paths["parity"].unlink()
    base = _commit(repo, "base without parity file")  # also the cohort commit

    # HEAD parity file exists with zero entries: 0 vs empty-at-base -> equal, PASS.
    _write_json(paths["parity"], {"missing_from_both_sdks": []})
    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["pr_delta"]["counts"]["sdk_missing_from_both"] == {
        "base": 0,
        "head": 0,
        "delta": 0,
    }
    assert result["passing"]

    # HEAD grows an entry: the increase is computed against the EMPTY base
    # (0 -> 1), proving a file missing at the ref hides nothing. Uninventoried,
    # the increase is unexplained and fails.
    _write_json(paths["parity"], {"missing_from_both_sdks": ["p-new"]})
    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["pr_delta"]["increased"] == ["sdk_missing_from_both"]
    assert result["pr_delta"]["unexplained_increase"] != []
    assert not result["passing"]

    # Fully inventoried, the same increase is explained discovered intake.
    _edit_inventory(
        paths,
        lambda inv: inv["items"].append(
            {
                "id": "missing_from_both_sdks:p-new",
                "source": "missing_from_both_sdks",
                "class": "discovered",
                "discovered_on": "2026-07-16",
                "provenance": "explained in #4242",
                "status": "open",
            }
        ),
    )
    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["pr_delta"]["increased"] == ["sdk_missing_from_both"]
    assert result["pr_delta"]["unexplained_increase"] == []
    assert result["passing"]


def test_target_decay_has_no_fixed_points_and_reaches_zero():
    """Regression: iterative int(round(n*0.9)) stuck at 1-4 forever (review P2

    on #9346); the one-shot floored decay must be monotonic to zero so small
    discovered batches cannot satisfy their clocks indefinitely.
    """
    for start in (1, 2, 3, 4, 10, 655):
        prev = ratchet._target_after_weeks(start, 0.1, 0)
        assert prev == start
        for weeks in range(1, 120):
            cur = ratchet._target_after_weeks(start, 0.1, weeks)
            assert cur <= prev
            prev = cur
        assert ratchet._target_after_weeks(start, 0.1, 120) == 0


def test_pr_mode_new_resolved_item_fails_birth_state(tmp_path: Path):
    """An item absent from the base inventory must be born open; a fabricated
    resolved item (delta-neutral by construction) is an integrity failure."""
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    _edit_inventory(
        paths,
        lambda inv: inv["items"].append(
            {
                "id": "typescript_sdk_drift:fake1",
                "source": "typescript_sdk_drift",
                "class": "discovered",
                "discovered_on": "2026-06-01",
                "provenance": "padding #666",
                "status": "resolved",
                "resolved_on": "2026-06-15",
            }
        ),
    )
    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["pr_delta"]["increased"] == []  # counts untouched by the fake
    assert not result["integrity"]["passing"]
    assert any("born open" in i for i in result["integrity"]["issues"])
    assert not result["passing"]

    # The sibling case: a new OPEN item without a baseline entry fails the
    # global sync check (open items must be baseline-backed, new ones too).
    def swap_to_ghost(inv):
        inv["items"] = [i for i in inv["items"] if i["id"] != "typescript_sdk_drift:fake1"]
        inv["items"].append(
            {
                "id": "typescript_sdk_drift:ghost",
                "source": "typescript_sdk_drift",
                "class": "discovered",
                "discovered_on": "2026-07-16",
                "provenance": "padding #666",
                "status": "open",
            }
        )

    _edit_inventory(paths, swap_to_ghost)
    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert not result["integrity"]["passing"]
    assert any("absent from baselines" in i for i in result["integrity"]["issues"])


def test_pr_mode_batch_inflation_attack_fails(tmp_path: Path):
    """Fake resolved items padding a batch's size + real new drift hidden in
    the same list (net-zero open delta) must fail on the birth-state rule."""
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    # Real new drift swapped in for a legitimately resolved item: same list,
    # count unchanged, so the delta gate alone would pass.
    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].remove("b")
    verify["python_sdk_drift"].append("evil1")
    _write_json(paths["verify"], verify)

    def mutate(inv):
        for item in inv["items"]:
            if item["id"] == "python_sdk_drift:b":
                item["status"] = "resolved"
                item["resolved_on"] = "2026-07-16"
        inv["items"].append(
            {
                "id": "python_sdk_drift:evil1",
                "source": "python_sdk_drift",
                "class": "discovered",
                "discovered_on": "2026-06-01",
                "provenance": "explained in #666",
                "status": "open",
            }
        )
        # Batch padding: five fabricated resolved items in the same batch
        # inflate batch_size 1 -> 6 and raise its scheduled target.
        for i in range(5):
            inv["items"].append(
                {
                    "id": f"python_sdk_drift:fake{i}",
                    "source": "python_sdk_drift",
                    "class": "discovered",
                    "discovered_on": "2026-06-01",
                    "provenance": "padding #666",
                    "status": "resolved",
                    "resolved_on": "2026-06-15",
                }
            )

    _edit_inventory(paths, mutate)
    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["pr_delta"]["increased"] == []  # the attack is delta-neutral
    assert not result["integrity"]["passing"]
    born_open_issues = [i for i in result["integrity"]["issues"] if "born open" in i]
    assert len(born_open_issues) == 5  # every fabricated item rejected
    assert not result["passing"]


def test_pr_mode_legitimate_lifecycle_two_generations(monkeypatch, tmp_path: Path):
    """born open -> resolved with history -> retained: two real generator runs
    bracketing a fix must pass pr mode end-to-end."""
    paths, repo, cohort = _seed(tmp_path, program=RED_PROGRAM)

    def run_gen(*extra: str) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "generate_contract_drift_inventory.py",
                "--repo-root",
                str(repo),
                "--cohort-commit",
                cohort,
                *extra,
            ],
        )
        return gen.main()

    assert run_gen("--as-of", "2026-07-10") == 0  # generation 1: all born open
    base = _commit(repo, "generation 1")

    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].remove("b")  # the item gets fixed
    _write_json(paths["verify"], verify)
    assert run_gen("--as-of", "2026-07-16") == 0  # generation 2: resolves it

    items = {i["id"]: i for i in json.loads(paths["inventory"].read_text())["items"]}
    assert items["python_sdk_drift:b"]["status"] == "resolved"
    assert items["python_sdk_drift:b"]["resolved_on"] == "2026-07-16"

    result = _result(paths, "2026-07-16", repo=repo, cohort=cohort, mode="pr", base_ref=base)
    assert result["integrity"]["passing"], result["integrity"]["issues"]
    assert result["pr_delta"]["counts"]["verify_python_sdk_drift"]["delta"] == -1
    assert result["passing"]


# ------------------------------------------------- pr mode: discovered intake
# Amendment after first contact (#9332): newly VISIBLE debt with clear
# provenance must have an intake path. A count increase is allowed iff EVERY
# baseline entry new vs the base ref is born in this PR as class=discovered
# with a PR/issue-referenced provenance and a valid discovered_on date.


def _discovered_route_item(name: str, *, provenance: str = "canary probe #9332") -> dict:
    return {
        "id": f"orphaned_in_spec:{name}",
        "source": "orphaned_in_spec",
        "class": "discovered",
        "discovered_on": "2026-07-16",
        "provenance": provenance,
        "status": "open",
    }


def test_pr_mode_increase_with_inventoried_discovered_intake_passes(tmp_path: Path):
    """The #9332 shape: canary-probe-exposed orphan routes land as a fully
    inventoried discovered batch -> explained intake, pr mode PASSES, and the
    batch starts its own program-mode burn-down clock."""
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    routes = json.loads(paths["routes"].read_text())
    routes["orphaned_in_spec"] += ["probe1", "probe2", "probe3"]
    _write_json(paths["routes"], routes)
    _edit_inventory(
        paths,
        lambda inv: inv["items"].extend(_discovered_route_item(f"probe{i}") for i in (1, 2, 3)),
    )

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["integrity"]["passing"], result["integrity"]["issues"]
    assert result["pr_delta"]["increased"] == ["routes_orphaned_in_spec"]
    assert result["pr_delta"]["unexplained_increase"] == []
    assert result["passing"]

    # The intake batch gets its own clock in program mode (already implemented).
    batch = {c["name"]: c for c in result["classes"]}["discovered:2026-07-16"]
    assert batch["batch_size"] == 3
    assert batch["target_max"] == 3  # week 0 of its own -10%/week schedule


def test_pr_mode_increase_with_one_uninventoried_entry_fails(tmp_path: Path):
    """A batch where even one new baseline entry lacks an inventory record is
    NOT explained intake: the increase fails (and sync integrity fails too)."""
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    routes = json.loads(paths["routes"].read_text())
    routes["orphaned_in_spec"] += ["probe1", "probe2"]
    _write_json(paths["routes"], routes)
    _edit_inventory(paths, lambda inv: inv["items"].append(_discovered_route_item("probe1")))

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert any(
        "orphaned_in_spec:probe2" in reason for reason in result["pr_delta"]["unexplained_increase"]
    )
    assert not result["integrity"]["passing"]  # sync: probe2 is unexplained debt
    assert not result["passing"]


def test_pr_mode_increase_with_free_text_provenance_fails(tmp_path: Path):
    """Provenance without a PR/issue reference is not intake-grade: the
    increase stays unexplained and the provenance-format invariant fires."""
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    routes = json.loads(paths["routes"].read_text())
    routes["orphaned_in_spec"].append("probe1")
    _write_json(paths["routes"], routes)
    _edit_inventory(
        paths,
        lambda inv: inv["items"].append(
            _discovered_route_item("probe1", provenance="found during a canary sweep")
        ),
    )

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert any(
        "orphaned_in_spec:probe1" in reason for reason in result["pr_delta"]["unexplained_increase"]
    )
    assert not result["integrity"]["passing"]  # provenance-reference invariant
    assert not result["passing"]


def test_pr_mode_increase_without_new_entries_fails(tmp_path: Path):
    """A count can increase with ZERO new ids (duplicate list entries). The
    'every new entry is explained' rule must not pass vacuously."""
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    routes = json.loads(paths["routes"].read_text())
    routes["orphaned_in_spec"].append("o1")  # duplicate of an existing entry
    _write_json(paths["routes"], routes)

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["pr_delta"]["increased"] == ["routes_orphaned_in_spec"]
    assert result["pr_delta"]["unexplained_increase"] != []
    assert not result["passing"]


def test_pr_mode_duplicate_bundled_with_valid_intake_fails(tmp_path: Path):
    """Round-1 review P2 on #9352 (both reviewers): a duplicate-entry increase
    must not ride along with a legitimate discovered entry in the SAME list —
    every unit of count increase needs its own distinct new inventoried entry."""
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    routes = json.loads(paths["routes"].read_text())
    routes["orphaned_in_spec"] += ["o1", "probe1"]  # dup of existing o1 + real probe1
    _write_json(paths["routes"], routes)
    _edit_inventory(paths, lambda inv: inv["items"].append(_discovered_route_item("probe1")))

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["pr_delta"]["counts"]["routes_orphaned_in_spec"]["delta"] == 2
    assert any(
        "routes_orphaned_in_spec" in reason for reason in result["pr_delta"]["unexplained_increase"]
    )
    assert not result["integrity"]["passing"]  # duplicated entry fails closed
    assert any("Duplicate baseline entry" in i for i in result["integrity"]["issues"])
    assert not result["passing"]


def test_pr_mode_cross_list_duplicate_masking_fails(tmp_path: Path):
    """Round-1 review P2 variant: a pure duplicate increase in one list must
    not be masked by the only new id coming from a net-zero swap in ANOTHER
    list — the explained-intake bound is per list, not repo-wide."""
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    routes = json.loads(paths["routes"].read_text())
    routes["orphaned_in_spec"].append("o1")  # duplicate: +1 with zero new route ids
    _write_json(paths["routes"], routes)

    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].remove("b")
    verify["python_sdk_drift"].append("swap1")  # net-zero swap: the sole new id
    _write_json(paths["verify"], verify)

    def mutate(inv):
        for item in inv["items"]:
            if item["id"] == "python_sdk_drift:b":
                item["status"] = "resolved"
                item["resolved_on"] = "2026-07-16"
        inv["items"].append(
            {
                "id": "python_sdk_drift:swap1",
                "source": "python_sdk_drift",
                "class": "discovered",
                "discovered_on": "2026-07-16",
                "provenance": "swap tracked in #4242",
                "status": "open",
            }
        )

    _edit_inventory(paths, mutate)

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["pr_delta"]["increased"] == ["routes_orphaned_in_spec"]
    assert any(
        "routes_orphaned_in_spec" in reason for reason in result["pr_delta"]["unexplained_increase"]
    )
    assert not result["passing"]


def test_pr_mode_delta_neutral_duplicate_smuggle_fails(tmp_path: Path):
    """Remove one entry and duplicate another in the same list: deltas are
    all zero, but the minted duplicate is slack a later PR could cash in as
    fake burn-down — must fail via integrity, independent of the delta gate.
    (#9354's unconditional duplicate check is what catches it.)"""
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    routes = json.loads(paths["routes"].read_text())
    routes["missing_in_spec"] = ["m1", "m1"]  # m2 removed, m1 duplicated: count still 2
    _write_json(paths["routes"], routes)

    def resolve_m2(inv):
        for item in inv["items"]:
            if item["id"] == "missing_in_spec:m2":
                item["status"] = "resolved"
                item["resolved_on"] = "2026-07-16"

    _edit_inventory(paths, resolve_m2)

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["pr_delta"]["increased"] == []  # the smuggle is delta-neutral
    assert not result["integrity"]["passing"]
    assert any(
        "Duplicate baseline entry" in i and "missing_in_spec:m1" in i
        for i in result["integrity"]["issues"]
    )
    assert not result["passing"]


def test_pr_mode_non_string_discovered_on_fails_closed_without_crash(tmp_path: Path):
    """Round-1 review P3 on #9352: a non-string discovered_on (e.g. a JSON
    number) must produce integrity/unexplained failures, not a TypeError."""
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    routes = json.loads(paths["routes"].read_text())
    routes["orphaned_in_spec"].append("probe1")
    _write_json(paths["routes"], routes)
    item = _discovered_route_item("probe1")
    item["discovered_on"] = 20260716  # number, not an ISO date string
    _edit_inventory(paths, lambda inv: inv["items"].append(item))

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert not result["integrity"]["passing"]
    assert any("discovered_on" in i for i in result["integrity"]["issues"])
    assert any("discovered_on" in reason for reason in result["pr_delta"]["unexplained_increase"])
    assert not result["passing"]


def test_pr_mode_reopen_is_regression_not_intake(tmp_path: Path):
    """An item with base-inventory history regressing back into the baseline
    is a regression, not newly discovered debt: honest reopen (original clock)
    must NOT ride the discovered-intake allowance."""
    paths, repo, cohort = _seed(tmp_path, program=RED_PROGRAM)

    _edit_inventory(
        paths,
        lambda inv: inv["items"].append(
            {
                "id": "typescript_sdk_drift:x1",
                "source": "typescript_sdk_drift",
                "class": "discovered",
                "discovered_on": "2026-06-01",
                "provenance": "tracked in #77",
                "status": "resolved",
                "resolved_on": "2026-06-10",
            }
        ),
    )
    base = _commit(repo, "base with resolved x1")

    verify = json.loads(paths["verify"].read_text())
    verify["typescript_sdk_drift"].append("x1")
    _write_json(paths["verify"], verify)

    def reopen_honest(inv):
        for item in inv["items"]:
            if item["id"] == "typescript_sdk_drift:x1":
                item["status"] = "open"
                item.pop("resolved_on", None)

    _edit_inventory(paths, reopen_honest)

    result = _result(paths, "2026-07-16", repo=repo, cohort=cohort, mode="pr", base_ref=base)
    assert result["integrity"]["passing"], result["integrity"]["issues"]
    assert any(
        "typescript_sdk_drift:x1" in reason for reason in result["pr_delta"]["unexplained_increase"]
    )
    assert not result["passing"]


def test_pr_mode_program_parameter_change_fails(tmp_path: Path):
    """Round-5 review P2: editing contract_drift_program.json in a PR is

    threshold inflation by definition and must fail pr-mode integrity even
    though counts and inventory are untouched.
    """
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    program = json.loads(paths["program"].read_text())
    program["start_total_items"] = 5000
    _write_json(paths["program"], program)

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    assert result["pr_delta"]["increased"] == []
    assert not result["integrity"]["passing"]
    assert any("Program baseline parameter changed" in i for i in result["integrity"]["issues"])
    assert not result["passing"]


# --------------------------------------------------------------- boundary mode


def _canonical_boundary_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _rule_suite_record(
    end_sha: str,
    **overrides: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "after_sha": end_sha,
        "before_sha": "0" * 40,
        "evaluation_result": "pass",
        "id": 987654,
        "pushed_at": "2026-07-20T00:00:00Z",
        "ref": "refs/heads/main",
        "repository_id": 1126097105,
        "repository_name": "aragora",
        "result": "pass",
        "rule_evaluations": [{"result": "pass", "rule_source": {"type": "repository"}}],
    }
    record.update(overrides)
    return record


def _rule_suite_claim(
    end_sha: str,
    *,
    delete: str | None = None,
    bypassed: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    record = _rule_suite_record(end_sha, **overrides)
    if delete is not None:
        record.pop(delete, None)
    raw = ratchet._canonical_json_bytes(record)
    claim = {
        field: copy.deepcopy(record[field])
        for field in (
            "after_sha",
            "before_sha",
            "evaluation_result",
            "id",
            "pushed_at",
            "ref",
            "repository_id",
            "repository_name",
            "result",
            "rule_evaluations",
        )
        if field in record
    }
    claim.update(
        {
            "authenticated": True,
            "available": True,
            "bypassed": bypassed,
            "raw_response": raw.decode("utf-8"),
            "raw_response_byte_length": len(raw),
            "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    return claim


def _write_canonical_boundary_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    raw = _canonical_boundary_bytes(payload)
    path.write_bytes(raw)
    return {
        "byte_length": len(raw),
        "name": path.stem,
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _canonical_fixture_artifact_paths() -> tuple[Path, Path] | None:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return (
            _DISCOVER_CANONICAL_ARTIFACT(None, ratchet.COHORT_ARTIFACT, repo_root),
            _DISCOVER_CANONICAL_ARTIFACT(None, ratchet.PROVENANCE_ARTIFACT, repo_root),
        )
    except ValueError:
        return None


def _write_synthetic_canonical_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    category_sizes = (
        ("python_sdk_drift", 74),
        ("routes_missing_in_spec", 11),
        ("routes_orphaned_in_spec", 17),
        ("sdk_missing_from_both", 29),
        ("typescript_sdk_drift", 524),
    )
    cohort_records: list[dict[str, Any]] = []
    sdk_records: list[dict[str, Any]] = []
    category_indices: dict[str, int] = {}
    for category, count in category_sizes:
        for _index in range(count):
            source_array_index = category_indices.get(category, 0)
            category_indices[category] = source_array_index + 1
            literal = f"fixture:{category}:{source_array_index:04d}"
            payload = {
                "category": category,
                "exact_historical_literal_record": literal,
                "schema": "cdg-original-record-id-v1",
            }
            payload_raw = ratchet._canonical_json_bytes(payload)
            payload_sha256 = hashlib.sha256(payload_raw).hexdigest()
            record: dict[str, Any] = {
                "category": category,
                "exact_historical_literal_record": literal,
                "id_payload_byte_length": len(payload_raw),
                "id_payload_sha256": payload_sha256,
                "original_record_id": f"cdg1:{payload_sha256}",
                "source_array_index": source_array_index,
            }
            if category in {"python_sdk_drift", "typescript_sdk_drift"}:
                record.update(
                    {
                        "method": "GET",
                        "sdk_language": [
                            "python" if category == "python_sdk_drift" else "typescript"
                        ],
                    }
                )
                sdk_records.append(record)
            cohort_records.append(record)

    provenance_records: list[dict[str, Any]] = []
    for index, cohort_record in enumerate(sdk_records):
        core = index < 75
        atoms = ["agents" if core else "billing"]
        if index < 12:
            atoms.append(f"fixture_{index}")
        partition, matches = ratchet._partition_from_atoms(atoms)
        occurrences = [
            {
                "provenance_atom": atoms[0],
                "sdk_language": cohort_record["sdk_language"][0],
            }
        ]
        if index < 92:
            occurrences.append(dict(occurrences[0]))
        record = {
            "category": cohort_record["category"],
            "exact_historical_literal_record": cohort_record["exact_historical_literal_record"],
            "id_payload_byte_length": cohort_record["id_payload_byte_length"],
            "id_payload_sha256": cohort_record["id_payload_sha256"],
            "matched_domains": matches,
            "original_record_id": cohort_record["original_record_id"],
            "partition": partition,
            "provenance_atoms": atoms,
            "sdk_language": cohort_record["sdk_language"][0],
            "source_array_index": cohort_record["source_array_index"],
            "source_occurrences": occurrences,
        }
        record_sha256 = hashlib.sha256(ratchet._canonical_json_bytes(record)).hexdigest()
        record["record_sha256"] = record_sha256
        cohort_record["sdk_provenance_record_sha256"] = record_sha256
        provenance_records.append(record)

    projection_records = []
    for index, cohort_record in enumerate(cohort_records):
        edge_count = 4 if index == 0 else 2 if index < 9 else 1
        record = {
            "operation_edges": [
                {
                    "evidence": [f"fixture:{index}:{edge_index}"],
                    "method": "GET",
                    "normalized_path": f"/fixture/{index}/{edge_index}",
                }
                for edge_index in range(edge_count)
            ],
            "original_record_id": cohort_record["original_record_id"],
        }
        record_sha256 = hashlib.sha256(ratchet._canonical_json_bytes(record)).hexdigest()
        record["record_sha256"] = record_sha256
        projection_records.append(record)

    original_ids = sorted(record["original_record_id"] for record in cohort_records)
    original_id_set_sha256 = ratchet._digest_set(
        "cdg-original-record-id-set-v1",
        original_ids,
        "original_record_ids",
    )
    projection_record_set_sha256 = ratchet._digest_set(
        "cdg-operation-projection-record-digest-set-v1",
        [record["record_sha256"] for record in projection_records],
        "record_sha256_values",
    )
    cohort = {
        "counts": {
            "by_category": ratchet.EXPECTED_CATEGORY_COUNTS,
            "method_bearing_sdk_records": 598,
            "method_null_route_parity_records": 57,
            "records": 655,
        },
        "id_encoding": "fixture",
        "membership_anchor": "fixture",
        "membership_sources": ["fixture"],
        "operation_projection": {
            "one_to_many_rule": "fixture",
            "record_digest_set_sha256": projection_record_set_sha256,
            "records": projection_records,
            "schema": "cdg-operation-projection-v1",
            "witness_dependencies": ["fixture"],
        },
        "original_record_id_set": {
            "original_record_ids": original_ids,
            "sha256": original_id_set_sha256,
        },
        "original_records": cohort_records,
        "schema": "contract-drift-original-cohort-v1",
    }

    provenance_record_set_sha256 = ratchet._digest_set(
        "cdg-sdk-provenance-record-digest-set-v1",
        [record["record_sha256"] for record in provenance_records],
        "record_sha256_values",
    )
    sdk_ids = [record["original_record_id"] for record in provenance_records]
    core_ids = [
        record["original_record_id"]
        for record in provenance_records
        if record["partition"] == "core"
    ]
    extended_ids = [
        record["original_record_id"]
        for record in provenance_records
        if record["partition"] == "extended"
    ]
    sdk_id_set_sha256 = ratchet._digest_set(
        "cdg-sdk-original-record-id-set-v1",
        sdk_ids,
        "original_record_ids",
    )
    core_id_set_sha256 = ratchet._digest_set(
        "cdg-core-original-record-id-set-v1",
        core_ids,
        "original_record_ids",
    )
    extended_id_set_sha256 = ratchet._digest_set(
        "cdg-extended-original-record-id-set-v1",
        extended_ids,
        "original_record_ids",
    )
    provenance = {
        "baseline_birth": "fixture",
        "counts": {
            "core": 75,
            "extended": 523,
            "python_sdk_drift": 74,
            "records": 598,
            "records_with_multiple_distinct_atoms": 12,
            "source_occurrences": 690,
            "typescript_sdk_drift": 524,
        },
        "dependencies": ["fixture"],
        "extraction_algorithm": "fixture",
        "partition": {
            "core_original_record_id_set_sha256": core_id_set_sha256,
            "extended_original_record_id_set_sha256": extended_id_set_sha256,
            "intersection_count": 0,
            "rule_schema": "cdg-sdk-partition-rule-v1",
            "sdk_original_record_id_set_sha256": sdk_id_set_sha256,
            "union_count": 598,
        },
        "record_digest_set_sha256": provenance_record_set_sha256,
        "records": provenance_records,
        "schema": "contract-drift-sdk-provenance-v1",
    }

    cohort_path = tmp_path / ratchet.COHORT_ARTIFACT["filename"]
    provenance_path = tmp_path / ratchet.PROVENANCE_ARTIFACT["filename"]
    cohort_raw = ratchet._canonical_json_bytes(cohort, terminal_lf=True)
    provenance_raw = ratchet._canonical_json_bytes(provenance, terminal_lf=True)
    cohort_path.write_bytes(cohort_raw)
    provenance_path.write_bytes(provenance_raw)
    monkeypatch.setitem(ratchet.COHORT_ARTIFACT, "byte_length", len(cohort_raw))
    monkeypatch.setitem(
        ratchet.COHORT_ARTIFACT,
        "sha256",
        hashlib.sha256(cohort_raw).hexdigest(),
    )
    monkeypatch.setitem(ratchet.PROVENANCE_ARTIFACT, "byte_length", len(provenance_raw))
    monkeypatch.setitem(
        ratchet.PROVENANCE_ARTIFACT,
        "sha256",
        hashlib.sha256(provenance_raw).hexdigest(),
    )
    monkeypatch.setattr(ratchet, "ORIGINAL_ID_SET_SHA256", original_id_set_sha256)
    monkeypatch.setattr(
        ratchet,
        "PROJECTION_RECORD_SET_SHA256",
        projection_record_set_sha256,
    )
    monkeypatch.setattr(
        ratchet,
        "PROVENANCE_RECORD_SET_SHA256",
        provenance_record_set_sha256,
    )
    monkeypatch.setattr(ratchet, "SDK_ID_SET_SHA256", sdk_id_set_sha256)
    monkeypatch.setattr(ratchet, "CORE_ID_SET_SHA256", core_id_set_sha256)
    monkeypatch.setattr(ratchet, "EXTENDED_ID_SET_SHA256", extended_id_set_sha256)
    return cohort_path, provenance_path


def _clone_repository_with_synthetic_accepted_authority(
    tmp_path: Path,
    *,
    cohort_path: Path,
    provenance_path: Path,
) -> tuple[Path, str, str, bool]:
    source_repo = Path(__file__).resolve().parents[2]
    repo_root = tmp_path / "synthetic-authority-repo"
    subprocess.run(
        ["git", "clone", "-q", "--no-hardlinks", str(source_repo), str(repo_root)],
        check=True,
    )
    production_sha = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    inventory_path = repo_root / gen.DEFAULT_INVENTORY
    inventory = json.loads(inventory_path.read_bytes())
    accepted_authority = inventory.get("accepted_authority")
    has_accepted_authority = isinstance(accepted_authority, dict)
    if accepted_authority is not None and not has_accepted_authority:
        raise AssertionError("production accepted authority is malformed")
    if has_accepted_authority:
        canonical_artifacts = accepted_authority.get("canonical_artifacts")
        if not isinstance(canonical_artifacts, dict):
            raise AssertionError("production accepted authority lacks canonical artifacts")
        before = copy.deepcopy(inventory)
        original_canonical_artifacts = copy.deepcopy(canonical_artifacts)
        cohort = json.loads(cohort_path.read_bytes())
        provenance = json.loads(provenance_path.read_bytes())
        canonical_artifacts["original_cohort"] = cohort
        canonical_artifacts["sdk_provenance"] = provenance
        reverted = copy.deepcopy(inventory)
        reverted_artifacts = reverted["accepted_authority"]["canonical_artifacts"]
        reverted_artifacts["original_cohort"] = original_canonical_artifacts["original_cohort"]
        reverted_artifacts["sdk_provenance"] = original_canonical_artifacts["sdk_provenance"]
        assert reverted == before
        _write_json(inventory_path, inventory)
    fixture_sha = _commit(repo_root, "bind synthetic accepted authority")
    return repo_root, production_sha, fixture_sha, has_accepted_authority


def _fixture_sdk_partitions() -> dict[str, list[str]]:
    artifact_paths = _canonical_fixture_artifact_paths()
    if artifact_paths is None:
        return {
            "core": [
                _fixture_original_record_id("python_sdk_drift", f"fixture-core-{index}")
                for index in range(75)
            ],
            "extended": [
                _fixture_original_record_id(
                    "typescript_sdk_drift",
                    f"fixture-extended-{index}",
                )
                for index in range(523)
            ],
        }
    provenance = json.loads(artifact_paths[1].read_bytes())
    partitions: dict[str, list[str]] = {"core": [], "extended": []}
    for record in provenance["records"]:
        partitions[record["partition"]].append(record["original_record_id"])
    for values in partitions.values():
        values.sort()
    return partitions


def _fixture_original_record_id(category: str, literal: str) -> str:
    payload = {
        "category": category,
        "exact_historical_literal_record": literal,
        "schema": "cdg-original-record-id-v1",
    }
    return f"cdg1:{hashlib.sha256(ratchet._canonical_json_bytes(payload)).hexdigest()}"


def _fixture_sdk_literal(partition: str) -> tuple[str, str]:
    artifact_paths = _canonical_fixture_artifact_paths()
    if artifact_paths is None:
        if partition == "core":
            return "python_sdk_drift", "fixture-core-0"
        return "typescript_sdk_drift", "fixture-extended-0"
    cohort_path, provenance_path = artifact_paths
    cohort = json.loads(cohort_path.read_bytes())
    provenance = json.loads(provenance_path.read_bytes())
    cohort_by_id = {record["original_record_id"]: record for record in cohort["original_records"]}
    record = next(record for record in provenance["records"] if record["partition"] == partition)
    cohort_record = cohort_by_id[record["original_record_id"]]
    return (
        cohort_record["category"],
        cohort_record["exact_historical_literal_record"],
    )


def _boundary_git_repo(
    tmp_path: Path,
    *,
    route_debt: bool = False,
    route_debt_at: str | None = None,
    sdk_debt_partition: str | None = None,
    sdk_debt_at: str | None = None,
) -> tuple[Path, str, dict[str, str]]:
    repo = tmp_path / "boundary-repo"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    baselines = repo / "scripts" / "baselines"
    baselines.mkdir(parents=True)
    _write_json(
        baselines / "check_sdk_parity.json",
        {"missing_from_both_sdks": []},
    )
    _write_json(
        baselines / "validate_openapi_routes.json",
        {"missing_in_spec": [], "orphaned_in_spec": []},
    )
    _write_json(
        baselines / "verify_sdk_contracts.json",
        {"python_sdk_drift": [], "typescript_sdk_drift": []},
    )
    (repo / "fixture.txt").write_text("start\n", encoding="utf-8")
    start_sha = _commit(repo, "accepted corrective bootstrap")
    shas: dict[str, str] = {}
    for index, boundary in enumerate(ratchet.BOUNDARY_NAMES, start=1):
        effective_route_debt_at = route_debt_at or ("route_truth" if route_debt else None)
        if boundary == effective_route_debt_at:
            _write_json(
                baselines / "validate_openapi_routes.json",
                {
                    "missing_in_spec": ["/api/contradiction"],
                    "orphaned_in_spec": [],
                },
            )
        if sdk_debt_partition is not None and boundary == sdk_debt_at:
            category, literal = _fixture_sdk_literal(sdk_debt_partition)
            verify = {"python_sdk_drift": [], "typescript_sdk_drift": []}
            verify[category] = [literal]
            _write_json(baselines / "verify_sdk_contracts.json", verify)
        (repo / "fixture.txt").write_text(f"boundary-{index}\n", encoding="utf-8")
        shas[boundary] = _commit(repo, boundary)
    return repo, start_sha, shas


def _boundary_payloads(
    boundary: str,
    start_sha: str,
    end_sha: str,
    boundary_shas: dict[str, str],
    *,
    repo: Path,
    release_immutability: bool = True,
) -> dict[str, dict[str, Any]]:
    chronology = [{"boundary": name, "sha": boundary_shas[name]} for name in ratchet.BOUNDARY_NAMES]
    common = {"boundary": boundary, "end_sha": end_sha, "start_sha": start_sha}

    def proof_interval(name: str) -> dict[str, str]:
        return {
            "predicate": name,
            "proof_end_sha": end_sha,
            "proof_for_boundary": boundary,
            "proof_start_sha": start_sha,
        }

    def fact(schema: str, value: dict[str, Any]) -> dict[str, Any]:
        return {
            "fact": value,
            "sha256": ratchet._fact_digest(schema, value),
        }

    sdk_partitions = _fixture_sdk_partitions()

    governed_records = []
    receipt_records = []
    prior_sha = start_sha
    for index, name in enumerate(
        ratchet.BOUNDARY_NAMES[: ratchet.BOUNDARY_NAMES.index(boundary) + 1],
        start=1,
    ):
        merge_sha = boundary_shas[name]
        merge_tree_sha = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", f"{merge_sha}^{{tree}}"],
            text=True,
        ).strip()
        governed_records.append(
            {
                "base_sha": prior_sha,
                "changed_files_complete": True,
                "head_sha": merge_sha,
                "head_tree_sha": merge_tree_sha,
                "pr": 9998 + index,
            }
        )
        receipt_records.append(
            {
                "base_sha": prior_sha,
                "first_parent_sha": prior_sha,
                "head_sha": merge_sha,
                "head_tree_sha": merge_tree_sha,
                "merge_sha": merge_sha,
                "merge_tree_sha": merge_tree_sha,
                "pr": 9998 + index,
            }
        )
        prior_sha = merge_sha

    return {
        "boundary_chronology": {
            **common,
            "boundaries": chronology,
            "schema": "contract-drift-boundary-chronology-v1",
        },
        "corrective_bootstrap": {
            **proof_interval("corrective_bootstrap"),
            "accepted_stage1_closure": fact(
                "contract-drift-stage1-closure-fact-v1",
                {
                    "authority_manifest_sha256": "1" * 64,
                    "boundary_verifier_sha256": "7" * 64,
                    "dependency_manifest_sha256": "2" * 64,
                    "inventory_sha256": "3" * 64,
                    "repo_file_count": 42,
                },
            ),
            "corrective_transition": fact(
                "contract-drift-corrective-transition-fact-v1",
                {
                    "commit_count": 1,
                    "end_sha": boundary_shas["corrective_bootstrap"],
                    "start_sha": start_sha,
                },
            ),
            "schema": "contract-drift-corrective-bootstrap-proof-v1",
            "stage2_verifier_chronology": fact(
                "contract-drift-stage2-verifier-chronology-fact-v1",
                {
                    "corrective_boundary_sha": boundary_shas["corrective_bootstrap"],
                    "ordered_after_stage1": True,
                    "start_sha": start_sha,
                    "verifier_sha256": "7" * 64,
                },
            ),
        },
        "route_truth": {
            **proof_interval("route_truth"),
            "openapi_truth": fact(
                "contract-drift-openapi-truth-fact-v1",
                {
                    "boundary_sha": boundary_shas["route_truth"],
                    "complete": True,
                    "route_boundary_sha256": "5" * 64,
                },
            ),
            "route_truth": fact(
                "contract-drift-route-truth-fact-v1",
                {
                    "authority_route_member_count": 2,
                    "boundary_sha": boundary_shas["route_truth"],
                    "complete": True,
                    "method_aware": True,
                    "route_boundary_sha256": "5" * 64,
                },
            ),
            "schema": "contract-drift-route-truth-proof-v1",
        },
        "core_sdk": {
            **proof_interval("core_sdk"),
            "qualifying_paydown": fact(
                "contract-drift-core-sdk-paydown-fact-v1",
                {
                    "added_units": [],
                    "boundary_sha": boundary_shas["core_sdk"],
                    "category_growth": [],
                    "max_pr_delta": 800,
                    "removed_original_record_ids": sdk_partitions["core"],
                    "replacement_units": [],
                },
            ),
            "schema": "contract-drift-core-sdk-proof-v1",
            "zero_core_debt": fact(
                "contract-drift-zero-core-debt-fact-v1",
                {
                    "boundary_sha": boundary_shas["core_sdk"],
                    "partition_set_sha256": "b3a1755f027c998d507f13f3ba9093f769cea8720d44bfac12be6beccd626787",
                    "remaining_original_units": 0,
                },
            ),
        },
        "extended_sdk": {
            **proof_interval("extended_sdk"),
            "qualifying_paydown": fact(
                "contract-drift-extended-sdk-paydown-fact-v1",
                {
                    "added_units": [],
                    "boundary_sha": boundary_shas["extended_sdk"],
                    "category_growth": [],
                    "max_pr_delta": 800,
                    "removed_original_record_ids": sdk_partitions["extended"],
                    "replacement_units": [],
                },
            ),
            "schema": "contract-drift-extended-sdk-proof-v1",
            "zero_sdk_debt": fact(
                "contract-drift-zero-sdk-debt-fact-v1",
                {
                    "boundary_sha": boundary_shas["extended_sdk"],
                    "partition_set_sha256": "51a963079136a92a86485b56f6cef42aafc7749bfad146ce5fb37293524c5762",
                    "remaining_original_units": 0,
                },
            ),
        },
        "final_seal": {
            **proof_interval("final_seal"),
            "complete_paydown": fact(
                "contract-drift-complete-paydown-fact-v1",
                {
                    "boundary_sha": boundary_shas["final_seal"],
                    "remaining_original_units": 0,
                },
            ),
            "dated_trajectory": fact(
                "contract-drift-dated-trajectory-fact-v1",
                {
                    "as_of": "2026-07-20",
                    "boundary_sha": boundary_shas["final_seal"],
                    "target": 0,
                    "total": 0,
                },
            ),
            "final_zero": fact(
                "contract-drift-final-zero-fact-v1",
                {
                    "all_categories_zero": True,
                    "boundary_sha": boundary_shas["final_seal"],
                },
            ),
            "publication": fact(
                "contract-drift-publication-fact-v1",
                {
                    "attestation_bundle_sha256": "6" * 64,
                    "boundary_sha": boundary_shas["final_seal"],
                    "release_api_id": 100,
                    "rule_suite_id": 987654,
                },
            ),
            "schema": "contract-drift-final-seal-proof-v1",
        },
        "external_prerequisites": {
            **common,
            "administration": {"authenticated": True, "available": True},
            "future_release_immutability": {
                "authenticated": True,
                "available": True,
                "enabled": release_immutability,
            },
            "rule_suite": _rule_suite_claim(end_sha),
            "schema": "contract-drift-external-prerequisites-v1",
        },
        "durable_capsule": {
            **common,
            "attestation": {
                "bundle_sha256": "6" * 64,
                "verified": True,
                "workflow": "actions/attest@v4",
            },
            "release": {
                "asset_api_ids": [101, 102, 103],
                "asset_names": ["manifest.json", "payload.json", "checksums.txt"],
                "exact_full_sha_tag": end_sha,
                "immutable": release_immutability,
                "release_api_id": 100,
                "verified": release_immutability,
            },
            "schema": "contract-drift-durable-capsule-v1",
        },
        "governed_prs": {
            **common,
            "records": governed_records,
            "schema": "contract-drift-governed-prs-v1",
        },
        "first_parent_receipts": {
            **common,
            "records": receipt_records,
            "schema": "contract-drift-first-parent-receipts-v1",
        },
    }


def _write_boundary_index(
    tmp_path: Path,
    boundary: str,
    start_sha: str,
    end_sha: str,
    boundary_shas: dict[str, str],
    *,
    repo: Path,
    release_immutability: bool = True,
    mutate: Any | None = None,
) -> tuple[Path, int, str]:
    resources_dir = tmp_path / f"resources-{boundary}"
    resources_dir.mkdir(parents=True)
    payloads = _boundary_payloads(
        boundary,
        start_sha,
        end_sha,
        boundary_shas,
        repo=repo,
        release_immutability=release_immutability,
    )
    if mutate is not None:
        mutate(payloads)
    payloads["boundary_chronology"]["boundaries"] = payloads["boundary_chronology"]["boundaries"][
        : ratchet.BOUNDARY_NAMES.index(boundary) + 1
    ]
    selected = set(ratchet.BOUNDARY_NAMES[: ratchet.BOUNDARY_NAMES.index(boundary) + 1])
    selected.update(
        {
            "boundary_chronology",
            "durable_capsule",
            "external_prerequisites",
            "first_parent_receipts",
            "governed_prs",
        }
    )
    payloads = {name: payload for name, payload in payloads.items() if name in selected}
    descriptors = []
    for name, payload in sorted(payloads.items()):
        descriptor = _write_canonical_boundary_json(resources_dir / f"{name}.json", payload)
        descriptor["name"] = name
        descriptor["path"] = f"{resources_dir.name}/{name}.json"
        descriptors.append(descriptor)
    index = {
        "boundary": boundary,
        "end_sha": end_sha,
        "resources": descriptors,
        "schema": ratchet.BOUNDARY_EVIDENCE_INDEX_SCHEMA,
        "start_sha": start_sha,
    }
    raw = _canonical_boundary_bytes(index)
    path = tmp_path / f"{boundary}-evidence-index.json"
    path.write_bytes(raw)
    return path, len(raw), hashlib.sha256(raw).hexdigest()


def _stub_boundary_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    partitions = _fixture_sdk_partitions()
    monkeypatch.setattr(
        ratchet,
        "_authenticate_canonical_artifacts",
        lambda **_kwargs: {
            "operation_projection": {"membership_count": 655},
            "original_cohort": {
                "byte_length": 1,
                "record_count": 655,
                "sha256": "8" * 64,
            },
            "sdk_provenance": {
                "byte_length": 1,
                "core_original_record_id_set_sha256": ratchet.CORE_ID_SET_SHA256,
                "core_original_record_ids": partitions["core"],
                "extended_original_record_id_set_sha256": ratchet.EXTENDED_ID_SET_SHA256,
                "extended_original_record_ids": partitions["extended"],
                "record_count": 598,
                "sdk_original_record_id_set_sha256": ratchet.SDK_ID_SET_SHA256,
                "sdk_original_record_ids": sorted(partitions["core"] + partitions["extended"]),
                "sha256": "9" * 64,
            },
        },
    )
    monkeypatch.setattr(
        ratchet,
        "_discover_canonical_artifact",
        lambda _explicit, _descriptor, _repo_root: Path(__file__).resolve(),
    )
    monkeypatch.setattr(
        ratchet,
        "_reauthenticate_canonical_input",
        lambda _path, **_kwargs: None,
    )
    monkeypatch.setattr(
        ratchet,
        "_authenticate_authority_manifest",
        lambda **_kwargs: {
            "authority_manifest_sha256": "1" * 64,
            "boundary_verifier_sha256": "7" * 64,
            "dependency_manifest_sha256": "2" * 64,
            "inventory_sha256": "3" * 64,
            "public_symbol_sha256": "4" * 64,
            "repo_file_count": 42,
            "route_authority_member_count": 2,
            "route_boundary_sha256": "5" * 64,
        },
    )


def _stub_boundary_evidence_index(
    monkeypatch: pytest.MonkeyPatch,
    *,
    index_path: Path,
    index_length: int,
    index_sha256: str,
    pr_additions: int = 400,
    pr_deletions: int = 400,
) -> None:
    state: dict[str, dict[str, Any]] = {}

    def collect(
        **kwargs: Any,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
    ]:
        resources, summary = ratchet._load_evidence_resources(
            evidence_index_path=index_path,
            evidence_index_byte_length=index_length,
            evidence_index_sha256=index_sha256,
            boundary=kwargs["boundary"],
            start_sha=kwargs["start_sha"],
            end_sha=kwargs["end_sha"],
            operation_log=kwargs["operation_log"],
        )
        state["summary"] = summary
        governed = resources["governed_prs"]["records"]
        authenticated_pr_changes = {
            record["pr"]: {
                "additions": pr_additions,
                "deletions": pr_deletions,
            }
            for record in governed
        }
        return (
            resources,
            summary,
            {
                "authenticated_pr_changes": authenticated_pr_changes,
                "expected_rule_suite_ref": "refs/heads/main",
                "fixture_evidence_index": True,
                "repository_id": 1126097105,
                "repository_name": "aragora",
            },
        )

    def reauthenticate(
        _context: dict[str, Any],
        *,
        operation_log: list[dict[str, Any]],
        end_sha: str,
    ) -> dict[str, Any]:
        del end_sha
        return ratchet._reauthenticate_evidence_resources(
            evidence_index_path=index_path,
            evidence_summary=state["summary"],
            operation_log=operation_log,
        )

    monkeypatch.setattr(ratchet, "_collect_live_evidence", collect)
    monkeypatch.setattr(ratchet, "_reauthenticate_live_context", reauthenticate)


def _boundary_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    *,
    release_immutability: bool = True,
    mutate: Any | None = None,
    pr_additions: int = 400,
    pr_deletions: int = 400,
) -> dict[str, Any]:
    repo, start_sha, boundary_shas = _boundary_git_repo(tmp_path)
    end_sha = boundary_shas[boundary]
    index_path, index_length, index_sha256 = _write_boundary_index(
        tmp_path,
        boundary,
        start_sha,
        end_sha,
        boundary_shas,
        repo=repo,
        release_immutability=release_immutability,
        mutate=mutate,
    )
    _stub_boundary_dependencies(monkeypatch)
    _stub_boundary_evidence_index(
        monkeypatch,
        index_path=index_path,
        index_length=index_length,
        index_sha256=index_sha256,
        pr_additions=pr_additions,
        pr_deletions=pr_deletions,
    )
    return ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary=boundary,
        start_ref=start_sha,
        end_ref=end_sha,
    )


def test_boundary_pass_fixture_uses_production_artifact_and_authority_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort_path, provenance_path = _write_synthetic_canonical_artifacts(
        tmp_path,
        monkeypatch,
    )
    repo_root, start_sha, end_sha, _has_accepted_authority = (
        _clone_repository_with_synthetic_accepted_authority(
            tmp_path,
            cohort_path=cohort_path,
            provenance_path=provenance_path,
        )
    )
    operation_log: list[dict[str, Any]] = []
    authority = ratchet._authenticate_authority_manifest(
        repo_root=repo_root,
        end_sha=end_sha,
        authority_manifest_path=None,
        authority_manifest_byte_length=None,
        authority_manifest_sha256=None,
        cohort_artifact_path=cohort_path,
        sdk_provenance_artifact_path=provenance_path,
        scratch_root=tmp_path,
        operation_log=operation_log,
    )
    boundary_shas = dict.fromkeys(ratchet.BOUNDARY_NAMES, end_sha)
    payloads = _boundary_payloads(
        "corrective_bootstrap",
        start_sha,
        end_sha,
        boundary_shas,
        repo=repo_root,
    )
    transition = payloads["corrective_bootstrap"]["corrective_transition"]
    transition["fact"]["commit_count"] = int(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-list",
                "--count",
                f"{start_sha}..{end_sha}",
            ],
            text=True,
        ).strip()
    )
    transition["sha256"] = ratchet._fact_digest(
        "contract-drift-corrective-transition-fact-v1",
        transition["fact"],
    )
    closure_fact = payloads["corrective_bootstrap"]["accepted_stage1_closure"]["fact"]
    closure_fact.update(
        {
            "authority_manifest_sha256": authority["authority_manifest_sha256"],
            "boundary_verifier_sha256": authority["boundary_verifier_sha256"],
            "dependency_manifest_sha256": authority["dependency_manifest_sha256"],
            "inventory_sha256": authority["inventory_sha256"],
            "repo_file_count": authority["repo_file_count"],
        }
    )
    payloads["corrective_bootstrap"]["accepted_stage1_closure"]["sha256"] = ratchet._fact_digest(
        "contract-drift-stage1-closure-fact-v1",
        closure_fact,
    )
    verifier_fact = payloads["corrective_bootstrap"]["stage2_verifier_chronology"]["fact"]
    verifier_fact["verifier_sha256"] = authority["boundary_verifier_sha256"]
    payloads["corrective_bootstrap"]["stage2_verifier_chronology"]["sha256"] = ratchet._fact_digest(
        "contract-drift-stage2-verifier-chronology-fact-v1",
        verifier_fact,
    )
    selected = {
        "boundary_chronology",
        "corrective_bootstrap",
        "durable_capsule",
        "external_prerequisites",
        "first_parent_receipts",
        "governed_prs",
    }
    resources = {name: value for name, value in payloads.items() if name in selected}
    resources["boundary_chronology"]["boundaries"] = resources["boundary_chronology"]["boundaries"][
        :1
    ]
    verification_raw = b'[{"verified":true}]'
    verification_identity = {
        "byte_length": len(verification_raw),
        "sha256": hashlib.sha256(verification_raw).hexdigest(),
    }
    attestation_digest = hashlib.sha256(
        ratchet._canonical_json_bytes([verification_identity] * 7)
    ).hexdigest()
    resources["durable_capsule"]["release"] = {
        "asset_api_ids": [102, 103, 101],
        "asset_names": ["manifest.json", "payload.json", "checksums.txt"],
        "exact_full_sha_tag": end_sha,
        "immutable": True,
        "release_api_id": 100,
        "verified": True,
    }
    resources["durable_capsule"]["attestation"] = {
        "bundle_sha256": attestation_digest,
        "verified": True,
        "workflow": "actions/attest@v4",
    }
    capsule_payload = {
        "boundary": "corrective_bootstrap",
        "end_sha": end_sha,
        "resources": [{"name": name, "value": value} for name, value in sorted(resources.items())],
        "schema": ratchet.BOUNDARY_CAPSULE_PAYLOAD_SCHEMA,
        "start_sha": start_sha,
    }
    payload_raw = _canonical_boundary_bytes(capsule_payload)
    capsule_manifest = {
        "boundary": "corrective_bootstrap",
        "end_sha": end_sha,
        "payload_byte_length": len(payload_raw),
        "payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
        "schema": ratchet.BOUNDARY_CAPSULE_MANIFEST_SCHEMA,
        "start_sha": start_sha,
    }
    manifest_raw = _canonical_boundary_bytes(capsule_manifest)
    checksums_raw = (
        f"{hashlib.sha256(manifest_raw).hexdigest()}  manifest.json\n"
        f"{hashlib.sha256(payload_raw).hexdigest()}  payload.json\n"
    ).encode()
    assets = {
        101: checksums_raw,
        102: manifest_raw,
        103: payload_raw,
    }
    tree_sha = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", f"{end_sha}^{{tree}}"],
        text=True,
    ).strip()
    real_subprocess_run = subprocess.run

    def github_transport(
        argv: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        if Path(argv[0]).name != "gh":
            return real_subprocess_run(argv, *args, **kwargs)
        if argv[1] != "api":
            return subprocess.CompletedProcess(argv, 0, stdout=verification_raw, stderr=b"")
        endpoint = argv[-1]
        if "/releases/assets/" in endpoint:
            body = assets[int(endpoint.rsplit("/", 1)[1])]
        elif endpoint == "repos/synaptent/aragora":
            body = ratchet._canonical_json_bytes(
                {
                    "full_name": "synaptent/aragora",
                    "id": 1126097105,
                    "name": "aragora",
                }
            )
        elif endpoint.endswith("/branches/main"):
            body = ratchet._canonical_json_bytes({"commit": {"sha": end_sha}, "name": "main"})
        elif endpoint.endswith("/branches/main/protection"):
            body = ratchet._canonical_json_bytes({"required_status_checks": {"strict": False}})
        elif endpoint.endswith("/immutable-releases"):
            body = ratchet._canonical_json_bytes({"enabled": True})
        elif endpoint.endswith("/releases?per_page=100&page=1"):
            body = ratchet._canonical_json_bytes([{"id": 100, "tag_name": end_sha}])
        elif endpoint.endswith("/releases/100"):
            body = ratchet._canonical_json_bytes(
                {
                    "assets": [
                        {"id": 101, "name": "checksums.txt"},
                        {"id": 102, "name": "manifest.json"},
                        {"id": 103, "name": "payload.json"},
                    ],
                    "draft": False,
                    "id": 100,
                    "immutable": True,
                    "prerelease": False,
                    "tag_name": end_sha,
                }
            )
        elif "/rulesets/rule-suites?ref=refs/heads/main&time_period=day" in endpoint:
            body = ratchet._canonical_json_bytes([_rule_suite_record(end_sha)])
        elif endpoint.endswith("/rulesets/rule-suites/987654"):
            body = ratchet._canonical_json_bytes(_rule_suite_record(end_sha))
        elif endpoint.endswith("/pulls/9999/files?per_page=100&page=1"):
            body = ratchet._canonical_json_bytes([{"filename": "fixture.txt", "id": 1}])
        elif endpoint.endswith("/pulls/9999"):
            body = ratchet._canonical_json_bytes(
                {
                    "additions": 400,
                    "base": {"sha": start_sha},
                    "changed_files": 1,
                    "deletions": 400,
                    "head": {"sha": end_sha},
                    "merge_commit_sha": end_sha,
                    "merged_at": "2026-07-20T00:00:00Z",
                    "number": 9999,
                }
            )
        elif endpoint.endswith(f"/git/commits/{end_sha}"):
            body = ratchet._canonical_json_bytes(
                {
                    "parents": [{"sha": start_sha}],
                    "sha": end_sha,
                    "tree": {"sha": tree_sha},
                }
            )
        else:
            raise AssertionError(endpoint)
        stdout = (
            b'HTTP/2 200 OK\r\nETag: "fixture"\r\n'
            b"Last-Modified: Mon, 20 Jul 2026 00:00:00 GMT\r\n\r\n" + body
        )
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(subprocess, "run", github_transport)

    result = ratchet.build_boundary_result(
        repo_root=repo_root,
        schema_version=1,
        boundary="corrective_bootstrap",
        start_ref=start_sha,
        end_ref=end_sha,
        cohort_artifact_path=cohort_path,
        sdk_provenance_artifact_path=provenance_path,
        scratch_root=tmp_path,
        output_root=tmp_path,
    )

    assert result["status"] == "pass", result.get("blocked_reason") or result.get("error")
    assert result["canonical_artifacts"]["original_cohort"]["record_count"] == 655
    assert result["canonical_artifacts"]["sdk_provenance"]["record_count"] == 598
    assert result["authority"]["repo_file_count"] > 0
    assert any(
        entry["resource"] == "sigstore-attestation:manifest.json"
        for entry in result["operation_log"]
    )
    authority_reads = [
        entry
        for entry in result["operation_log"]
        if entry["resource"] == "authority_manifest"
        and entry["kind"] in {"authority_reconstruction", "external_authority_manifest"}
    ]
    assert len(authority_reads) == 2
    assert authority_reads[0]["byte_length"] > 0
    assert authority_reads[0]["sha256"] == authority_reads[1]["sha256"]


def test_original_cohort_descriptor_is_immutable_across_authority_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort_path, provenance_path = _write_synthetic_canonical_artifacts(
        tmp_path,
        monkeypatch,
    )
    original_raw = cohort_path.read_bytes()

    def authenticate() -> dict[str, Any]:
        return ratchet._authenticate_canonical_artifacts(
            repo_root=tmp_path,
            cohort_artifact_path=cohort_path,
            sdk_provenance_artifact_path=provenance_path,
            scratch_root=tmp_path,
            operation_log=[],
        )

    def bind_cohort(raw: bytes) -> None:
        cohort_path.write_bytes(raw)
        monkeypatch.setitem(ratchet.COHORT_ARTIFACT, "byte_length", len(raw))
        monkeypatch.setitem(
            ratchet.COHORT_ARTIFACT,
            "sha256",
            hashlib.sha256(raw).hexdigest(),
        )

    with pytest.raises(ValueError, match="byte-length mismatch"):
        cohort_path.write_bytes(original_raw + b" ")
        authenticate()

    cohort_path.write_bytes(original_raw.replace(b'"fixture"', b'"fixturx"', 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        authenticate()

    cohort = json.loads(original_raw)
    cohort["original_record_id_set"]["sha256"] = "0" * 64
    bind_cohort(ratchet._canonical_json_bytes(cohort, terminal_lf=True))
    with pytest.raises(ValueError, match="original-record ID-set digest mismatch"):
        authenticate()

    cohort = json.loads(original_raw)
    cohort["original_records"][1] = copy.deepcopy(cohort["original_records"][0])
    bind_cohort(ratchet._canonical_json_bytes(cohort, terminal_lf=True))
    with pytest.raises(ValueError, match="duplicate original record IDs"):
        authenticate()


def test_canonical_cohort_and_provenance_artifacts_are_in_authority_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort_path, provenance_path = _write_synthetic_canonical_artifacts(
        tmp_path,
        monkeypatch,
    )
    original_raw = provenance_path.read_bytes()
    provenance = json.loads(original_raw)
    provenance["records"][1] = copy.deepcopy(provenance["records"][0])
    duplicate_raw = ratchet._canonical_json_bytes(provenance, terminal_lf=True)
    provenance_path.write_bytes(duplicate_raw)
    monkeypatch.setitem(ratchet.PROVENANCE_ARTIFACT, "byte_length", len(duplicate_raw))
    monkeypatch.setitem(
        ratchet.PROVENANCE_ARTIFACT,
        "sha256",
        hashlib.sha256(duplicate_raw).hexdigest(),
    )

    with pytest.raises(ValueError, match="does not biject"):
        ratchet._authenticate_canonical_artifacts(
            repo_root=tmp_path,
            cohort_artifact_path=cohort_path,
            sdk_provenance_artifact_path=provenance_path,
            scratch_root=tmp_path,
            operation_log=[],
        )

    provenance_path.write_bytes(original_raw)
    monkeypatch.setitem(ratchet.PROVENANCE_ARTIFACT, "byte_length", len(original_raw))
    monkeypatch.setitem(
        ratchet.PROVENANCE_ARTIFACT,
        "sha256",
        hashlib.sha256(original_raw).hexdigest(),
    )
    repo_root, production_sha, end_sha, has_accepted_authority = (
        _clone_repository_with_synthetic_accepted_authority(
            tmp_path,
            cohort_path=cohort_path,
            provenance_path=provenance_path,
        )
    )
    if has_accepted_authority:
        with pytest.raises(
            gen.AuthorityClosureError,
            match="accepted authority differs from authenticated canonical artifacts",
        ):
            ratchet._authenticate_authority_manifest(
                repo_root=repo_root,
                end_sha=production_sha,
                authority_manifest_path=None,
                authority_manifest_byte_length=None,
                authority_manifest_sha256=None,
                cohort_artifact_path=cohort_path,
                sdk_provenance_artifact_path=provenance_path,
                scratch_root=tmp_path,
                operation_log=[],
            )
    captured: dict[str, Any] = {}
    original_build = ratchet.inventory_mod.build_authority_manifest

    def capture_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
        manifest = original_build(*args, **kwargs)
        captured["manifest"] = manifest
        return manifest

    monkeypatch.setattr(
        ratchet.inventory_mod,
        "build_authority_manifest",
        capture_manifest,
    )
    ratchet._authenticate_authority_manifest(
        repo_root=repo_root,
        end_sha=end_sha,
        authority_manifest_path=None,
        authority_manifest_byte_length=None,
        authority_manifest_sha256=None,
        cohort_artifact_path=cohort_path,
        sdk_provenance_artifact_path=provenance_path,
        scratch_root=tmp_path,
        operation_log=[],
    )

    external_artifacts = captured["manifest"]["inventory"]["external_artifacts"]
    assert external_artifacts == sorted(
        [
            {
                "byte_length": cohort_path.stat().st_size,
                "canonical_bytes": True,
                "path": cohort_path.name,
                "sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
            },
            {
                "byte_length": provenance_path.stat().st_size,
                "canonical_bytes": True,
                "path": provenance_path.name,
                "sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            },
        ],
        key=lambda item: item["path"],
    )


def test_external_authority_manifest_and_evidence_index_bytes_are_canonical_before_semantic_digest(
    tmp_path: Path,
):
    canonical = tmp_path / "canonical.json"
    descriptor = _write_canonical_boundary_json(canonical, {"schema": "fixture", "value": 1})
    payload, authenticated = ratchet._load_canonical_json_bytes(
        canonical,
        label="fixture",
        expected_byte_length=descriptor["byte_length"],
        expected_sha256=descriptor["sha256"],
        terminal_lf=True,
    )
    assert payload == {"schema": "fixture", "value": 1}
    assert authenticated["canonical_bytes_valid"] is True

    for raw, message in (
        (b"\xef\xbb\xbf" + canonical.read_bytes(), "BOM"),
        (b'{ "schema": "fixture", "value": 1 }\n', "canonical"),
        (b'{"schema":"fixture","value":1}', "terminal LF"),
        (b'{"value":1,"schema":"fixture"}\n', "canonical"),
    ):
        candidate = tmp_path / f"bad-{hashlib.sha256(raw).hexdigest()}.json"
        candidate.write_bytes(raw)
        with pytest.raises(ValueError, match=message):
            ratchet._load_canonical_json_bytes(
                candidate,
                label="fixture",
                expected_byte_length=len(raw),
                expected_sha256=hashlib.sha256(raw).hexdigest(),
                terminal_lf=True,
            )


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--evidence-index", "forged.json"),
        ("--github-repository", "attacker/mirror"),
        ("--github-branch", "forged"),
    ),
)
def test_boundary_cli_rejects_caller_supplied_authority(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_contract_drift_ratchet.py",
            "--mode",
            "boundary",
            "--schema-version",
            "1",
            "--boundary",
            "corrective_bootstrap",
            "--start-ref",
            "0" * 40,
            "--end-ref",
            "1" * 40,
            flag,
            value,
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        ratchet.main()
    assert exc_info.value.code == 2


def test_boundary_python_api_rejects_caller_supplied_evidence_or_github_trust_roots():
    parameters = inspect.signature(ratchet.build_boundary_result).parameters
    assert "evidence_index_path" not in parameters
    assert "evidence_index_byte_length" not in parameters
    assert "evidence_index_sha256" not in parameters
    assert "github_repository" not in parameters
    assert "github_branch" not in parameters


def test_parse_http_response_preserves_exact_body_bytes():
    body = b"binary\r\nbody\n\nHTTP/1.1 200 OK\r\nnot-a-header"
    headers, parsed = ratchet._parse_http_response(
        b'HTTP/1.1 200 OK\r\nETag: "fixture"\r\nContent-Type: application/octet-stream'
        b"\r\n\r\n" + body
    )
    assert headers["etag"] == '"fixture"'
    assert parsed == body

    headers, parsed = ratchet._parse_http_response(
        b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\nETag: final\n\npayload\r\nbytes"
    )
    assert headers["etag"] == "final"
    assert parsed == b"payload\r\nbytes"

    with pytest.raises(ValueError, match="authenticated HTTP headers"):
        ratchet._parse_http_response(b"headerless")


def test_boundary_rule_suite_binds_repository_main_ref_and_end_sha():
    end_sha = "1" * 40
    rule_suite = _rule_suite_claim(end_sha)
    operation_log: list[dict[str, Any]] = []

    authenticated = ratchet._authenticate_persisted_rule_suite_claim(
        rule_suite,
        repository_id=1126097105,
        repository_name="aragora",
        expected_ref="refs/heads/main",
        end_sha=end_sha,
        operation_log=operation_log,
    )
    ratchet._validate_current_rule_suite_binding(
        rule_suite,
        observed_rule_suite=_rule_suite_record(end_sha),
        observed_identity={
            "byte_length": rule_suite["raw_response_byte_length"],
            "raw_response": rule_suite["raw_response"],
            "response_fields": _rule_suite_record(end_sha),
            "sha256": rule_suite["raw_response_sha256"],
        },
        repository_id=1126097105,
        repository_name="aragora",
        expected_ref="refs/heads/main",
        end_sha=end_sha,
        operation_log=operation_log,
    )

    assert authenticated["repository_id"] == 1126097105
    assert authenticated["repository_name"] == "aragora"
    assert authenticated["ref"] == "refs/heads/main"
    assert authenticated["after_sha"] == end_sha
    assert authenticated["result"] == "pass"
    raw_entries = [
        entry
        for entry in operation_log
        if entry["resource"] == "github-rule-suite-seal-time-response"
    ]
    assert raw_entries
    assert raw_entries[0]["raw_response"] == rule_suite["raw_response"]
    assert raw_entries[0]["response_fields"] == _rule_suite_record(end_sha)


@pytest.mark.parametrize(
    ("label", "claim"),
    (
        pytest.param(
            "stale-after-sha",
            _rule_suite_claim("2" * 40),
            id="stale-after-sha",
        ),
        pytest.param(
            "wrong-repository-id",
            _rule_suite_claim("1" * 40, repository_id=7),
            id="wrong-repository-id",
        ),
        pytest.param(
            "wrong-repository-name",
            _rule_suite_claim("1" * 40, repository_name="mirror"),
            id="wrong-repository-name",
        ),
        pytest.param(
            "wrong-ref",
            _rule_suite_claim("1" * 40, ref="refs/heads/feature"),
            id="wrong-ref",
        ),
        pytest.param(
            "masked-ref",
            _rule_suite_claim("1" * 40, ref="refs/__gh__/UNKNOWN"),
            id="masked-ref",
        ),
        pytest.param(
            "missing-after-sha",
            _rule_suite_claim("1" * 40, delete="after_sha"),
            id="missing-after-sha",
        ),
        pytest.param(
            "null-repository-id",
            _rule_suite_claim("1" * 40, repository_id=None),
            id="null-repository-id",
        ),
        pytest.param(
            "plain-result-fail",
            _rule_suite_claim("1" * 40, result="fail"),
            id="plain-result-fail",
        ),
        pytest.param(
            "result-bypass",
            _rule_suite_claim("1" * 40, result="bypass"),
            id="result-bypass",
        ),
        pytest.param(
            "evaluation-bypass",
            _rule_suite_claim("1" * 40, evaluation_result="bypass"),
            id="evaluation-bypass",
        ),
        pytest.param(
            "nested-evaluation-bypass",
            _rule_suite_claim(
                "1" * 40,
                rule_evaluations=[{"result": "bypass", "rule_source": {"type": "repository"}}],
            ),
            id="nested-evaluation-bypass",
        ),
        pytest.param(
            "capsule-bypassed",
            _rule_suite_claim("1" * 40, bypassed=True),
            id="capsule-bypassed",
        ),
    ),
)
def test_stale_wrong_repository_wrong_ref_missing_fields_or_bypassed_rule_suite_fails_closed(
    label: str,
    claim: dict[str, Any],
):
    if label != "capsule-bypassed":
        with pytest.raises(ValueError):
            ratchet._select_current_rule_suite_candidate(
                [json.loads(claim["raw_response"])],
                rule_suite_id=987654,
                repository_id=1126097105,
                repository_name="aragora",
                expected_ref="refs/heads/main",
                end_sha="1" * 40,
            )
    with pytest.raises(ValueError) as exc_info:
        ratchet._authenticate_persisted_rule_suite_claim(
            claim,
            repository_id=1126097105,
            repository_name="aragora",
            expected_ref="refs/heads/main",
            end_sha="1" * 40,
            operation_log=[],
        )
    assert str(exc_info.value), label


def test_blocked_boundary_exit_code_is_distinct_from_argparse_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        ratchet,
        "build_boundary_result",
        lambda **_kwargs: {
            "blocked_reason": "authenticated prerequisite unavailable",
            "manifest_sha256": "a" * 64,
            "status": "blocked",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_contract_drift_ratchet.py",
            "--mode",
            "boundary",
            "--schema-version",
            "1",
            "--boundary",
            "corrective_bootstrap",
            "--start-ref",
            "0" * 40,
            "--end-ref",
            "1" * 40,
        ],
    )
    assert ratchet.main() == 3


def test_scratch_asset_write_rejects_preexisting_symlink(tmp_path: Path):
    target = tmp_path / "target"
    target.write_bytes(b"preserve")
    candidate = tmp_path / "asset"
    candidate.symlink_to(target)

    with pytest.raises(ValueError, match="created exclusively"):
        ratchet._write_exclusive_private_file(
            candidate,
            b"replacement",
            scratch_root=tmp_path,
            output_root=tmp_path,
        )

    assert target.read_bytes() == b"preserve"


def test_nonzero_read_only_probe_is_not_logged_as_authenticated_pass(tmp_path: Path):
    repo, start_sha, boundary_shas = _boundary_git_repo(tmp_path)
    operation_log: list[dict[str, Any]] = []
    proc = ratchet._run_read_only(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            boundary_shas["final_seal"],
            start_sha,
        ],
        operation_log=operation_log,
        resource="negative-ancestry-probe",
        check=False,
    )

    assert proc.returncode == 1
    assert operation_log[-1]["authentication"] == "observed_nonzero"


def test_boundary_verifier_independently_reads_resources_and_emits_own_operation_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    result = _boundary_result(tmp_path, monkeypatch, "corrective_bootstrap")
    assert result["status"] == "pass"
    resources = {
        entry["resource"]
        for entry in result["operation_log"]
        if entry["kind"] == "external_resource"
    }
    assert "corrective_bootstrap" in resources
    assert "external_prerequisites" in resources
    assert all(entry["authentication"] == "pass" for entry in result["operation_log"])
    assert result["evidence"]["resource_count"] == len(resources)


def test_caller_summaries_and_parse_reserialize_are_not_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, start_sha, boundary_shas = _boundary_git_repo(tmp_path)
    boundary = "corrective_bootstrap"
    end_sha = boundary_shas[boundary]
    index_path, index_length, index_sha256 = _write_boundary_index(
        tmp_path,
        boundary,
        start_sha,
        end_sha,
        boundary_shas,
        repo=repo,
    )
    index = json.loads(index_path.read_text())
    index["summary"] = {"status": "pass", "resource_count": len(index["resources"])}
    raw = _canonical_boundary_bytes(index)
    index_path.write_bytes(raw)
    _stub_boundary_dependencies(monkeypatch)
    _stub_boundary_evidence_index(
        monkeypatch,
        index_path=index_path,
        index_length=len(raw),
        index_sha256=hashlib.sha256(raw).hexdigest(),
    )
    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary=boundary,
        start_ref=start_sha,
        end_ref=end_sha,
    )
    assert result["status"] == "fail"
    assert "caller-supplied" in result["error"]

    pretty = (
        json.dumps(json.loads(index_path.read_text()), indent=2, sort_keys=True).encode() + b"\n"
    )
    index_path.write_bytes(pretty)
    _stub_boundary_evidence_index(
        monkeypatch,
        index_path=index_path,
        index_length=len(pretty),
        index_sha256=hashlib.sha256(pretty).hexdigest(),
    )
    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary=boundary,
        start_ref=start_sha,
        end_ref=end_sha,
    )
    assert result["status"] == "fail"
    assert "terminal LF" in result["error"] or "canonical" in result["error"]


def test_boundary_predicates_are_distinct_nonempty_strictly_ordered_and_start_differs_from_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    results = [
        _boundary_result(tmp_path / boundary, monkeypatch, boundary)
        for boundary in ratchet.BOUNDARY_NAMES
    ]
    predicate_shapes = []
    for result, boundary in zip(results, ratchet.BOUNDARY_NAMES, strict=True):
        assert result["start_sha"] != result["end_sha"]
        assert result["status"] == "pass"
        selected = result["predicates"]
        assert selected
        assert list(selected) == list(
            ratchet.BOUNDARY_NAMES[: ratchet.BOUNDARY_NAMES.index(boundary) + 1]
        )
        assert all(selected[name]["proven"] is True for name in selected)
        predicate_shapes.append(tuple(selected[boundary]["checks"]))
    assert len(set(predicate_shapes)) == len(ratchet.BOUNDARY_NAMES)


def test_boundary_status_pass_requires_all_predicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def break_route(payloads: dict[str, dict[str, Any]]) -> None:
        payloads["route_truth"]["route_truth"]["fact"]["complete"] = False

    result = _boundary_result(
        tmp_path,
        monkeypatch,
        "core_sdk",
        mutate=break_route,
    )
    assert result["status"] == "fail"
    assert not result["passing"]
    assert "route truth" in result["error"]


def test_boundary_status_blocked_is_only_verified_external_prerequisite_or_movement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    blocked = _boundary_result(
        tmp_path / "prerequisite",
        monkeypatch,
        "corrective_bootstrap",
        release_immutability=False,
    )
    assert blocked["status"] == "blocked"
    assert "future GitHub Release immutability" in blocked["blocked_reason"]

    repo, start_sha, boundary_shas = _boundary_git_repo(tmp_path / "movement")
    _stub_boundary_dependencies(monkeypatch)
    monkeypatch.setattr(
        ratchet,
        "_collect_live_evidence",
        lambda **_kwargs: (_ for _ in ()).throw(
            ratchet.BoundaryBlocked("authenticated remote resource moved concurrently")
        ),
    )
    movement = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary="corrective_bootstrap",
        start_ref=start_sha,
        end_ref=boundary_shas["corrective_bootstrap"],
    )
    assert movement["status"] == "blocked"
    assert "moved concurrently" in movement["blocked_reason"]

    repo, start_sha, boundary_shas = _boundary_git_repo(tmp_path / "no-main-evaluation")
    monkeypatch.setattr(
        ratchet,
        "_collect_live_evidence",
        lambda **_kwargs: ratchet._select_current_rule_suite_candidate(
            [],
            rule_suite_id=987654,
            repository_id=1126097105,
            repository_name="aragora",
            expected_ref="refs/heads/main",
            end_sha=boundary_shas["corrective_bootstrap"],
        ),
    )
    no_main_evaluation = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary="corrective_bootstrap",
        start_ref=start_sha,
        end_ref=boundary_shas["corrective_bootstrap"],
    )
    assert no_main_evaluation["status"] == "blocked"
    assert "absence of a main rule evaluation" in no_main_evaluation["blocked_reason"]


def test_boundary_status_fail_covers_malformed_false_missing_bypass_and_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    mutations = {
        "malformed": lambda payloads: payloads["core_sdk"].update(
            {"qualifying_paydown": "trusted summary"}
        ),
        "false": lambda payloads: payloads["core_sdk"]["zero_core_debt"]["fact"].update(
            {"remaining_original_units": 1}
        ),
        "missing": lambda payloads: payloads.pop("core_sdk"),
        "bypass": lambda payloads: payloads["external_prerequisites"]["rule_suite"].update(
            {"bypassed": True}
        ),
        "mutation": lambda payloads: payloads["external_prerequisites"].update(
            {"mutation_tainted": True}
        ),
    }
    for label, mutate in mutations.items():
        result = _boundary_result(
            tmp_path / label,
            monkeypatch,
            "core_sdk",
            mutate=mutate,
        )
        assert result["status"] == "fail", (label, result)
        assert not result["passing"]


@pytest.mark.parametrize(
    ("pr_additions", "pr_deletions", "max_pr_delta", "expected_error"),
    (
        pytest.param(401, 400, 800, "exceeds the 800-line cap", id="live-pr-over-cap"),
        pytest.param(400, 400, 799, "max_pr_delta", id="paydown-max-mismatch"),
    ),
)
def test_paydown_pr_delta_is_authenticated_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pr_additions: int,
    pr_deletions: int,
    max_pr_delta: int,
    expected_error: str,
):
    def mutate(payloads: dict[str, dict[str, Any]]) -> None:
        paydown = payloads["core_sdk"]["qualifying_paydown"]
        paydown["fact"]["max_pr_delta"] = max_pr_delta
        paydown["sha256"] = ratchet._fact_digest(
            "contract-drift-core-sdk-paydown-fact-v1",
            paydown["fact"],
        )

    result = _boundary_result(
        tmp_path,
        monkeypatch,
        "core_sdk",
        mutate=mutate,
        pr_additions=pr_additions,
        pr_deletions=pr_deletions,
    )

    assert result["status"] == "fail"
    assert expected_error in result["error"]


def test_canonical_route_fact_fails_when_exact_ref_baseline_contradicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, start_sha, boundary_shas = _boundary_git_repo(tmp_path, route_debt=True)
    end_sha = boundary_shas["route_truth"]
    index_path, index_length, index_sha256 = _write_boundary_index(
        tmp_path,
        "route_truth",
        start_sha,
        end_sha,
        boundary_shas,
        repo=repo,
    )
    _stub_boundary_dependencies(monkeypatch)
    _stub_boundary_evidence_index(
        monkeypatch,
        index_path=index_path,
        index_length=index_length,
        index_sha256=index_sha256,
    )

    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary="route_truth",
        start_ref=start_sha,
        end_ref=end_sha,
    )

    assert result["status"] == "fail"
    assert "contradicted by exact-ref route baselines" in result["error"]


@pytest.mark.parametrize(
    ("category", "path_name"),
    (
        ("python_sdk_drift", "verify_sdk_contracts.json"),
        ("typescript_sdk_drift", "verify_sdk_contracts.json"),
        ("missing_in_spec", "validate_openapi_routes.json"),
        ("orphaned_in_spec", "validate_openapi_routes.json"),
        ("missing_from_both_sdks", "check_sdk_parity.json"),
    ),
)
@pytest.mark.parametrize(
    ("mode", "value"),
    (
        ("missing", None),
        ("null", None),
        ("string", "not-a-list"),
        ("object", {"not": "a-list"}),
        ("mixed", ["valid", 7]),
    ),
)
def test_exact_ref_baseline_categories_require_lists_of_strings(
    tmp_path: Path,
    category: str,
    path_name: str,
    mode: str,
    value: Any,
):
    repo, _start_sha, _boundary_shas = _boundary_git_repo(tmp_path)
    path = repo / "scripts" / "baselines" / path_name
    payload = json.loads(path.read_text())
    if mode == "missing":
        payload.pop(category)
    else:
        payload[category] = value
    _write_json(path, payload)
    ref = _commit(repo, f"malformed {category} {mode}")

    with pytest.raises(ValueError, match=category):
        ratchet._baseline_category_counts_at_ref(
            repo,
            ref,
            operation_log=[],
        )


def test_exact_ref_baseline_categories_allow_empty_string_lists(tmp_path: Path):
    repo, _start_sha, boundary_shas = _boundary_git_repo(tmp_path)

    assert ratchet._baseline_category_counts_at_ref(
        repo,
        boundary_shas["final_seal"],
        operation_log=[],
    ) == {
        "python_sdk_drift": 0,
        "routes_missing_in_spec": 0,
        "routes_orphaned_in_spec": 0,
        "sdk_missing_from_both": 0,
        "typescript_sdk_drift": 0,
    }


def test_later_boundary_fails_when_route_debt_is_reintroduced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, start_sha, boundary_shas = _boundary_git_repo(
        tmp_path,
        route_debt_at="core_sdk",
    )
    end_sha = boundary_shas["core_sdk"]
    index_path, index_length, index_sha256 = _write_boundary_index(
        tmp_path,
        "core_sdk",
        start_sha,
        end_sha,
        boundary_shas,
        repo=repo,
    )
    _stub_boundary_dependencies(monkeypatch)
    _stub_boundary_evidence_index(
        monkeypatch,
        index_path=index_path,
        index_length=index_length,
        index_sha256=index_sha256,
    )

    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary="core_sdk",
        start_ref=start_sha,
        end_ref=end_sha,
    )

    assert result["status"] == "fail"
    assert "contradicted by exact-ref route baselines" in result["error"]


@pytest.mark.parametrize(
    ("boundary", "partition"),
    (("core_sdk", "core"), ("extended_sdk", "extended")),
)
def test_sdk_zero_debt_fails_when_exact_ref_baseline_contradicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    partition: str,
):
    repo, start_sha, boundary_shas = _boundary_git_repo(
        tmp_path,
        sdk_debt_partition=partition,
        sdk_debt_at=boundary,
    )
    end_sha = boundary_shas[boundary]
    index_path, index_length, index_sha256 = _write_boundary_index(
        tmp_path,
        boundary,
        start_sha,
        end_sha,
        boundary_shas,
        repo=repo,
    )
    _stub_boundary_dependencies(monkeypatch)
    _stub_boundary_evidence_index(
        monkeypatch,
        index_path=index_path,
        index_length=index_length,
        index_sha256=index_sha256,
    )

    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary=boundary,
        start_ref=start_sha,
        end_ref=end_sha,
    )

    assert result["status"] == "fail"
    assert "contradicted by exact-ref SDK category baselines" in result["error"]


def test_core_sdk_allows_remaining_extended_exact_ref_baseline_debt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, start_sha, boundary_shas = _boundary_git_repo(
        tmp_path,
        sdk_debt_partition="extended",
        sdk_debt_at="core_sdk",
    )
    end_sha = boundary_shas["core_sdk"]
    index_path, index_length, index_sha256 = _write_boundary_index(
        tmp_path,
        "core_sdk",
        start_sha,
        end_sha,
        boundary_shas,
        repo=repo,
    )
    _stub_boundary_dependencies(monkeypatch)
    _stub_boundary_evidence_index(
        monkeypatch,
        index_path=index_path,
        index_length=index_length,
        index_sha256=index_sha256,
    )

    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary="core_sdk",
        start_ref=start_sha,
        end_ref=end_sha,
    )

    assert result["status"] == "pass", result


def test_governed_prs_and_receipts_must_reconcile_exact_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    result = _boundary_result(
        tmp_path,
        monkeypatch,
        "core_sdk",
        mutate=lambda payloads: payloads["first_parent_receipts"]["records"][1].update(
            {"pr": 123456}
        ),
    )

    assert result["status"] == "fail"
    assert "do not reconcile" in result["error"]


def test_evidence_reauthentication_blocks_toctou_movement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, start_sha, boundary_shas = _boundary_git_repo(tmp_path)
    end_sha = boundary_shas["corrective_bootstrap"]
    index_path, index_length, index_sha256 = _write_boundary_index(
        tmp_path,
        "corrective_bootstrap",
        start_sha,
        end_sha,
        boundary_shas,
        repo=repo,
    )
    _stub_boundary_dependencies(monkeypatch)
    _stub_boundary_evidence_index(
        monkeypatch,
        index_path=index_path,
        index_length=index_length,
        index_sha256=index_sha256,
    )
    original_evaluate = ratchet._evaluate_boundary_evidence

    def evaluate_and_move(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original_evaluate(*args, **kwargs)
        resource = tmp_path / "resources-corrective_bootstrap" / "governed_prs.json"
        resource.write_bytes(resource.read_bytes() + b" ")
        return result

    monkeypatch.setattr(ratchet, "_evaluate_boundary_evidence", evaluate_and_move)

    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary="corrective_bootstrap",
        start_ref=start_sha,
        end_ref=end_sha,
    )

    assert result["status"] == "blocked"
    assert "moved concurrently" in result["blocked_reason"]


def test_deterministic_boundary_fixtures_reach_pass_while_live_release_immutability_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        sys.modules[__name__],
        "_canonical_fixture_artifact_paths",
        lambda: None,
    )
    for boundary in ratchet.BOUNDARY_NAMES:
        result = _boundary_result(tmp_path / boundary, monkeypatch, boundary)
        assert result["status"] == "pass", result

    live = _boundary_result(
        tmp_path / "live-blocked",
        monkeypatch,
        "final_seal",
        release_immutability=False,
    )
    assert live["status"] == "blocked"
    assert live["passing"] is False


def test_boundary_uses_private_scratch_child_and_removes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, start_sha, boundary_shas = _boundary_git_repo(tmp_path / "repo")
    boundary = "corrective_bootstrap"
    end_sha = boundary_shas[boundary]
    index_path, index_length, index_sha256 = _write_boundary_index(
        tmp_path,
        boundary,
        start_sha,
        end_sha,
        boundary_shas,
        repo=repo,
    )
    _stub_boundary_dependencies(monkeypatch)
    _stub_boundary_evidence_index(
        monkeypatch,
        index_path=index_path,
        index_length=index_length,
        index_sha256=index_sha256,
    )
    original_collect = ratchet._collect_live_evidence
    observed: dict[str, Path] = {}

    def collect(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        observed["scratch_root"] = kwargs["scratch_root"]
        return original_collect(**kwargs)

    monkeypatch.setattr(ratchet, "_collect_live_evidence", collect)
    shared_parent = tmp_path / "shared-scratch"
    shared_parent.mkdir()
    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary=boundary,
        start_ref=start_sha,
        end_ref=end_sha,
        scratch_root=shared_parent,
    )

    private_root = observed["scratch_root"]
    assert result["status"] == "pass", result
    assert private_root.parent == shared_parent.resolve()
    assert private_root.name.startswith("contract-drift-boundary-")
    assert not private_root.exists()


def test_unexpected_boundary_exception_fails_closed_and_cleans_private_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, start_sha, boundary_shas = _boundary_git_repo(tmp_path / "repo")
    observed: dict[str, Path] = {}
    original_temporary_directory = ratchet.tempfile.TemporaryDirectory

    def tracking_temporary_directory(*args: Any, **kwargs: Any) -> Any:
        directory = original_temporary_directory(*args, **kwargs)
        observed["scratch_root"] = Path(directory.name)
        return directory

    monkeypatch.setattr(
        ratchet.tempfile,
        "TemporaryDirectory",
        tracking_temporary_directory,
    )
    monkeypatch.setattr(
        ratchet,
        "_snapshot_repository",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("hostile boundary crash")),
    )

    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary="corrective_bootstrap",
        start_ref=start_sha,
        end_ref=boundary_shas["corrective_bootstrap"],
        scratch_root=tmp_path,
    )

    assert result["status"] == "fail"
    assert result["error_code"] == "boundary_unexpected_exception"
    assert result["error"] == "unexpected boundary exception: RuntimeError"
    assert not observed["scratch_root"].exists()


@pytest.mark.parametrize(
    ("wrong_first_parent", "expected_error"),
    (
        pytest.param(False, None, id="ignores-pr-api-base-sha"),
        pytest.param(
            True,
            "lacks first-parent or tree equality",
            id="rejects-wrong-merge-first-parent",
        ),
    ),
)
def test_live_evidence_authenticates_base_from_merge_first_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wrong_first_parent: bool,
    expected_error: str | None,
):
    _repo, start_sha, boundary_shas = _boundary_git_repo(tmp_path)
    end_sha = boundary_shas["corrective_bootstrap"]
    resources = _boundary_payloads(
        "corrective_bootstrap",
        start_sha,
        end_sha,
        boundary_shas,
        repo=_repo,
    )
    selected = {
        "boundary_chronology",
        "corrective_bootstrap",
        "durable_capsule",
        "external_prerequisites",
        "first_parent_receipts",
        "governed_prs",
    }
    resources = {name: value for name, value in resources.items() if name in selected}
    resources["boundary_chronology"]["boundaries"] = resources["boundary_chronology"]["boundaries"][
        :1
    ]

    verification_identity = {"byte_length": 18, "sha256": "a" * 64}
    verification_identities = [verification_identity] * 7
    attestation_digest = hashlib.sha256(
        ratchet._canonical_json_bytes(verification_identities)
    ).hexdigest()
    resources["durable_capsule"]["release"] = {
        "asset_api_ids": [102, 103, 101],
        "asset_names": ["manifest.json", "payload.json", "checksums.txt"],
        "exact_full_sha_tag": end_sha,
        "immutable": True,
        "release_api_id": 100,
        "verified": True,
    }
    resources["durable_capsule"]["attestation"] = {
        "bundle_sha256": attestation_digest,
        "verified": True,
        "workflow": "actions/attest@v4",
    }
    payload = {
        "boundary": "corrective_bootstrap",
        "end_sha": end_sha,
        "resources": [{"name": name, "value": value} for name, value in sorted(resources.items())],
        "schema": ratchet.BOUNDARY_CAPSULE_PAYLOAD_SCHEMA,
        "start_sha": start_sha,
    }
    payload_raw = _canonical_boundary_bytes(payload)
    manifest = {
        "boundary": "corrective_bootstrap",
        "end_sha": end_sha,
        "payload_byte_length": len(payload_raw),
        "payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
        "schema": ratchet.BOUNDARY_CAPSULE_MANIFEST_SCHEMA,
        "start_sha": start_sha,
    }
    manifest_raw = _canonical_boundary_bytes(manifest)
    checksums_raw = (
        f"{hashlib.sha256(manifest_raw).hexdigest()}  manifest.json\n"
        f"{hashlib.sha256(payload_raw).hexdigest()}  payload.json\n"
    ).encode()
    assets = {
        101: checksums_raw,
        102: manifest_raw,
        103: payload_raw,
    }
    identity = {
        "byte_length": 2,
        "etag": '"stable"',
        "sha256": hashlib.sha256(b"{}").hexdigest(),
        "updated_at": "2026-07-20T00:00:00Z",
    }

    def stable_get(
        endpoint: str,
        *,
        operation_log: list[dict[str, Any]],
        attempts: int = 3,
        preserve_raw: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        del operation_log, attempts
        if endpoint == "repos/synaptent/aragora":
            return {
                "full_name": "synaptent/aragora",
                "id": 1126097105,
                "name": "aragora",
            }, identity
        if endpoint.endswith("/branches/main"):
            return {"commit": {"sha": end_sha}, "name": "main"}, identity
        if endpoint.endswith("/branches/main/protection"):
            return {"required_status_checks": {"strict": False}}, identity
        if endpoint.endswith("/immutable-releases"):
            return {"enabled": True}, identity
        if endpoint.endswith("/releases/100"):
            return {
                "assets": [
                    {"id": 101, "name": "checksums.txt"},
                    {"id": 102, "name": "manifest.json"},
                    {"id": 103, "name": "payload.json"},
                ],
                "draft": False,
                "id": 100,
                "immutable": True,
                "prerelease": False,
                "tag_name": end_sha,
            }, identity
        if endpoint.endswith("/rulesets/rule-suites/987654"):
            payload = _rule_suite_record(end_sha)
            raw = ratchet._canonical_json_bytes(payload).decode("utf-8")
            rule_suite_identity = dict(identity)
            if preserve_raw:
                rule_suite_identity.update(
                    {
                        "byte_length": len(raw.encode("utf-8")),
                        "raw_response": raw,
                        "response_fields": payload,
                        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    }
                )
            return payload, rule_suite_identity
        if endpoint.endswith("/pulls/9999"):
            return {
                "additions": 400,
                "base": {"sha": "f" * 40},
                "changed_files": 1,
                "deletions": 400,
                "head": {"sha": end_sha},
                "merge_commit_sha": end_sha,
                "merged_at": "2026-07-20T00:00:00Z",
                "number": 9999,
            }, identity
        if endpoint.endswith(f"/git/commits/{end_sha}"):
            tree_sha = resources["governed_prs"]["records"][0]["head_tree_sha"]
            return {
                "parents": [{"sha": "e" * 40 if wrong_first_parent else start_sha}],
                "sha": end_sha,
                "tree": {"sha": tree_sha},
            }, identity
        raise AssertionError(endpoint)

    def paginated_get(
        endpoint: str,
        *,
        operation_log: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        del operation_log
        if endpoint.endswith("/releases"):
            return [{"id": 100, "tag_name": end_sha}], {f"{endpoint}?page=1": identity}
        if endpoint.endswith("/rulesets/rule-suites?ref=refs/heads/main&time_period=day"):
            return [_rule_suite_record(end_sha)], {f"{endpoint}&page=1": identity}
        if endpoint.endswith("/pulls/9999/files"):
            return [{"filename": "fixture.txt", "id": 1}], {f"{endpoint}?page=1": identity}
        raise AssertionError(endpoint)

    def raw_get(
        endpoint: str,
        *,
        operation_log: list[dict[str, Any]],
        attempts: int = 3,
    ) -> tuple[bytes, dict[str, Any]]:
        del operation_log, attempts
        asset_id = int(endpoint.rsplit("/", 1)[1])
        raw = assets[asset_id]
        return raw, {
            "byte_length": len(raw),
            "etag": f'"asset-{asset_id}"',
            "sha256": hashlib.sha256(raw).hexdigest(),
            "updated_at": "2026-07-20T00:00:00Z",
        }

    monkeypatch.setattr(ratchet, "_gh_api_get_stable", stable_get)
    monkeypatch.setattr(ratchet, "_gh_api_paginated", paginated_get)
    monkeypatch.setattr(ratchet, "_gh_api_get_raw_stable", raw_get)
    verification_commands: list[list[str]] = []

    def verify(
        argv: list[str],
        *,
        operation_log: list[dict[str, Any]],
        resource: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del operation_log
        verification_commands.append(argv)
        if argv[1:3] == ["attestation", "verify"]:
            source_index = argv.index("--source-digest")
            assert argv[source_index + 1] == end_sha
        return {"resource": resource, "verified": bool(argv)}, verification_identity

    monkeypatch.setattr(ratchet, "_run_live_verification", verify)

    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            ratchet._collect_live_evidence(
                github_repository="synaptent/aragora",
                github_branch="main",
                boundary="corrective_bootstrap",
                start_sha=start_sha,
                end_sha=end_sha,
                scratch_root=tmp_path,
                operation_log=[],
            )
        return

    discovered, summary, context = ratchet._collect_live_evidence(
        github_repository="synaptent/aragora",
        github_branch="main",
        boundary="corrective_bootstrap",
        start_sha=start_sha,
        end_sha=end_sha,
        scratch_root=tmp_path,
        operation_log=[],
    )

    assert set(discovered) == selected
    assert summary["source"] == "immutable_github_release"
    assert summary["resource_count"] == len(selected)
    assert len(context["asset_identities"]) == 3
    assert any(endpoint.endswith("/pulls/9999") for endpoint in context["endpoint_identities"])
    assert (
        len([argv for argv in verification_commands if argv[1:3] == ["attestation", "verify"]]) == 3
    )
    assert all(
        argv[argv.index("--source-digest") + 1] == end_sha
        for argv, _identity, _resource in context["verification_commands"]
        if argv[1:3] == ["attestation", "verify"]
    )


def test_live_verification_rejects_falsey_json_for_initial_and_replay_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    end_sha = "1" * 40
    commands = (
        [
            "gh",
            "release",
            "verify-asset",
            end_sha,
            str(tmp_path / "asset.json"),
            "-R",
            "synaptent/aragora",
            "--format",
            "json",
        ],
        ratchet._attestation_verify_argv(
            tmp_path / "asset.json",
            github_repository="synaptent/aragora",
            end_sha=end_sha,
        ),
    )
    for raw in (b"null", b"{}", b"[]", b"false", b'""', b"0"):
        monkeypatch.setattr(
            ratchet,
            "_run_read_only",
            lambda argv, **_kwargs: subprocess.CompletedProcess(
                argv,
                0,
                stdout=raw,
                stderr=b"",
            ),
        )
        for argv in commands:
            with pytest.raises(ValueError, match="returned empty verification JSON"):
                ratchet._run_live_verification(
                    argv,
                    operation_log=[],
                    resource="initial-verification",
                )
            context = {
                "asset_identities": {},
                "endpoint_identities": {},
                "github_repository": "synaptent/aragora",
                "local_asset_identities": {},
                "verification_commands": [
                    (
                        argv,
                        {
                            "byte_length": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                        },
                        "replay-verification",
                    )
                ],
            }
            with pytest.raises(ValueError, match="returned empty verification JSON"):
                ratchet._reauthenticate_live_context(
                    context,
                    operation_log=[],
                    end_sha=end_sha,
                )


@pytest.mark.parametrize("source_digest", (None, "0" * 40))
def test_live_evidence_replay_rejects_missing_or_wrong_attestation_source_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_digest: str | None,
):
    end_sha = "1" * 40
    argv = [
        "gh",
        "attestation",
        "verify",
        str(tmp_path / "asset.json"),
        "-R",
        "synaptent/aragora",
        "--signer-workflow",
        "synaptent/aragora/.github/workflows/contract-drift-boundary.yml",
        "--format",
        "json",
    ]
    if source_digest is not None:
        argv.extend(["--source-digest", source_digest])
    context = {
        "asset_identities": {},
        "endpoint_identities": {},
        "github_repository": "synaptent/aragora",
        "local_asset_identities": {},
        "verification_commands": [
            (
                argv,
                {"byte_length": 1, "sha256": "2" * 64},
                "sigstore-attestation:asset.json",
            )
        ],
    }
    monkeypatch.setattr(
        ratchet,
        "_run_live_verification",
        lambda *_args, **_kwargs: pytest.fail(
            "hostile replay command must fail before transport execution"
        ),
    )

    with pytest.raises(ValueError, match="source digest"):
        ratchet._reauthenticate_live_context(
            context,
            operation_log=[],
            end_sha=end_sha,
        )


def test_live_release_pagination_runs_to_exhaustion(monkeypatch: pytest.MonkeyPatch):
    requests: list[str] = []

    def stable_get(
        endpoint: str,
        *,
        operation_log: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del operation_log
        requests.append(endpoint)
        page = int(endpoint.rsplit("page=", 1)[1])
        payload = [{"id": index} for index in range(100)] if page == 1 else [{"id": 100}]
        return payload, {
            "byte_length": page,
            "etag": f'"page-{page}"',
            "sha256": f"{page:064x}",
            "updated_at": None,
        }

    monkeypatch.setattr(ratchet, "_gh_api_get_stable", stable_get)
    records, identities = ratchet._gh_api_paginated(
        "repos/synaptent/aragora/releases",
        operation_log=[],
    )

    assert len(records) == 101
    assert requests == [
        "repos/synaptent/aragora/releases?per_page=100&page=1",
        "repos/synaptent/aragora/releases?per_page=100&page=2",
    ]
    assert set(identities) == set(requests)


def test_stage2_reruns_full_stage1_matrix():
    stage1_names = {
        "test_all_loaded_repository_modules_are_under_exact_ref_extraction_root",
        "test_authority_roots_are_tier4",
        "test_canonical_tier_cli_is_read_only_and_digest_bound",
        "test_classifier_and_merge_train_closure_match",
        "test_deterministic_bounded_authority_dependency_closure_has_incoming_edges_and_exact_ref_digests",
        "test_local_reusable_workflows_and_composite_actions_join_closure",
        "test_measured_sdk_handler_openapi_subjects_are_not_authority_dependencies",
        "test_merge_train_mirror_is_normal_repo_file_authority_member",
        "test_standalone_classifier_extracts_and_calls_exact_ref_canonical_review_queue_policy_under_I_S",
        "test_workflows_yml_and_yaml_recurse_through_structural_run_uses_and_path_filters",
    }
    assert ratchet.STAGE1_REQUIRED_TESTS == tuple(sorted(stage1_names))
    repo = Path(ratchet.__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *ratchet.STAGE1_TEST_MATRIX, "-q"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_read_only_cli_hashes_worktree_index_gitdirs_objects_refs_and_reflogs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, _start_sha, _boundary_shas = _boundary_git_repo(tmp_path)
    operation_log: list[dict[str, Any]] = []
    calls: list[tuple[Path, frozenset[str]]] = []
    original_path_manifest = ratchet._path_manifest

    def recording_path_manifest(
        path: Path,
        *,
        content: bool,
        exclude_top_level: frozenset[str] = frozenset(),
    ) -> bytes:
        calls.append((path.resolve(), exclude_top_level))
        return original_path_manifest(
            path,
            content=content,
            exclude_top_level=exclude_top_level,
        )

    monkeypatch.setattr(ratchet, "_path_manifest", recording_path_manifest)
    snapshot = ratchet._snapshot_repository(repo, operation_log)
    assert set(snapshot) == {
        "common_git_dir",
        "index",
        "object_database",
        "refs",
        "reflogs",
        "worktree",
        "worktree_git_dir",
    }
    assert all(
        isinstance(value["sha256"], str) and len(value["sha256"]) == 64
        for value in snapshot.values()
    )
    assert operation_log

    git_dir = Path(
        subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-dir"],
            text=True,
        ).strip()
    ).resolve()
    common_dir = Path(
        subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            text=True,
        ).strip()
    ).resolve()
    assert git_dir == common_dir
    expected_excludes = frozenset({"logs", "objects", "refs", "worktrees"})
    assert [excludes for path, excludes in calls if path == git_dir].count(expected_excludes) == 2

    fake_common = tmp_path / "fake-common"
    fake_common.mkdir()
    metadata = fake_common / "config"
    metadata.write_text("metadata=v1\n", encoding="utf-8")
    for subtree in expected_excludes:
        child = fake_common / subtree / "nested"
        child.mkdir(parents=True)
        (child / "separately-captured").write_bytes(b"before")
    before = original_path_manifest(
        fake_common,
        content=True,
        exclude_top_level=expected_excludes,
    )
    for subtree in expected_excludes:
        (fake_common / subtree / "nested" / "separately-captured").write_bytes(b"after")
    after_common_mutation = original_path_manifest(
        fake_common,
        content=True,
        exclude_top_level=expected_excludes,
    )
    assert after_common_mutation == before
    metadata.write_text("metadata=v2\n", encoding="utf-8")
    after_metadata_mutation = original_path_manifest(
        fake_common,
        content=True,
        exclude_top_level=expected_excludes,
    )
    assert after_metadata_mutation != before

    linked = tmp_path / "linked"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(linked), "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    calls.clear()
    linked_before = ratchet._snapshot_repository(linked, [])
    linked_git_dir = Path(
        subprocess.check_output(
            ["git", "-C", str(linked), "rev-parse", "--path-format=absolute", "--git-dir"],
            text=True,
        ).strip()
    ).resolve()
    assert linked_git_dir != common_dir
    linked_excludes = [excludes for path, excludes in calls if path == linked_git_dir]
    assert linked_excludes == [frozenset()]
    linked_entries = json.loads(
        original_path_manifest(
            linked_git_dir,
            content=True,
            exclude_top_level=linked_excludes[0],
        )
    )
    linked_paths = {entry["path"] for entry in linked_entries}
    assert {"HEAD", "commondir", "gitdir", "index"} <= linked_paths
    (linked_git_dir / "verifier-sentinel").write_bytes(b"linked metadata changed")
    linked_after = ratchet._snapshot_repository(linked, [])
    assert linked_after["worktree_git_dir"] != linked_before["worktree_git_dir"]
    assert {name: value for name, value in linked_after.items() if name != "worktree_git_dir"} == {
        name: value for name, value in linked_before.items() if name != "worktree_git_dir"
    }


def test_read_only_cli_allows_only_scratch_and_output_writes(tmp_path: Path):
    scratch = tmp_path / "scratch"
    output = tmp_path / "output"
    ratchet._guard_write_path(scratch / "child.json", scratch, output)
    ratchet._guard_write_path(output / "manifest.json", scratch, output)
    with pytest.raises(ValueError, match="outside"):
        ratchet._guard_write_path(tmp_path / "escape.json", scratch, output)


def test_read_only_cli_is_deterministic_across_double_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    first = _boundary_result(tmp_path, monkeypatch, "extended_sdk")
    repo = tmp_path / "boundary-repo"
    start_sha = first["start_sha"]
    end_sha = first["end_sha"]
    second = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary="extended_sdk",
        start_ref=start_sha,
        end_ref=end_sha,
    )
    assert ratchet._canonical_json_bytes(first) == ratchet._canonical_json_bytes(second)


def test_read_only_cli_preserves_etag_and_updated_at():
    before = {"etag": '"stable"', "updated_at": "2026-07-20T00:00:00Z"}
    after = copy.deepcopy(before)
    assert ratchet._remote_identity_moved(before, after) is False
    after["etag"] = '"moved"'
    assert ratchet._remote_identity_moved(before, after) is True


def test_read_only_cli_retries_or_blocks_on_concurrent_mutation():
    calls = 0

    def probe() -> tuple[dict[str, str], dict[str, str]]:
        nonlocal calls
        calls += 1
        before = {"etag": f'"{calls}"', "updated_at": "2026-07-20T00:00:00Z"}
        after = dict(before)
        if calls < 3:
            after["etag"] = '"moved"'
        return before, after

    before, after = ratchet._retry_stable_remote_probe(probe, attempts=3)
    assert calls == 3
    assert before == after

    with pytest.raises(ratchet.BoundaryBlocked, match="moved"):
        ratchet._retry_stable_remote_probe(
            lambda: ({"etag": '"a"'}, {"etag": '"b"'}),
            attempts=2,
        )


def test_read_only_cli_rejects_mutating_http_verbs():
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        with pytest.raises(ValueError, match="mutating HTTP"):
            ratchet._guard_http_method(method)
    ratchet._guard_http_method("GET")
    ratchet._guard_http_method("HEAD")


def test_read_only_cli_rejects_mutating_git_and_subprocess_actions():
    for argv in (
        ["git", "merge", "main"],
        ["git", "push", "origin", "HEAD"],
        ["git", "config", "user.name", "unsafe"],
        ["gh", "pr", "merge", "1"],
        ["gh", "run", "rerun", "1"],
        ["gh", "release", "upload", "tag", "asset"],
        ["gh", "api", "-XPOST", "repos/o/r"],
        ["gh", "api", "-f", "key=value", "repos/o/r"],
    ):
        with pytest.raises(ValueError, match="mutating|unsupported"):
            ratchet._guard_subprocess_argv(argv)
    ratchet._guard_subprocess_argv(["git", "status", "--porcelain=v1"])
    ratchet._guard_subprocess_argv(["gh", "api", "--method", "GET", "repos/o/r"])
