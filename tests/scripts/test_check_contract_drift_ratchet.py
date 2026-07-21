"""Tests for scripts/check_contract_drift_ratchet.py."""

from __future__ import annotations

import copy
import hashlib
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
            ratchet._discover_canonical_artifact(None, ratchet.COHORT_ARTIFACT, repo_root),
            ratchet._discover_canonical_artifact(None, ratchet.PROVENANCE_ARTIFACT, repo_root),
        )
    except ValueError:
        return None


def _require_canonical_fixture_artifacts() -> tuple[Path, Path]:
    paths = _canonical_fixture_artifact_paths()
    if paths is None:
        pytest.skip("canonical Contract Drift mission artifacts are unavailable")
    return paths


def _fixture_sdk_partitions() -> dict[str, list[str]]:
    artifact_paths = _canonical_fixture_artifact_paths()
    if artifact_paths is None:
        return {
            "core": [
                f"cdg1:{hashlib.sha256(f'fixture-core-{index}'.encode()).hexdigest()}"
                for index in range(75)
            ],
            "extended": [
                f"cdg1:{hashlib.sha256(f'fixture-extended-{index}'.encode()).hexdigest()}"
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


def _boundary_git_repo(
    tmp_path: Path,
    *,
    route_debt: bool = False,
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
        if route_debt and boundary == "route_truth":
            _write_json(
                baselines / "validate_openapi_routes.json",
                {
                    "missing_in_spec": ["/api/contradiction"],
                    "orphaned_in_spec": [],
                },
            )
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
            "rule_suite": {
                "authenticated": True,
                "available": True,
                "bypassed": False,
                "id": 987654,
                "result": "pass",
            },
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


def _boundary_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    *,
    release_immutability: bool = True,
    mutate: Any | None = None,
) -> dict[str, Any]:
    _require_canonical_fixture_artifacts()
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
    return ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary=boundary,
        start_ref=start_sha,
        end_ref=end_sha,
        evidence_index_path=index_path,
        evidence_index_byte_length=index_length,
        evidence_index_sha256=index_sha256,
    )


def test_boundary_pass_fixture_uses_production_artifact_and_authority_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cohort_path, provenance_path = _require_canonical_fixture_artifacts()
    monkeypatch.setattr(
        ratchet,
        "_snapshot_repository",
        lambda _repo_root, _operation_log: {
            "fixture_repository": {
                "byte_length": 1,
                "sha256": hashlib.sha256(b"\0").hexdigest(),
            }
        },
    )
    repo_root = Path(__file__).resolve().parents[2]
    start_sha = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD^"],
        text=True,
    ).strip()
    end_sha = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
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

    resources_dir = tmp_path / "real-authentication-resources"
    resources_dir.mkdir()
    selected = {
        "boundary_chronology",
        "corrective_bootstrap",
        "durable_capsule",
        "external_prerequisites",
        "first_parent_receipts",
        "governed_prs",
    }
    descriptors = []
    for name, payload in sorted(payloads.items()):
        if name not in selected:
            continue
        if name == "boundary_chronology":
            payload["boundaries"] = payload["boundaries"][:1]
        descriptor = _write_canonical_boundary_json(resources_dir / f"{name}.json", payload)
        descriptor.update(
            {
                "name": name,
                "path": f"{resources_dir.name}/{name}.json",
            }
        )
        descriptors.append(descriptor)
    index = {
        "boundary": "corrective_bootstrap",
        "end_sha": end_sha,
        "resources": descriptors,
        "schema": ratchet.BOUNDARY_EVIDENCE_INDEX_SCHEMA,
        "start_sha": start_sha,
    }
    index_raw = _canonical_boundary_bytes(index)
    index_path = tmp_path / "real-authentication-index.json"
    index_path.write_bytes(index_raw)

    result = ratchet.build_boundary_result(
        repo_root=repo_root,
        schema_version=1,
        boundary="corrective_bootstrap",
        start_ref=start_sha,
        end_ref=end_sha,
        cohort_artifact_path=cohort_path,
        sdk_provenance_artifact_path=provenance_path,
        evidence_index_path=index_path,
        evidence_index_byte_length=len(index_raw),
        evidence_index_sha256=hashlib.sha256(index_raw).hexdigest(),
        scratch_root=tmp_path,
        output_root=tmp_path,
    )

    assert result["status"] == "pass", result.get("blocked_reason") or result.get("error")
    assert result["canonical_artifacts"]["original_cohort"]["record_count"] == 655
    assert result["canonical_artifacts"]["sdk_provenance"]["record_count"] == 598
    assert result["authority"]["repo_file_count"] > 0
    authority_reads = [
        entry
        for entry in result["operation_log"]
        if entry["resource"] == "authority_manifest"
        and entry["kind"] in {"authority_reconstruction", "external_authority_manifest"}
    ]
    assert len(authority_reads) == 2
    assert authority_reads[0]["byte_length"] > 0
    assert authority_reads[0]["sha256"] == authority_reads[1]["sha256"]


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
    _require_canonical_fixture_artifacts()
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
    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary=boundary,
        start_ref=start_sha,
        end_ref=end_sha,
        evidence_index_path=index_path,
        evidence_index_byte_length=len(raw),
        evidence_index_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert result["status"] == "fail"
    assert "caller-supplied" in result["error"]

    pretty = (
        json.dumps(json.loads(index_path.read_text()), indent=2, sort_keys=True).encode() + b"\n"
    )
    index_path.write_bytes(pretty)
    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary=boundary,
        start_ref=start_sha,
        end_ref=end_sha,
        evidence_index_path=index_path,
        evidence_index_byte_length=len(pretty),
        evidence_index_sha256=hashlib.sha256(pretty).hexdigest(),
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
        "_load_evidence_resources",
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
        evidence_index_path=tmp_path / "movement/index.json",
        evidence_index_byte_length=1,
        evidence_index_sha256="0" * 64,
    )
    assert movement["status"] == "blocked"
    assert "moved concurrently" in movement["blocked_reason"]


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


def test_canonical_route_fact_fails_when_exact_ref_baseline_contradicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _require_canonical_fixture_artifacts()
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

    result = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary="route_truth",
        start_ref=start_sha,
        end_ref=end_sha,
        evidence_index_path=index_path,
        evidence_index_byte_length=index_length,
        evidence_index_sha256=index_sha256,
    )

    assert result["status"] == "fail"
    assert "contradicted by exact-ref route baselines" in result["error"]


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
    _require_canonical_fixture_artifacts()
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
        evidence_index_path=index_path,
        evidence_index_byte_length=index_length,
        evidence_index_sha256=index_sha256,
    )

    assert result["status"] == "blocked"
    assert "moved concurrently" in result["blocked_reason"]


def test_deterministic_boundary_fixtures_reach_pass_while_live_release_immutability_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
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


def test_live_evidence_discovers_release_assets_rule_suite_and_prs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    ) -> tuple[Any, dict[str, Any]]:
        del operation_log, attempts
        if endpoint == "repos/synaptent/aragora":
            return {"full_name": "synaptent/aragora"}, identity
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
            return {"id": 987654, "result": "pass"}, identity
        if endpoint.endswith("/pulls/9999"):
            return {
                "base": {"sha": start_sha},
                "changed_files": 1,
                "head": {"sha": end_sha},
                "merge_commit_sha": end_sha,
                "merged_at": "2026-07-20T00:00:00Z",
                "number": 9999,
            }, identity
        if endpoint.endswith(f"/git/commits/{end_sha}"):
            tree_sha = resources["governed_prs"]["records"][0]["head_tree_sha"]
            return {
                "parents": [{"sha": start_sha}],
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
    monkeypatch.setattr(
        ratchet,
        "_run_live_verification",
        lambda argv, *, operation_log, resource: (
            {"resource": resource, "verified": bool(argv)},
            verification_identity,
        ),
    )

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
):
    repo, _start_sha, _boundary_shas = _boundary_git_repo(tmp_path)
    operation_log: list[dict[str, Any]] = []
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
    index_path = tmp_path / "extended_sdk-evidence-index.json"
    second = ratchet.build_boundary_result(
        repo_root=repo,
        schema_version=1,
        boundary="extended_sdk",
        start_ref=start_sha,
        end_ref=end_sha,
        evidence_index_path=index_path,
        evidence_index_byte_length=first["evidence"]["index"]["byte_length"],
        evidence_index_sha256=first["evidence"]["index"]["sha256"],
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
    ):
        with pytest.raises(ValueError, match="mutating|unsupported"):
            ratchet._guard_subprocess_argv(argv)
    ratchet._guard_subprocess_argv(["git", "status", "--porcelain=v1"])
    ratchet._guard_subprocess_argv(["gh", "api", "--method", "GET", "repos/o/r"])
