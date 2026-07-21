"""Tests for scripts/check_contract_drift_ratchet.py."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

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
