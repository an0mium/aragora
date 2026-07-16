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


def test_pr_mode_fails_on_any_single_list_increase(tmp_path: Path):
    paths, repo, base = _seed(tmp_path, program=RED_PROGRAM)

    routes = json.loads(paths["routes"].read_text())
    routes["orphaned_in_spec"].append("o-new")
    _write_json(paths["routes"], routes)
    _edit_inventory(
        paths,
        lambda inv: inv["items"].append(
            {
                "id": "orphaned_in_spec:o-new",
                "source": "orphaned_in_spec",
                "class": "discovered",
                "discovered_on": "2026-07-16",
                "provenance": "explained in #4242",
                "status": "open",
            }
        ),
    )

    result = _result(paths, "2026-07-16", repo=repo, cohort=base, mode="pr", base_ref=base)
    # Inventory is in sync (provenance recorded) yet the delta gate still fails.
    assert result["integrity"]["passing"]
    assert result["pr_delta"]["increased"] == ["routes_orphaned_in_spec"]
    assert not result["passing"]


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
    # fails, but only via the count-increase delta gate (the regression).
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

    # HEAD grows an entry: increase vs empty base -> FAIL (with inventory synced).
    _write_json(paths["parity"], {"missing_from_both_sdks": ["p-new"]})
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
    assert not result["passing"]
