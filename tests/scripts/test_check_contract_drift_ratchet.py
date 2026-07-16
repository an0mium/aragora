"""Tests for scripts/check_contract_drift_ratchet.py."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import scripts.check_contract_drift_ratchet as ratchet
import scripts.generate_contract_drift_inventory as gen


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _cohort_items(docs: dict[str, dict]) -> list[dict]:
    return [
        {
            "id": item_id,
            "source": list_key,
            "class": "start_cohort",
            "discovered_on": "2026-04-17",
            "provenance": gen.COHORT_PROVENANCE,
            "status": "open",
        }
        for item_id, list_key in sorted(gen.collect_ids(docs).items())
    ]


def _write_inventory(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gen.render_inventory(sorted(items, key=lambda i: i["id"]), "test"))


def _seed(
    tmp_path: Path,
    *,
    verify: dict | None = None,
    routes: dict | None = None,
    parity: dict | None = None,
    program: dict | None = None,
    inventory_items: list[dict] | None = None,
) -> dict[str, Path]:
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
        else {
            "missing_in_spec": ["m1", "m2"],
            "orphaned_in_spec": ["o1"],
        }
    )
    parity = parity if parity is not None else {"missing_from_both_sdks": ["p1", "p2"]}
    docs = {"verify": verify, "routes": routes, "parity": parity}

    paths = {
        "verify": tmp_path / "verify.json",
        "routes": tmp_path / "routes.json",
        "parity": tmp_path / "parity.json",
        "program": tmp_path / "program.json",
        "inventory": tmp_path / "inventory.json",
    }
    _write_json(paths["verify"], verify)
    _write_json(paths["routes"], routes)
    _write_json(paths["parity"], parity)
    if program is not None:
        _write_json(paths["program"], program)
    items = inventory_items if inventory_items is not None else _cohort_items(docs)
    _write_inventory(paths["inventory"], items)
    return paths


def _argv(paths: dict[str, Path], *extra: str) -> list[str]:
    return [
        "check_contract_drift_ratchet.py",
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


def _result(paths: dict[str, Path], as_of: str, **kwargs) -> dict:
    return ratchet.build_ratchet_result(
        mode=kwargs.pop("mode", "program"),
        program_baseline=paths["program"],
        verify_baseline=paths["verify"],
        routes_baseline=paths["routes"],
        parity_baseline=paths["parity"],
        inventory_path=paths["inventory"],
        repo_root=kwargs.pop("repo_root", paths["verify"].parent),
        as_of=date.fromisoformat(as_of),
        **kwargs,
    )


# ---------------------------------------------------------------- program mode


def test_strict_passes_on_program_start(monkeypatch, tmp_path: Path):
    today = date.today().isoformat()
    paths = _seed(
        tmp_path,
        program={
            "start_date": today,
            "start_total_items": 10,
            "weekly_reduction": 0.1,
            "grace_weeks": 0,
        },
    )
    monkeypatch.setattr(sys, "argv", _argv(paths, "--strict", "--as-of", today))
    assert ratchet.main() == 0


def test_strict_fails_when_above_target(monkeypatch, tmp_path: Path):
    today = date.today()
    paths = _seed(
        tmp_path,
        program={
            "start_date": today.isoformat(),
            "start_total_items": 10,
            "weekly_reduction": 0.1,
            "grace_weeks": 0,
        },
    )
    as_of = (today + timedelta(days=8)).isoformat()
    monkeypatch.setattr(sys, "argv", _argv(paths, "--strict", "--as-of", as_of))
    assert ratchet.main() == 1


def test_program_numbers_read_only_from_program_baseline(tmp_path: Path):
    """Changing contract_drift_program.json (and nothing else) moves the target."""
    program = {
        "start_date": "2026-06-01",
        "start_total_items": 40,
        "weekly_reduction": 0.5,
        "grace_weeks": 0,
    }
    paths = _seed(tmp_path, program=program)
    result = _result(paths, "2026-06-08")
    assert result["program"]["start_total_items"] == 40
    assert result["target"]["max_open_items"] == 20  # 40 * 0.5 after one week

    _write_json(paths["program"], dict(program, start_total_items=80))
    assert _result(paths, "2026-06-08")["target"]["max_open_items"] == 40


def test_program_schedule_math_per_class_and_batch_clocks(tmp_path: Path):
    verify = {
        "python_sdk_drift": ["a", "b"],
        "typescript_sdk_drift": [f"d{i}" for i in range(1, 9)],  # d1..d8 open
    }
    routes = {"missing_in_spec": [], "orphaned_in_spec": []}
    parity = {"missing_from_both_sdks": []}
    docs = {"verify": verify, "routes": routes, "parity": parity}

    cohort = [i for i in _cohort_items(docs) if i["id"].startswith("python_sdk_drift")]
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
        for i in range(1, 11)  # batch_size 10, 8 open + 2 resolved
    ]
    paths = _seed(
        tmp_path,
        verify=verify,
        routes=routes,
        parity=parity,
        program={
            "start_date": "2026-06-01",
            "start_total_items": 30,
            "weekly_reduction": 0.1,
            "grace_weeks": 0,
        },
        inventory_items=cohort + discovered,
    )

    result = _result(paths, "2026-06-15")  # 2 weeks after both clocks start
    classes = {cls["name"]: cls for cls in result["classes"]}

    cohort_cls = classes["start_cohort"]
    assert cohort_cls["batch_size"] == 30
    assert cohort_cls["target_max"] == ratchet._target_after_weeks(30, 0.1, 2)
    assert cohort_cls["open_items"] == 2
    assert cohort_cls["passing"]

    batch = classes["discovered:2026-06-01"]
    assert batch["batch_size"] == 10  # open + resolved, batch clock at its own date
    assert batch["weeks_elapsed"] == 2
    assert batch["target_max"] == 8  # 10 -> 9 -> 8
    assert batch["open_items"] == 8  # resolved items excluded from open count
    assert batch["passing"]
    assert result["passing"]

    # One week later the batch target drops to 7 while 8 remain open -> FAIL.
    later = _result(paths, "2026-06-22")
    batch_later = {c["name"]: c for c in later["classes"]}["discovered:2026-06-01"]
    assert batch_later["target_max"] == 7
    assert not batch_later["passing"]
    assert not later["passing"]


def test_fail_closed_missing_inventory(monkeypatch, tmp_path: Path):
    today = date.today().isoformat()
    paths = _seed(
        tmp_path,
        program={"start_date": today, "start_total_items": 10, "weekly_reduction": 0.1},
    )
    paths["inventory"].unlink()
    # Fails even without --strict: integrity violations always fail closed.
    monkeypatch.setattr(sys, "argv", _argv(paths, "--as-of", today))
    assert ratchet.main() == 1


def test_fail_closed_missing_program_baseline(monkeypatch, tmp_path: Path):
    today = date.today().isoformat()
    paths = _seed(tmp_path)  # no program file written
    monkeypatch.setattr(sys, "argv", _argv(paths, "--as-of", today))
    assert ratchet.main() == 1


def test_fail_closed_unexplained_baseline_entry(monkeypatch, tmp_path: Path):
    today = date.today().isoformat()
    paths = _seed(
        tmp_path,
        program={"start_date": today, "start_total_items": 10, "weekly_reduction": 0.1},
    )
    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].append("sneaky-new-item")  # baseline grows, no inventory
    _write_json(paths["verify"], verify)
    monkeypatch.setattr(sys, "argv", _argv(paths, "--as-of", today))
    assert ratchet.main() == 1


def test_fail_closed_unknown_class(monkeypatch, tmp_path: Path):
    today = date.today().isoformat()
    paths = _seed(
        tmp_path,
        program={"start_date": today, "start_total_items": 10, "weekly_reduction": 0.1},
    )
    inventory = json.loads(paths["inventory"].read_text())
    inventory["items"][0]["class"] = "grandfathered"
    paths["inventory"].write_text(json.dumps(inventory))
    monkeypatch.setattr(sys, "argv", _argv(paths, "--as-of", today))
    assert ratchet.main() == 1


def test_resolved_items_excluded_but_retained(tmp_path: Path):
    today = date.today().isoformat()
    docs = {
        "verify": {"python_sdk_drift": ["a"], "typescript_sdk_drift": []},
        "routes": {"missing_in_spec": [], "orphaned_in_spec": []},
        "parity": {"missing_from_both_sdks": []},
    }
    items = _cohort_items(docs) + [
        {
            "id": "python_sdk_drift:gone",
            "source": "python_sdk_drift",
            "class": "start_cohort",
            "discovered_on": "2026-04-17",
            "provenance": gen.COHORT_PROVENANCE,
            "status": "resolved",
            "resolved_on": "2026-05-01",
        }
    ]
    paths = _seed(
        tmp_path,
        verify=docs["verify"],
        routes=docs["routes"],
        parity=docs["parity"],
        program={"start_date": today, "start_total_items": 5, "weekly_reduction": 0.1},
        inventory_items=items,
    )
    result = _result(paths, today)
    assert result["integrity"]["passing"]
    cohort = {c["name"]: c for c in result["classes"]}["start_cohort"]
    assert cohort["open_items"] == 1  # resolved item not counted


# --------------------------------------------------------------------- pr mode


def _git_repo_with_base(tmp_path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
        cwd=tmp_path,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()


# 10 items @ -10%/week from 2026-04-17: by 2026-07-16 the target is well below
# the 10 seeded open items, so the program schedule is red at that as-of date.
RED_PROGRAM = {
    "start_date": "2026-04-17",
    "start_total_items": 10,
    "weekly_reduction": 0.1,
    "grace_weeks": 0,
}


def test_pr_mode_passes_on_equal_counts_while_program_red(tmp_path: Path):
    paths = _seed(tmp_path, program=RED_PROGRAM)
    base = _git_repo_with_base(tmp_path)
    result = _result(paths, "2026-07-16", mode="pr", base_ref=base, repo_root=tmp_path)
    assert result["passing"]
    assert not result["program_passing"]  # program schedule still honestly red
    assert result["pr_delta"]["increased"] == []


def test_pr_mode_passes_on_decrease(tmp_path: Path):
    paths = _seed(tmp_path, program=RED_PROGRAM)
    base = _git_repo_with_base(tmp_path)

    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].remove("b")
    _write_json(paths["verify"], verify)
    inventory = json.loads(paths["inventory"].read_text())
    for item in inventory["items"]:
        if item["id"] == "python_sdk_drift:b":
            item["status"] = "resolved"
            item["resolved_on"] = "2026-07-16"
    paths["inventory"].write_text(json.dumps(inventory))

    result = _result(paths, "2026-07-16", mode="pr", base_ref=base, repo_root=tmp_path)
    assert result["passing"]
    assert result["pr_delta"]["counts"]["verify_python_sdk_drift"]["delta"] == -1


def test_pr_mode_fails_on_any_single_list_increase(tmp_path: Path):
    paths = _seed(tmp_path, program=RED_PROGRAM)
    base = _git_repo_with_base(tmp_path)

    routes = json.loads(paths["routes"].read_text())
    routes["orphaned_in_spec"].append("o-new")
    _write_json(paths["routes"], routes)
    inventory = json.loads(paths["inventory"].read_text())
    inventory["items"].append(
        {
            "id": "orphaned_in_spec:o-new",
            "source": "orphaned_in_spec",
            "class": "discovered",
            "discovered_on": "2026-07-16",
            "provenance": "explained in #4242",
            "status": "open",
        }
    )
    paths["inventory"].write_text(json.dumps(inventory))

    result = _result(paths, "2026-07-16", mode="pr", base_ref=base, repo_root=tmp_path)
    # Inventory is in sync (provenance recorded) yet the delta gate still fails.
    assert result["integrity"]["passing"]
    assert result["pr_delta"]["increased"] == ["routes_orphaned_in_spec"]
    assert not result["passing"]


def test_pr_mode_fails_on_integrity_violation(monkeypatch, tmp_path: Path):
    paths = _seed(tmp_path, program=RED_PROGRAM)
    base = _git_repo_with_base(tmp_path)

    verify = json.loads(paths["verify"].read_text())
    verify["python_sdk_drift"].append("unexplained")  # not added to inventory
    _write_json(paths["verify"], verify)

    result = _result(paths, "2026-07-16", mode="pr", base_ref=base, repo_root=tmp_path)
    assert not result["integrity"]["passing"]
    assert not result["passing"]

    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            paths,
            "--mode",
            "pr",
            "--base-ref",
            base,
            "--repo-root",
            str(tmp_path),
            "--as-of",
            "2026-07-16",
        ),
    )
    assert ratchet.main() == 1  # fails closed even without --strict


def test_pr_mode_missing_file_at_base_treated_as_empty(tmp_path: Path):
    paths = _seed(tmp_path, parity={"missing_from_both_sdks": []}, program=RED_PROGRAM)
    paths["parity"].unlink()
    base = _git_repo_with_base(tmp_path)  # base commit lacks parity file

    # HEAD parity file exists with zero entries: 0 vs empty-at-base -> equal, PASS.
    _write_json(paths["parity"], {"missing_from_both_sdks": []})
    result = _result(paths, "2026-07-16", mode="pr", base_ref=base, repo_root=tmp_path)
    assert result["pr_delta"]["counts"]["sdk_missing_from_both"] == {
        "base": 0,
        "head": 0,
        "delta": 0,
    }
    assert result["passing"]

    # HEAD grows an entry: increase vs empty base -> FAIL (with inventory synced).
    _write_json(paths["parity"], {"missing_from_both_sdks": ["p-new"]})
    inventory = json.loads(paths["inventory"].read_text())
    inventory["items"].append(
        {
            "id": "missing_from_both_sdks:p-new",
            "source": "missing_from_both_sdks",
            "class": "discovered",
            "discovered_on": "2026-07-16",
            "provenance": "explained in #4242",
            "status": "open",
        }
    )
    paths["inventory"].write_text(json.dumps(inventory))
    result = _result(paths, "2026-07-16", mode="pr", base_ref=base, repo_root=tmp_path)
    assert result["pr_delta"]["increased"] == ["sdk_missing_from_both"]
    assert not result["passing"]
