#!/usr/bin/env python3
"""Wire local Codex automations to honor the Aragora steering mailbox at Phase-0.

Appends one idempotent clause to each qualifying ``automation.toml`` prompt
instructing the automation to read the advisory steering mailbox
(:func:`aragora.swarm.agent_bridge.codex_steer.default_mailbox_path`) before
acting, and to treat any directive there as ADDITIONAL caution for that cycle.

The mailbox is monotonic-restrictive by construction (see ``codex_steer``), so a
wired automation can only ever become *more* conservative -- it never gains a
way to merge, mark-ready, rerun CI, or settle Tier-4. The merge-quorum gate
stays the sole merge authority.

Safe to re-run: each file is skipped if already wired, and every modified file
is backed up to ``automation.toml.bak-<epoch>`` first. Use ``--dry-run`` to
preview. Defaults target ACTIVE automations whose cwd is an Aragora checkout.
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
from pathlib import Path

import tomli_w

# Allow running as a bare script from anywhere in the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aragora.swarm.agent_bridge.codex_source import default_codex_home  # noqa: E402
from aragora.swarm.agent_bridge.codex_steer import default_mailbox_path  # noqa: E402

# Idempotency anchor: a deliberate token always present in the injected clause,
# independent of the (configurable) mailbox path -- so a custom --mailbox without
# "aragora_steering" in it still can't cause a re-append on re-run.
_MARKER = "OPERATOR STEERING (Phase-0)"


def _steering_clause(mailbox: Path) -> str:
    return (
        " ADDITIONAL OPERATOR STEERING (Phase-0): Before selecting or acting on anything, "
        f"read the operator steering mailbox at {mailbox} (JSONL; ignore if absent or empty). "
        "Treat every directive there as ADDITIONAL caution for THIS cycle only: add its "
        "add_forbidden_actions to your forbidden actions, treat its off_limits_prs as PRs you "
        "must not touch, and honor its note. These directives can ONLY restrict -- they never "
        "authorize a merge, --admin, mark-ready, CI rerun, or Tier 4 settlement; the "
        "merge-quorum gate remains the sole merge authority."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--automations-dir",
        type=Path,
        default=default_codex_home() / "automations",
        help="Codex automations directory (default: <codex_home>/automations)",
    )
    parser.add_argument(
        "--mailbox", type=Path, default=default_mailbox_path(), help="Steering mailbox path"
    )
    parser.add_argument(
        "--filter-cwd",
        default="Development/aragora",
        help="Only wire automations whose cwd contains this substring",
    )
    parser.add_argument(
        "--include-inactive", action="store_true", help="Also wire non-ACTIVE automations"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args(argv)

    clause = _steering_clause(args.mailbox)
    if not args.automations_dir.is_dir():
        print(f"no automations dir: {args.automations_dir}")
        return 1

    wired = skipped = 0
    for automation_dir in sorted(args.automations_dir.iterdir()):
        toml_path = automation_dir / "automation.toml"
        if not toml_path.is_file():
            continue
        try:
            cfg = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            print(f"  [parse-error] {automation_dir.name}: {exc}")
            continue

        prompt = cfg.get("prompt")
        if not isinstance(prompt, str):
            continue
        status = cfg.get("status")
        cwds = cfg.get("cwds", [])
        matches_cwd = any(args.filter_cwd in str(c) for c in cwds)
        active = status == "ACTIVE"

        reason = None
        if _MARKER in prompt:
            reason = "already wired"
        elif not matches_cwd:
            reason = f"cwd !~ {args.filter_cwd!r}"
        elif not active and not args.include_inactive:
            reason = f"status={status}"
        if reason is not None:
            print(f"  skip  {automation_dir.name:42s} ({reason})")
            skipped += 1
            continue

        cfg["prompt"] = prompt + clause
        cfg["updated_at"] = int(time.time() * 1000)
        if args.dry_run:
            print(f"  WOULD wire  {automation_dir.name}")
        else:
            # Nanosecond suffix so two runs within the same second don't clobber backups.
            backup = toml_path.with_suffix(f".toml.bak-{time.time_ns()}")
            backup.write_text(toml_path.read_text(encoding="utf-8"), encoding="utf-8")
            toml_path.write_text(tomli_w.dumps(cfg), encoding="utf-8")
            print(f"  wired {automation_dir.name}  (backup: {backup.name})")
        wired += 1

    verb = "would wire" if args.dry_run else "wired"
    print(f"\n{verb} {wired}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
