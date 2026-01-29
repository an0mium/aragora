#!/usr/bin/env python3
"""
Golden-path harness for Aragora.

Runs three canonical workflows with deterministic demo agents:
1) Debate/Decision (aragora ask)
2) Gauntlet stress-test
3) Code review (multi-agent critique)

Outputs JSON artifacts to the chosen output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from aragora.agents.base import create_agent
from aragora.cli.main import run_debate
from aragora.cli.review import build_review_prompt, extract_review_findings
from aragora.core import Environment
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.gauntlet import AttackCategory, GauntletConfig, GauntletRunner, ProbeCategory

logger = logging.getLogger(__name__)

DEFAULT_TASK = "Design a rate limiter for 1M requests/sec with audit-ready decisions."

DEFAULT_SPEC = """\
# Payment Webhook Service (Draft)

## Goals
- Accept payment webhooks from multiple providers.
- Normalize events and write to a ledger table.
- Expose an API for finance to query event status.

## Constraints
- Must be idempotent.
- Must handle spikes (10k events/min).
- Must log audit trails for compliance.

## Open Questions
- How do we validate webhook signatures consistently?
- What is the retry/backoff strategy for provider outages?
"""

DEFAULT_DIFF = """\
diff --git a/app/users.py b/app/users.py
index 5a1c2b3..9d8e7f6 100644
--- a/app/users.py
+++ b/app/users.py
@@ -14,7 +14,10 @@ def search_users(query):
-    sql = f"SELECT * FROM users WHERE name = '{query}'"
-    return db.execute(sql)
+    # TODO: switch to parameterized queries
+    sql = "SELECT * FROM users WHERE name = '%s'" % query
+    return db.execute(sql)
"""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _demo_agent_factory() -> Callable[[str], Any]:
    counter = itertools.count(1)

    def factory(agent_type: str) -> Any:
        idx = next(counter)
        return create_agent(
            model_type=agent_type,  # type: ignore[arg-type]
            name=f"{agent_type}_{idx}",
            role="critic",
        )

    return factory


def _protocol_overrides(mode: str, enable_trending: bool) -> dict[str, Any]:
    # Keep these in sync with DebateProtocol fields.
    overrides = {
        "enable_research": False,
        "enable_trending_injection": enable_trending,
        "enable_rhetorical_observer": False,
        "enable_trickster": False,
        "enable_evolution": False,
        "verify_claims_during_consensus": False,
        "enable_evidence_weighting": False,
        "enable_breakpoints": False,
        "role_rotation": False,
        "role_matching": False,
        "use_structured_phases": False,
        "convergence_detection": False,
        "vote_grouping": False,
        "early_stopping": False,
    }

    if mode == "full":
        overrides.update(
            {
                "enable_rhetorical_observer": True,
                "role_rotation": True,
                "role_matching": True,
                "use_structured_phases": True,
            }
        )

    return overrides


def _configure_trending(enable_trending: bool) -> None:
    if enable_trending:
        os.environ.pop("ARAGORA_DISABLE_TRENDING", None)
    else:
        os.environ["ARAGORA_DISABLE_TRENDING"] = "1"


def run_ask(output_dir: Path, mode: str = "fast", enable_trending: bool = False) -> dict[str, Any]:
    rounds = 1 if mode == "fast" else 3
    result = asyncio.run(
        run_debate(
            task=DEFAULT_TASK,
            agents_str="demo,demo,demo",
            rounds=rounds,
            consensus="majority",
            context="",
            learn=False,
            enable_audience=False,
            protocol_overrides=_protocol_overrides(mode, enable_trending),
            mode=None,
        )
    )
    payload = result.to_dict()
    _write_json(output_dir / "ask_result.json", payload)
    return payload


async def _run_review_debate(
    diff: str,
    agents_str: str,
    rounds: int,
    protocol_overrides: dict[str, Any],
) -> Any:
    agent_types = [spec.strip() for spec in agents_str.split(",") if spec.strip()]
    if not agent_types:
        agent_types = ["demo", "demo"]

    roles = ["security_reviewer", "performance_reviewer", "quality_reviewer"]
    agents = []
    for i, agent_type in enumerate(agent_types):
        role = roles[i % len(roles)]
        agents.append(
            create_agent(
                model_type=agent_type,  # type: ignore[arg-type]
                name=f"{agent_type}_{role}",
                role=role,
            )
        )

    task = build_review_prompt(diff)
    env = Environment(task=task, max_rounds=rounds)
    protocol = DebateProtocol(rounds=rounds, consensus="majority", **protocol_overrides)
    arena = Arena(env, agents, protocol)
    return await arena.run()


def run_review(
    output_dir: Path, mode: str = "fast", enable_trending: bool = False
) -> dict[str, Any]:
    rounds = 1 if mode == "fast" else 2
    result = asyncio.run(
        _run_review_debate(
            diff=DEFAULT_DIFF,
            agents_str="demo,demo",
            rounds=rounds,
            protocol_overrides=_protocol_overrides(mode, enable_trending),
        )
    )
    findings = extract_review_findings(result)
    # Remove non-serializable objects for JSON output
    findings.pop("all_critiques", None)
    payload = {
        "summary": result.final_answer,
        "findings": findings,
    }
    _write_json(output_dir / "review_result.json", payload)
    return payload


def run_gauntlet(output_dir: Path, mode: str = "fast") -> dict[str, Any]:
    config = GauntletConfig(
        agents=["demo", "demo", "demo"],
        attack_categories=[AttackCategory.LOGIC],
        probe_categories=[ProbeCategory.CONTRADICTION],
        attack_rounds=1 if mode == "fast" else 2,
        attacks_per_category=1,
        probes_per_category=1 if mode == "fast" else 2,
        run_scenario_matrix=False if mode == "fast" else True,
    )
    runner = GauntletRunner(config=config, agent_factory=_demo_agent_factory())
    result = asyncio.run(runner.run(DEFAULT_SPEC))
    payload = result.to_dict()
    _write_json(output_dir / "gauntlet_result.json", payload)
    return payload


def run_all(output_dir: Path, mode: str = "fast", enable_trending: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "ask": run_ask(output_dir, mode=mode, enable_trending=enable_trending),
        "gauntlet": run_gauntlet(output_dir, mode=mode),
        "review": run_review(output_dir, mode=mode, enable_trending=enable_trending),
    }

    summary = {
        "mode": mode,
        "artifacts": {
            "ask": "ask_result.json",
            "gauntlet": "gauntlet_result.json",
            "review": "review_result.json",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Aragora golden-path workflows.")
    parser.add_argument(
        "--output-dir",
        default="output/golden_paths",
        help="Directory to write JSON artifacts (default: output/golden_paths)",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="Execution mode (fast disables heavy features).",
    )
    parser.add_argument(
        "--only",
        choices=["all", "ask", "gauntlet", "review"],
        default="all",
        help="Run only a single workflow.",
    )
    parser.add_argument(
        "--enable-trending",
        action="store_true",
        help="Enable Pulse trending context (disabled by default for deterministic runs).",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    _configure_trending(args.enable_trending)

    if args.only == "ask":
        run_ask(output_dir, mode=args.mode, enable_trending=args.enable_trending)
    elif args.only == "gauntlet":
        run_gauntlet(output_dir, mode=args.mode)
    elif args.only == "review":
        run_review(output_dir, mode=args.mode, enable_trending=args.enable_trending)
    else:
        run_all(output_dir, mode=args.mode, enable_trending=args.enable_trending)

    print(f"[golden-paths] artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
