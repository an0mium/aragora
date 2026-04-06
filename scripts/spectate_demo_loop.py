#!/usr/bin/env python3
"""Run continuous public debates for the spectate bridge on the landing page.

Usage:
    # Against local server
    python scripts/spectate_demo_loop.py

    # Against production
    python scripts/spectate_demo_loop.py --api-url https://api.aragora.ai

    # Custom interval
    python scripts/spectate_demo_loop.py --interval 30 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUESTIONS = [
    "Should a fast-growing software org split the monolith now or sequence the migration later?",
    "How should teams balance technical debt reduction with feature velocity?",
    "What is the ideal deployment frequency for mission-critical systems?",
    "Should AI code review replace human code review, or complement it?",
    "How should a startup approach SOC 2 certification — fast-track or build organically?",
    "Is it better to build an internal ML platform or use managed services?",
    "Should engineering teams adopt trunk-based development or feature branches?",
    "How should organizations handle the transition from REST to GraphQL APIs?",
    "What's the right observability strategy — logs, metrics, traces, or all three?",
    "Should companies build their own LLM infrastructure or use API providers?",
]


def run_debate(api_url: str, question: str, rounds: int = 2, timeout: int = 120) -> dict:
    """Trigger a playground debate via the API."""
    url = f"{api_url.rstrip('/')}/api/v1/playground/debate"
    payload = json.dumps(
        {
            "question": question,
            "agent_count": 3,
            "max_rounds": rounds,
        }
    ).encode()

    headers = {
        "Content-Type": "application/json",
    }
    token = os.environ.get("ARAGORA_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, data=payload, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except URLError as e:
        logger.error("Debate request failed: %s", e)
        return {"error": str(e)}


def check_bridge_status(api_url: str) -> dict:
    """Check spectate bridge status."""
    url = f"{api_url.rstrip('/')}/api/v1/spectate/status"
    try:
        with urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except URLError:
        return {"active": False, "error": "unreachable"}


def main():
    parser = argparse.ArgumentParser(
        description="Run continuous public debates for spectate bridge"
    )
    parser.add_argument("--api-url", default="http://localhost:8080", help="API base URL")
    parser.add_argument("--interval", type=int, default=15, help="Seconds between debates")
    parser.add_argument("--rounds", type=int, default=2, help="Debate rounds per question")
    parser.add_argument("--max-debates", type=int, default=0, help="Max debates (0=infinite)")
    args = parser.parse_args()

    logger.info("Starting spectate demo loop against %s", args.api_url)

    # Check bridge status first
    status = check_bridge_status(args.api_url)
    if status.get("active"):
        logger.info("Bridge active: %d recent events", status.get("recent_event_count", 0))
    else:
        logger.warning("Bridge not active: %s", status.get("error", "unknown"))
        logger.info("Proceeding anyway — debates will buffer once bridge starts")

    debate_count = 0
    while True:
        # Append timestamp to bypass debate cache — each debate must be unique
        base_question = QUESTIONS[debate_count % len(QUESTIONS)]
        question = f"{base_question} (run {debate_count + 1}, {time.strftime('%H:%M')})"
        debate_count += 1
        logger.info("Debate %d: %s", debate_count, question[:60])

        result = run_debate(args.api_url, question, rounds=args.rounds)
        if "error" in result:
            logger.error("  Failed: %s", result["error"])
        else:
            debate_id = result.get("debate_id", "?")
            conclusion = str(result.get("conclusion", ""))[:100]
            logger.info("  Done: %s — %s", debate_id, conclusion)

        if args.max_debates and debate_count >= args.max_debates:
            logger.info("Reached max debates (%d), stopping", args.max_debates)
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
