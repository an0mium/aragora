"""
Slack Code Review Example
=========================

Uses Aragora's multi-agent debate to review a code diff, then posts
the structured review to a Slack channel via webhook.

Requirements:
    - pip install aragora
    - ANTHROPIC_API_KEY or OPENAI_API_KEY set in environment
    - SLACK_WEBHOOK_URL set in environment (for Slack posting)

Usage:
    python examples/slack_code_review/main.py --diff path/to/diff.patch
    python examples/slack_code_review/main.py --diff - < diff.patch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# --- API key check -------------------------------------------------------


def _check_api_keys() -> None:
    """Exit early with a helpful message if no API keys are configured."""
    keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
    if not any(os.environ.get(k) for k in keys):
        print(
            "ERROR: No API key found. Set at least one of:\n"
            "  ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY\n"
            "See the README for setup instructions.",
            file=sys.stderr,
        )
        sys.exit(1)


# --- Debate setup ---------------------------------------------------------


def build_review_task(diff_text: str) -> str:
    """Build the debate task prompt from a code diff."""
    return (
        "You are reviewing the following code diff. Identify:\n"
        "1. Bugs or logic errors\n"
        "2. Security vulnerabilities\n"
        "3. Performance concerns\n"
        "4. Style and maintainability issues\n"
        "5. Missing tests or documentation\n\n"
        "Provide a structured review with severity ratings "
        "(critical / warning / info) for each finding.\n\n"
        f"```diff\n{diff_text[:8000]}\n```"
    )


async def run_code_review(diff_text: str) -> dict:
    """Run a multi-agent code review debate and return structured findings."""
    from aragora import Arena, Environment, DebateProtocol

    # Configure a quick 3-round debate focused on code review
    env = Environment(
        task=build_review_task(diff_text),
        context="This is a code review. Be specific and cite line numbers.",
        roles=["proposer", "critic", "synthesizer"],
        max_rounds=3,
    )

    protocol = DebateProtocol(
        rounds=3,
        consensus="majority",
        use_structured_phases=False,
    )

    # Run the debate -- Arena selects agents based on available API keys
    arena = Arena(env, protocol=protocol)
    result = await arena.run()

    # Structure the output
    return {
        "consensus_reached": result.consensus_reached,
        "confidence": result.confidence,
        "rounds_used": result.rounds_used,
        "review": result.final_answer,
        "participants": result.participants,
        "winner": result.winner,
    }


# --- Slack posting --------------------------------------------------------


async def post_to_slack(review: dict) -> bool:
    """Post the review to Slack using the Aragora Slack integration."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("SLACK_WEBHOOK_URL not set -- skipping Slack post.", file=sys.stderr)
        return False

    from aragora.integrations.slack import SlackConfig, SlackIntegration, SlackMessage

    config = SlackConfig(
        webhook_url=webhook_url,
        channel="#code-reviews",
        bot_name="Aragora Code Reviewer",
    )
    slack = SlackIntegration(config)

    # Build a rich Slack message from the review findings
    status = "Approved" if review["consensus_reached"] else "Needs Discussion"
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Code Review: {status}"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Confidence:* {review['confidence']:.0%}"},
                {"type": "mrkdwn", "text": f"*Rounds:* {review['rounds_used']}"},
                {"type": "mrkdwn", "text": f"*Agents:* {', '.join(review['participants'])}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Review Summary:*\n```{review['review'][:2500]}```",
            },
        },
    ]

    message = SlackMessage(
        text=f"Code Review: {status}",
        blocks=blocks,
    )

    try:
        success = await slack._send_message(message)
        return success
    finally:
        await slack.close()


# --- CLI entry point ------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent code review via Aragora")
    parser.add_argument(
        "--diff",
        required=True,
        help="Path to diff file, or '-' to read from stdin",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable text",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    _check_api_keys()

    # Read the diff
    if args.diff == "-":
        diff_text = sys.stdin.read()
    else:
        with open(args.diff) as f:
            diff_text = f.read()

    if not diff_text.strip():
        print("ERROR: Empty diff provided.", file=sys.stderr)
        sys.exit(1)

    print(f"Reviewing diff ({len(diff_text)} chars) with multi-agent debate...")

    # Run the review
    review = await run_code_review(diff_text)

    # Output results
    if args.json:
        print(json.dumps(review, indent=2))
    else:
        status = "APPROVED" if review["consensus_reached"] else "NEEDS DISCUSSION"
        print(f"\n{'=' * 60}")
        print(f"  Code Review Result: {status}")
        print(f"  Confidence: {review['confidence']:.0%}")
        print(f"  Agents: {', '.join(review['participants'])}")
        print(f"{'=' * 60}")
        print(f"\n{review['review']}")

    # Post to Slack if configured
    if os.environ.get("SLACK_WEBHOOK_URL"):
        posted = await post_to_slack(review)
        if posted:
            print("\nReview posted to Slack.")


if __name__ == "__main__":
    asyncio.run(main())
