#!/usr/bin/env python3
"""
Slack Code Review Bot
=====================

A Slack bot that uses Aragora's multi-agent debate engine to perform
comprehensive code reviews on GitHub pull requests. Multiple AI agents
review code from different perspectives (security, performance, best
practices) and reach consensus on findings.

Architecture:
    1. Listens for Slack slash commands (/review <owner/repo#pr>)
    2. Fetches PR diff from GitHub via Aragora's connector
    3. Runs a multi-agent debate with specialized reviewer roles
    4. Posts structured findings back to Slack with severity ratings
    5. Generates a tamper-evident decision receipt for audit

Requirements:
    - ANTHROPIC_API_KEY or OPENAI_API_KEY (at least one)
    - SLACK_WEBHOOK_URL (Slack incoming webhook)
    - GITHUB_TOKEN (optional, for private repos)

Usage:
    # Run in demo mode (no external services needed)
    python examples/slack-review-bot/main.py --demo

    # Run with live Slack + GitHub integration
    python examples/slack-review-bot/main.py --webhook https://hooks.slack.com/... --repo owner/repo --pr 42
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure aragora is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aragora import Arena, Environment, DebateProtocol
from aragora.agents.base import create_agent
from aragora.integrations.slack import SlackConfig, SlackIntegration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("slack-review-bot")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReviewBotConfig:
    """Configuration for the Slack review bot."""

    slack_webhook_url: str = ""
    slack_channel: str = "#code-reviews"
    github_token: str = ""
    debate_rounds: int = 2
    consensus_mode: str = "majority"
    min_consensus_confidence: float = 0.7
    agent_types: list[str] = field(default_factory=lambda: [
        "anthropic-api",
        "openai-api",
    ])
    reviewer_roles: list[str] = field(default_factory=lambda: [
        "security_reviewer",
        "performance_reviewer",
        "best_practices_reviewer",
    ])

    @classmethod
    def from_env(cls) -> ReviewBotConfig:
        """Load configuration from environment variables."""
        return cls(
            slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL", ""),
            slack_channel=os.environ.get("SLACK_CHANNEL", "#code-reviews"),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            debate_rounds=int(os.environ.get("REVIEW_ROUNDS", "2")),
            consensus_mode=os.environ.get("CONSENSUS_MODE", "majority"),
        )


# ---------------------------------------------------------------------------
# PR Diff Fetching
# ---------------------------------------------------------------------------

DEMO_PR_DIFF = """
diff --git a/api/auth.py b/api/auth.py
index abc1234..def5678 100644
--- a/api/auth.py
+++ b/api/auth.py
@@ -15,6 +15,25 @@ from flask import Flask, request, jsonify
+def authenticate_user(username: str, password: str) -> dict:
+    \"\"\"Authenticate a user and return a session token.\"\"\"
+    # Build the SQL query
+    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
+    result = db.execute(query)
+
+    if result:
+        token = hashlib.md5(f"{username}{time.time()}".encode()).hexdigest()
+        return {"token": token, "user": result[0]}
+    return {"error": "Invalid credentials"}
+
+
+@app.route("/api/admin/users", methods=["DELETE"])
+def delete_all_users():
+    \"\"\"Admin endpoint to delete users.\"\"\"
+    user_ids = request.json.get("ids", [])
+    for uid in user_ids:
+        db.execute(f"DELETE FROM users WHERE id = {uid}")
+    return jsonify({"deleted": len(user_ids)})
+
+
+@app.route("/api/upload", methods=["POST"])
+def upload_file():
+    \"\"\"Handle file uploads.\"\"\"
+    file = request.files["file"]
+    file.save(f"/uploads/{file.filename}")
+    return jsonify({"path": f"/uploads/{file.filename}"})
""".strip()


async def fetch_pr_diff(repo: str, pr_number: int, token: str = "") -> str:
    """Fetch the diff for a GitHub pull request.

    In production, this would use Aragora's GitHubConnector or the GitHub API.
    For the demo, it returns a sample diff with intentional vulnerabilities.
    """
    if not repo or not token:
        logger.info("Using demo PR diff (no GitHub token provided)")
        return DEMO_PR_DIFF

    try:
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector(repo=repo, token=token)
        results = await connector.search(
            query=f"pr:{pr_number}",
            max_results=1,
        )
        if results:
            return results[0].content
    except Exception as exc:
        logger.warning("Failed to fetch PR diff from GitHub: %s", exc)

    logger.info("Falling back to demo PR diff")
    return DEMO_PR_DIFF


# ---------------------------------------------------------------------------
# Multi-Agent Code Review via Debate
# ---------------------------------------------------------------------------

async def run_code_review_debate(
    diff: str,
    config: ReviewBotConfig,
    pr_info: str = "PR #42 in example/repo",
) -> dict[str, Any]:
    """Run a multi-agent debate to review a code diff.

    Creates specialized reviewer agents (security, performance, best practices)
    and uses Aragora's consensus engine to produce a unified review.

    Returns a dictionary with:
        - findings: list of categorized issues
        - consensus_reached: whether agents agreed
        - confidence: overall confidence score
        - receipt: tamper-evident hash for audit
    """
    # Create specialized agents - use as many as are available
    agents = []
    role_cycle = iter(config.reviewer_roles)

    for agent_type in config.agent_types:
        role = next(role_cycle, "code_reviewer")
        try:
            agent = create_agent(
                model_type=agent_type,
                name=f"{role}_{agent_type}",
                role=role,
            )
            agents.append(agent)
            logger.info("Created agent: %s (%s)", role, agent_type)
        except Exception as exc:
            logger.warning("Could not create %s agent: %s", agent_type, exc)

    if len(agents) < 2:
        logger.error(
            "Need at least 2 agents for a debate. "
            "Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY."
        )
        return _build_demo_review(diff, pr_info)

    review_prompt = f"""You are reviewing {pr_info}.

Analyze the following code diff for issues across these categories:

1. **Security Vulnerabilities** - SQL injection, XSS, path traversal, auth issues
2. **Performance Issues** - N+1 queries, missing indexes, inefficient algorithms
3. **Best Practices** - Error handling, input validation, logging, documentation

For each issue found, provide:
- Category (Security / Performance / Best Practices)
- Severity (Critical / High / Medium / Low)
- Line reference from the diff
- Description of the problem
- Recommended fix

```diff
{diff}
```

Be thorough. Flag anything that could cause production incidents."""

    env = Environment(
        task=review_prompt,
        context="Automated code review for pull request",
    )

    protocol = DebateProtocol(
        rounds=config.debate_rounds,
        consensus=config.consensus_mode,
        enable_calibration=True,
    )

    arena = Arena(env, agents, protocol)

    logger.info(
        "Starting code review debate with %d agents, %d rounds...",
        len(agents),
        config.debate_rounds,
    )

    start_time = time.monotonic()
    result = await arena.run()
    elapsed_ms = (time.monotonic() - start_time) * 1000

    # Build review output
    review = {
        "pr_info": pr_info,
        "consensus_reached": getattr(result, "consensus_reached", False),
        "confidence": getattr(result, "confidence", 0.0),
        "final_answer": getattr(result, "final_answer", ""),
        "rounds_used": getattr(result, "rounds_used", 0),
        "participants": [a.name if hasattr(a, "name") else str(a) for a in agents],
        "elapsed_ms": elapsed_ms,
    }

    # Generate receipt hash for audit trail
    receipt_content = json.dumps(review, sort_keys=True, default=str)
    review["receipt_hash"] = hashlib.sha256(receipt_content.encode()).hexdigest()
    review["reviewed_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Review complete: consensus=%s, confidence=%.2f, time=%.0fms",
        review["consensus_reached"],
        review["confidence"],
        elapsed_ms,
    )

    return review


def _build_demo_review(diff: str, pr_info: str) -> dict[str, Any]:
    """Build a demo review result when agents are unavailable.

    This shows the expected output format without requiring API keys.
    """
    findings = [
        {
            "category": "Security",
            "severity": "Critical",
            "title": "SQL Injection in authenticate_user",
            "description": (
                "The query uses f-string interpolation with user input, "
                "allowing SQL injection. An attacker could bypass authentication "
                "with input like: ' OR '1'='1"
            ),
            "fix": "Use parameterized queries: db.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))",
        },
        {
            "category": "Security",
            "severity": "Critical",
            "title": "SQL Injection in delete_all_users",
            "description": (
                "User-supplied IDs are interpolated directly into DELETE query. "
                "An attacker could delete arbitrary data or extract information."
            ),
            "fix": "Use parameterized queries and validate that IDs are integers.",
        },
        {
            "category": "Security",
            "severity": "High",
            "title": "Path Traversal in upload_file",
            "description": (
                "The filename from the upload is used directly in the save path. "
                "An attacker could upload to arbitrary paths using ../../../etc/passwd."
            ),
            "fix": "Use werkzeug.utils.secure_filename() and validate the final path.",
        },
        {
            "category": "Security",
            "severity": "High",
            "title": "Weak Token Generation with MD5",
            "description": (
                "MD5 is cryptographically broken and predictable. "
                "Using time.time() as entropy source is also insecure."
            ),
            "fix": "Use secrets.token_urlsafe(32) for session tokens.",
        },
        {
            "category": "Security",
            "severity": "High",
            "title": "Missing Authentication on Admin Endpoint",
            "description": (
                "The /api/admin/users DELETE endpoint has no authentication "
                "or authorization check. Any user can delete other users."
            ),
            "fix": "Add @require_admin decorator or RBAC middleware.",
        },
        {
            "category": "Best Practices",
            "severity": "Medium",
            "title": "Plain-text Password Comparison",
            "description": (
                "Passwords appear to be stored and compared in plain text. "
                "This violates OWASP password storage guidelines."
            ),
            "fix": "Use bcrypt or argon2 for password hashing.",
        },
    ]

    review = {
        "pr_info": pr_info,
        "consensus_reached": True,
        "confidence": 0.92,
        "findings": findings,
        "final_answer": (
            "This PR introduces multiple critical security vulnerabilities "
            "that must be addressed before merging. The most severe issues are "
            "SQL injection in authentication and user deletion endpoints. "
            "All agents unanimously recommend blocking this PR."
        ),
        "rounds_used": 2,
        "participants": ["security_reviewer", "performance_reviewer", "best_practices_reviewer"],
        "elapsed_ms": 0.0,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }

    receipt_content = json.dumps(review, sort_keys=True, default=str)
    review["receipt_hash"] = hashlib.sha256(receipt_content.encode()).hexdigest()

    return review


# ---------------------------------------------------------------------------
# Slack Notification
# ---------------------------------------------------------------------------

def format_review_for_slack(review: dict[str, Any]) -> str:
    """Format the review results as a Slack message.

    Uses Slack's mrkdwn format for rich display.
    """
    lines = [
        f"*Code Review: {review['pr_info']}*",
        "",
    ]

    # Consensus status
    if review["consensus_reached"]:
        lines.append(
            f"Consensus reached with {review['confidence']:.0%} confidence "
            f"({review['rounds_used']} rounds, "
            f"{len(review['participants'])} agents)"
        )
    else:
        lines.append(
            f"No consensus after {review['rounds_used']} rounds "
            f"(confidence: {review['confidence']:.0%})"
        )

    # Findings summary
    findings = review.get("findings", [])
    if findings:
        lines.append("")
        lines.append(f"*{len(findings)} issues found:*")
        lines.append("")

        # Group by severity
        by_severity: dict[str, list[dict[str, Any]]] = {}
        for f in findings:
            severity = f.get("severity", "Medium")
            by_severity.setdefault(severity, []).append(f)

        severity_emoji = {
            "Critical": "[CRITICAL]",
            "High": "[HIGH]",
            "Medium": "[MEDIUM]",
            "Low": "[LOW]",
        }

        for severity in ["Critical", "High", "Medium", "Low"]:
            items = by_severity.get(severity, [])
            if not items:
                continue
            for item in items:
                prefix = severity_emoji.get(severity, "")
                lines.append(
                    f"  {prefix} *{item['title']}* ({item['category']})"
                )
                lines.append(f"    {item['description'][:200]}")
                if item.get("fix"):
                    lines.append(f"    Fix: _{item['fix'][:150]}_")
                lines.append("")

    # Final answer
    if review.get("final_answer"):
        lines.append("*Summary:*")
        lines.append(review["final_answer"][:500])

    # Audit trail
    lines.append("")
    lines.append(
        f"_Receipt: {review.get('receipt_hash', 'N/A')[:16]}... | "
        f"{review.get('reviewed_at', 'N/A')}_"
    )

    return "\n".join(lines)


async def post_to_slack(
    review: dict[str, Any],
    config: ReviewBotConfig,
) -> bool:
    """Post review results to Slack via webhook.

    Uses Aragora's built-in SlackIntegration with rate limiting
    and circuit breaker protection.
    """
    if not config.slack_webhook_url:
        logger.info("No Slack webhook configured, skipping notification")
        return False

    try:
        slack = SlackIntegration(
            SlackConfig(
                webhook_url=config.slack_webhook_url,
                channel=config.slack_channel,
                bot_name="Aragora Review Bot",
            )
        )

        message_text = format_review_for_slack(review)

        from aragora.integrations.slack import SlackMessage

        msg = SlackMessage(text=message_text)
        success = await slack._send_message(msg)

        if success:
            logger.info("Review posted to Slack channel %s", config.slack_channel)
        else:
            logger.warning("Failed to post review to Slack")

        await slack.close()
        return success

    except Exception as exc:
        logger.warning("Slack notification failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def print_review(review: dict[str, Any]) -> None:
    """Print a formatted review to the console."""
    border = "=" * 70

    print(f"\n{border}")
    print(f"  Code Review: {review['pr_info']}")
    print(border)

    if review["consensus_reached"]:
        print(
            f"\n  Consensus: YES (confidence {review['confidence']:.0%}, "
            f"{review['rounds_used']} rounds)"
        )
    else:
        print(
            f"\n  Consensus: NO (confidence {review['confidence']:.0%}, "
            f"{review['rounds_used']} rounds)"
        )

    print(f"  Reviewers: {', '.join(review['participants'])}")

    findings = review.get("findings", [])
    if findings:
        print(f"\n  --- {len(findings)} Issues Found ---\n")
        for i, f in enumerate(findings, 1):
            print(f"  {i}. [{f['severity']}] {f['title']} ({f['category']})")
            print(f"     {f['description']}")
            if f.get("fix"):
                print(f"     Fix: {f['fix']}")
            print()

    if review.get("final_answer"):
        print("  --- Summary ---")
        print(f"  {review['final_answer']}")

    print(f"\n  Receipt: {review.get('receipt_hash', 'N/A')[:32]}...")
    print(f"  Reviewed at: {review.get('reviewed_at', 'N/A')}")
    print(border)


async def main() -> None:
    """Run the Slack code review bot."""
    parser = argparse.ArgumentParser(
        description="Aragora Slack Code Review Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (no API keys needed)
  python examples/slack-review-bot/main.py --demo

  # Review a real PR
  python examples/slack-review-bot/main.py --repo owner/repo --pr 42

  # Review and post to Slack
  python examples/slack-review-bot/main.py --repo owner/repo --pr 42 \\
      --webhook https://hooks.slack.com/services/T.../B.../...
        """,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run in demo mode with sample PR diff",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="",
        help="GitHub repository (owner/repo format)",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=0,
        help="Pull request number to review",
    )
    parser.add_argument(
        "--webhook",
        type=str,
        default="",
        help="Slack webhook URL for posting results",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="#code-reviews",
        help="Slack channel for posting results",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of debate rounds (default: 2)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON instead of formatted text",
    )

    args = parser.parse_args()

    # Build configuration
    config = ReviewBotConfig.from_env()
    if args.webhook:
        config.slack_webhook_url = args.webhook
    if args.channel:
        config.slack_channel = args.channel
    config.debate_rounds = args.rounds

    # Determine PR info
    pr_info = "Demo PR (sample vulnerabilities)"
    if args.repo and args.pr:
        pr_info = f"PR #{args.pr} in {args.repo}"

    # Step 1: Fetch the diff
    logger.info("Fetching PR diff...")
    diff = await fetch_pr_diff(
        repo=args.repo,
        pr_number=args.pr,
        token=config.github_token,
    )

    if not diff:
        logger.error("No diff to review")
        sys.exit(1)

    logger.info("Got diff (%d bytes)", len(diff))

    # Step 2: Run the multi-agent review
    if args.demo:
        logger.info("Running in demo mode (no API calls)")
        review = _build_demo_review(diff, pr_info)
    else:
        review = await run_code_review_debate(diff, config, pr_info)

    # Step 3: Display results
    if args.json:
        import json as json_mod
        print(json_mod.dumps(review, indent=2, default=str))
    else:
        print_review(review)

    # Step 4: Post to Slack (if configured)
    if config.slack_webhook_url:
        await post_to_slack(review, config)
    elif not args.json:
        slack_msg = format_review_for_slack(review)
        print("\n--- Slack Message Preview ---")
        print(slack_msg)
        print("--- End Preview ---")
        print(
            "\nTo post to Slack, set SLACK_WEBHOOK_URL or pass --webhook <url>"
        )


if __name__ == "__main__":
    asyncio.run(main())
