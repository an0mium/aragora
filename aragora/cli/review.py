#!/usr/bin/env python3
"""
Aragora PR Review - Multi Agent Code Review

Run multi-agent code review debates on diffs/PRs:
    git diff main | aragora review
    aragora review https://github.com/owner/repo/pull/123
    aragora review --diff-file pr.diff --output-dir ./artifacts
    aragora review --demo  # Try without API keys
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

from aragora.agents.base import AgentType, create_agent
from aragora.core import Agent, DebateResult, Environment
from aragora.debate.disagreement import DisagreementReporter
from aragora.debate.orchestrator import Arena, DebateProtocol

# Default agents for code review (fast, diverse perspectives)
DEFAULT_REVIEW_AGENTS = "anthropic-api,openai-api"
DEFAULT_ROUNDS = 2  # Fast reviews
MAX_DIFF_SIZE = 50000  # 50KB max diff size
REVIEWS_DIR = Path.home() / ".aragora" / "reviews"
SHARE_BASE_URL = "https://aragora.ai/reviews"


def generate_review_id(findings: dict, diff_hash: str) -> str:
    """Generate a short, unique review ID."""
    # Use first 8 chars of UUID combined with diff hash for uniqueness
    uid = uuid.uuid4().hex[:8]
    return f"{uid}"


def save_review_for_sharing(
    review_id: str,
    findings: dict,
    diff: str,
    agents: str,
    pr_url: Optional[str] = None,
) -> Path:
    """Save review to local storage for sharing.

    Reviews are stored at ~/.aragora/reviews/{id}.json
    The server can serve these for shareable links.
    """
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

    # Create review record
    review_data = {
        "id": review_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "agents": agents.split(","),
        "pr_url": pr_url,
        "diff_preview": diff[:500] + "..." if len(diff) > 500 else diff,
        "diff_hash": hashlib.sha256(diff.encode()).hexdigest()[:16],
        "findings": {
            "unanimous_critiques": findings.get("unanimous_critiques", []),
            "split_opinions": [
                {"issue": d, "majority": m, "minority": mi}
                for d, m, mi in findings.get("split_opinions", [])
            ],
            "risk_areas": findings.get("risk_areas", []),
            "agreement_score": findings.get("agreement_score", 0),
            "critical_issues": findings.get("critical_issues", []),
            "high_issues": findings.get("high_issues", []),
            "medium_issues": findings.get("medium_issues", []),
            "low_issues": findings.get("low_issues", []),
            "summary": findings.get("final_summary", ""),
        },
    }

    # Save to file
    review_path = REVIEWS_DIR / f"{review_id}.json"
    review_path.write_text(json.dumps(review_data, indent=2))

    return review_path


def get_shareable_url(review_id: str) -> str:
    """Get the shareable URL for a review."""
    return f"{SHARE_BASE_URL}/{review_id}"


def get_available_agents() -> str:
    """Get available agents based on configured API keys.

    Falls back gracefully if not all providers are configured.
    """
    agents = []

    # Check Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        agents.append("anthropic-api")

    # Check OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        agents.append("openai-api")

    # Check OpenRouter as fallback
    if os.environ.get("OPENROUTER_API_KEY"):
        if len(agents) < 2:
            agents.append("openrouter-api")

    # Check other providers
    if os.environ.get("GEMINI_API_KEY") and len(agents) < 2:
        agents.append("gemini-api")

    if os.environ.get("MISTRAL_API_KEY") and len(agents) < 2:
        agents.append("mistral-api")

    if not agents:
        return ""

    return ",".join(agents)


def get_demo_findings() -> dict:
    """Get demo review findings for trying without API keys."""
    return {
        "unanimous_critiques": [
            "SQL injection vulnerability in user search - query built with string concatenation",
            "Missing input validation on file upload endpoint",
        ],
        "split_opinions": [
            ("Add request rate limiting", ["anthropic-api", "openai-api"], ["gemini-api"]),
            ("Cache database queries", ["anthropic-api"], ["openai-api", "gemini-api"]),
        ],
        "risk_areas": [
            "Error handling in payment flow may expose sensitive data",
            "Session management needs manual review",
        ],
        "agreement_score": 0.75,
        "agent_alignment": {
            "anthropic-api": {"openai-api": 0.8, "gemini-api": 0.6},
            "openai-api": {"anthropic-api": 0.8, "gemini-api": 0.7},
        },
        "critical_issues": [
            {
                "agent": "anthropic-api",
                "issue": "SQL injection in search_users()",
                "target": "api/users.py:45",
            },
        ],
        "high_issues": [
            {
                "agent": "openai-api",
                "issue": "Missing CSRF protection on POST endpoints",
                "target": "api/routes.py",
            },
        ],
        "medium_issues": [
            {
                "agent": "gemini-api",
                "issue": "Unbounded query results - add pagination",
                "target": "api/products.py:102",
            },
        ],
        "low_issues": [],
        "all_critiques": [],
        "final_summary": """## Multi Agent Review Summary

This code review identified **2 critical security issues** that all AI models agree on.

**Unanimous Findings (High Confidence):**
1. SQL injection in `search_users()` - user input directly concatenated into query
2. Missing input validation on file upload - allows arbitrary file types

**Split Opinions:**
- Rate limiting: 2/3 models recommend adding, 1 suggests it's premature
- Query caching: Models disagree on whether caching adds complexity without benefit

**Recommendation:** Address the SQL injection immediately before merging.""",
        "agents_used": ["anthropic-api", "openai-api", "gemini-api"],
    }


def build_review_prompt(diff: str, focus_areas: Optional[list[str]] = None) -> str:
    """Build a focused code review prompt."""
    focus = focus_areas or ["security", "performance", "quality"]

    focus_instructions = []
    if "security" in focus:
        focus_instructions.append(
            """
**Security** - Look for:
- SQL/NoSQL injection, XSS, CSRF
- Authentication/authorization bypass
- Secrets/credentials in code
- Insecure deserialization
- Path traversal"""
        )

    if "performance" in focus:
        focus_instructions.append(
            """
**Performance** - Look for:
- N+1 query patterns
- O(n^2) or worse algorithms
- Memory leaks, unbounded collections
- Missing pagination
- Blocking operations in async code"""
        )

    if "quality" in focus:
        focus_instructions.append(
            """
**Code Quality** - Look for:
- Missing error handling
- Edge cases not covered
- Unclear or complex logic
- Missing input validation
- Resource cleanup issues"""
        )

    focus_text = "\n".join(focus_instructions)

    # Truncate diff if too large
    if len(diff) > MAX_DIFF_SIZE:
        diff = diff[:MAX_DIFF_SIZE] + "\n\n[... diff truncated ...]"

    return f"""You are reviewing a pull request. Analyze the diff carefully and identify issues.

## Review Focus
{focus_text}

## Diff to Review
```diff
{diff}
```

## Response Format
For each issue found, specify:
1. **Category**: Security, Performance, or Quality
2. **Severity**: CRITICAL, HIGH, MEDIUM, or LOW
3. **Location**: File and line number if identifiable
4. **Issue**: Clear description of the problem
5. **Suggestion**: How to fix it

If no issues found in a category, say "No issues found."

Be thorough but avoid false positives. Focus on real, actionable issues."""


async def run_review_debate(
    diff: str,
    agents_str: str = DEFAULT_REVIEW_AGENTS,
    rounds: int = DEFAULT_ROUNDS,
    focus_areas: Optional[list[str]] = None,
) -> DebateResult:
    """Run a code review debate on the given diff."""

    # Parse and create agents
    agent_specs = []
    for spec in agents_str.split(","):
        spec = spec.strip()
        if spec:
            agent_specs.append(spec)

    if len(agent_specs) < 2:
        agent_specs = DEFAULT_REVIEW_AGENTS.split(",")

    # Create agents with reviewer roles
    agents: list[Agent] = []
    roles = ["security_reviewer", "performance_reviewer", "quality_reviewer"]
    for i, agent_type in enumerate(agent_specs):
        role = roles[i % len(roles)]
        agent = create_agent(
            model_type=cast(AgentType, agent_type),
            name=f"{agent_type}_{role}",
            role=role,
        )
        agents.append(agent)

    # Build review prompt
    task = build_review_prompt(diff, focus_areas)

    # Create environment and protocol
    env = Environment(task=task, max_rounds=rounds)
    protocol = DebateProtocol(rounds=rounds, consensus="majority")

    # Run debate
    arena = Arena(env, agents, protocol)
    result = await arena.run()

    return result


def extract_review_findings(result: DebateResult) -> dict:
    """Extract structured findings from debate result."""
    reporter = DisagreementReporter()
    report = reporter.generate_report(
        votes=result.votes,
        critiques=result.critiques,
        winner=result.final_answer[:100] if result.final_answer else None,
    )

    # Categorize critiques by severity
    critical_issues = []
    high_issues = []
    medium_issues = []
    low_issues = []

    for critique in result.critiques:
        severity = critique.severity if hasattr(critique, "severity") else 0.5
        for issue in critique.issues:
            issue_data = {
                "agent": critique.agent,
                "issue": issue,
                "target": critique.target_agent,
                "suggestions": critique.suggestions,
            }
            if severity >= 0.9:
                critical_issues.append(issue_data)
            elif severity >= 0.7:
                high_issues.append(issue_data)
            elif severity >= 0.4:
                medium_issues.append(issue_data)
            else:
                low_issues.append(issue_data)

    return {
        "unanimous_critiques": report.unanimous_critiques,
        "split_opinions": report.split_opinions,
        "risk_areas": report.risk_areas,
        "agreement_score": report.agreement_score,
        "agent_alignment": report.agent_alignment,
        "critical_issues": critical_issues,
        "high_issues": high_issues,
        "medium_issues": medium_issues,
        "low_issues": low_issues,
        "all_critiques": result.critiques,
        "final_summary": result.final_answer,
        "agents_used": list(set(m.agent for m in result.messages)) if result.messages else [],
    }


def format_github_comment(result: Optional[DebateResult], findings: dict[str, Any]) -> str:
    """Format findings as a GitHub PR comment."""
    agents_used = findings.get("agents_used", [])
    agent_names = (
        ", ".join(set(a.split("_")[0] for a in agents_used)) if agents_used else "AI agents"
    )

    lines = [
        "## Multi Agent Code Review",
        "",
        f"**{len(agents_used)} agents reviewed this PR** ({agent_names})",
        "",
    ]

    # Unanimous issues (high confidence)
    unanimous = findings.get("unanimous_critiques", [])
    if unanimous:
        lines.extend(
            [
                "### Unanimous Issues",
                "> All AI models agree - address these first",
                "",
            ]
        )
        for issue in unanimous[:5]:  # Limit to top 5
            lines.append(f"- {issue}")
        lines.append("")

    # Critical/High issues
    critical = findings.get("critical_issues", [])
    high = findings.get("high_issues", [])
    if critical or high:
        lines.extend(
            [
                "### Critical & High Severity Issues",
                "",
            ]
        )
        for issue in (critical + high)[:5]:
            severity = "CRITICAL" if issue in critical else "HIGH"
            lines.append(f"- **{severity}**: {issue['issue'][:200]}")
        lines.append("")

    # Split opinions
    split = findings.get("split_opinions", [])
    if split:
        lines.extend(
            [
                "### Split Opinions",
                "> Agents disagree - your call on the tradeoff",
                "",
                "| Topic | For | Against |",
                "|-------|-----|---------|",
            ]
        )
        for desc, majority, minority in split[:3]:
            topic = desc[:50] + "..." if len(desc) > 50 else desc
            lines.append(f"| {topic} | {', '.join(majority)} | {', '.join(minority)} |")
        lines.append("")

    # Risk areas
    risks = findings.get("risk_areas", [])
    if risks:
        lines.extend(
            [
                "### Risk Areas",
                "> Low confidence - manual review recommended",
                "",
            ]
        )
        for risk in risks[:3]:
            lines.append(f"- {risk}")
        lines.append("")

    # Summary if available
    summary = findings.get("final_summary", "")
    if summary and len(summary) > 50:
        lines.extend(
            [
                "### Summary",
                "",
                summary[:500] + ("..." if len(summary) > 500 else ""),
                "",
            ]
        )

    # Footer
    agreement = findings.get("agreement_score", 0)
    lines.extend(
        [
            "---",
            f"*Agreement score: {agreement:.0%} | Powered by [Aragora](https://github.com/an0mium/aragora) - Multi Agent Decision Making*",
        ]
    )

    return "\n".join(lines)


def cmd_review(args: argparse.Namespace) -> int:
    """Handle 'review' command."""

    # Demo mode - show sample output without API keys
    if getattr(args, "demo", False):
        print("Running in demo mode (no API calls)...", file=sys.stderr)
        findings = get_demo_findings()

        output_dir = Path(args.output_dir) if args.output_dir else None

        if args.output_format == "github":
            comment = format_github_comment(None, findings)
            comment = "**[DEMO MODE]** " + comment
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "comment.md").write_text(comment)
            print(comment)
        elif args.output_format == "json":
            import json

            output = {
                "demo_mode": True,
                "unanimous_critiques": findings["unanimous_critiques"],
                "split_opinions": [(d, m, mi) for d, m, mi in findings["split_opinions"]],
                "risk_areas": findings["risk_areas"],
                "agreement_score": findings["agreement_score"],
                "critical_issues": findings["critical_issues"],
                "high_issues": findings["high_issues"],
                "summary": findings["final_summary"],
            }
            json_output = json.dumps(output, indent=2)
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "review.json").write_text(json_output)
            print(json_output)
        else:
            print("Demo mode only supports github and json output formats", file=sys.stderr)
            return 1

        print("\n---", file=sys.stderr)
        print("This was a demo. To run a real review, configure API keys:", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        print("  export OPENAI_API_KEY=sk-...", file=sys.stderr)
        return 0

    # Get diff content
    diff = ""

    if args.diff_file:
        # Read from file
        diff_path = Path(args.diff_file)
        if not diff_path.exists():
            print(f"Error: Diff file not found: {args.diff_file}", file=sys.stderr)
            return 1
        diff = diff_path.read_text()
    elif args.pr_url:
        # Fetch from GitHub PR
        print(f"Fetching PR diff from: {args.pr_url}", file=sys.stderr)
        try:
            # Extract owner/repo/number from URL
            # Supports: https://github.com/owner/repo/pull/123
            parts = args.pr_url.rstrip("/").split("/")
            if len(parts) >= 5 and parts[-2] == "pull":
                pr_number = parts[-1]
                # Extract owner/repo for cross-repo support
                try:
                    # Find github.com index and extract owner/repo
                    gh_idx = next(i for i, p in enumerate(parts) if "github.com" in p)
                    owner = parts[gh_idx + 1]
                    repo = parts[gh_idx + 2]
                    repo_arg: Optional[str] = f"{owner}/{repo}"
                except (StopIteration, IndexError):
                    repo_arg = None

                # Build gh command with repo context if available
                gh_cmd = ["gh", "pr", "diff", pr_number]
                if repo_arg:
                    gh_cmd.extend(["--repo", repo_arg])

                gh_result = subprocess.run(
                    gh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    shell=False,
                )
                if gh_result.returncode == 0:
                    diff = gh_result.stdout
                else:
                    print(f"Error fetching PR: {gh_result.stderr}", file=sys.stderr)
                    return 1
            else:
                print(f"Invalid PR URL format: {args.pr_url}", file=sys.stderr)
                print("Expected: https://github.com/owner/repo/pull/123", file=sys.stderr)
                return 1
        except subprocess.TimeoutExpired:
            print("Timeout fetching PR diff", file=sys.stderr)
            return 1
        except FileNotFoundError:
            print("Error: 'gh' CLI not found. Install GitHub CLI.", file=sys.stderr)
            return 1
    elif not sys.stdin.isatty():
        # Read from stdin
        diff = sys.stdin.read()
    else:
        print(
            "Error: No diff provided. Use --diff-file, PR URL, or pipe diff to stdin.",
            file=sys.stderr,
        )
        return 1

    if not diff.strip():
        print("Error: Empty diff", file=sys.stderr)
        return 1

    # Determine which agents to use
    agents_str = args.agents
    if agents_str == DEFAULT_REVIEW_AGENTS:
        # Check if default agents are available, fall back if not
        available = get_available_agents()
        if not available:
            print("Error: No API keys configured.", file=sys.stderr)
            print(
                "Set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY",
                file=sys.stderr,
            )
            print("\nTry demo mode instead: aragora review --demo", file=sys.stderr)
            return 1
        if available != DEFAULT_REVIEW_AGENTS:
            print(f"Note: Using available agents: {available}", file=sys.stderr)
            agents_str = available

    # Run review debate
    print(f"Running AI code review ({agents_str}, {args.rounds} rounds)...", file=sys.stderr)

    try:
        result = asyncio.run(
            run_review_debate(
                diff=diff,
                agents_str=agents_str,
                rounds=args.rounds,
                focus_areas=args.focus.split(",") if args.focus else None,
            )
        )
    except Exception as e:
        print(f"Error running review: {e}", file=sys.stderr)
        return 1

    # Extract findings
    findings = extract_review_findings(result)

    # Generate shareable link if requested
    share_url = None
    if getattr(args, "share", False):
        diff_hash = hashlib.sha256(diff.encode()).hexdigest()[:16]
        review_id = generate_review_id(findings, diff_hash)
        save_review_for_sharing(
            review_id=review_id,
            findings=findings,
            diff=diff,
            agents=agents_str,
            pr_url=getattr(args, "pr_url", None),
        )
        share_url = get_shareable_url(review_id)
        print(f"\nShareable link: {share_url}", file=sys.stderr)
        print(f"Review ID: {review_id}", file=sys.stderr)

    # Output based on format
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.output_format == "github":
        comment = format_github_comment(result, findings)
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "comment.md").write_text(comment)
            print(f"Comment saved to: {output_dir / 'comment.md'}", file=sys.stderr)
        print(comment)

    elif args.output_format == "json":
        import json

        # Convert to JSON-serializable format
        output = {
            "unanimous_critiques": findings["unanimous_critiques"],
            "split_opinions": [(d, m, mi) for d, m, mi in findings["split_opinions"]],
            "risk_areas": findings["risk_areas"],
            "agreement_score": findings["agreement_score"],
            "critical_issues": findings["critical_issues"],
            "high_issues": findings["high_issues"],
            "medium_issues": findings["medium_issues"],
            "low_issues": findings["low_issues"],
            "summary": findings["final_summary"],
        }
        json_output = json.dumps(output, indent=2)
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "review.json").write_text(json_output)
        print(json_output)

    elif args.output_format == "html":
        # Use existing static HTML exporter
        try:
            from aragora.export.artifact import ArtifactBuilder
            from aragora.export.static_html import StaticHTMLExporter

            artifact = ArtifactBuilder().from_result(result).build()
            exporter = StaticHTMLExporter(artifact)
            html = exporter.generate()
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "review.html").write_text(html)
                print(f"HTML report saved to: {output_dir / 'review.html'}", file=sys.stderr)
            else:
                print(html)
        except ImportError:
            print("Error: HTML export not available", file=sys.stderr)
            return 1

    return 0


def create_review_parser(subparsers) -> None:
    """Add review subcommand to argument parser."""
    parser = subparsers.add_parser(
        "review",
        help="Run AI code review on a diff or PR",
        description="Multi-agent AI code review for pull requests",
    )

    parser.add_argument(
        "pr_url",
        nargs="?",
        help="GitHub PR URL (e.g., https://github.com/owner/repo/pull/123)",
    )

    parser.add_argument(
        "--diff-file",
        help="Path to diff file (alternative to PR URL or stdin)",
    )

    parser.add_argument(
        "--agents",
        default=DEFAULT_REVIEW_AGENTS,
        help=f"Comma-separated list of agents (default: {DEFAULT_REVIEW_AGENTS})",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Number of debate rounds (default: {DEFAULT_ROUNDS})",
    )

    parser.add_argument(
        "--focus",
        default="security,performance,quality",
        help="Focus areas: security,performance,quality (default: all)",
    )

    parser.add_argument(
        "--output-format",
        choices=["github", "json", "html"],
        default="github",
        help="Output format (default: github)",
    )

    parser.add_argument(
        "--output-dir",
        help="Directory to save output artifacts",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (no API keys required, shows sample output)",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a shareable link for this review",
    )

    parser.set_defaults(func=cmd_review)


# For direct module execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aragora PR Review")
    subparsers = parser.add_subparsers(dest="command")
    create_review_parser(subparsers)
    args = parser.parse_args()

    if hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)
