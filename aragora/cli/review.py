#!/usr/bin/env python3
"""
Aragora PR Review - AI Red Team Code Review

Run multi-agent code review debates on diffs/PRs:
    git diff main | aragora review
    aragora review https://github.com/owner/repo/pull/123
    aragora review --diff-file pr.diff --output-dir ./artifacts
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from aragora.agents.base import create_agent
from aragora.core import Environment, DebateResult, Critique
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.debate.disagreement import DisagreementReporter


# Default agents for code review (fast, diverse perspectives)
DEFAULT_REVIEW_AGENTS = "anthropic-api,openai-api"
DEFAULT_ROUNDS = 2  # Fast reviews
MAX_DIFF_SIZE = 50000  # 50KB max diff size


def build_review_prompt(diff: str, focus_areas: Optional[list[str]] = None) -> str:
    """Build a focused code review prompt."""
    focus = focus_areas or ["security", "performance", "quality"]

    focus_instructions = []
    if "security" in focus:
        focus_instructions.append("""
**Security** - Look for:
- SQL/NoSQL injection, XSS, CSRF
- Authentication/authorization bypass
- Secrets/credentials in code
- Insecure deserialization
- Path traversal""")

    if "performance" in focus:
        focus_instructions.append("""
**Performance** - Look for:
- N+1 query patterns
- O(n^2) or worse algorithms
- Memory leaks, unbounded collections
- Missing pagination
- Blocking operations in async code""")

    if "quality" in focus:
        focus_instructions.append("""
**Code Quality** - Look for:
- Missing error handling
- Edge cases not covered
- Unclear or complex logic
- Missing input validation
- Resource cleanup issues""")

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
    agents = []
    roles = ["security_reviewer", "performance_reviewer", "quality_reviewer"]
    for i, agent_type in enumerate(agent_specs):
        role = roles[i % len(roles)]
        agent = create_agent(
            model_type=agent_type,
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
        severity = critique.severity if hasattr(critique, 'severity') else 0.5
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


def format_github_comment(result: DebateResult, findings: dict) -> str:
    """Format findings as a GitHub PR comment."""
    agents_used = findings.get("agents_used", [])
    agent_names = ", ".join(set(a.split("_")[0] for a in agents_used)) if agents_used else "AI agents"

    lines = [
        "## AI Red Team Code Review",
        "",
        f"**{len(agents_used)} agents reviewed this PR** ({agent_names})",
        "",
    ]

    # Unanimous issues (high confidence)
    unanimous = findings.get("unanimous_critiques", [])
    if unanimous:
        lines.extend([
            "### Unanimous Issues",
            "> All AI models agree - address these first",
            "",
        ])
        for issue in unanimous[:5]:  # Limit to top 5
            lines.append(f"- {issue}")
        lines.append("")

    # Critical/High issues
    critical = findings.get("critical_issues", [])
    high = findings.get("high_issues", [])
    if critical or high:
        lines.extend([
            "### Critical & High Severity Issues",
            "",
        ])
        for issue in (critical + high)[:5]:
            severity = "CRITICAL" if issue in critical else "HIGH"
            lines.append(f"- **{severity}**: {issue['issue'][:200]}")
        lines.append("")

    # Split opinions
    split = findings.get("split_opinions", [])
    if split:
        lines.extend([
            "### Split Opinions",
            "> Agents disagree - your call on the tradeoff",
            "",
            "| Topic | For | Against |",
            "|-------|-----|---------|",
        ])
        for desc, majority, minority in split[:3]:
            topic = desc[:50] + "..." if len(desc) > 50 else desc
            lines.append(f"| {topic} | {', '.join(majority)} | {', '.join(minority)} |")
        lines.append("")

    # Risk areas
    risks = findings.get("risk_areas", [])
    if risks:
        lines.extend([
            "### Risk Areas",
            "> Low confidence - manual review recommended",
            "",
        ])
        for risk in risks[:3]:
            lines.append(f"- {risk}")
        lines.append("")

    # Summary if available
    summary = findings.get("final_summary", "")
    if summary and len(summary) > 50:
        lines.extend([
            "### Summary",
            "",
            summary[:500] + ("..." if len(summary) > 500 else ""),
            "",
        ])

    # Footer
    agreement = findings.get("agreement_score", 0)
    lines.extend([
        "---",
        f"*Agreement score: {agreement:.0%} | Powered by [Aragora](https://github.com/an0mium/aragora) - AI Red Team*",
    ])

    return "\n".join(lines)


def cmd_review(args: argparse.Namespace) -> int:
    """Handle 'review' command."""
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
            import subprocess
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
                    repo_arg = f"{owner}/{repo}"
                except (StopIteration, IndexError):
                    repo_arg = None

                # Build gh command with repo context if available
                gh_cmd = ["gh", "pr", "diff", pr_number]
                if repo_arg:
                    gh_cmd.extend(["--repo", repo_arg])

                result = subprocess.run(
                    gh_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    diff = result.stdout
                else:
                    print(f"Error fetching PR: {result.stderr}", file=sys.stderr)
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
        print("Error: No diff provided. Use --diff-file, PR URL, or pipe diff to stdin.", file=sys.stderr)
        return 1

    if not diff.strip():
        print("Error: Empty diff", file=sys.stderr)
        return 1

    # Run review debate
    print(f"Running AI code review ({args.agents}, {args.rounds} rounds)...", file=sys.stderr)

    try:
        result = asyncio.run(
            run_review_debate(
                diff=diff,
                agents_str=args.agents,
                rounds=args.rounds,
                focus_areas=args.focus.split(",") if args.focus else None,
            )
        )
    except Exception as e:
        print(f"Error running review: {e}", file=sys.stderr)
        return 1

    # Extract findings
    findings = extract_review_findings(result)

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
            from aragora.export.static_html import StaticHTMLExporter
            exporter = StaticHTMLExporter()
            html = exporter.export(result)
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
