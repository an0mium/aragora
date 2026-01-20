"""
PR Comment Formatter for Multi Agent Code Review.

Generates GitHub-flavored markdown comments from debate results.
Designed to be scannable in 30 seconds with clear severity hierarchy.
"""

from __future__ import annotations

__all__ = [
    "PRCommentConfig",
    "format_compact_comment",
    "format_pr_comment",
    "format_slack_message",
]

from dataclasses import dataclass
from typing import Optional


@dataclass
class PRCommentConfig:
    """Configuration for PR comment formatting."""

    max_unanimous_issues: int = 5
    max_critical_high_issues: int = 5
    max_split_opinions: int = 3
    max_risk_areas: int = 3
    max_summary_length: int = 500
    include_footer: bool = True
    include_artifact_link: bool = True
    artifact_url: Optional[str] = None


def format_pr_comment(
    findings: dict,
    config: Optional[PRCommentConfig] = None,
) -> str:
    """Format findings as a GitHub PR comment.

    Args:
        findings: Dict with keys:
            - unanimous_critiques: List of issues all agents agree on
            - split_opinions: List of (description, majority, minority) tuples
            - risk_areas: List of low-confidence areas
            - agreement_score: Float 0-1
            - critical_issues: List of critical severity issues
            - high_issues: List of high severity issues
            - agents_used: List of agent names
            - final_summary: Optional summary text
        config: Optional formatting configuration

    Returns:
        GitHub-flavored markdown string
    """
    config = config or PRCommentConfig()

    agents_used = findings.get("agents_used", [])
    agent_names = _format_agent_names(agents_used)

    lines = [
        "## Multi Agent Code Review",
        "",
        f"**{len(agents_used)} agents reviewed this PR** ({agent_names})",
    ]

    if config.include_artifact_link and config.artifact_url:
        lines.append(f" | [Full Report]({config.artifact_url})")

    lines.append("")

    # Unanimous issues (highest confidence - address first)
    unanimous = findings.get("unanimous_critiques", [])
    if unanimous:
        lines.extend(
            [
                "### Unanimous Issues",
                "> All AI models agree - address these first",
                "",
            ]
        )
        for issue in unanimous[: config.max_unanimous_issues]:
            lines.append(f"- {issue}")
        if len(unanimous) > config.max_unanimous_issues:
            lines.append(f"- *...and {len(unanimous) - config.max_unanimous_issues} more*")
        lines.append("")

    # Critical & High severity issues
    critical = findings.get("critical_issues", [])
    high = findings.get("high_issues", [])
    if critical or high:
        lines.extend(
            [
                "### Critical & High Severity Issues",
                "",
            ]
        )
        combined = []
        for issue in critical:
            combined.append(("CRITICAL", issue))
        for issue in high:
            combined.append(("HIGH", issue))

        for severity, issue in combined[: config.max_critical_high_issues]:
            issue_text = issue.get("issue", str(issue)) if isinstance(issue, dict) else str(issue)
            # Truncate long issues
            if len(issue_text) > 200:
                issue_text = issue_text[:200] + "..."
            lines.append(f"- **{severity}**: {issue_text}")

        if len(combined) > config.max_critical_high_issues:
            lines.append(f"- *...and {len(combined) - config.max_critical_high_issues} more*")
        lines.append("")

    # Split opinions (agents disagree - user decides)
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
        for item in split[: config.max_split_opinions]:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                desc, majority, minority = item[0], item[1], item[2]
            else:
                continue

            topic = desc[:50] + "..." if len(str(desc)) > 50 else str(desc)
            majority_str = ", ".join(majority) if isinstance(majority, list) else str(majority)
            minority_str = ", ".join(minority) if isinstance(minority, list) else str(minority)
            lines.append(f"| {topic} | {majority_str} | {minority_str} |")

        if len(split) > config.max_split_opinions:
            lines.append(f"\n*...and {len(split) - config.max_split_opinions} more split opinions*")
        lines.append("")

    # Risk areas (low confidence - manual review needed)
    risks = findings.get("risk_areas", [])
    if risks:
        lines.extend(
            [
                "### Risk Areas",
                "> Low confidence - manual review recommended",
                "",
            ]
        )
        for risk in risks[: config.max_risk_areas]:
            lines.append(f"- {risk}")
        if len(risks) > config.max_risk_areas:
            lines.append(f"- *...and {len(risks) - config.max_risk_areas} more*")
        lines.append("")

    # Summary
    summary = findings.get("final_summary", "")
    if summary and len(summary) > 50:
        lines.extend(
            [
                "### Summary",
                "",
            ]
        )
        if len(summary) > config.max_summary_length:
            lines.append(summary[: config.max_summary_length] + "...")
        else:
            lines.append(summary)
        lines.append("")

    # Footer
    if config.include_footer:
        agreement = findings.get("agreement_score", 0)
        lines.extend(
            [
                "---",
                f"*Agreement score: {agreement:.0%} | Powered by [Aragora](https://github.com/an0mium/aragora) - Multi Agent Decision Making*",
            ]
        )

    return "\n".join(lines)


def format_compact_comment(findings: dict) -> str:
    """Format a compact single-line summary for quick status checks.

    Returns something like:
    "AI Review: 2 critical, 3 high, 1 unanimous | 85% agreement"
    """
    critical = len(findings.get("critical_issues", []))
    high = len(findings.get("high_issues", []))
    unanimous = len(findings.get("unanimous_critiques", []))
    agreement = findings.get("agreement_score", 0)

    parts = []
    if critical:
        parts.append(f"{critical} critical")
    if high:
        parts.append(f"{high} high")
    if unanimous:
        parts.append(f"{unanimous} unanimous")

    if not parts:
        return f"AI Review: No major issues found | {agreement:.0%} agreement"

    return f"AI Review: {', '.join(parts)} | {agreement:.0%} agreement"


def format_slack_message(findings: dict) -> dict:
    """Format findings as a Slack message payload.

    Returns dict ready for Slack webhook POST.
    """
    critical = len(findings.get("critical_issues", []))
    high = len(findings.get("high_issues", []))
    unanimous = findings.get("unanimous_critiques", [])
    agreement = findings.get("agreement_score", 0)

    # Determine color based on severity
    if critical > 0:
        color = "danger"  # red
    elif high > 0:
        color = "warning"  # yellow
    else:
        color = "good"  # green

    blocks = []

    # Header
    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Multi Agent Code Review",
            },
        }
    )

    # Summary section
    summary_text = f"*Agreement:* {agreement:.0%}"
    if critical:
        summary_text += f" | *Critical:* {critical}"
    if high:
        summary_text += f" | *High:* {high}"

    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": summary_text,
            },
        }
    )

    # Unanimous issues
    if unanimous:
        issues_text = "\n".join(f"• {issue}" for issue in unanimous[:3])
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Unanimous Issues:*\n{issues_text}",
                },
            }
        )

    return {
        "attachments": [
            {
                "color": color,
                "blocks": blocks,
            }
        ]
    }


def _format_agent_names(agents_used: list) -> str:
    """Extract and deduplicate agent provider names."""
    if not agents_used:
        return "AI agents"

    # Extract provider names (e.g., "anthropic-api_security_reviewer" -> "anthropic")
    providers = set()
    for agent in agents_used:
        name = str(agent).split("_")[0]
        # Clean up common suffixes
        if "-api" in name:
            name = name.replace("-api", "")
        providers.add(name.title())

    return ", ".join(sorted(providers))
