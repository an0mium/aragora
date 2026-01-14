"""
aragora publish - Generate shareable stress-test reports.

Creates beautiful, interactive HTML/Markdown reports from debate traces
that can be shared, embedded, or published as decision receipts.

Usage:
    aragora publish <debate-id> --format html --output ./reports/
    aragora publish latest --format md  # Most recent debate
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from aragora.core import DebateResult
from aragora.debate.traces import DebateTrace


def generate_html_report(result: DebateResult, trace: Optional[DebateTrace] = None) -> str:
    """Generate an interactive HTML report from a debate result."""

    # Build message timeline
    messages_html = ""
    for msg in result.messages:
        role_class = msg.role.replace(" ", "-")
        messages_html += f"""
        <div class="message {role_class}">
            <div class="message-header">
                <span class="agent">{msg.agent}</span>
                <span class="role">{msg.role}</span>
                <span class="round">Round {msg.round}</span>
            </div>
            <div class="message-content">{msg.content[:500]}{"..." if len(msg.content) > 500 else ""}</div>
        </div>
        """

    # Build critiques section
    critiques_html = ""
    for critique in result.critiques:
        issues = "".join(f"<li>{i}</li>" for i in critique.issues[:3])
        critiques_html += f"""
        <div class="critique">
            <div class="critique-header">
                <span class="critic">{critique.agent}</span>
                <span class="target">→ {critique.target_agent}</span>
                <span class="severity" style="--severity: {critique.severity}">
                    Severity: {critique.severity:.1f}
                </span>
            </div>
            <ul class="issues">{issues}</ul>
        </div>
        """

    consensus_class = "reached" if result.consensus_reached else "not-reached"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>aragora Stress-Test: {result.task[:50]}...</title>
    <style>
        :root {{
            --primary: #6366f1;
            --secondary: #8b5cf6;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #0f172a;
            --card: #1e293b;
            --text: #e2e8f0;
            --muted: #94a3b8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            border-radius: 1rem;
        }}
        h1 {{ font-size: 2rem; margin-bottom: 0.5rem; }}
        .meta {{ color: rgba(255,255,255,0.8); font-size: 0.9rem; }}
        .task {{
            background: var(--card);
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
            border-left: 4px solid var(--primary);
        }}
        .task h2 {{ font-size: 1rem; color: var(--muted); margin-bottom: 0.5rem; }}
        .section {{ margin-bottom: 2rem; }}
        .section-title {{
            font-size: 1.25rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .message {{
            background: var(--card);
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.75rem;
        }}
        .message-header {{
            display: flex;
            gap: 1rem;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }}
        .agent {{ color: var(--primary); font-weight: 600; }}
        .role {{ color: var(--muted); }}
        .round {{ color: var(--secondary); }}
        .message-content {{
            font-size: 0.95rem;
            white-space: pre-wrap;
        }}
        .message.proposer {{ border-left: 3px solid var(--primary); }}
        .message.critic {{ border-left: 3px solid var(--warning); }}
        .message.synthesizer {{ border-left: 3px solid var(--success); }}
        .critique {{
            background: var(--card);
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.75rem;
            border-left: 3px solid var(--warning);
        }}
        .critique-header {{
            display: flex;
            gap: 1rem;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }}
        .critic {{ color: var(--warning); font-weight: 600; }}
        .target {{ color: var(--muted); }}
        .severity {{
            color: hsl(calc(120 - var(--severity) * 120), 70%, 50%);
        }}
        .issues {{ margin-left: 1.5rem; font-size: 0.9rem; }}
        .final-answer {{
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(99, 102, 241, 0.1));
            border: 1px solid var(--success);
            padding: 1.5rem;
            border-radius: 0.5rem;
            white-space: pre-wrap;
        }}
        .consensus {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-weight: 600;
        }}
        .consensus.reached {{ background: rgba(34, 197, 94, 0.2); color: var(--success); }}
        .consensus.not-reached {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 2rem;
        }}
        .stat {{
            background: var(--card);
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        }}
        .stat-value {{ font-size: 1.5rem; font-weight: 700; color: var(--primary); }}
        .stat-label {{ font-size: 0.8rem; color: var(--muted); }}
        footer {{
            text-align: center;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid var(--card);
            color: var(--muted);
            font-size: 0.85rem;
        }}
        footer a {{ color: var(--primary); text-decoration: none; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏛️ aragora Stress-Test</h1>
            <div class="meta">Adversarial Validation • {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
        </header>

        <div class="task">
            <h2>TASK</h2>
            <p>{result.task}</p>
        </div>

        <div class="section">
            <h2 class="section-title">💬 Debate Timeline</h2>
            {messages_html}
        </div>

        <div class="section">
            <h2 class="section-title">🔍 Critiques</h2>
            {critiques_html if critiques_html else '<p style="color: var(--muted)">No critiques recorded.</p>'}
        </div>

        <div class="section">
            <h2 class="section-title">
                ✨ Final Answer
                <span class="consensus {consensus_class}">
                    {"✓ Consensus" if result.consensus_reached else "⚠ No Consensus"}
                    ({result.confidence:.0%})
                </span>
            </h2>
            <div class="final-answer">{result.final_answer}</div>
        </div>

        <div class="stats">
            <div class="stat">
                <div class="stat-value">{result.rounds_used}</div>
                <div class="stat-label">Rounds</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(result.messages)}</div>
                <div class="stat-label">Messages</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(result.critiques)}</div>
                <div class="stat-label">Critiques</div>
            </div>
            <div class="stat">
                <div class="stat-value">{result.duration_seconds:.0f}s</div>
                <div class="stat-label">Duration</div>
            </div>
        </div>

        <footer>
            Generated by <a href="https://github.com/an0mium/aragora">aragora</a> v0.8.0 •
            AI Red Team for Decision Stress-Testing
        </footer>
    </div>
</body>
</html>
"""
    return html


def generate_markdown_report(result: DebateResult, trace: Optional[DebateTrace] = None) -> str:
    """Generate a Markdown report from a debate result."""

    messages_md = ""
    for msg in result.messages:
        messages_md += f"\n### {msg.agent} ({msg.role}) - Round {msg.round}\n\n"
        messages_md += f"{msg.content[:800]}{'...' if len(msg.content) > 800 else ''}\n"

    critiques_md = ""
    for critique in result.critiques:
        critiques_md += f"\n**{critique.agent} → {critique.target_agent}** (severity: {critique.severity:.1f})\n\n"
        for issue in critique.issues[:3]:
            critiques_md += f"- {issue}\n"

    consensus = "✓ Consensus Reached" if result.consensus_reached else "⚠ No Consensus"

    md = f"""# 🏛️ aragora Debate Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

---

## Task

{result.task}

---

## Debate Timeline

{messages_md}

---

## Critiques

{critiques_md if critiques_md else "*No critiques recorded.*"}

---

## Final Answer

**{consensus}** ({result.confidence:.0%} confidence)

{result.final_answer}

---

## Statistics

| Metric | Value |
|--------|-------|
| Rounds | {result.rounds_used} |
| Messages | {len(result.messages)} |
| Critiques | {len(result.critiques)} |
| Duration | {result.duration_seconds:.0f}s |

---

*Generated by [aragora](https://github.com/an0mium/aragora) v0.8.0 - AI Red Team for Decision Stress-Testing*
"""
    return md


def generate_json_report(result: DebateResult, trace: Optional[DebateTrace] = None) -> str:
    """Generate a JSON report from a debate result."""
    data = {
        "id": result.id,
        "task": result.task,
        "final_answer": result.final_answer,
        "consensus_reached": result.consensus_reached,
        "confidence": result.confidence,
        "rounds_used": result.rounds_used,
        "duration_seconds": result.duration_seconds,
        "messages": [
            {
                "agent": m.agent,
                "role": m.role,
                "content": m.content,
                "round": m.round,
            }
            for m in result.messages
        ],
        "critiques": [
            {
                "agent": c.agent,
                "target": c.target_agent,
                "issues": c.issues,
                "suggestions": c.suggestions,
                "severity": c.severity,
            }
            for c in result.critiques
        ],
        "generated_at": datetime.now().isoformat(),
        "generator": "aragora v0.8.0",
    }
    return json.dumps(data, indent=2)


def publish_debate(
    result: DebateResult,
    output_dir: str = ".",
    format: str = "html",
    trace: Optional[DebateTrace] = None,
) -> Path:
    """
    Publish a debate result to a file.

    Args:
        result: The debate result to publish
        output_dir: Directory to write the report
        format: Output format (html, md, json)
        trace: Optional trace for additional details

    Returns:
        Path to the generated file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debate_id = result.id[:8] if result.id else timestamp

    if format == "html":
        content = generate_html_report(result, trace)
        filename = f"debate_{debate_id}.html"
    elif format == "md":
        content = generate_markdown_report(result, trace)
        filename = f"debate_{debate_id}.md"
    elif format == "json":
        content = generate_json_report(result, trace)
        filename = f"debate_{debate_id}.json"
    else:
        raise ValueError(f"Unknown format: {format}")

    filepath = output_path / filename
    filepath.write_text(content)

    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Generate shareable debate reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    aragora publish --format html --output ./reports/
    aragora publish --format md
    aragora publish --format json --output ./api/
        """,
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["html", "md", "json"],
        default="html",
        help="Output format (default: html)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=".",
        help="Output directory (default: current)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a demo report",
    )

    args = parser.parse_args()

    if args.demo:
        # Create a demo debate result
        from aragora.core import Critique, Message

        demo_result = DebateResult(
            task="What is the best programming language for AI development?",
            final_answer="Python remains the best choice for AI development due to its rich ecosystem (PyTorch, TensorFlow, JAX), ease of prototyping, and strong community support. However, Rust is emerging for production systems requiring performance, and Julia for scientific computing.",
            confidence=0.85,
            consensus_reached=True,
            rounds_used=2,
            duration_seconds=45.3,
            messages=[
                Message(
                    role="proposer",
                    agent="gemini",
                    content="Python is the clear winner...",
                    round=0,
                ),
                Message(
                    role="proposer", agent="codex", content="We should consider Rust...", round=0
                ),
                Message(
                    role="critic", agent="claude", content="Both have valid points...", round=1
                ),
            ],
            critiques=[
                Critique(
                    agent="claude",
                    target_agent="gemini",
                    target_content="Python proposal",
                    issues=["Ignores performance concerns", "Doesn't consider GIL limitations"],
                    suggestions=["Address when Rust/C++ bindings are needed"],
                    severity=0.4,
                    reasoning="Valid but incomplete",
                ),
            ],
        )

        filepath = publish_debate(demo_result, args.output, args.format)
        print(f"Demo report generated: {filepath}")
    else:
        print("No debate to publish. Use --demo for a demo report.")
        print("In a full implementation, this would load from debate history.")


if __name__ == "__main__":
    main()
