#!/usr/bin/env python3
"""Generate canonical CLI reference docs from the live argparse parser.

Outputs:
- docs/reference/CLI_REFERENCE.md
- docs-site/docs/api/cli.md
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from aragora.cli.parser import build_parser

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "reference" / "CLI_REFERENCE.md"
DOCS_SITE_PATH = REPO_ROOT / "docs-site" / "docs" / "api" / "cli.md"


def _find_top_level_subparsers(parser: argparse.ArgumentParser):
    for action in parser._actions:  # noqa: SLF001 - argparse internals are stable here
        if isinstance(getattr(action, "choices", None), dict):
            return action
    raise RuntimeError("No top-level subparsers found")


def _extract_global_options(parser: argparse.ArgumentParser) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for action in parser._actions:  # noqa: SLF001
        if not action.option_strings:
            continue
        if action.dest == "help":
            continue

        opts = ", ".join(action.option_strings)
        default = "-"
        if isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001
            default = "false"
        elif isinstance(action, argparse._StoreFalseAction):  # noqa: SLF001
            default = "true"
        elif action.default not in (None, argparse.SUPPRESS):
            default = str(action.default)

        help_text = (action.help or "").replace("|", "\\|").strip() or "-"
        rows.append((opts, default, help_text))
    return rows


def _extract_commands(parser: argparse.ArgumentParser) -> tuple[list[dict], int]:
    top = _find_top_level_subparsers(parser)

    parser_to_names: dict[int, list[str]] = defaultdict(list)
    for name, sub in top.choices.items():
        parser_to_names[id(sub)].append(name)

    commands: list[dict] = []
    for choice_action in top._choices_actions:  # noqa: SLF001
        name = choice_action.dest
        sub_parser = top.choices[name]
        aliases = sorted(n for n in parser_to_names[id(sub_parser)] if n != name)

        nested: list[str] = []
        for action in sub_parser._actions:  # noqa: SLF001
            if isinstance(getattr(action, "choices", None), dict):
                nested = sorted(action.choices.keys())
                break

        commands.append(
            {
                "name": name,
                "aliases": aliases,
                "help": (choice_action.help or "").strip(),
                "subcommands": nested,
            }
        )

    commands.sort(key=lambda c: c["name"])
    total_invocations = len(top.choices)
    return commands, total_invocations


def _render_markdown(*, include_frontmatter: bool) -> str:
    parser = build_parser()
    global_options = _extract_global_options(parser)
    commands, total_invocations = _extract_commands(parser)
    canonical_count = len(commands)

    lines: list[str] = []
    if include_frontmatter:
        lines.extend(
            [
                "---",
                "title: Aragora CLI Reference",
                "description: Generated Aragora CLI command catalog from live parser",
                "---",
                "",
            ]
        )

    environment_link = "ENVIRONMENT.md"
    sdk_link = "../SDK_GUIDE.md"
    api_reference_link = "../api/API_REFERENCE.md"
    receipt_link = "../debate/GAUNTLET.md"
    if include_frontmatter:
        environment_link = "../getting-started/environment"
        sdk_link = "../guides/sdk"
        api_reference_link = "./reference"
        receipt_link = "../guides/gauntlet"

    lines.extend(
        [
            "# Aragora CLI Reference",
            "",
            "> Source of truth: generated from `aragora/cli/parser.py` via `python scripts/generate_cli_reference.py`.",
            "",
            "## Scope",
            "",
            "This reference documents the command surface as implemented in code. It includes all top-level commands and known aliases.",
            "",
            f"- Canonical top-level commands: **{canonical_count}**",
            f"- Total top-level invocations (including aliases): **{total_invocations}**",
            "",
            "## Installation",
            "",
            "```bash",
            "pip install aragora",
            "```",
            "",
            "## Global Usage",
            "",
            "```bash",
            "aragora [--version] [--db PATH] [--verbose] <command> [options]",
            "```",
            "",
            "### Global Options",
            "",
            "| Option | Default | Description |",
            "|--------|---------|-------------|",
        ]
    )

    for opt, default, help_text in global_options:
        lines.append(f"| `{opt}` | `{default}` | {help_text} |")

    lines.extend(
        [
            "",
            f"For full runtime configuration, see [ENVIRONMENT]({environment_link}).",
            "",
            "## Command Catalog",
            "",
            "| Command | Aliases | Summary | Subcommands |",
            "|---------|---------|---------|-------------|",
        ]
    )

    for cmd in commands:
        aliases = ", ".join(f"`{a}`" for a in cmd["aliases"]) if cmd["aliases"] else "-"
        summary = (cmd["help"] or "-").replace("|", "\\|")
        subcommands = ", ".join(f"`{s}`" for s in cmd["subcommands"]) if cmd["subcommands"] else "-"
        lines.append(f"| `{cmd['name']}` | {aliases} | {summary} | {subcommands} |")

    lines.extend(
        [
            "",
            "## Core Workflows",
            "",
            "```bash",
            "# Fast onboarding",
            "aragora quickstart --demo",
            "",
            "# Debate",
            'aragora ask "Design a rate limiter" --agents anthropic-api,openai-api --rounds 3',
            "",
            "# Full decision pipeline",
            'aragora decide "Roll out SSO" --auto-approve --budget-limit 10.00',
            "",
            "# Receipt validation",
            "aragora receipt verify receipt.json",
            "aragora verify receipt.json",
            "",
            "# Start API + WebSocket server",
            "aragora serve --api-port 8080 --ws-port 8765",
            "```",
            "",
            "## Notes",
            "",
            "- There is **no** top-level `training` CLI command in the current parser.",
            "- For any command-specific flags, use `aragora <command> --help`.",
            "- For nested commands, use `aragora <command> <subcommand> --help`.",
            "",
            "## See Also",
            "",
            f"- [SDK Guide]({sdk_link})",
            f"- [Receipt and Gauntlet Guidance]({receipt_link})",
            f"- [API Reference]({api_reference_link})",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CLI reference documentation")
    parser.add_argument("--check", action="store_true", help="Check whether docs are up to date")
    parser.add_argument(
        "--check-site",
        action="store_true",
        help="Also check docs-site/docs/api/cli.md (skip by default because sync-docs rewrites it)",
    )
    args = parser.parse_args()

    docs_md = _render_markdown(include_frontmatter=False)
    site_md = _render_markdown(include_frontmatter=True)

    if args.check:
        ok = True
        if DOC_PATH.exists() and DOC_PATH.read_text(encoding="utf-8") != docs_md:
            ok = False
            print(f"Out of date: {DOC_PATH}")
        if (
            args.check_site
            and DOCS_SITE_PATH.exists()
            and DOCS_SITE_PATH.read_text(encoding="utf-8") != site_md
        ):
            ok = False
            print(f"Out of date: {DOCS_SITE_PATH}")
        if not ok:
            print("Run: python scripts/generate_cli_reference.py")
            raise SystemExit(1)
        print("CLI reference files are up to date.")
        return

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_SITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(docs_md, encoding="utf-8")
    DOCS_SITE_PATH.write_text(site_md, encoding="utf-8")
    print(f"Wrote {DOC_PATH}")
    print(f"Wrote {DOCS_SITE_PATH}")


if __name__ == "__main__":
    main()
