"""
Template CLI commands for managing workflow templates.

Provides commands to list, show, run, and validate workflow templates.

Usage:
    aragora template list [--category] [--tags]
    aragora template show <template-id>
    aragora template run <template-id> [--input]
    aragora template validate <path>
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def create_template_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create the template subcommand parser."""
    parser = subparsers.add_parser(
        "template",
        help="Manage workflow templates",
        description="List, show, run, and validate workflow templates.",
    )

    template_subparsers = parser.add_subparsers(dest="template_command", help="Template commands")

    # List command
    list_parser = template_subparsers.add_parser("list", help="List available templates")
    list_parser.add_argument(
        "--category",
        "-c",
        help="Filter by category (legal, healthcare, code, etc.)",
    )
    list_parser.add_argument(
        "--tags",
        "-t",
        nargs="+",
        help="Filter by tags (space-separated)",
    )
    list_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )
    list_parser.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include deprecated templates",
    )
    list_parser.set_defaults(func=cmd_template_list)

    # Show command
    show_parser = template_subparsers.add_parser("show", help="Show template details")
    show_parser.add_argument("template_id", help="Template ID to show")
    show_parser.add_argument(
        "--version",
        "-v",
        help="Specific version (default: latest)",
    )
    show_parser.add_argument(
        "--format",
        "-f",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format (default: text)",
    )
    show_parser.set_defaults(func=cmd_template_show)

    # Run command
    run_parser = template_subparsers.add_parser("run", help="Run a template")
    run_parser.add_argument("template_id", help="Template ID to run")
    run_parser.add_argument(
        "--input",
        "-i",
        help="Input data (JSON string or @filepath)",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        help="Output file path",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show execution plan without running",
    )
    run_parser.set_defaults(func=cmd_template_run)

    # Validate command
    validate_parser = template_subparsers.add_parser("validate", help="Validate a template file")
    validate_parser.add_argument("path", help="Path to template file (JSON or YAML)")
    validate_parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation",
    )
    validate_parser.set_defaults(func=cmd_template_validate)

    # Package command
    package_parser = template_subparsers.add_parser("package", help="Package a template")
    package_parser.add_argument("path", help="Path to template file")
    package_parser.add_argument(
        "--output",
        "-o",
        help="Output package file path",
    )
    package_parser.add_argument(
        "--version",
        "-v",
        default="1.0.0",
        help="Package version (default: 1.0.0)",
    )
    package_parser.add_argument(
        "--author",
        "-a",
        help="Package author",
    )
    package_parser.set_defaults(func=cmd_template_package)

    parser.set_defaults(func=lambda args: parser.print_help())


def cmd_template_list(args: argparse.Namespace) -> int:
    """List available templates."""
    try:
        from aragora.workflow.templates import WORKFLOW_TEMPLATES
        from aragora.workflow.templates.package import (
            list_packages,
            TemplateCategory,
            package_all_templates,
        )

        # Try package registry first, fall back to raw templates
        packages = list_packages(
            category=args.category,
            tags=args.tags,
            include_deprecated=args.include_deprecated,
        )

        # If no packages registered, create them from templates
        if not packages:
            all_packages = package_all_templates()
            packages = list(all_packages.values())

            # Apply filters
            if args.category:
                try:
                    cat = TemplateCategory(args.category)
                    packages = [p for p in packages if p.metadata.category == cat]
                except ValueError:
                    pass

            if args.tags:
                tag_set = set(args.tags)
                packages = [
                    p for p in packages
                    if any(t in p.metadata.tags for t in tag_set)
                ]

        if args.format == "json":
            output = [
                {
                    "id": p.metadata.id,
                    "name": p.metadata.name,
                    "version": p.metadata.version,
                    "category": p.metadata.category.value,
                    "status": p.metadata.status.value,
                    "description": p.metadata.description,
                    "tags": p.metadata.tags,
                }
                for p in packages
            ]
            print(json.dumps(output, indent=2))
            return 0

        # Table format
        print("\n" + "=" * 70)
        print(" WORKFLOW TEMPLATES")
        print("=" * 70 + "\n")

        if not packages:
            print("No templates found matching criteria.")
            return 0

        # Group by category
        categories: dict[str, list] = {}
        for p in packages:
            cat = p.metadata.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(p)

        for cat_name, cat_packages in sorted(categories.items()):
            print(f"\n[{cat_name.upper()}]")
            print("-" * 40)
            for p in cat_packages:
                status = ""
                if p.is_deprecated:
                    status = " [DEPRECATED]"
                print(f"  {p.metadata.id} (v{p.metadata.version}){status}")
                print(f"    {p.metadata.description[:60]}")
                if p.metadata.tags:
                    print(f"    Tags: {', '.join(p.metadata.tags[:5])}")

        print(f"\nTotal: {len(packages)} templates")
        print("\nUse 'aragora template show <id>' for details.\n")
        return 0

    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_template_show(args: argparse.Namespace) -> int:
    """Show template details."""
    try:
        from aragora.workflow.templates import WORKFLOW_TEMPLATES
        from aragora.workflow.templates.package import get_package, create_package

        # Try to get from package registry
        package = get_package(args.template_id, version=args.version)

        # Fall back to raw template
        if package is None:
            template = WORKFLOW_TEMPLATES.get(args.template_id)
            if template is None:
                # Try with category prefix
                for key, t in WORKFLOW_TEMPLATES.items():
                    if key.endswith(f"/{args.template_id}") or t.get("name") == args.template_id:
                        template = t
                        break

            if template is None:
                print(f"Template not found: {args.template_id}", file=sys.stderr)
                return 1

            package = create_package(template, version="1.0.0")

        if args.format == "json":
            print(json.dumps(package.to_dict(), indent=2))
            return 0

        if args.format == "yaml":
            try:
                import yaml
                print(yaml.safe_dump(package.to_dict(), default_flow_style=False))
            except ImportError:
                print("YAML output requires PyYAML: pip install pyyaml", file=sys.stderr)
                return 1
            return 0

        # Text format
        m = package.metadata
        print("\n" + "=" * 60)
        print(f" {m.name} (v{m.version})")
        print("=" * 60)
        print(f"\nID:          {m.id}")
        print(f"Category:    {m.category.value}")
        print(f"Status:      {m.status.value}")
        print(f"Description: {m.description}")

        if m.tags:
            print(f"Tags:        {', '.join(m.tags)}")

        if m.author:
            print(f"Author:      {m.author.name}")
            if m.author.organization:
                print(f"             ({m.author.organization})")

        if m.dependencies:
            print("\nDependencies:")
            for dep in m.dependencies:
                req = "required" if dep.required else "optional"
                print(f"  - {dep.name} ({dep.type}, {req})")

        if m.estimated_duration:
            print(f"\nEstimated Duration: {m.estimated_duration}")

        if m.complexity:
            print(f"Complexity:         {m.complexity}")

        if m.recommended_agents:
            print(f"Recommended Agents: {', '.join(m.recommended_agents)}")

        if package.readme:
            print("\n" + "-" * 40)
            print("README:")
            print("-" * 40)
            print(package.readme[:500])
            if len(package.readme) > 500:
                print("... [truncated]")

        if m.successor and m.status.value == "deprecated":
            print(f"\n[!] This template is deprecated. Migrate to: {m.successor}")

        print()
        return 0

    except Exception as e:
        logger.error(f"Failed to show template: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_template_run(args: argparse.Namespace) -> int:
    """Run a template."""
    try:
        from aragora.workflow.templates import WORKFLOW_TEMPLATES
        from aragora.workflow.engine import WorkflowEngine

        # Find template
        template = WORKFLOW_TEMPLATES.get(args.template_id)
        if template is None:
            for key, t in WORKFLOW_TEMPLATES.items():
                if key.endswith(f"/{args.template_id}"):
                    template = t
                    break

        if template is None:
            print(f"Template not found: {args.template_id}", file=sys.stderr)
            return 1

        # Parse input
        input_data = {}
        if args.input:
            if args.input.startswith("@"):
                # Load from file
                input_path = Path(args.input[1:])
                if not input_path.exists():
                    print(f"Input file not found: {input_path}", file=sys.stderr)
                    return 1
                input_data = json.loads(input_path.read_text())
            else:
                input_data = json.loads(args.input)

        if args.dry_run:
            print("\n" + "=" * 60)
            print(" DRY RUN - Execution Plan")
            print("=" * 60)
            print(f"\nTemplate: {args.template_id}")
            print(f"Input: {json.dumps(input_data, indent=2)}")

            if "steps" in template:
                print("\nSteps:")
                for i, step in enumerate(template["steps"], 1):
                    step_name = step.get("name", f"Step {i}")
                    step_type = step.get("type", "unknown")
                    print(f"  {i}. {step_name} ({step_type})")

            print("\n[Dry run complete - no execution performed]")
            return 0

        # Run template
        print(f"\nRunning template: {args.template_id}...")

        async def run():
            engine = WorkflowEngine()
            result = await engine.execute(template, input_data)
            return result

        result = asyncio.run(run())

        # Output result
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(result, indent=2, default=str))
            print(f"Result saved to: {output_path}")
        else:
            print("\n" + "=" * 60)
            print(" RESULT")
            print("=" * 60)
            print(json.dumps(result, indent=2, default=str))

        return 0

    except Exception as e:
        logger.error(f"Failed to run template: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_template_validate(args: argparse.Namespace) -> int:
    """Validate a template file."""
    try:
        path = Path(args.path)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

        # Load template
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                template = yaml.safe_load(content)
            except ImportError:
                print("YAML validation requires PyYAML: pip install pyyaml", file=sys.stderr)
                return 1
        else:
            template = json.loads(content)

        errors = []
        warnings = []

        # Required fields
        required = ["name", "steps"]
        for field in required:
            if field not in template:
                errors.append(f"Missing required field: {field}")

        # Validate steps
        if "steps" in template:
            steps = template["steps"]
            if not isinstance(steps, list):
                errors.append("'steps' must be a list")
            else:
                for i, step in enumerate(steps):
                    if not isinstance(step, dict):
                        errors.append(f"Step {i+1}: must be an object")
                        continue
                    if "type" not in step:
                        errors.append(f"Step {i+1}: missing 'type' field")
                    if "name" not in step:
                        warnings.append(f"Step {i+1}: missing 'name' field (recommended)")

        # Strict mode checks
        if args.strict:
            if "description" not in template:
                warnings.append("Missing 'description' field")
            if "version" not in template:
                warnings.append("Missing 'version' field")
            if "category" not in template:
                warnings.append("Missing 'category' field")

        # Output results
        print("\n" + "=" * 60)
        print(f" VALIDATION: {path.name}")
        print("=" * 60)

        if errors:
            print(f"\nErrors ({len(errors)}):")
            for e in errors:
                print(f"  [X] {e}")

        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for w in warnings:
                print(f"  [!] {w}")

        if not errors and not warnings:
            print("\n  [OK] Template is valid")

        print()
        return 1 if errors else 0

    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_template_package(args: argparse.Namespace) -> int:
    """Package a template file."""
    try:
        from aragora.workflow.templates.package import create_package, TemplateAuthor

        path = Path(args.path)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

        # Load template
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                template = yaml.safe_load(content)
            except ImportError:
                print("YAML requires PyYAML: pip install pyyaml", file=sys.stderr)
                return 1
        else:
            template = json.loads(content)

        # Create package
        author = TemplateAuthor(name=args.author) if args.author else None
        package = create_package(
            template=template,
            version=args.version,
            author=author,
        )

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = path.with_suffix(".pkg.json")

        package.save(output_path)
        print(f"Package created: {output_path}")
        print(f"  ID: {package.metadata.id}")
        print(f"  Version: {package.metadata.version}")
        print(f"  Checksum: {package.checksum}")

        return 0

    except Exception as e:
        logger.error(f"Packaging failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main(args: argparse.Namespace) -> int:
    """Main entry point for template CLI."""
    if not hasattr(args, "template_command") or args.template_command is None:
        # Show help if no subcommand
        print("Usage: aragora template <command>")
        print("\nCommands:")
        print("  list      List available templates")
        print("  show      Show template details")
        print("  run       Run a template")
        print("  validate  Validate a template file")
        print("  package   Package a template for distribution")
        print("\nUse 'aragora template <command> --help' for more information.")
        return 0

    return args.func(args)


__all__ = [
    "create_template_parser",
    "cmd_template_list",
    "cmd_template_show",
    "cmd_template_run",
    "cmd_template_validate",
    "cmd_template_package",
    "main",
]
