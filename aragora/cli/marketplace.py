"""
Marketplace CLI commands.

Provides commands for managing agent templates locally,
including listing, searching, importing, and exporting templates.

Usage:
    aragora marketplace list
    aragora marketplace search "code review"
    aragora marketplace get devil-advocate
    aragora marketplace export devil-advocate -o template.json
    aragora marketplace import template.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click


@click.group()
def marketplace():
    """Manage agent template marketplace."""
    pass


@marketplace.command("list")
@click.option("--category", "-c", help="Filter by category")
@click.option(
    "--type",
    "-t",
    "template_type",
    help="Filter by type (AgentTemplate, DebateTemplate, WorkflowTemplate)",
)
@click.option("--limit", "-l", default=20, help="Max results to show")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def list_templates(
    category: Optional[str], template_type: Optional[str], limit: int, json_output: bool
):
    """List available templates."""
    from aragora.marketplace import TemplateRegistry, TemplateCategory

    registry = TemplateRegistry()

    # Parse category
    cat = None
    if category:
        try:
            cat = TemplateCategory(category)
        except ValueError:
            click.echo(f"Invalid category: {category}", err=True)
            click.echo(f"Valid categories: {[c.value for c in TemplateCategory]}", err=True)
            sys.exit(1)

    templates = registry.search(
        category=cat,
        template_type=template_type,
        limit=limit,
    )

    if json_output:
        click.echo(json.dumps([t.to_dict() for t in templates], indent=2))
        return

    if not templates:
        click.echo("No templates found.")
        return

    click.echo(f"\n{'ID':<25} {'Name':<30} {'Category':<12} {'Type'}")
    click.echo("-" * 85)

    for t in templates:
        template_type_name = type(t).__name__.replace("Template", "")
        click.echo(
            f"{t.metadata.id:<25} "
            f"{t.metadata.name[:28]:<30} "
            f"{t.metadata.category.value:<12} "
            f"{template_type_name}"
        )

    click.echo(f"\nTotal: {len(templates)} templates")


@marketplace.command("search")
@click.argument("query")
@click.option("--category", "-c", help="Filter by category")
@click.option("--tags", "-t", help="Comma-separated tags")
@click.option("--limit", "-l", default=20, help="Max results")
def search_templates(query: str, category: Optional[str], tags: Optional[str], limit: int):
    """Search templates by keyword."""
    from aragora.marketplace import TemplateRegistry, TemplateCategory

    registry = TemplateRegistry()

    cat = None
    if category:
        try:
            cat = TemplateCategory(category)
        except ValueError:
            click.echo(f"Invalid category: {category}", err=True)
            sys.exit(1)

    tag_list = tags.split(",") if tags else None

    templates = registry.search(
        query=query,
        category=cat,
        tags=tag_list,
        limit=limit,
    )

    if not templates:
        click.echo(f"No templates found for '{query}'")
        return

    click.echo(f"\nResults for '{query}':\n")

    for t in templates:
        stars = "★" * min(t.metadata.stars, 5) if t.metadata.stars else ""
        click.echo(f"  {t.metadata.id}")
        click.echo(f"    {t.metadata.name} - {t.metadata.description[:60]}...")
        click.echo(
            f"    Category: {t.metadata.category.value} | Downloads: {t.metadata.downloads} {stars}"
        )
        click.echo()


@marketplace.command("get")
@click.argument("template_id")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def get_template(template_id: str, json_output: bool):
    """Get details of a specific template."""
    from aragora.marketplace import TemplateRegistry, AgentTemplate, DebateTemplate

    registry = TemplateRegistry()
    template = registry.get(template_id)

    if template is None:
        click.echo(f"Template not found: {template_id}", err=True)
        sys.exit(1)

    if json_output:
        click.echo(json.dumps(template.to_dict(), indent=2))
        return

    m = template.metadata
    click.echo(f"\n{'='*60}")
    click.echo(f"  {m.name}")
    click.echo(f"{'='*60}")
    click.echo(f"\n  ID:          {m.id}")
    click.echo(f"  Version:     {m.version}")
    click.echo(f"  Author:      {m.author}")
    click.echo(f"  Category:    {m.category.value}")
    click.echo(f"  Tags:        {', '.join(m.tags) if m.tags else 'none'}")
    click.echo(f"  Downloads:   {m.downloads}")
    click.echo(f"  Stars:       {m.stars}")
    click.echo("\n  Description:")
    click.echo(f"    {m.description}")

    if isinstance(template, AgentTemplate):
        click.echo(f"\n  Agent Type:  {template.agent_type}")
        click.echo(f"  Capabilities: {', '.join(template.capabilities)}")
        click.echo(f"  Constraints:  {', '.join(template.constraints)}")
        click.echo("\n  System Prompt:")
        for line in template.system_prompt.split("\n")[:10]:
            click.echo(f"    {line}")
        if template.system_prompt.count("\n") > 10:
            click.echo("    ...")

    elif isinstance(template, DebateTemplate):
        click.echo(f"\n  Task Template: {template.task_template}")
        click.echo(f"  Roles: {len(template.agent_roles)}")
        click.echo(f"  Protocol: {json.dumps(template.protocol, indent=4)}")

    click.echo()


@marketplace.command("export")
@click.argument("template_id")
@click.option("--output", "-o", help="Output file path")
def export_template(template_id: str, output: Optional[str]):
    """Export a template to JSON file."""
    from aragora.marketplace import TemplateRegistry

    registry = TemplateRegistry()
    json_str = registry.export_template(template_id)

    if json_str is None:
        click.echo(f"Template not found: {template_id}", err=True)
        sys.exit(1)

    if output:
        Path(output).write_text(json_str)
        click.echo(f"Exported to {output}")
    else:
        click.echo(json_str)


@marketplace.command("import")
@click.argument("file_path", type=click.Path(exists=True))
def import_template(file_path: str):
    """Import a template from JSON file."""
    from aragora.marketplace import TemplateRegistry

    registry = TemplateRegistry()

    try:
        json_str = Path(file_path).read_text()
        template_id = registry.import_template(json_str)
        click.echo(f"Imported template: {template_id}")
    except json.JSONDecodeError as e:
        click.echo(f"Invalid JSON: {e}", err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Invalid template: {e}", err=True)
        sys.exit(1)


@marketplace.command("categories")
def list_categories():
    """List all template categories."""
    from aragora.marketplace import TemplateRegistry

    registry = TemplateRegistry()
    categories = registry.list_categories()

    click.echo("\nCategories:")
    click.echo("-" * 30)

    for cat in categories:
        click.echo(f"  {cat['category']:<15} ({cat['count']} templates)")


@marketplace.command("rate")
@click.argument("template_id")
@click.argument("score", type=click.IntRange(1, 5))
@click.option("--review", "-r", help="Review text")
def rate_template(template_id: str, score: int, review: Optional[str]):
    """Rate a template (1-5 stars)."""
    from aragora.marketplace import TemplateRegistry, TemplateRating
    import os

    registry = TemplateRegistry()

    # Check template exists
    if registry.get(template_id) is None:
        click.echo(f"Template not found: {template_id}", err=True)
        sys.exit(1)

    user_id = os.environ.get("USER", "anonymous")

    rating = TemplateRating(
        user_id=user_id,
        template_id=template_id,
        score=score,
        review=review,
    )
    registry.rate(rating)

    avg = registry.get_average_rating(template_id)
    click.echo(f"Rated {template_id}: {'★' * score} ({score}/5)")
    click.echo(f"Average rating: {avg:.1f}")


@marketplace.command("use")
@click.argument("template_id")
@click.argument("task")
@click.option("--rounds", "-r", default=3, help="Number of debate rounds")
def use_template(template_id: str, task: str, rounds: int):
    """Use a debate template to start a debate."""
    from aragora.marketplace import TemplateRegistry, DebateTemplate

    registry = TemplateRegistry()
    template = registry.get(template_id)

    if template is None:
        click.echo(f"Template not found: {template_id}", err=True)
        sys.exit(1)

    if not isinstance(template, DebateTemplate):
        click.echo(f"Template {template_id} is not a debate template", err=True)
        sys.exit(1)

    # Format the task using the template
    formatted_task = template.task_template.format(
        topic=task,
        motion=task,
        problem=task,
        code=task,
    )

    click.echo(f"\nStarting debate with template: {template.metadata.name}")
    click.echo(f"Task: {formatted_task}")
    click.echo(f"Roles: {len(template.agent_roles)}")
    click.echo(f"Rounds: {template.protocol.get('rounds', rounds)}")
    click.echo("\nTo run this debate, use:")
    click.echo(f'  aragora debate run "{formatted_task}" --rounds {rounds}')


if __name__ == "__main__":
    marketplace()
