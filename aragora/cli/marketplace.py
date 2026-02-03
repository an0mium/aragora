"""
Marketplace CLI commands.

Provides commands for managing agent templates via server API,
with local fallback for offline use.

Usage:
    aragora marketplace list
    aragora marketplace search "code review"
    aragora marketplace get devil-advocate
    aragora marketplace export devil-advocate -o template.json
    aragora marketplace import template.json
    aragora marketplace list --local  # Use local registry instead of API
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click


def _get_api_client():
    """Get API client if available and server is reachable."""
    try:
        from aragora.client import AragoraClient

        client = AragoraClient(
            base_url=os.environ.get("ARAGORA_API_URL", "http://localhost:8080"),
            api_key=os.environ.get("ARAGORA_API_KEY"),
        )
        # Quick health check
        client.get("/api/health")
        return client
    except Exception:
        return None


def _use_local_registry(local_flag: bool) -> bool:
    """Determine if we should use local registry."""
    if local_flag:
        return True
    # If ARAGORA_MARKETPLACE_LOCAL is set, use local
    if os.environ.get("ARAGORA_MARKETPLACE_LOCAL", "").lower() in ("1", "true", "yes"):
        return True
    return False


@click.group()
@click.option("--local", is_flag=True, help="Use local registry instead of server API")
@click.pass_context
def marketplace(ctx, local: bool):
    """Manage agent template marketplace."""
    ctx.ensure_object(dict)
    ctx.obj["use_local"] = _use_local_registry(local)

    if not ctx.obj["use_local"]:
        client = _get_api_client()
        if client is None:
            click.echo("Warning: Server unavailable, falling back to local registry", err=True)
            ctx.obj["use_local"] = True
        else:
            ctx.obj["client"] = client


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
@click.pass_context
def list_templates(
    ctx, category: str | None, template_type: str | None, limit: int, json_output: bool
):
    """List available templates."""
    if ctx.obj.get("use_local"):
        _list_templates_local(category, template_type, limit, json_output)
    else:
        _list_templates_api(ctx.obj["client"], category, template_type, limit, json_output)


def _list_templates_api(
    client, category: str | None, template_type: str | None, limit: int, json_output: bool
):
    """List templates via API."""
    params: dict[str, Any] = {"limit": limit}
    if category:
        params["category"] = category
    if template_type:
        params["type"] = template_type

    try:
        response = client.get("/api/v1/marketplace/templates", params=params)
        templates = response.get("templates", [])
    except Exception as e:
        click.echo(f"API error: {e}", err=True)
        sys.exit(1)

    if json_output:
        click.echo(json.dumps(templates, indent=2))
        return

    if not templates:
        click.echo("No templates found.")
        return

    click.echo(f"\n{'ID':<25} {'Name':<30} {'Category':<12} {'Type'}")
    click.echo("-" * 85)

    for t in templates:
        click.echo(
            f"{t.get('id', ''):<25} "
            f"{t.get('name', '')[:28]:<30} "
            f"{t.get('category', ''):<12} "
            f"{t.get('type', '')}"
        )

    click.echo(f"\nTotal: {len(templates)} templates")


def _list_templates_local(
    category: str | None, template_type: str | None, limit: int, json_output: bool
):
    """List templates from local registry."""
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
@click.pass_context
def search_templates(ctx, query: str, category: str | None, tags: str | None, limit: int):
    """Search templates by keyword."""
    if ctx.obj.get("use_local"):
        _search_templates_local(query, category, tags, limit)
    else:
        _search_templates_api(ctx.obj["client"], query, category, tags, limit)


def _search_templates_api(client, query: str, category: str | None, tags: str | None, limit: int):
    """Search templates via API."""
    params: dict[str, Any] = {"query": query, "limit": limit}
    if category:
        params["category"] = category
    if tags:
        params["tags"] = tags

    try:
        response = client.get("/api/v1/marketplace/search", params=params)
        templates = response.get("templates", [])
    except Exception as e:
        click.echo(f"API error: {e}", err=True)
        sys.exit(1)

    if not templates:
        click.echo(f"No templates found for '{query}'")
        return

    click.echo(f"\nResults for '{query}':\n")

    for t in templates:
        stars = "★" * min(t.get("stars", 0), 5) if t.get("stars") else ""
        click.echo(f"  {t.get('id', '')}")
        desc = t.get("description", "")[:60]
        click.echo(f"    {t.get('name', '')} - {desc}...")
        click.echo(
            f"    Category: {t.get('category', '')} | Downloads: {t.get('downloads', 0)} {stars}"
        )
        click.echo()


def _search_templates_local(query: str, category: str | None, tags: str | None, limit: int):
    """Search templates from local registry."""
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
@click.pass_context
def get_template(ctx, template_id: str, json_output: bool):
    """Get details of a specific template."""
    if ctx.obj.get("use_local"):
        _get_template_local(template_id, json_output)
    else:
        _get_template_api(ctx.obj["client"], template_id, json_output)


def _get_template_api(client, template_id: str, json_output: bool):
    """Get template details via API."""
    try:
        template = client.get(f"/api/v1/marketplace/templates/{template_id}")
    except Exception as e:
        click.echo(f"API error: {e}", err=True)
        sys.exit(1)

    if template is None:
        click.echo(f"Template not found: {template_id}", err=True)
        sys.exit(1)

    if json_output:
        click.echo(json.dumps(template, indent=2))
        return

    click.echo(f"\n{'=' * 60}")
    click.echo(f"  {template.get('name', 'Unknown')}")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n  ID:          {template.get('id', '')}")
    click.echo(f"  Version:     {template.get('version', '')}")
    click.echo(f"  Author:      {template.get('author', '')}")
    click.echo(f"  Category:    {template.get('category', '')}")
    click.echo(f"  Tags:        {', '.join(template.get('tags', [])) or 'none'}")
    click.echo(f"  Downloads:   {template.get('downloads', 0)}")
    click.echo(f"  Stars:       {template.get('stars', 0)}")
    click.echo("\n  Description:")
    click.echo(f"    {template.get('description', '')}")

    template_type = template.get("type", "")
    if template_type == "AgentTemplate":
        click.echo(f"\n  Agent Type:  {template.get('agent_type', '')}")
        click.echo(f"  Capabilities: {', '.join(template.get('capabilities', []))}")
        click.echo(f"  Constraints:  {', '.join(template.get('constraints', []))}")
        prompt = template.get("system_prompt", "")
        if prompt:
            click.echo("\n  System Prompt:")
            for line in prompt.split("\n")[:10]:
                click.echo(f"    {line}")
            if prompt.count("\n") > 10:
                click.echo("    ...")

    elif template_type == "DebateTemplate":
        click.echo(f"\n  Task Template: {template.get('task_template', '')}")
        click.echo(f"  Roles: {len(template.get('agent_roles', []))}")
        click.echo(f"  Protocol: {json.dumps(template.get('protocol', {}), indent=4)}")

    click.echo()


def _get_template_local(template_id: str, json_output: bool):
    """Get template details from local registry."""
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
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  {m.name}")
    click.echo(f"{'=' * 60}")
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
@click.pass_context
def export_template(ctx, template_id: str, output: str | None):
    """Export a template to JSON file."""
    if ctx.obj.get("use_local"):
        _export_template_local(template_id, output)
    else:
        _export_template_api(ctx.obj["client"], template_id, output)


def _export_template_api(client, template_id: str, output: str | None):
    """Export template via API."""
    try:
        response = client.get(f"/api/v1/marketplace/templates/{template_id}/export")
        json_str = json.dumps(response, indent=2)
    except Exception as e:
        click.echo(f"API error: {e}", err=True)
        sys.exit(1)

    if output:
        Path(output).write_text(json_str)
        click.echo(f"Exported to {output}")
    else:
        click.echo(json_str)


def _export_template_local(template_id: str, output: str | None):
    """Export template from local registry."""
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
@click.pass_context
def import_template(ctx, file_path: str):
    """Import a template from JSON file."""
    if ctx.obj.get("use_local"):
        _import_template_local(file_path)
    else:
        _import_template_api(ctx.obj["client"], file_path)


def _import_template_api(client, file_path: str):
    """Import template via API."""
    try:
        json_str = Path(file_path).read_text()
        template_data = json.loads(json_str)
        response = client.post("/api/v1/marketplace/templates", json=template_data)
        template_id = response.get("id", response.get("template_id", "unknown"))
        click.echo(f"Imported template: {template_id}")
    except json.JSONDecodeError as e:
        click.echo(f"Invalid JSON: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"API error: {e}", err=True)
        sys.exit(1)


def _import_template_local(file_path: str):
    """Import template to local registry."""
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
@click.pass_context
def list_categories(ctx):
    """List all template categories."""
    if ctx.obj.get("use_local"):
        _list_categories_local()
    else:
        _list_categories_api(ctx.obj["client"])


def _list_categories_api(client):
    """List categories via API."""
    try:
        response = client.get("/api/v1/marketplace/categories")
        categories = response.get("categories", [])
    except Exception as e:
        click.echo(f"API error: {e}", err=True)
        sys.exit(1)

    click.echo("\nCategories:")
    click.echo("-" * 30)

    for cat in categories:
        click.echo(f"  {cat.get('category', ''):<15} ({cat.get('count', 0)} templates)")


def _list_categories_local():
    """List categories from local registry."""
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
@click.pass_context
def rate_template(ctx, template_id: str, score: int, review: str | None):
    """Rate a template (1-5 stars)."""
    if ctx.obj.get("use_local"):
        _rate_template_local(template_id, score, review)
    else:
        _rate_template_api(ctx.obj["client"], template_id, score, review)


def _rate_template_api(client, template_id: str, score: int, review: str | None):
    """Rate template via API."""
    try:
        payload = {"score": score}
        if review:
            payload["review"] = review
        response = client.post(f"/api/v1/marketplace/templates/{template_id}/rate", json=payload)
        avg = response.get("average_rating", score)
        click.echo(f"Rated {template_id}: {'★' * score} ({score}/5)")
        click.echo(f"Average rating: {avg:.1f}")
    except Exception as e:
        click.echo(f"API error: {e}", err=True)
        sys.exit(1)


def _rate_template_local(template_id: str, score: int, review: str | None):
    """Rate template in local registry."""
    from aragora.marketplace import TemplateRegistry, TemplateRating

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
@click.pass_context
def use_template(ctx, template_id: str, task: str, rounds: int):
    """Use a debate template to start a debate."""
    if ctx.obj.get("use_local"):
        _use_template_local(template_id, task, rounds)
    else:
        _use_template_api(ctx.obj["client"], template_id, task, rounds)


def _use_template_api(client, template_id: str, task: str, rounds: int):
    """Use template via API."""
    try:
        template = client.get(f"/api/v1/marketplace/templates/{template_id}")
    except Exception as e:
        click.echo(f"API error: {e}", err=True)
        sys.exit(1)

    if template is None:
        click.echo(f"Template not found: {template_id}", err=True)
        sys.exit(1)

    if template.get("type") != "DebateTemplate":
        click.echo(f"Template {template_id} is not a debate template", err=True)
        sys.exit(1)

    # Format the task using the template
    task_template = template.get("task_template", "{topic}")
    formatted_task = task_template.format(
        topic=task,
        motion=task,
        problem=task,
        code=task,
    )

    protocol = template.get("protocol", {})
    click.echo(f"\nStarting debate with template: {template.get('name', template_id)}")
    click.echo(f"Task: {formatted_task}")
    click.echo(f"Roles: {len(template.get('agent_roles', []))}")
    click.echo(f"Rounds: {protocol.get('rounds', rounds)}")
    click.echo("\nTo run this debate, use:")
    click.echo(f'  aragora debate run "{formatted_task}" --rounds {rounds}')


def _use_template_local(template_id: str, task: str, rounds: int):
    """Use template from local registry."""
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
