"""
Workflow Templates API Handler.

Endpoints:
- GET /api/workflow/templates - List available templates
- GET /api/workflow/templates/:id - Get template details
- GET /api/workflow/templates/:id/package - Get full package
- POST /api/workflow/templates/run - Execute a template

Provides marketplace-style access to workflow templates.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    get_bounded_string_param,
    get_int_param,
    handle_errors,
    json_response,
)
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter (60 requests per minute)
_template_limiter = RateLimiter(requests_per_minute=60)


class WorkflowTemplatesHandler(BaseHandler):
    """Handler for workflow templates API endpoints."""

    ROUTES: list[str] = [
        "/api/workflow/templates",
        "/api/workflow/templates/*",
        "/api/v1/workflow/templates",
        "/api/v1/workflow/templates/*",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/workflow/templates") or path.startswith(
            "/api/v1/workflow/templates"
        )

    def handle(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Route workflow template requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _template_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for templates endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # Parse path
        method = handler.command if hasattr(handler, "command") else "GET"

        # Handle list/search
        if path in ("/api/workflow/templates", "/api/v1/workflow/templates"):
            if method == "GET":
                return self._list_templates(query_params)
            elif method == "POST":
                # Run a template
                return self._run_template(handler)
            else:
                return error_response(f"Method {method} not allowed", 405)

        # Handle specific template requests
        parts = path.split("/")
        if len(parts) >= 4:
            # Extract template ID (could be like "legal/contract-review")
            template_parts = parts[4:] if path.startswith("/api/v1/") else parts[3:]

            if not template_parts:
                return error_response("Template ID required", 400)

            # Check for special routes
            if template_parts[-1] == "package":
                template_id = "/".join(template_parts[:-1])
                return self._get_package(template_id)
            elif template_parts[-1] == "run":
                template_id = "/".join(template_parts[:-1])
                return self._run_specific_template(template_id, handler)
            else:
                template_id = "/".join(template_parts)
                return self._get_template(template_id)

        return error_response("Invalid path", 400)

    @handle_errors("list templates")
    def _list_templates(self, query_params: dict) -> HandlerResult:
        """List available workflow templates."""
        from aragora.workflow.templates import list_templates, WORKFLOW_TEMPLATES
        from aragora.workflow.templates.package import package_all_templates

        # Get filters
        category = get_bounded_string_param(
            query_params, "category", None, max_length=50
        )
        tag = get_bounded_string_param(query_params, "tag", None, max_length=50)
        search = get_bounded_string_param(query_params, "search", None, max_length=100)
        limit = get_int_param(query_params, "limit", 50, min_val=1, max_val=100)
        offset = get_int_param(query_params, "offset", 0, min_val=0)

        # Get templates
        templates = list_templates(category=category)

        # Apply tag filter
        if tag:
            templates = [t for t in templates if tag in t.get("tags", [])]

        # Apply search filter
        if search:
            search_lower = search.lower()
            templates = [
                t
                for t in templates
                if search_lower in t["name"].lower()
                or search_lower in t.get("description", "").lower()
            ]

        # Count total before pagination
        total = len(templates)

        # Apply pagination
        templates = templates[offset : offset + limit]

        # Enrich with additional metadata
        enriched = []
        for t in templates:
            template_def = WORKFLOW_TEMPLATES.get(t["id"])
            enriched.append(
                {
                    **t,
                    "steps_count": len(template_def.get("steps", []))
                    if template_def
                    else 0,
                    "pattern": template_def.get("pattern") if template_def else None,
                    "estimated_duration": template_def.get("estimated_duration")
                    if template_def
                    else None,
                }
            )

        return json_response(
            {
                "templates": enriched,
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    @handle_errors("get template")
    def _get_template(self, template_id: str) -> HandlerResult:
        """Get details of a specific template."""
        from aragora.workflow.templates import get_template, WORKFLOW_TEMPLATES

        template = get_template(template_id)
        if not template:
            return error_response(f"Template not found: {template_id}", 404)

        # Get metadata
        category = template_id.split("/")[0] if "/" in template_id else "general"

        return json_response(
            {
                "id": template_id,
                "name": template.get("name", template_id),
                "description": template.get("description", ""),
                "category": category,
                "pattern": template.get("pattern"),
                "steps": template.get("steps", []),
                "inputs": template.get("inputs", {}),
                "outputs": template.get("outputs", {}),
                "estimated_duration": template.get("estimated_duration"),
                "recommended_agents": template.get("recommended_agents", []),
                "tags": template.get("tags", []),
            }
        )

    @handle_errors("get template package")
    def _get_package(self, template_id: str) -> HandlerResult:
        """Get the full package for a template."""
        from aragora.workflow.templates import get_template
        from aragora.workflow.templates.package import (
            create_package,
            TemplateAuthor,
        )

        template = get_template(template_id)
        if not template:
            return error_response(f"Template not found: {template_id}", 404)

        # Create package on-the-fly
        category = template_id.split("/")[0] if "/" in template_id else "general"
        package = create_package(
            template=template,
            version="1.0.0",
            author=TemplateAuthor(name="Aragora Team", organization="Aragora"),
            category=category,
        )

        return json_response(package.to_dict())

    @handle_errors("run template")
    def _run_template(self, handler: Any) -> HandlerResult:
        """Run a workflow template with provided inputs."""
        from aragora.workflow.engine import WorkflowEngine
        from aragora.workflow.templates import get_template

        # Parse request body
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        template_id = data.get("template_id")
        if not template_id:
            return error_response("template_id is required", 400)

        template = get_template(template_id)
        if not template:
            return error_response(f"Template not found: {template_id}", 404)

        inputs = data.get("inputs", {})
        agents = data.get("agents")

        # Execute template
        try:
            engine = WorkflowEngine()
            result = engine.execute_sync(
                workflow=template,
                inputs=inputs,
                agents=agents,
            )

            return json_response(
                {
                    "status": "completed",
                    "template_id": template_id,
                    "result": result.to_dict() if hasattr(result, "to_dict") else result,
                }
            )
        except Exception as e:
            logger.error(f"Template execution failed: {e}")
            return json_response(
                {
                    "status": "failed",
                    "template_id": template_id,
                    "error": str(e),
                },
                status=500,
            )

    @handle_errors("run specific template")
    def _run_specific_template(
        self, template_id: str, handler: Any
    ) -> HandlerResult:
        """Run a specific workflow template."""
        from aragora.workflow.templates import get_template

        template = get_template(template_id)
        if not template:
            return error_response(f"Template not found: {template_id}", 404)

        # Parse request body
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        # Add template_id to data and delegate
        data["template_id"] = template_id

        # Simulate the body being available again
        class MockHandler:
            def __init__(self, original, data):
                self.headers = original.headers
                self._data = data

            def read_body(self):
                return self._data

        # Return response indicating async execution would start
        return json_response(
            {
                "status": "accepted",
                "template_id": template_id,
                "message": "Template execution started",
            },
            status=202,
        )


# Categories endpoint
class WorkflowCategoriesHandler(BaseHandler):
    """Handler for workflow template categories."""

    ROUTES: list[str] = [
        "/api/workflow/categories",
        "/api/v1/workflow/categories",
    ]

    def can_handle(self, path: str) -> bool:
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Return available template categories."""
        from aragora.workflow.templates.package import TemplateCategory
        from aragora.workflow.templates import WORKFLOW_TEMPLATES

        # Count templates per category
        category_counts: dict[str, int] = {}
        for template_id in WORKFLOW_TEMPLATES:
            category = template_id.split("/")[0] if "/" in template_id else "general"
            category_counts[category] = category_counts.get(category, 0) + 1

        categories = [
            {
                "id": cat.value,
                "name": cat.value.replace("_", " ").title(),
                "template_count": category_counts.get(cat.value, 0),
            }
            for cat in TemplateCategory
            if category_counts.get(cat.value, 0) > 0
        ]

        return json_response({"categories": categories})


# Patterns endpoint
class WorkflowPatternsHandler(BaseHandler):
    """Handler for workflow patterns listing."""

    ROUTES: list[str] = [
        "/api/workflow/patterns",
        "/api/v1/workflow/patterns",
    ]

    def can_handle(self, path: str) -> bool:
        return path in self.ROUTES

    def handle(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Return available workflow patterns."""
        from aragora.workflow.patterns import PATTERN_REGISTRY
        from aragora.workflow.patterns.base import PatternType

        patterns = []
        for pattern_type in PatternType:
            pattern_class = PATTERN_REGISTRY.get(pattern_type)
            patterns.append(
                {
                    "id": pattern_type.value,
                    "name": pattern_type.value.replace("_", " ").title(),
                    "description": pattern_class.__doc__.split("\n")[0]
                    if pattern_class and pattern_class.__doc__
                    else "",
                    "available": pattern_class is not None,
                }
            )

        return json_response({"patterns": patterns})


class WorkflowPatternTemplatesHandler(BaseHandler):
    """Handler for pattern-based workflow template operations."""

    ROUTES: list[str] = [
        "/api/workflow/pattern-templates",
        "/api/workflow/pattern-templates/*",
        "/api/v1/workflow/pattern-templates",
        "/api/v1/workflow/pattern-templates/*",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path.startswith("/api/workflow/pattern-templates") or path.startswith(
            "/api/v1/workflow/pattern-templates"
        )

    def handle(
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Route pattern template requests."""
        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _template_limiter.is_allowed(client_ip):
            return error_response("Rate limit exceeded", 429)

        method = handler.command if hasattr(handler, "command") else "GET"

        # List pattern templates
        if path in ("/api/workflow/pattern-templates", "/api/v1/workflow/pattern-templates"):
            if method == "GET":
                return self._list_pattern_templates()
            else:
                return error_response(f"Method {method} not allowed", 405)

        # Handle specific pattern template requests
        parts = path.split("/")
        pattern_id = parts[-1]

        # Check for instantiate route
        if len(parts) >= 2 and parts[-1] == "instantiate":
            pattern_id = parts[-2]
            return self._instantiate_pattern(pattern_id, handler)

        return self._get_pattern_template(pattern_id)

    @handle_errors("list pattern templates")
    def _list_pattern_templates(self) -> HandlerResult:
        """List available pattern-based workflow templates."""
        from aragora.workflow.templates.patterns import list_pattern_templates

        templates = list_pattern_templates()

        return json_response({
            "pattern_templates": templates,
            "total": len(templates),
        })

    @handle_errors("get pattern template")
    def _get_pattern_template(self, pattern_id: str) -> HandlerResult:
        """Get details of a specific pattern template."""
        from aragora.workflow.templates.patterns import get_pattern_template, PATTERN_TEMPLATES

        # Try with pattern/ prefix if not found
        template = get_pattern_template(pattern_id)
        if not template:
            template = get_pattern_template(f"pattern/{pattern_id}")

        if not template:
            return error_response(f"Pattern template not found: {pattern_id}", 404)

        return json_response({
            "id": template.get("id", pattern_id),
            "name": template.get("name", pattern_id),
            "description": template.get("description", ""),
            "pattern": template.get("pattern"),
            "version": template.get("version", "1.0.0"),
            "config": template.get("config", {}),
            "inputs": template.get("inputs", {}),
            "outputs": template.get("outputs", {}),
            "tags": template.get("tags", []),
        })

    @handle_errors("instantiate pattern")
    def _instantiate_pattern(self, pattern_id: str, handler: Any) -> HandlerResult:
        """Create a workflow definition from a pattern template."""
        from aragora.workflow.templates.patterns import (
            create_hive_mind_workflow,
            create_map_reduce_workflow,
            create_review_cycle_workflow,
        )

        # Parse request body
        try:
            content_length = int(handler.headers.get("Content-Length", 0))
            body = handler.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError) as e:
            return error_response(f"Invalid JSON: {e}", 400)

        # Map pattern IDs to factory functions
        pattern_factories = {
            "hive-mind": create_hive_mind_workflow,
            "hive_mind": create_hive_mind_workflow,
            "map-reduce": create_map_reduce_workflow,
            "map_reduce": create_map_reduce_workflow,
            "review-cycle": create_review_cycle_workflow,
            "review_cycle": create_review_cycle_workflow,
        }

        factory = pattern_factories.get(pattern_id)
        if not factory:
            return error_response(
                f"Unknown pattern: {pattern_id}. Available: {list(pattern_factories.keys())}",
                404
            )

        # Extract configuration from request
        name = data.get("name", f"{pattern_id.replace('-', ' ').title()} Workflow")
        task = data.get("task", "")
        config = data.get("config", {})

        # Merge task and config
        workflow_args = {"name": name, "task": task, **config}

        try:
            workflow = factory(**workflow_args)

            # Convert workflow to serializable dict
            workflow_dict = {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "pattern": pattern_id,
                "steps": [
                    {
                        "id": step.id,
                        "name": step.name,
                        "type": step.step_type,
                        "config": step.config,
                        "next_steps": step.next_steps,
                    }
                    for step in workflow.steps
                ],
                "entry_step": workflow.entry_step,
                "tags": workflow.tags,
                "metadata": workflow.metadata,
            }

            return json_response({
                "status": "created",
                "workflow": workflow_dict,
            }, status=201)

        except Exception as e:
            logger.error(f"Failed to instantiate pattern {pattern_id}: {e}")
            return error_response(f"Failed to instantiate pattern: {e}", 500)
