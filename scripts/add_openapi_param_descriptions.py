#!/usr/bin/env python3
"""
Add descriptions to OpenAPI spec parameters.

Uses a mapping of common parameter names to descriptions.
"""

import argparse
import json
import sys
from pathlib import Path

# Parameter description mapping
PARAM_DESCRIPTIONS = {
    # Pagination
    "limit": "Maximum number of items to return (default: 20, max: 100)",
    "offset": "Number of items to skip for pagination",
    "page": "Page number for pagination (1-indexed)",
    "page_size": "Number of items per page",
    "cursor": "Cursor for pagination (opaque token from previous response)",
    # Identifiers
    "id": "Unique identifier of the resource",
    "debate_id": "Unique identifier of the debate",
    "workspace_id": "Workspace to scope the query to",
    "user_id": "Unique identifier of the user",
    "agent_id": "Unique identifier of the agent",
    "session_id": "Unique identifier of the session",
    "message_id": "Unique identifier of the message",
    "workflow_id": "Unique identifier of the workflow",
    "template_id": "Unique identifier of the template",
    "budget_id": "Unique identifier of the budget",
    "receipt_id": "Unique identifier of the receipt",
    "run_id": "Unique identifier of the run",
    "loop_id": "Unique identifier of the loop",
    "tenant_id": "Unique identifier of the tenant",
    "policy_id": "Unique identifier of the policy",
    "role_id": "Unique identifier of the role",
    "team_id": "Unique identifier of the team",
    "channel_id": "Unique identifier of the channel",
    "finding_id": "Unique identifier of the finding",
    "insight_id": "Unique identifier of the insight",
    # Filters
    "status": "Filter by status (e.g., pending, active, completed, failed)",
    "category": "Filter by category",
    "domain": "Filter by domain",
    "type": "Filter by type",
    "format": "Output format (e.g., json, yaml, csv)",
    "level": "Filter by level (e.g., debug, info, warning, error)",
    "severity": "Filter by severity (e.g., low, medium, high, critical)",
    "priority": "Filter by priority",
    "provider": "Filter by provider",
    "model": "Filter by model name",
    "source": "Filter by source",
    "target": "Filter by target",
    "scope": "Filter by scope",
    # Time
    "start_date": "Start date for filtering (ISO 8601 format)",
    "end_date": "End date for filtering (ISO 8601 format)",
    "start_time": "Start time for filtering (Unix timestamp or ISO 8601)",
    "end_time": "End time for filtering (Unix timestamp or ISO 8601)",
    "since": "Filter results newer than this timestamp",
    "until": "Filter results older than this timestamp",
    "period": "Time period for aggregation (e.g., hour, day, week, month)",
    "days": "Number of days to include",
    # Names and text
    "name": "Name of the resource",
    "query": "Search query string",
    "q": "Search query string",
    "topic": "Topic to filter or search by",
    "question": "Question text",
    "description": "Description text",
    "label": "Label or tag to filter by",
    "tag": "Tag to filter by",
    "tags": "Comma-separated list of tags to filter by",
    # Git/VCS
    "repo": "Repository name or path",
    "branch": "Git branch name",
    "commit": "Git commit SHA",
    "ref": "Git reference (branch, tag, or commit SHA)",
    "path": "File or directory path",
    "file": "File name or path",
    "checkpoint_name": "Name of the checkpoint",
    # Boolean flags
    "active": "Filter to only active items",
    "enabled": "Filter to only enabled items",
    "include_deleted": "Include soft-deleted items in results",
    "include_archived": "Include archived items in results",
    "with_content": "Include full content in response",
    "with_metadata": "Include metadata in response",
    "recursive": "Process recursively",
    "force": "Force the operation even if it would normally be prevented",
    "dry_run": "Simulate the operation without making changes",
    "verbose": "Include additional details in response",
    "deep": "Perform deep analysis",
    # Sorting
    "sort": "Field to sort by",
    "sort_by": "Field to sort by",
    "order": "Sort order (asc or desc)",
    "order_by": "Field and direction to order by",
    # Counts and limits
    "count": "Number of items to return",
    "max": "Maximum value or count",
    "min": "Minimum value or count",
    "rounds": "Number of debate rounds",
    "threshold": "Threshold value",
    # Specific to Aragora
    "consensus_threshold": "Minimum agreement level for consensus (0.0-1.0)",
    "confidence": "Confidence level (0.0-1.0)",
    "risk_level": "Risk level (low, medium, high, critical)",
    "capabilities": "Required agent capabilities",
    "protocol": "Debate protocol to use",
    "mode": "Operating mode",
    "phase": "Phase to filter by",
    "version": "Version string or number",
}


def add_param_descriptions(spec: dict) -> tuple[dict, int, int]:
    """Add descriptions to parameters missing them.

    Returns: (updated_spec, added_count, existing_count)
    """
    added = 0
    existing = 0

    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue
            for param in details.get("parameters", []):
                if param.get("description"):
                    existing += 1
                else:
                    param_name = param.get("name", "").lower()
                    # Try exact match first
                    if param_name in PARAM_DESCRIPTIONS:
                        param["description"] = PARAM_DESCRIPTIONS[param_name]
                        added += 1
                    # Try removing common prefixes/suffixes
                    elif param_name.endswith("_id") and param_name in PARAM_DESCRIPTIONS:
                        param["description"] = PARAM_DESCRIPTIONS[param_name]
                        added += 1
                    # Generic fallback based on type
                    elif param.get("schema", {}).get("type") == "boolean":
                        param["description"] = f"Whether to enable {param_name.replace('_', ' ')}"
                        added += 1
                    elif param.get("schema", {}).get("type") == "integer":
                        param["description"] = f"The {param_name.replace('_', ' ')} value"
                        added += 1
                    elif param.get("schema", {}).get("type") == "string":
                        param["description"] = f"The {param_name.replace('_', ' ')}"
                        added += 1
                    else:
                        # Keep track of unmatched for manual review
                        param["description"] = f"{param_name.replace('_', ' ').title()} parameter"
                        added += 1

    return spec, added, existing


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Add parameter descriptions to OpenAPI spec")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("docs/api/openapi.json"),
        help="Path to OpenAPI JSON spec",
    )
    args = parser.parse_args()

    spec_path = args.spec

    if not spec_path.exists():
        print(f"Error: {spec_path} not found")
        sys.exit(1)

    print(f"Reading {spec_path}...")
    with open(spec_path, "r") as f:
        spec = json.load(f)

    print("Adding parameter descriptions...")
    spec, added, existing = add_param_descriptions(spec)

    print(f"Writing {spec_path}...")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    print("\nResults:")
    print(f"  - Already had description: {existing}")
    print(f"  - Added description: {added}")
    print(f"  - Total parameters: {added + existing}")


if __name__ == "__main__":
    main()
