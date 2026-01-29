"""
Workflow Automation Example

Shows how to create and execute workflow templates that chain debates
with conditional logic and external integrations.

Usage:
    python examples/workflow_automation.py

Environment:
    ARAGORA_API_KEY - Your API key
    ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
"""

from __future__ import annotations

import os
import time

from aragora import AragoraClient


def main() -> None:
    client = AragoraClient(
        base_url=os.environ.get("ARAGORA_API_URL", "https://api.aragora.ai"),
        api_key=os.environ.get("ARAGORA_API_KEY"),
    )

    # List available workflow templates
    print("=== Available Templates ===\n")
    templates = client.workflows.list_templates(category="analysis")
    for tmpl in templates.get("templates", [])[:5]:
        print(f"  {tmpl['name']} ({tmpl['id']})")
        print(f"    {tmpl.get('description', 'No description')}")
        print()

    # Create a workflow from a pattern
    print("Creating workflow...\n")
    workflow = client.workflows.instantiate_pattern(
        pattern_id="code-review",
        name="Code Review Pipeline",
        description="Multi-stage code review with AI agents",
        category="security",
        agents=["claude", "gpt-4"],
    )

    workflow_id = workflow["workflow_id"]
    print(f"Workflow created: {workflow_id}")

    # Execute the workflow
    print("Executing workflow...\n")
    execution = client.workflows.execute(
        workflow_id=workflow_id,
        input_data="""
def authenticate(username, password):
    query = f"SELECT * FROM users WHERE username='{username}'"
    return db.execute(query)
""",
    )

    execution_id = execution["execution_id"]
    print(f"Execution started: {execution_id}")

    # Poll for completion
    status = execution
    while status.get("status") in ("running", "pending"):
        time.sleep(3)
        status = client.workflows.get_execution(execution_id)
        step = status.get("current_step", "unknown")
        print(f"  Status: {status['status']}, Step: {step}")

    # Print results
    print("\n--- Workflow Results ---")
    if status.get("status") == "completed":
        output = status.get("output", {})
        print(f"  Final output: {output}")
    else:
        print(f"  Workflow ended with status: {status.get('status')}")
        if status.get("error"):
            print(f"  Error: {status['error']}")


if __name__ == "__main__":
    main()
