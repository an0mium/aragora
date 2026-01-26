"""
MCP Tool Definitions for Aragora.

This package organizes MCP tools into logical modules:
- debate: Core debate operations
- gauntlet: Document stress-testing
- agent: Agent management and breeding
- memory: Memory tier operations
- checkpoint: Debate checkpoint management
- verification: Consensus verification and proofs
- evidence: Evidence collection and citation
- trending: Trending topic analysis
- browser: Browser automation with Playwright
"""

from aragora.mcp.tools_module.debate import (
    run_debate_tool,
    get_debate_tool,
    search_debates_tool,
    fork_debate_tool,
    get_forks_tool,
)
from aragora.mcp.tools_module.gauntlet import run_gauntlet_tool
from aragora.mcp.tools_module.agent import (
    list_agents_tool,
    get_agent_history_tool,
    get_agent_lineage_tool,
    breed_agents_tool,
)
from aragora.mcp.tools_module.memory import (
    query_memory_tool,
    store_memory_tool,
    get_memory_pressure_tool,
)
from aragora.mcp.tools_module.checkpoint import (
    create_checkpoint_tool,
    list_checkpoints_tool,
    resume_checkpoint_tool,
    delete_checkpoint_tool,
)
from aragora.mcp.tools_module.verification import (
    get_consensus_proofs_tool,
    verify_consensus_tool,
    generate_proof_tool,
)
from aragora.mcp.tools_module.evidence import (
    search_evidence_tool,
    cite_evidence_tool,
    verify_citation_tool,
)
from aragora.mcp.tools_module.trending import list_trending_topics_tool
from aragora.mcp.tools_module.audit import (
    list_audit_presets_tool,
    list_audit_types_tool,
    get_audit_preset_tool,
    create_audit_session_tool,
    run_audit_tool,
    get_audit_status_tool,
    get_audit_findings_tool,
    update_finding_status_tool,
    run_quick_audit_tool,
)
from aragora.mcp.tools_module.knowledge import (
    query_knowledge_tool,
    store_knowledge_tool,
    get_knowledge_stats_tool,
    get_decision_receipt_tool,
)
from aragora.mcp.tools_module.workflow import (
    run_workflow_tool,
    get_workflow_status_tool,
    list_workflow_templates_tool,
    cancel_workflow_tool,
)
from aragora.mcp.tools_module.integrations import (
    trigger_external_webhook_tool,
    list_integrations_tool,
    test_integration_tool,
    get_integration_events_tool,
)
from aragora.mcp.tools_module.control_plane import (
    register_agent_tool,
    unregister_agent_tool,
    list_registered_agents_tool,
    get_agent_health_tool,
    submit_task_tool,
    get_task_status_tool,
    cancel_task_tool,
    list_pending_tasks_tool,
    get_control_plane_status_tool,
    trigger_health_check_tool,
    get_resource_utilization_tool,
)
from aragora.mcp.tools_module.browser import (
    browser_navigate_tool,
    browser_click_tool,
    browser_fill_tool,
    browser_screenshot_tool,
    browser_get_text_tool,
    browser_extract_tool,
    browser_execute_script_tool,
    browser_wait_for_tool,
    browser_get_html_tool,
    browser_close_tool,
    browser_get_cookies_tool,
    browser_clear_cookies_tool,
)
from aragora.mcp.tools_module.canvas import (
    canvas_create_tool,
    canvas_get_tool,
    canvas_add_node_tool,
    canvas_add_edge_tool,
    canvas_execute_action_tool,
    canvas_list_tool,
    canvas_delete_node_tool,
)

__all__ = [
    # Debate tools
    "run_debate_tool",
    "get_debate_tool",
    "search_debates_tool",
    "fork_debate_tool",
    "get_forks_tool",
    # Gauntlet tools
    "run_gauntlet_tool",
    # Agent tools
    "list_agents_tool",
    "get_agent_history_tool",
    "get_agent_lineage_tool",
    "breed_agents_tool",
    # Memory tools
    "query_memory_tool",
    "store_memory_tool",
    "get_memory_pressure_tool",
    # Checkpoint tools
    "create_checkpoint_tool",
    "list_checkpoints_tool",
    "resume_checkpoint_tool",
    "delete_checkpoint_tool",
    # Verification tools
    "get_consensus_proofs_tool",
    "verify_consensus_tool",
    "generate_proof_tool",
    # Evidence tools
    "search_evidence_tool",
    "cite_evidence_tool",
    "verify_citation_tool",
    # Trending tools
    "list_trending_topics_tool",
    # Audit tools
    "list_audit_presets_tool",
    "list_audit_types_tool",
    "get_audit_preset_tool",
    "create_audit_session_tool",
    "run_audit_tool",
    "get_audit_status_tool",
    "get_audit_findings_tool",
    "update_finding_status_tool",
    "run_quick_audit_tool",
    # Knowledge tools
    "query_knowledge_tool",
    "store_knowledge_tool",
    "get_knowledge_stats_tool",
    "get_decision_receipt_tool",
    # Workflow tools
    "run_workflow_tool",
    "get_workflow_status_tool",
    "list_workflow_templates_tool",
    "cancel_workflow_tool",
    # External integration tools
    "trigger_external_webhook_tool",
    "list_integrations_tool",
    "test_integration_tool",
    "get_integration_events_tool",
    # Control plane tools
    "register_agent_tool",
    "unregister_agent_tool",
    "list_registered_agents_tool",
    "get_agent_health_tool",
    "submit_task_tool",
    "get_task_status_tool",
    "cancel_task_tool",
    "list_pending_tasks_tool",
    "get_control_plane_status_tool",
    "trigger_health_check_tool",
    "get_resource_utilization_tool",
    # Browser automation tools
    "browser_navigate_tool",
    "browser_click_tool",
    "browser_fill_tool",
    "browser_screenshot_tool",
    "browser_get_text_tool",
    "browser_extract_tool",
    "browser_execute_script_tool",
    "browser_wait_for_tool",
    "browser_get_html_tool",
    "browser_close_tool",
    "browser_get_cookies_tool",
    "browser_clear_cookies_tool",
    # Canvas tools
    "canvas_create_tool",
    "canvas_get_tool",
    "canvas_add_node_tool",
    "canvas_add_edge_tool",
    "canvas_execute_action_tool",
    "canvas_list_tool",
    "canvas_delete_node_tool",
]
