"""Canvas to Workflow synchronization.

Converts visual canvas graph state (UniversalGraph Stage 4 nodes)
into executable WorkflowDefinitions, ensuring canvas edits are
reflected in the actual workflow that gets executed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CanvasChange:
    """Describes a single diff between canvas and workflow state."""
    change_type: str  # 'added', 'removed', 'modified'
    node_id: str
    field: str
    old_value: Any = None
    new_value: Any = None


def sync_canvas_to_workflow(graph: Any) -> dict[str, Any]:
    """Convert Stage 4 (orchestration) nodes from a UniversalGraph into a WorkflowDefinition dict.

    Reads orchestration nodes, converts:
    - agent_task -> task StepDefinition
    - parallel_fan -> parallel StepDefinition
    - human_gate -> human_checkpoint StepDefinition
    - verification -> verification StepDefinition
    - debate -> debate StepDefinition
    - merge -> merge StepDefinition

    Edges between orchestration nodes become TransitionRules.

    Returns a dict matching WorkflowDefinition schema with 'steps' and 'transitions'.
    """
    try:
        from aragora.canvas.stages import PipelineStage
    except ImportError:
        PipelineStage = None

    steps: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    # Get orchestration nodes
    orch_nodes: dict[str, Any] = {}
    if hasattr(graph, 'nodes'):
        for node_id, node in graph.nodes.items():
            stage = getattr(node, 'stage', None)
            if stage and hasattr(stage, 'value') and stage.value == 'orchestration':
                orch_nodes[node_id] = node
            elif PipelineStage and stage == PipelineStage.ORCHESTRATION:
                orch_nodes[node_id] = node

    # Map orch_type to workflow step_type
    _ORCH_TO_STEP = {
        'agent_task': 'task',
        'debate': 'debate',
        'human_gate': 'human_checkpoint',
        'parallel_fan': 'parallel',
        'merge': 'merge',
        'verification': 'verification',
    }

    for node_id, node in orch_nodes.items():
        data = getattr(node, 'data', {}) or {}
        if isinstance(data, dict):
            node_data = data
        elif hasattr(data, '__dict__'):
            node_data = data.__dict__
        else:
            node_data = {}

        orch_type = node_data.get('orchType', node_data.get('orch_type', 'agent_task'))
        step_type = _ORCH_TO_STEP.get(orch_type, 'task')

        step = {
            'id': node_id,
            'name': node_data.get('label', getattr(node, 'label', f'Step {node_id}')),
            'description': node_data.get('description', ''),
            'step_type': step_type,
            'config': {
                'assigned_agent': node_data.get('assignedAgent', node_data.get('assigned_agent', '')),
                'capabilities': node_data.get('capabilities', []),
                'elo_score': node_data.get('eloScore', node_data.get('elo_score')),
            },
            'timeout_seconds': node_data.get('timeoutSeconds', 3600),
            'retries': node_data.get('retries', 1),
            'optional': node_data.get('optional', False),
        }
        steps.append(step)

    # Convert edges to transitions
    if hasattr(graph, 'edges'):
        for edge_id, edge in (graph.edges.items() if isinstance(graph.edges, dict) else enumerate(graph.edges)):
            source = getattr(edge, 'source_id', edge.get('source_id', '')) if isinstance(edge, dict) else getattr(edge, 'source_id', '')
            target = getattr(edge, 'target_id', edge.get('target_id', '')) if isinstance(edge, dict) else getattr(edge, 'target_id', '')

            # Only include edges between orchestration nodes
            if source in orch_nodes and target in orch_nodes:
                transitions.append({
                    'id': getattr(edge, 'id', edge_id) if not isinstance(edge_id, int) else f'tr-{source}-{target}',
                    'from_step': source,
                    'to_step': target,
                    'condition': getattr(edge, 'condition', '') if not isinstance(edge, dict) else edge.get('condition', ''),
                    'label': getattr(edge, 'label', '') if not isinstance(edge, dict) else edge.get('label', ''),
                    'priority': 0,
                })

    return {
        'name': getattr(graph, 'name', 'Canvas Workflow'),
        'steps': steps,
        'transitions': transitions,
        'metadata': {
            'source': 'canvas_sync',
            'graph_id': getattr(graph, 'id', ''),
            'node_count': len(steps),
        },
    }


def diff_canvas_workflow(
    graph: Any,
    existing_workflow: dict[str, Any],
) -> list[CanvasChange]:
    """Compare canvas state against existing workflow and return changes.

    Useful for detecting what the user modified in the canvas since
    the workflow was last synced.
    """
    changes: list[CanvasChange] = []
    current = sync_canvas_to_workflow(graph)

    current_steps = {s['id']: s for s in current.get('steps', [])}
    existing_steps = {s['id']: s for s in existing_workflow.get('steps', [])}

    # Find added steps
    for step_id in current_steps:
        if step_id not in existing_steps:
            changes.append(CanvasChange(
                change_type='added',
                node_id=step_id,
                field='step',
                new_value=current_steps[step_id],
            ))

    # Find removed steps
    for step_id in existing_steps:
        if step_id not in current_steps:
            changes.append(CanvasChange(
                change_type='removed',
                node_id=step_id,
                field='step',
                old_value=existing_steps[step_id],
            ))

    # Find modified steps
    for step_id in current_steps:
        if step_id in existing_steps:
            curr = current_steps[step_id]
            prev = existing_steps[step_id]
            for key in ('name', 'step_type', 'config', 'description'):
                if curr.get(key) != prev.get(key):
                    changes.append(CanvasChange(
                        change_type='modified',
                        node_id=step_id,
                        field=key,
                        old_value=prev.get(key),
                        new_value=curr.get(key),
                    ))

    return changes
