import {
  DAG_STAGE_ORDER,
  STAGE_LANE_GAP,
  STAGE_LANE_PADDING_X,
  STAGE_LANE_WIDTH,
  buildFlowFromSnapshot,
  normalizeDagSnapshot,
} from '../useUnifiedDAG';

describe('useUnifiedDAG layout helpers', () => {
  it('normalizes raw graph payloads into stage-aware dependencies', () => {
    const snapshot = normalizeDagSnapshot({
      id: 'dag-1',
      name: 'Pipeline',
      nodes: [
        { id: 'idea-1', stage: 'ideas', node_subtype: 'concept', label: 'Seed idea', execution_status: 'completed' },
        { id: 'principle-1', stage: 'principles', node_subtype: 'principle', label: 'Guardrail', execution_status: 'completed' },
        { id: 'goal-1', stage: 'goals', node_subtype: 'goal', label: 'Goal', execution_status: 'pending' },
        { id: 'action-1', stage: 'actions', node_subtype: 'task', label: 'Action', execution_status: 'completed' },
        { id: 'orch-1', stage: 'orchestration', node_subtype: 'agent_task', label: 'Execute', execution_status: 'pending' },
      ],
      edges: [
        { id: 'e1', source_id: 'idea-1', target_id: 'principle-1', edge_type: 'derived_from' },
        { id: 'e2', source_id: 'principle-1', target_id: 'goal-1', edge_type: 'constrains' },
        { id: 'e3', source_id: 'goal-1', target_id: 'action-1', edge_type: 'implements' },
        { id: 'e4', source_id: 'action-1', target_id: 'orch-1', edge_type: 'executes' },
      ],
    });

    expect(snapshot.graphId).toBe('dag-1');
    expect(snapshot.stages.map((stage) => stage.stage)).toEqual(DAG_STAGE_ORDER);
    expect(snapshot.dependencies).toHaveLength(4);
    expect(snapshot.dependencies[1].source_stage).toBe('principles');
    expect(snapshot.stageStatus.orchestration).toBe('pending');
  });

  it('lays out nodes in deterministic stage lanes and marks ready orchestration work', () => {
    const snapshot = normalizeDagSnapshot({
      graph_id: 'dag-2',
      name: 'Pipeline',
      nodes: [
        { id: 'idea-1', stage: 'ideas', node_subtype: 'concept', label: 'Seed idea', execution_status: 'completed' },
        { id: 'principle-1', stage: 'principles', node_subtype: 'principle', label: 'Guardrail', execution_status: 'completed' },
        { id: 'goal-1', stage: 'goals', node_subtype: 'goal', label: 'Goal', execution_status: 'completed' },
        { id: 'action-1', stage: 'actions', node_subtype: 'task', label: 'Action', execution_status: 'completed' },
        { id: 'orch-1', stage: 'orchestration', node_subtype: 'agent_task', label: 'Execute', execution_status: 'pending' },
      ],
      dependencies: [
        { id: 'e1', source_id: 'idea-1', target_id: 'principle-1', edge_type: 'derived_from', source_stage: 'ideas', target_stage: 'principles' },
        { id: 'e2', source_id: 'principle-1', target_id: 'goal-1', edge_type: 'constrains', source_stage: 'principles', target_stage: 'goals' },
        { id: 'e3', source_id: 'goal-1', target_id: 'action-1', edge_type: 'implements', source_stage: 'goals', target_stage: 'actions' },
        { id: 'e4', source_id: 'action-1', target_id: 'orch-1', edge_type: 'executes', source_stage: 'actions', target_stage: 'orchestration' },
      ],
      stage_status: {
        ideas: 'complete',
        principles: 'complete',
        goals: 'complete',
        actions: 'complete',
        orchestration: 'pending',
      },
    });
    const flow = buildFlowFromSnapshot(snapshot);
    const byId = Object.fromEntries(flow.nodes.map((node) => [node.id, node]));

    expect(byId['principle-1'].position.x).toBe(STAGE_LANE_WIDTH + STAGE_LANE_GAP + STAGE_LANE_PADDING_X);
    expect(byId['goal-1'].position.x).toBe((STAGE_LANE_WIDTH + STAGE_LANE_GAP) * 2 + STAGE_LANE_PADDING_X);
    expect(byId['orch-1'].data.status).toBe('ready');
    expect(byId['orch-1'].data.canExecute).toBe(true);
    expect(flow.edges[3].data?.edgeType).toBe('executes');
  });
});
