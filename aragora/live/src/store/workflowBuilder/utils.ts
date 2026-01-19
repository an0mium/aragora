/**
 * Utility functions for Workflow Builder Store
 */

import type { StepType } from './types';

export function generateId(prefix: string = 'node'): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

export function getDefaultStepConfig(type: StepType): Record<string, unknown> {
  switch (type) {
    case 'agent':
      return { agent_type: 'claude', prompt_template: '' };
    case 'debate':
      return { agents: ['claude', 'gpt4'], rounds: 3 };
    case 'quick_debate':
      return { agents: ['claude', 'gemini'], rounds: 2, fast_mode: true };
    case 'parallel':
      return { steps: [], max_concurrent: 3 };
    case 'conditional':
      return { condition: '', true_step: '', false_step: '' };
    case 'loop':
      return { max_iterations: 5, condition: '', step: '' };
    case 'human_checkpoint':
      return { prompt: 'Please review and approve', timeout_hours: 24 };
    case 'memory_read':
      return { query: '', workspace_id: 'default', limit: 10 };
    case 'memory_write':
      return { node_type: 'fact', content_field: 'output' };
    case 'task':
      return { action: '' };
    default:
      return {};
  }
}

export function getDefaultStepName(type: StepType): string {
  const names: Record<StepType, string> = {
    agent: 'Agent Step',
    debate: 'Multi-Agent Debate',
    quick_debate: 'Quick Debate',
    parallel: 'Parallel Execution',
    conditional: 'Condition Check',
    loop: 'Loop',
    human_checkpoint: 'Human Approval',
    memory_read: 'Read from Knowledge',
    memory_write: 'Write to Knowledge',
    task: 'Task',
  };
  return names[type] || 'New Step';
}
