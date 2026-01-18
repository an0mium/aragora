'use client';

import { create } from 'zustand';
import { devtools, subscribeWithSelector, persist } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export type StepType =
  | 'agent'
  | 'debate'
  | 'quick_debate'
  | 'parallel'
  | 'conditional'
  | 'loop'
  | 'human_checkpoint'
  | 'memory_read'
  | 'memory_write'
  | 'task';

export type NodeCategory = 'agents' | 'control' | 'memory' | 'integration';

export interface Position {
  x: number;
  y: number;
}

export interface StepDefinition {
  id: string;
  name: string;
  step_type: StepType;
  config: Record<string, unknown>;
  next_steps: string[];
  position?: Position;
}

export interface TransitionRule {
  id: string;
  from_step: string;
  to_step: string;
  condition?: string;
  label?: string;
}

export interface WorkflowDefinition {
  id: string;
  name: string;
  description?: string;
  category?: string;
  steps: StepDefinition[];
  transitions: TransitionRule[];
  config?: {
    timeout_seconds?: number;
    max_tokens?: number;
    max_cost_usd?: number;
  };
  version?: string;
  created_at?: string;
  updated_at?: string;
}

export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  workflow: WorkflowDefinition;
  preview_image?: string;
}

export interface WorkflowSimulationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  estimated_cost?: number;
  estimated_duration_ms?: number;
  step_order: string[];
}

// ============================================================================
// Store State
// ============================================================================

interface WorkflowBuilderState {
  // Current workflow being edited
  currentWorkflow: WorkflowDefinition | null;
  originalWorkflow: WorkflowDefinition | null;

  // Canvas state
  canvas: {
    zoom: number;
    panX: number;
    panY: number;
    selectedNodeIds: Set<string>;
    selectedEdgeIds: Set<string>;
    isDragging: boolean;
    draggedNodeId: string | null;
  };

  // Node palette
  nodePalette: {
    searchQuery: string;
    selectedCategory: NodeCategory | null;
    expandedCategories: Set<string>;
  };

  // Configuration panel
  configPanel: {
    isOpen: boolean;
    selectedNodeId: string | null;
    pendingChanges: Partial<StepDefinition> | null;
  };

  // Workflow list
  workflows: WorkflowDefinition[];
  templates: WorkflowTemplate[];

  // UI state
  isDirty: boolean;
  isSaving: boolean;
  isLoading: boolean;
  saveError: string | null;
  loadError: string | null;
  validationErrors: string[];

  // Execution preview
  executionPreview: {
    isOpen: boolean;
    isRunning: boolean;
    result: WorkflowSimulationResult | null;
  };

  // Undo/redo history
  _history: WorkflowDefinition[];
  _historyIndex: number;
}

interface WorkflowBuilderActions {
  // Workflow CRUD
  setCurrentWorkflow: (workflow: WorkflowDefinition | null) => void;
  createNewWorkflow: (name: string, description?: string) => WorkflowDefinition;
  updateWorkflowMetadata: (updates: Partial<Pick<WorkflowDefinition, 'name' | 'description' | 'category' | 'config'>>) => void;

  // Node operations
  addNode: (type: StepType, position: Position) => string;
  updateNode: (id: string, updates: Partial<StepDefinition>) => void;
  deleteNode: (id: string) => void;
  duplicateNode: (id: string) => string | null;

  // Edge operations
  addEdge: (fromId: string, toId: string, condition?: string) => string;
  updateEdge: (id: string, updates: Partial<TransitionRule>) => void;
  deleteEdge: (id: string) => void;

  // Canvas operations
  setZoom: (zoom: number) => void;
  setPan: (x: number, y: number) => void;
  selectNodes: (ids: string[]) => void;
  selectEdges: (ids: string[]) => void;
  clearSelection: () => void;
  setDragging: (isDragging: boolean, nodeId?: string | null) => void;

  // Node palette operations
  setSearchQuery: (query: string) => void;
  setSelectedCategory: (category: NodeCategory | null) => void;
  toggleCategory: (category: string) => void;

  // Config panel operations
  openConfigPanel: (nodeId: string) => void;
  closeConfigPanel: () => void;
  setPendingChanges: (changes: Partial<StepDefinition> | null) => void;
  applyPendingChanges: () => void;

  // Validation
  validate: () => string[];

  // Execution preview
  openExecutionPreview: () => void;
  closeExecutionPreview: () => void;
  setSimulationResult: (result: WorkflowSimulationResult | null) => void;
  setSimulationRunning: (running: boolean) => void;

  // Workflow list
  setWorkflows: (workflows: WorkflowDefinition[]) => void;
  setTemplates: (templates: WorkflowTemplate[]) => void;

  // Loading states
  setLoading: (loading: boolean) => void;
  setSaving: (saving: boolean) => void;
  setSaveError: (error: string | null) => void;
  setLoadError: (error: string | null) => void;

  // Undo/redo
  undo: () => void;
  redo: () => void;
  pushHistory: () => void;

  // Reset
  resetBuilder: () => void;
  markClean: () => void;
}

type WorkflowBuilderStore = WorkflowBuilderState & WorkflowBuilderActions;

// ============================================================================
// Constants
// ============================================================================

const MAX_HISTORY = 50;

const initialCanvasState = {
  zoom: 1,
  panX: 0,
  panY: 0,
  selectedNodeIds: new Set<string>(),
  selectedEdgeIds: new Set<string>(),
  isDragging: false,
  draggedNodeId: null,
};

const initialPaletteState = {
  searchQuery: '',
  selectedCategory: null,
  expandedCategories: new Set<string>(['agents', 'control']),
};

const initialConfigPanelState = {
  isOpen: false,
  selectedNodeId: null,
  pendingChanges: null,
};

const initialPreviewState = {
  isOpen: false,
  isRunning: false,
  result: null,
};

// ============================================================================
// Utility Functions
// ============================================================================

function generateId(prefix: string = 'node'): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function getDefaultStepConfig(type: StepType): Record<string, unknown> {
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

function getDefaultStepName(type: StepType): string {
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

// ============================================================================
// Store Implementation
// ============================================================================

export const useWorkflowBuilderStore = create<WorkflowBuilderStore>()(
  devtools(
    subscribeWithSelector(
      persist(
        (set, get) => ({
          // Initial state
          currentWorkflow: null,
          originalWorkflow: null,
          canvas: { ...initialCanvasState },
          nodePalette: { ...initialPaletteState },
          configPanel: { ...initialConfigPanelState },
          workflows: [],
          templates: [],
          isDirty: false,
          isSaving: false,
          isLoading: false,
          saveError: null,
          loadError: null,
          validationErrors: [],
          executionPreview: { ...initialPreviewState },
          _history: [],
          _historyIndex: -1,

          // Workflow CRUD
          setCurrentWorkflow: (workflow) => {
            set({
              currentWorkflow: workflow,
              originalWorkflow: workflow ? JSON.parse(JSON.stringify(workflow)) : null,
              isDirty: false,
              validationErrors: [],
              configPanel: { ...initialConfigPanelState },
              canvas: {
                ...initialCanvasState,
                selectedNodeIds: new Set(),
                selectedEdgeIds: new Set(),
              },
              _history: workflow ? [JSON.parse(JSON.stringify(workflow))] : [],
              _historyIndex: 0,
            }, false, 'setCurrentWorkflow');
          },

          createNewWorkflow: (name, description) => {
            const workflow: WorkflowDefinition = {
              id: generateId('wf'),
              name,
              description,
              steps: [],
              transitions: [],
              config: {
                timeout_seconds: 600,
                max_tokens: 100000,
                max_cost_usd: 10,
              },
              version: '1.0.0',
              created_at: new Date().toISOString(),
            };

            set({
              currentWorkflow: workflow,
              originalWorkflow: JSON.parse(JSON.stringify(workflow)),
              isDirty: false,
              _history: [JSON.parse(JSON.stringify(workflow))],
              _historyIndex: 0,
            }, false, 'createNewWorkflow');

            return workflow;
          },

          updateWorkflowMetadata: (updates) => {
            const state = get();
            if (!state.currentWorkflow) return;

            state.pushHistory();
            set({
              currentWorkflow: {
                ...state.currentWorkflow,
                ...updates,
                updated_at: new Date().toISOString(),
              },
              isDirty: true,
            }, false, 'updateWorkflowMetadata');
          },

          // Node operations
          addNode: (type, position) => {
            const state = get();
            if (!state.currentWorkflow) return '';

            state.pushHistory();
            const id = generateId('step');
            const newStep: StepDefinition = {
              id,
              name: getDefaultStepName(type),
              step_type: type,
              config: getDefaultStepConfig(type),
              next_steps: [],
              position,
            };

            set({
              currentWorkflow: {
                ...state.currentWorkflow,
                steps: [...state.currentWorkflow.steps, newStep],
                updated_at: new Date().toISOString(),
              },
              isDirty: true,
            }, false, 'addNode');

            return id;
          },

          updateNode: (id, updates) => {
            const state = get();
            if (!state.currentWorkflow) return;

            state.pushHistory();
            set({
              currentWorkflow: {
                ...state.currentWorkflow,
                steps: state.currentWorkflow.steps.map(step =>
                  step.id === id ? { ...step, ...updates } : step
                ),
                updated_at: new Date().toISOString(),
              },
              isDirty: true,
            }, false, 'updateNode');
          },

          deleteNode: (id) => {
            const state = get();
            if (!state.currentWorkflow) return;

            state.pushHistory();
            set({
              currentWorkflow: {
                ...state.currentWorkflow,
                steps: state.currentWorkflow.steps.filter(step => step.id !== id),
                transitions: state.currentWorkflow.transitions.filter(
                  t => t.from_step !== id && t.to_step !== id
                ),
                updated_at: new Date().toISOString(),
              },
              isDirty: true,
              configPanel: state.configPanel.selectedNodeId === id
                ? { ...initialConfigPanelState }
                : state.configPanel,
              canvas: {
                ...state.canvas,
                selectedNodeIds: new Set(
                  Array.from(state.canvas.selectedNodeIds).filter(nodeId => nodeId !== id)
                ),
              },
            }, false, 'deleteNode');
          },

          duplicateNode: (id) => {
            const state = get();
            if (!state.currentWorkflow) return null;

            const step = state.currentWorkflow.steps.find(s => s.id === id);
            if (!step) return null;

            state.pushHistory();
            const newId = generateId('step');
            const newStep: StepDefinition = {
              ...JSON.parse(JSON.stringify(step)),
              id: newId,
              name: `${step.name} (copy)`,
              position: step.position ? {
                x: step.position.x + 50,
                y: step.position.y + 50,
              } : undefined,
              next_steps: [],
            };

            set({
              currentWorkflow: {
                ...state.currentWorkflow,
                steps: [...state.currentWorkflow.steps, newStep],
                updated_at: new Date().toISOString(),
              },
              isDirty: true,
            }, false, 'duplicateNode');

            return newId;
          },

          // Edge operations
          addEdge: (fromId, toId, condition) => {
            const state = get();
            if (!state.currentWorkflow) return '';

            // Check if edge already exists
            const exists = state.currentWorkflow.transitions.some(
              t => t.from_step === fromId && t.to_step === toId
            );
            if (exists) return '';

            state.pushHistory();
            const id = generateId('edge');
            const newTransition: TransitionRule = {
              id,
              from_step: fromId,
              to_step: toId,
              condition,
            };

            // Also update next_steps on the source node
            const updatedSteps = state.currentWorkflow.steps.map(step => {
              if (step.id === fromId && !step.next_steps.includes(toId)) {
                return { ...step, next_steps: [...step.next_steps, toId] };
              }
              return step;
            });

            set({
              currentWorkflow: {
                ...state.currentWorkflow,
                steps: updatedSteps,
                transitions: [...state.currentWorkflow.transitions, newTransition],
                updated_at: new Date().toISOString(),
              },
              isDirty: true,
            }, false, 'addEdge');

            return id;
          },

          updateEdge: (id, updates) => {
            const state = get();
            if (!state.currentWorkflow) return;

            state.pushHistory();
            set({
              currentWorkflow: {
                ...state.currentWorkflow,
                transitions: state.currentWorkflow.transitions.map(t =>
                  t.id === id ? { ...t, ...updates } : t
                ),
                updated_at: new Date().toISOString(),
              },
              isDirty: true,
            }, false, 'updateEdge');
          },

          deleteEdge: (id) => {
            const state = get();
            if (!state.currentWorkflow) return;

            const edge = state.currentWorkflow.transitions.find(t => t.id === id);
            if (!edge) return;

            state.pushHistory();
            // Also remove from next_steps
            const updatedSteps = state.currentWorkflow.steps.map(step => {
              if (step.id === edge.from_step) {
                return {
                  ...step,
                  next_steps: step.next_steps.filter(s => s !== edge.to_step),
                };
              }
              return step;
            });

            set({
              currentWorkflow: {
                ...state.currentWorkflow,
                steps: updatedSteps,
                transitions: state.currentWorkflow.transitions.filter(t => t.id !== id),
                updated_at: new Date().toISOString(),
              },
              isDirty: true,
              canvas: {
                ...state.canvas,
                selectedEdgeIds: new Set(
                  Array.from(state.canvas.selectedEdgeIds).filter(edgeId => edgeId !== id)
                ),
              },
            }, false, 'deleteEdge');
          },

          // Canvas operations
          setZoom: (zoom) => set(
            (state) => ({ canvas: { ...state.canvas, zoom: Math.max(0.25, Math.min(2, zoom)) } }),
            false,
            'setZoom'
          ),

          setPan: (x, y) => set(
            (state) => ({ canvas: { ...state.canvas, panX: x, panY: y } }),
            false,
            'setPan'
          ),

          selectNodes: (ids) => set(
            (state) => ({
              canvas: { ...state.canvas, selectedNodeIds: new Set(ids), selectedEdgeIds: new Set() },
            }),
            false,
            'selectNodes'
          ),

          selectEdges: (ids) => set(
            (state) => ({
              canvas: { ...state.canvas, selectedEdgeIds: new Set(ids), selectedNodeIds: new Set() },
            }),
            false,
            'selectEdges'
          ),

          clearSelection: () => set(
            (state) => ({
              canvas: {
                ...state.canvas,
                selectedNodeIds: new Set(),
                selectedEdgeIds: new Set(),
              },
            }),
            false,
            'clearSelection'
          ),

          setDragging: (isDragging, nodeId = null) => set(
            (state) => ({
              canvas: { ...state.canvas, isDragging, draggedNodeId: nodeId },
            }),
            false,
            'setDragging'
          ),

          // Node palette
          setSearchQuery: (query) => set(
            (state) => ({ nodePalette: { ...state.nodePalette, searchQuery: query } }),
            false,
            'setSearchQuery'
          ),

          setSelectedCategory: (category) => set(
            (state) => ({ nodePalette: { ...state.nodePalette, selectedCategory: category } }),
            false,
            'setSelectedCategory'
          ),

          toggleCategory: (category) => set(
            (state) => {
              const expanded = new Set(state.nodePalette.expandedCategories);
              if (expanded.has(category)) {
                expanded.delete(category);
              } else {
                expanded.add(category);
              }
              return { nodePalette: { ...state.nodePalette, expandedCategories: expanded } };
            },
            false,
            'toggleCategory'
          ),

          // Config panel
          openConfigPanel: (nodeId) => set(
            { configPanel: { isOpen: true, selectedNodeId: nodeId, pendingChanges: null } },
            false,
            'openConfigPanel'
          ),

          closeConfigPanel: () => set(
            { configPanel: { ...initialConfigPanelState } },
            false,
            'closeConfigPanel'
          ),

          setPendingChanges: (changes) => set(
            (state) => ({ configPanel: { ...state.configPanel, pendingChanges: changes } }),
            false,
            'setPendingChanges'
          ),

          applyPendingChanges: () => {
            const state = get();
            if (!state.configPanel.selectedNodeId || !state.configPanel.pendingChanges) return;

            state.updateNode(state.configPanel.selectedNodeId, state.configPanel.pendingChanges);
            set(
              (s) => ({ configPanel: { ...s.configPanel, pendingChanges: null } }),
              false,
              'applyPendingChanges'
            );
          },

          // Validation
          validate: () => {
            const state = get();
            if (!state.currentWorkflow) return ['No workflow loaded'];

            const errors: string[] = [];
            const workflow = state.currentWorkflow;

            // Check for empty workflow
            if (workflow.steps.length === 0) {
              errors.push('Workflow has no steps');
            }

            // Check for cycles
            const visited = new Set<string>();
            const recursionStack = new Set<string>();

            function hasCycle(stepId: string): boolean {
              if (recursionStack.has(stepId)) return true;
              if (visited.has(stepId)) return false;

              visited.add(stepId);
              recursionStack.add(stepId);

              const step = workflow.steps.find(s => s.id === stepId);
              if (step) {
                for (const nextId of step.next_steps) {
                  if (hasCycle(nextId)) return true;
                }
              }

              recursionStack.delete(stepId);
              return false;
            }

            for (const step of workflow.steps) {
              if (!visited.has(step.id) && hasCycle(step.id)) {
                errors.push('Workflow contains cycles');
                break;
              }
            }

            // Check for orphan steps (no incoming or outgoing edges)
            const hasIncoming = new Set<string>();
            const hasOutgoing = new Set<string>();

            for (const step of workflow.steps) {
              if (step.next_steps.length > 0) {
                hasOutgoing.add(step.id);
                step.next_steps.forEach(id => hasIncoming.add(id));
              }
            }

            // First step shouldn't have incoming edges
            const potentialStarts = workflow.steps.filter(s => !hasIncoming.has(s.id));
            if (potentialStarts.length === 0 && workflow.steps.length > 0) {
              errors.push('No start step found (all steps have incoming edges)');
            } else if (potentialStarts.length > 1) {
              errors.push(`Multiple potential start steps: ${potentialStarts.map(s => s.name).join(', ')}`);
            }

            // Check step configurations
            for (const step of workflow.steps) {
              if (!step.name || step.name.trim() === '') {
                errors.push(`Step ${step.id} has no name`);
              }

              // Type-specific validation
              if (step.step_type === 'agent' && !step.config.agent_type) {
                errors.push(`Agent step "${step.name}" has no agent type specified`);
              }

              if (step.step_type === 'debate' && (!step.config.agents || (step.config.agents as string[]).length < 2)) {
                errors.push(`Debate step "${step.name}" needs at least 2 agents`);
              }
            }

            set({ validationErrors: errors }, false, 'validate');
            return errors;
          },

          // Execution preview
          openExecutionPreview: () => set(
            (state) => ({ executionPreview: { ...state.executionPreview, isOpen: true } }),
            false,
            'openExecutionPreview'
          ),

          closeExecutionPreview: () => set(
            { executionPreview: { ...initialPreviewState } },
            false,
            'closeExecutionPreview'
          ),

          setSimulationResult: (result) => set(
            (state) => ({ executionPreview: { ...state.executionPreview, result, isRunning: false } }),
            false,
            'setSimulationResult'
          ),

          setSimulationRunning: (running) => set(
            (state) => ({ executionPreview: { ...state.executionPreview, isRunning: running } }),
            false,
            'setSimulationRunning'
          ),

          // Workflow list
          setWorkflows: (workflows) => set({ workflows }, false, 'setWorkflows'),
          setTemplates: (templates) => set({ templates }, false, 'setTemplates'),

          // Loading states
          setLoading: (loading) => set({ isLoading: loading }, false, 'setLoading'),
          setSaving: (saving) => set({ isSaving: saving }, false, 'setSaving'),
          setSaveError: (error) => set({ saveError: error }, false, 'setSaveError'),
          setLoadError: (error) => set({ loadError: error }, false, 'setLoadError'),

          // Undo/redo
          pushHistory: () => {
            const state = get();
            if (!state.currentWorkflow) return;

            const newHistory = state._history.slice(0, state._historyIndex + 1);
            newHistory.push(JSON.parse(JSON.stringify(state.currentWorkflow)));

            // Trim history if too long
            if (newHistory.length > MAX_HISTORY) {
              newHistory.shift();
            }

            set({
              _history: newHistory,
              _historyIndex: newHistory.length - 1,
            }, false, 'pushHistory');
          },

          undo: () => {
            const state = get();
            if (state._historyIndex <= 0) return;

            const newIndex = state._historyIndex - 1;
            const workflow = JSON.parse(JSON.stringify(state._history[newIndex]));

            set({
              currentWorkflow: workflow,
              _historyIndex: newIndex,
              isDirty: true,
            }, false, 'undo');
          },

          redo: () => {
            const state = get();
            if (state._historyIndex >= state._history.length - 1) return;

            const newIndex = state._historyIndex + 1;
            const workflow = JSON.parse(JSON.stringify(state._history[newIndex]));

            set({
              currentWorkflow: workflow,
              _historyIndex: newIndex,
              isDirty: true,
            }, false, 'redo');
          },

          // Reset
          resetBuilder: () => set({
            currentWorkflow: null,
            originalWorkflow: null,
            canvas: { ...initialCanvasState, selectedNodeIds: new Set(), selectedEdgeIds: new Set() },
            nodePalette: { ...initialPaletteState, expandedCategories: new Set(['agents', 'control']) },
            configPanel: { ...initialConfigPanelState },
            isDirty: false,
            isSaving: false,
            isLoading: false,
            saveError: null,
            loadError: null,
            validationErrors: [],
            executionPreview: { ...initialPreviewState },
            _history: [],
            _historyIndex: -1,
          }, false, 'resetBuilder'),

          markClean: () => set({ isDirty: false }, false, 'markClean'),
        }),
        {
          name: 'workflow-builder-storage',
          partialize: (state) => ({
            // Only persist certain state
            nodePalette: {
              expandedCategories: Array.from(state.nodePalette.expandedCategories),
            },
          }),
        }
      )
    ),
    { name: 'workflow-builder-store' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectCurrentWorkflow = (state: WorkflowBuilderStore) => state.currentWorkflow;
export const selectCanvas = (state: WorkflowBuilderStore) => state.canvas;
export const selectConfigPanel = (state: WorkflowBuilderStore) => state.configPanel;
export const selectNodePalette = (state: WorkflowBuilderStore) => state.nodePalette;
export const selectIsDirty = (state: WorkflowBuilderStore) => state.isDirty;
export const selectIsSaving = (state: WorkflowBuilderStore) => state.isSaving;
export const selectIsLoading = (state: WorkflowBuilderStore) => state.isLoading;
export const selectValidationErrors = (state: WorkflowBuilderStore) => state.validationErrors;
export const selectWorkflows = (state: WorkflowBuilderStore) => state.workflows;
export const selectTemplates = (state: WorkflowBuilderStore) => state.templates;
export const selectExecutionPreview = (state: WorkflowBuilderStore) => state.executionPreview;
export const selectCanUndo = (state: WorkflowBuilderStore) => state._historyIndex > 0;
export const selectCanRedo = (state: WorkflowBuilderStore) => state._historyIndex < state._history.length - 1;
