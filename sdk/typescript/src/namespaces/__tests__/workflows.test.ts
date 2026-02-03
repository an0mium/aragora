/**
 * Workflows Namespace Tests
 *
 * Comprehensive tests for the workflows namespace API including:
 * - Workflow CRUD operations
 * - Workflow execution
 * - Templates and patterns
 * - Executions and approvals
 * - SME workflows
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { WorkflowsAPI } from '../workflows';

interface MockClient {
  listWorkflows: Mock;
  getWorkflow: Mock;
  createWorkflow: Mock;
  updateWorkflow: Mock;
  deleteWorkflow: Mock;
  executeWorkflow: Mock;
  listWorkflowTemplates: Mock;
  getWorkflowTemplate: Mock;
  runWorkflowTemplate: Mock;
  listWorkflowCategories: Mock;
  listWorkflowPatterns: Mock;
  getWorkflowTemplatePackage: Mock;
  listWorkflowExecutions: Mock;
  getWorkflowExecution: Mock;
  getWorkflowStatus: Mock;
  getWorkflowVersions: Mock;
  simulateWorkflow: Mock;
  listWorkflowApprovals: Mock;
  resolveWorkflowApproval: Mock;
  restoreWorkflowVersion: Mock;
  deleteWorkflowExecution: Mock;
  listPatternTemplates: Mock;
  getPatternTemplate: Mock;
  instantiatePattern: Mock;
  listRecommendedTemplates: Mock;
  listSMEWorkflows: Mock;
  getSMEWorkflow: Mock;
  executeSMEWorkflow: Mock;
}

describe('WorkflowsAPI Namespace', () => {
  let api: WorkflowsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      listWorkflows: vi.fn(),
      getWorkflow: vi.fn(),
      createWorkflow: vi.fn(),
      updateWorkflow: vi.fn(),
      deleteWorkflow: vi.fn(),
      executeWorkflow: vi.fn(),
      listWorkflowTemplates: vi.fn(),
      getWorkflowTemplate: vi.fn(),
      runWorkflowTemplate: vi.fn(),
      listWorkflowCategories: vi.fn(),
      listWorkflowPatterns: vi.fn(),
      getWorkflowTemplatePackage: vi.fn(),
      listWorkflowExecutions: vi.fn(),
      getWorkflowExecution: vi.fn(),
      getWorkflowStatus: vi.fn(),
      getWorkflowVersions: vi.fn(),
      simulateWorkflow: vi.fn(),
      listWorkflowApprovals: vi.fn(),
      resolveWorkflowApproval: vi.fn(),
      restoreWorkflowVersion: vi.fn(),
      deleteWorkflowExecution: vi.fn(),
      listPatternTemplates: vi.fn(),
      getPatternTemplate: vi.fn(),
      instantiatePattern: vi.fn(),
      listRecommendedTemplates: vi.fn(),
      listSMEWorkflows: vi.fn(),
      getSMEWorkflow: vi.fn(),
      executeSMEWorkflow: vi.fn(),
    };
    api = new WorkflowsAPI(mockClient as any);
  });

  // ===========================================================================
  // Workflow CRUD
  // ===========================================================================

  describe('Workflow CRUD', () => {
    it('should list workflows', async () => {
      const mockWorkflows = {
        workflows: [
          { id: 'wf1', name: 'Approval Workflow', status: 'active' },
          { id: 'wf2', name: 'Onboarding', status: 'active' },
        ],
      };
      mockClient.listWorkflows.mockResolvedValue(mockWorkflows);

      const result = await api.list();

      expect(mockClient.listWorkflows).toHaveBeenCalled();
      expect(result.workflows).toHaveLength(2);
    });

    it('should list workflows with pagination', async () => {
      const mockWorkflows = { workflows: [{ id: 'wf3' }] };
      mockClient.listWorkflows.mockResolvedValue(mockWorkflows);

      const result = await api.list({ limit: 10, offset: 20 });

      expect(mockClient.listWorkflows).toHaveBeenCalledWith({ limit: 10, offset: 20 });
      expect(result.workflows).toHaveLength(1);
    });

    it('should get workflow by ID', async () => {
      const mockWorkflow = {
        id: 'wf1',
        name: 'Approval Workflow',
        description: 'Multi-stage approval process',
        nodes: [
          { id: 'n1', type: 'trigger' },
          { id: 'n2', type: 'approval' },
          { id: 'n3', type: 'action' },
        ],
        status: 'active',
        version: 3,
      };
      mockClient.getWorkflow.mockResolvedValue(mockWorkflow);

      const result = await api.get('wf1');

      expect(mockClient.getWorkflow).toHaveBeenCalledWith('wf1');
      expect(result.name).toBe('Approval Workflow');
      expect(result.nodes).toHaveLength(3);
    });

    it('should create workflow', async () => {
      const newWorkflow = {
        name: 'New Workflow',
        description: 'Test workflow',
        nodes: [{ id: 'n1', type: 'trigger' }],
      };
      const mockCreated = { id: 'wf_new', ...newWorkflow, status: 'draft', version: 1 };
      mockClient.createWorkflow.mockResolvedValue(mockCreated);

      const result = await api.create(newWorkflow);

      expect(mockClient.createWorkflow).toHaveBeenCalledWith(newWorkflow);
      expect(result.id).toBe('wf_new');
      expect(result.status).toBe('draft');
    });

    it('should update workflow', async () => {
      const updates = { name: 'Updated Workflow', status: 'active' };
      const mockUpdated = { id: 'wf1', ...updates, version: 4 };
      mockClient.updateWorkflow.mockResolvedValue(mockUpdated);

      const result = await api.update('wf1', updates);

      expect(mockClient.updateWorkflow).toHaveBeenCalledWith('wf1', updates);
      expect(result.name).toBe('Updated Workflow');
    });

    it('should delete workflow', async () => {
      mockClient.deleteWorkflow.mockResolvedValue(undefined);

      await api.delete('wf1');

      expect(mockClient.deleteWorkflow).toHaveBeenCalledWith('wf1');
    });
  });

  // ===========================================================================
  // Workflow Execution
  // ===========================================================================

  describe('Workflow Execution', () => {
    it('should execute workflow', async () => {
      const mockExecution = { execution_id: 'exec_123' };
      mockClient.executeWorkflow.mockResolvedValue(mockExecution);

      const result = await api.execute('wf1', { input_param: 'value' });

      expect(mockClient.executeWorkflow).toHaveBeenCalledWith('wf1', { input_param: 'value' });
      expect(result.execution_id).toBe('exec_123');
    });

    it('should execute workflow without inputs', async () => {
      const mockExecution = { execution_id: 'exec_124' };
      mockClient.executeWorkflow.mockResolvedValue(mockExecution);

      const result = await api.execute('wf1');

      expect(mockClient.executeWorkflow).toHaveBeenCalledWith('wf1', undefined);
      expect(result.execution_id).toBe('exec_124');
    });

    it('should list executions', async () => {
      const mockExecutions = {
        executions: [
          { id: 'exec_1', workflow_id: 'wf1', status: 'completed' },
          { id: 'exec_2', workflow_id: 'wf1', status: 'running' },
        ],
      };
      mockClient.listWorkflowExecutions.mockResolvedValue(mockExecutions);

      const result = await api.listExecutions({ workflow_id: 'wf1' });

      expect(mockClient.listWorkflowExecutions).toHaveBeenCalledWith({ workflow_id: 'wf1' });
      expect(result.executions).toHaveLength(2);
    });

    it('should get execution by ID', async () => {
      const mockExecution = {
        id: 'exec_1',
        workflow_id: 'wf1',
        status: 'completed',
        started_at: '2024-01-20T10:00:00Z',
        completed_at: '2024-01-20T10:05:00Z',
        outputs: { result: 'approved' },
      };
      mockClient.getWorkflowExecution.mockResolvedValue(mockExecution);

      const result = await api.getExecution('exec_1');

      expect(mockClient.getWorkflowExecution).toHaveBeenCalledWith('exec_1');
      expect(result.status).toBe('completed');
    });

    it('should get workflow status', async () => {
      const mockStatus = {
        id: 'exec_latest',
        workflow_id: 'wf1',
        status: 'running',
        current_node: 'n2',
      };
      mockClient.getWorkflowStatus.mockResolvedValue(mockStatus);

      const result = await api.getStatus('wf1');

      expect(mockClient.getWorkflowStatus).toHaveBeenCalledWith('wf1');
      expect(result.status).toBe('running');
    });

    it('should simulate workflow', async () => {
      const mockSimulation = {
        would_execute: true,
        estimated_duration: '5m',
        nodes_to_execute: ['n1', 'n2', 'n3'],
        potential_issues: [],
      };
      mockClient.simulateWorkflow.mockResolvedValue(mockSimulation);

      const result = await api.simulate('wf1', { test_input: 'value' });

      expect(mockClient.simulateWorkflow).toHaveBeenCalledWith('wf1', { test_input: 'value' });
      expect(result.would_execute).toBe(true);
    });

    it('should delete execution', async () => {
      mockClient.deleteWorkflowExecution.mockResolvedValue(undefined);

      await api.deleteExecution('exec_1');

      expect(mockClient.deleteWorkflowExecution).toHaveBeenCalledWith('exec_1');
    });
  });

  // ===========================================================================
  // Templates
  // ===========================================================================

  describe('Templates', () => {
    it('should list templates', async () => {
      const mockTemplates = {
        templates: [
          { id: 'tpl1', name: 'Approval', category: 'governance' },
          { id: 'tpl2', name: 'Onboarding', category: 'hr' },
        ],
      };
      mockClient.listWorkflowTemplates.mockResolvedValue(mockTemplates);

      const result = await api.listTemplates();

      expect(mockClient.listWorkflowTemplates).toHaveBeenCalled();
      expect(result.templates).toHaveLength(2);
    });

    it('should list templates by category', async () => {
      const mockTemplates = {
        templates: [{ id: 'tpl1', name: 'Approval', category: 'governance' }],
      };
      mockClient.listWorkflowTemplates.mockResolvedValue(mockTemplates);

      const result = await api.listTemplates({ category: 'governance' });

      expect(mockClient.listWorkflowTemplates).toHaveBeenCalledWith({ category: 'governance' });
      expect(result.templates).toHaveLength(1);
    });

    it('should get template by ID', async () => {
      const mockTemplate = {
        id: 'tpl1',
        name: 'Multi-Stage Approval',
        description: 'Configurable approval workflow',
        inputs: [{ name: 'approvers', type: 'array' }],
      };
      mockClient.getWorkflowTemplate.mockResolvedValue(mockTemplate);

      const result = await api.getTemplate('tpl1');

      expect(mockClient.getWorkflowTemplate).toHaveBeenCalledWith('tpl1');
      expect(result.name).toBe('Multi-Stage Approval');
    });

    it('should run template', async () => {
      const mockResult = {
        execution_id: 'exec_from_tpl',
        workflow_id: 'wf_from_tpl',
      };
      mockClient.runWorkflowTemplate.mockResolvedValue(mockResult);

      const result = await api.runTemplate('tpl1', {
        inputs: { approvers: ['user1', 'user2'] },
      });

      expect(mockClient.runWorkflowTemplate).toHaveBeenCalledWith('tpl1', {
        inputs: { approvers: ['user1', 'user2'] },
      });
      expect(result.execution_id).toBe('exec_from_tpl');
    });

    it('should get template package', async () => {
      const mockPackage = {
        template: { id: 'tpl1', name: 'Approval' },
        examples: [{ name: 'Basic', config: {} }],
        documentation: '# Approval Workflow\n...',
      };
      mockClient.getWorkflowTemplatePackage.mockResolvedValue(mockPackage);

      const result = await api.getTemplatePackage('tpl1', { include_examples: true });

      expect(mockClient.getWorkflowTemplatePackage).toHaveBeenCalledWith('tpl1', {
        include_examples: true,
      });
      expect(result.examples).toHaveLength(1);
    });

    it('should list recommended templates', async () => {
      const mockTemplates = {
        templates: [{ id: 'tpl_rec1', name: 'Quick Start' }],
      };
      mockClient.listRecommendedTemplates.mockResolvedValue(mockTemplates);

      const result = await api.listRecommended();

      expect(mockClient.listRecommendedTemplates).toHaveBeenCalled();
      expect(result.templates).toHaveLength(1);
    });
  });

  // ===========================================================================
  // Categories and Patterns
  // ===========================================================================

  describe('Categories and Patterns', () => {
    it('should list categories', async () => {
      const mockCategories = { categories: ['governance', 'hr', 'finance', 'operations'] };
      mockClient.listWorkflowCategories.mockResolvedValue(mockCategories);

      const result = await api.listCategories();

      expect(mockClient.listWorkflowCategories).toHaveBeenCalled();
      expect(result.categories).toContain('governance');
    });

    it('should list patterns', async () => {
      const mockPatterns = { patterns: ['sequential', 'parallel', 'conditional', 'loop'] };
      mockClient.listWorkflowPatterns.mockResolvedValue(mockPatterns);

      const result = await api.listPatterns();

      expect(mockClient.listWorkflowPatterns).toHaveBeenCalled();
      expect(result.patterns).toContain('parallel');
    });

    it('should list pattern templates', async () => {
      const mockPatterns = {
        patterns: [
          { id: 'p1', name: 'Fan-Out', description: 'Parallel execution' },
          { id: 'p2', name: 'Pipeline', description: 'Sequential stages' },
        ],
      };
      mockClient.listPatternTemplates.mockResolvedValue(mockPatterns);

      const result = await api.listPatternTemplates();

      expect(mockClient.listPatternTemplates).toHaveBeenCalled();
      expect(result.patterns).toHaveLength(2);
    });

    it('should get pattern template', async () => {
      const mockPattern = {
        id: 'p1',
        name: 'Fan-Out',
        schema: { type: 'object' },
      };
      mockClient.getPatternTemplate.mockResolvedValue(mockPattern);

      const result = await api.getPatternTemplate('p1');

      expect(mockClient.getPatternTemplate).toHaveBeenCalledWith('p1');
      expect(result.name).toBe('Fan-Out');
    });

    it('should instantiate pattern', async () => {
      const mockResult = {
        template_id: 'tpl_from_pattern',
        workflow: { id: 'wf_new', name: 'My Fan-Out' },
      };
      mockClient.instantiatePattern.mockResolvedValue(mockResult);

      const result = await api.instantiatePattern('p1', {
        name: 'My Fan-Out',
        description: 'Custom fan-out workflow',
        agents: ['claude', 'gpt4'],
      });

      expect(mockClient.instantiatePattern).toHaveBeenCalledWith('p1', {
        name: 'My Fan-Out',
        description: 'Custom fan-out workflow',
        agents: ['claude', 'gpt4'],
      });
      expect(result.workflow.name).toBe('My Fan-Out');
    });
  });

  // ===========================================================================
  // Versions
  // ===========================================================================

  describe('Versions', () => {
    it('should get workflow versions', async () => {
      const mockVersions = {
        versions: [
          { version: 3, created_at: '2024-01-20', changes: 'Added approval step' },
          { version: 2, created_at: '2024-01-15', changes: 'Initial release' },
          { version: 1, created_at: '2024-01-10', changes: 'Draft' },
        ],
      };
      mockClient.getWorkflowVersions.mockResolvedValue(mockVersions);

      const result = await api.getVersions('wf1');

      expect(mockClient.getWorkflowVersions).toHaveBeenCalledWith('wf1');
      expect(result.versions).toHaveLength(3);
    });

    it('should restore workflow version', async () => {
      const mockRestored = { id: 'wf1', name: 'Approval', version: 4 };
      mockClient.restoreWorkflowVersion.mockResolvedValue(mockRestored);

      const result = await api.restoreVersion('wf1', 2);

      expect(mockClient.restoreWorkflowVersion).toHaveBeenCalledWith('wf1', 2);
      expect(result.version).toBe(4);
    });
  });

  // ===========================================================================
  // Approvals
  // ===========================================================================

  describe('Approvals', () => {
    it('should list pending approvals', async () => {
      const mockApprovals = {
        approvals: [
          { id: 'apr1', workflow_id: 'wf1', status: 'pending', requested_by: 'user1' },
          { id: 'apr2', workflow_id: 'wf2', status: 'pending', requested_by: 'user2' },
        ],
      };
      mockClient.listWorkflowApprovals.mockResolvedValue(mockApprovals);

      const result = await api.listApprovals({ status: 'pending' });

      expect(mockClient.listWorkflowApprovals).toHaveBeenCalledWith({ status: 'pending' });
      expect(result.approvals).toHaveLength(2);
    });

    it('should approve request', async () => {
      const mockResolved = {
        id: 'apr1',
        status: 'approved',
        resolved_by: 'admin1',
        resolved_at: '2024-01-20T10:00:00Z',
      };
      mockClient.resolveWorkflowApproval.mockResolvedValue(mockResolved);

      const result = await api.resolveApproval('apr1', { approved: true, comment: 'LGTM' });

      expect(mockClient.resolveWorkflowApproval).toHaveBeenCalledWith('apr1', {
        approved: true,
        comment: 'LGTM',
      });
      expect(result.status).toBe('approved');
    });

    it('should reject request', async () => {
      const mockResolved = {
        id: 'apr1',
        status: 'rejected',
        resolved_by: 'admin1',
      };
      mockClient.resolveWorkflowApproval.mockResolvedValue(mockResolved);

      const result = await api.resolveApproval('apr1', {
        approved: false,
        comment: 'Needs more detail',
      });

      expect(result.status).toBe('rejected');
    });
  });

  // ===========================================================================
  // SME Workflows
  // ===========================================================================

  describe('SME Workflows', () => {
    it('should list SME workflows', async () => {
      const mockWorkflows = {
        workflows: [
          { id: 'sme1', name: 'Invoice Approval', industry: 'finance' },
          { id: 'sme2', name: 'Customer Onboarding', industry: 'retail' },
        ],
      };
      mockClient.listSMEWorkflows.mockResolvedValue(mockWorkflows);

      const result = await api.listSME();

      expect(mockClient.listSMEWorkflows).toHaveBeenCalled();
      expect(result.workflows).toHaveLength(2);
    });

    it('should list SME workflows by industry', async () => {
      const mockWorkflows = {
        workflows: [{ id: 'sme1', name: 'Invoice Approval', industry: 'finance' }],
      };
      mockClient.listSMEWorkflows.mockResolvedValue(mockWorkflows);

      const result = await api.listSME({ industry: 'finance' });

      expect(mockClient.listSMEWorkflows).toHaveBeenCalledWith({ industry: 'finance' });
      expect(result.workflows).toHaveLength(1);
    });

    it('should get SME workflow', async () => {
      const mockWorkflow = {
        id: 'sme1',
        name: 'Invoice Approval',
        description: 'Automated invoice processing',
        estimated_savings: '$5,000/month',
      };
      mockClient.getSMEWorkflow.mockResolvedValue(mockWorkflow);

      const result = await api.getSME('sme1');

      expect(mockClient.getSMEWorkflow).toHaveBeenCalledWith('sme1');
      expect(result.name).toBe('Invoice Approval');
    });

    it('should execute SME workflow', async () => {
      const mockExecution = { execution_id: 'exec_sme1' };
      mockClient.executeSMEWorkflow.mockResolvedValue(mockExecution);

      const result = await api.executeSME('sme1', {
        inputs: { invoice_id: 'inv_123' },
        execute: true,
        tenant_id: 'tenant_456',
      });

      expect(mockClient.executeSMEWorkflow).toHaveBeenCalledWith('sme1', {
        inputs: { invoice_id: 'inv_123' },
        execute: true,
        tenant_id: 'tenant_456',
      });
      expect(result.execution_id).toBe('exec_sme1');
    });
  });
});
