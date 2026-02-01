/**
 * Workflows Namespace Tests
 *
 * Tests for the workflows namespace API including:
 * - Core CRUD operations (create, get, list, update, delete)
 * - Workflow execution and status tracking
 * - Templates and patterns
 * - Approvals and versioning
 * - SME-specific workflows
 * - Simulation and validation
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AragoraClient, createClient } from '../client';
import { AragoraError } from '../types';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('Workflows Namespace', () => {
  let client: AragoraClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = createClient({
      baseUrl: 'https://api.aragora.ai',
      apiKey: 'test-api-key',
      retryEnabled: false,
    });
  });

  // ===========================================================================
  // Core CRUD Operations
  // ===========================================================================

  describe('Core CRUD Operations', () => {
    it('should list workflows', async () => {
      const mockWorkflows = {
        workflows: [
          { id: 'wf-1', name: 'Data Processing', status: 'active', created_at: '2024-01-01T00:00:00Z' },
          { id: 'wf-2', name: 'Code Review', status: 'active', created_at: '2024-01-02T00:00:00Z' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockWorkflows)),
      });

      const result = await client.workflows.list();

      expect(result.workflows).toHaveLength(2);
      expect(result.workflows[0].name).toBe('Data Processing');
    });

    it('should list workflows with pagination', async () => {
      const mockWorkflows = {
        workflows: [
          { id: 'wf-1', name: 'Workflow 1', status: 'active', created_at: '2024-01-01T00:00:00Z' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockWorkflows)),
      });

      const result = await client.workflows.list({ limit: 10, offset: 0 });

      expect(result.workflows).toHaveLength(1);
    });

    it('should get a workflow by ID', async () => {
      const mockWorkflow = {
        id: 'wf-123',
        name: 'Data Processing Pipeline',
        description: 'Processes incoming data',
        status: 'active',
        nodes: [
          { id: 'n1', type: 'input', name: 'Data Input' },
          { id: 'n2', type: 'transform', name: 'Transform' },
          { id: 'n3', type: 'output', name: 'Data Output' },
        ],
        edges: [
          { source: 'n1', target: 'n2' },
          { source: 'n2', target: 'n3' },
        ],
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-05T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockWorkflow)),
      });

      const result = await client.workflows.get('wf-123');

      expect(result.id).toBe('wf-123');
      expect(result.name).toBe('Data Processing Pipeline');
      expect(result.nodes).toHaveLength(3);
    });

    it('should create a workflow', async () => {
      const mockCreated = {
        id: 'wf-new',
        name: 'New Workflow',
        description: 'A new workflow',
        status: 'draft',
        nodes: [],
        edges: [],
        created_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockCreated)),
      });

      const result = await client.workflows.create({
        name: 'New Workflow',
        description: 'A new workflow',
      });

      expect(result.id).toBe('wf-new');
      expect(result.status).toBe('draft');
    });

    it('should update a workflow', async () => {
      const mockUpdated = {
        id: 'wf-123',
        name: 'Updated Workflow',
        description: 'Updated description',
        status: 'active',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-10T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockUpdated)),
      });

      const result = await client.workflows.update('wf-123', {
        name: 'Updated Workflow',
        description: 'Updated description',
      });

      expect(result.name).toBe('Updated Workflow');
    });

    it('should delete a workflow', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(''),
      });

      await expect(client.workflows.delete('wf-123')).resolves.toBeUndefined();
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/workflow'),
        expect.objectContaining({ method: 'DELETE' })
      );
    });
  });

  // ===========================================================================
  // Workflow Execution
  // ===========================================================================

  describe('Workflow Execution', () => {
    it('should execute a workflow', async () => {
      const mockExecution = {
        execution_id: 'exec-123',
        workflow_id: 'wf-123',
        status: 'running',
        started_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExecution)),
      });

      const result = await client.workflows.execute('wf-123', { input_param: 'value' });

      expect(result.execution_id).toBe('exec-123');
    });

    it('should execute a workflow without inputs', async () => {
      const mockExecution = {
        execution_id: 'exec-456',
        status: 'running',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExecution)),
      });

      const result = await client.workflows.execute('wf-123');

      expect(result.execution_id).toBe('exec-456');
    });

    it('should get execution status', async () => {
      const mockExecution = {
        execution_id: 'exec-123',
        workflow_id: 'wf-123',
        status: 'completed',
        started_at: '2024-01-01T00:00:00Z',
        completed_at: '2024-01-01T00:05:00Z',
        outputs: { result: 'success' },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExecution)),
      });

      const result = await client.workflows.getExecution('exec-123');

      expect(result.status).toBe('completed');
      expect(result.outputs?.result).toBe('success');
    });

    it('should list workflow executions', async () => {
      const mockExecutions = {
        executions: [
          { execution_id: 'exec-1', workflow_id: 'wf-123', status: 'completed' },
          { execution_id: 'exec-2', workflow_id: 'wf-123', status: 'running' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExecutions)),
      });

      const result = await client.workflows.listExecutions({ workflow_id: 'wf-123' });

      expect(result.executions).toHaveLength(2);
    });

    it('should filter executions by status', async () => {
      const mockExecutions = {
        executions: [
          { execution_id: 'exec-1', status: 'completed' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExecutions)),
      });

      const result = await client.workflows.listExecutions({ status: 'completed' });

      expect(result.executions).toHaveLength(1);
      expect(result.executions[0].status).toBe('completed');
    });

    it('should get workflow status', async () => {
      const mockStatus = {
        workflow_id: 'wf-123',
        status: 'active',
        last_execution: 'exec-123',
        execution_count: 50,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockStatus)),
      });

      const result = await client.workflows.getStatus('wf-123');

      expect(result.status).toBe('active');
    });

    it('should delete an execution', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(''),
      });

      // deleteExecution may return void or an empty object depending on implementation
      const result = await client.workflows.deleteExecution('exec-123');
      expect(result === undefined || (typeof result === 'object' && Object.keys(result as object).length === 0)).toBe(true);
    });
  });

  // ===========================================================================
  // Workflow Templates
  // ===========================================================================

  describe('Workflow Templates', () => {
    it('should list templates', async () => {
      const mockTemplates = {
        templates: [
          { id: 'tmpl-1', name: 'Code Review', description: 'Automated code review', category: 'development' },
          { id: 'tmpl-2', name: 'Data Analysis', description: 'Analyze datasets', category: 'analytics' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockTemplates)),
      });

      const result = await client.workflows.listTemplates();

      expect(result.templates).toHaveLength(2);
    });

    it('should list templates by category', async () => {
      const mockTemplates = {
        templates: [
          { id: 'tmpl-1', name: 'Code Review', category: 'development' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockTemplates)),
      });

      const result = await client.workflows.listTemplates({ category: 'development' });

      expect(result.templates).toHaveLength(1);
      expect(result.templates[0].category).toBe('development');
    });

    it('should get a template by ID', async () => {
      const mockTemplate = {
        id: 'tmpl-123',
        name: 'Code Review Template',
        description: 'Review code changes',
        category: 'development',
        nodes: [
          { id: 'n1', type: 'input', name: 'Code Input' },
          { id: 'n2', type: 'agent', name: 'Review Agent' },
        ],
        inputs_schema: {
          type: 'object',
          properties: {
            repo: { type: 'string' },
            pr_number: { type: 'number' },
          },
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockTemplate)),
      });

      const result = await client.workflows.getTemplate('tmpl-123');

      expect(result.id).toBe('tmpl-123');
      expect(result.category).toBe('development');
    });

    it('should run a template', async () => {
      const mockResult = {
        execution_id: 'exec-789',
        workflow_id: 'wf-created',
        status: 'running',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockResult)),
      });

      const result = await client.workflows.runTemplate('tmpl-123', {
        inputs: { repo: 'my-repo', pr_number: 42 },
      });

      expect(result.execution_id).toBe('exec-789');
    });

    it('should get template package', async () => {
      const mockPackage = {
        id: 'tmpl-123',
        name: 'Code Review',
        version: '1.0.0',
        files: [
          { path: 'workflow.json', content: '{}' },
          { path: 'README.md', content: '# Code Review' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockPackage)),
      });

      const result = await client.workflows.getTemplatePackage('tmpl-123', { include_examples: true });

      expect(result.version).toBe('1.0.0');
      expect(result.files).toHaveLength(2);
    });

    it('should list recommended templates', async () => {
      const mockRecommended = {
        templates: [
          { id: 'tmpl-1', name: 'Popular Template', category: 'general' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRecommended)),
      });

      const result = await client.workflows.listRecommended();

      expect(result.templates).toHaveLength(1);
    });
  });

  // ===========================================================================
  // Categories and Patterns
  // ===========================================================================

  describe('Categories and Patterns', () => {
    it('should list categories', async () => {
      const mockCategories = {
        categories: ['development', 'analytics', 'operations', 'security', 'compliance'],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockCategories)),
      });

      const result = await client.workflows.listCategories();

      expect(result.categories).toContain('development');
      expect(result.categories).toContain('analytics');
    });

    it('should list patterns', async () => {
      const mockPatterns = {
        patterns: ['sequential', 'parallel', 'conditional', 'loop', 'fan-out'],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockPatterns)),
      });

      const result = await client.workflows.listPatterns();

      expect(result.patterns).toContain('sequential');
      expect(result.patterns).toContain('parallel');
    });

    it('should list pattern templates', async () => {
      const mockPatternTemplates = {
        patterns: [
          { id: 'seq-1', name: 'Sequential Pipeline', description: 'Run steps in sequence' },
          { id: 'par-1', name: 'Parallel Execution', description: 'Run steps in parallel' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockPatternTemplates)),
      });

      const result = await client.workflows.listPatternTemplates();

      expect(result.patterns).toHaveLength(2);
    });

    it('should get a pattern template', async () => {
      const mockPattern = {
        id: 'seq-1',
        name: 'Sequential Pipeline',
        description: 'Run steps in sequence',
        schema: {
          nodes: ['input', 'step1', 'step2', 'output'],
          edges: [['input', 'step1'], ['step1', 'step2'], ['step2', 'output']],
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockPattern)),
      });

      const result = await client.workflows.getPatternTemplate('seq-1');

      expect(result.name).toBe('Sequential Pipeline');
    });

    it('should instantiate a pattern', async () => {
      const mockInstantiated = {
        template_id: 'tmpl-new',
        workflow: {
          id: 'wf-new',
          name: 'My Sequential Workflow',
          status: 'draft',
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockInstantiated)),
      });

      const result = await client.workflows.instantiatePattern('seq-1', {
        name: 'My Sequential Workflow',
        description: 'A sequential workflow',
        category: 'development',
      });

      expect(result.template_id).toBe('tmpl-new');
      expect(result.workflow.name).toBe('My Sequential Workflow');
    });
  });

  // ===========================================================================
  // Versioning
  // ===========================================================================

  describe('Versioning', () => {
    it('should get workflow versions', async () => {
      const mockVersions = {
        versions: [
          { version: 3, created_at: '2024-01-03T00:00:00Z', created_by: 'user-1', message: 'Added logging' },
          { version: 2, created_at: '2024-01-02T00:00:00Z', created_by: 'user-1', message: 'Fixed bug' },
          { version: 1, created_at: '2024-01-01T00:00:00Z', created_by: 'user-1', message: 'Initial version' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockVersions)),
      });

      const result = await client.workflows.getVersions('wf-123');

      expect(result.versions).toHaveLength(3);
      expect(result.versions[0].version).toBe(3);
    });

    it('should restore a specific version', async () => {
      const mockRestored = {
        id: 'wf-123',
        name: 'Workflow',
        version: 2,
        status: 'active',
        restored_from: 1,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRestored)),
      });

      const result = await client.workflows.restoreVersion('wf-123', 1);

      expect(result.version).toBe(2);
    });
  });

  // ===========================================================================
  // Approvals
  // ===========================================================================

  describe('Approvals', () => {
    it('should list pending approvals', async () => {
      const mockApprovals = {
        approvals: [
          {
            id: 'apr-1',
            workflow_id: 'wf-123',
            execution_id: 'exec-123',
            status: 'pending',
            requested_at: '2024-01-01T00:00:00Z',
            requested_by: 'system',
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockApprovals)),
      });

      const result = await client.workflows.listApprovals({ status: 'pending' });

      expect(result.approvals).toHaveLength(1);
      expect(result.approvals[0].status).toBe('pending');
    });

    it('should list approvals for a workflow', async () => {
      const mockApprovals = {
        approvals: [
          { id: 'apr-1', workflow_id: 'wf-123', status: 'approved' },
          { id: 'apr-2', workflow_id: 'wf-123', status: 'pending' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockApprovals)),
      });

      const result = await client.workflows.listApprovals({ workflow_id: 'wf-123' });

      expect(result.approvals).toHaveLength(2);
    });

    it('should approve a workflow', async () => {
      const mockApproved = {
        id: 'apr-1',
        status: 'approved',
        resolved_at: '2024-01-01T00:00:00Z',
        resolved_by: 'user-1',
        comment: 'Looks good!',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockApproved)),
      });

      const result = await client.workflows.resolveApproval('apr-1', {
        approved: true,
        comment: 'Looks good!',
      });

      expect(result.status).toBe('approved');
    });

    it('should reject an approval', async () => {
      const mockRejected = {
        id: 'apr-1',
        status: 'rejected',
        resolved_at: '2024-01-01T00:00:00Z',
        comment: 'Needs more testing',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockRejected)),
      });

      const result = await client.workflows.resolveApproval('apr-1', {
        approved: false,
        comment: 'Needs more testing',
      });

      expect(result.status).toBe('rejected');
    });
  });

  // ===========================================================================
  // Simulation
  // ===========================================================================

  describe('Simulation', () => {
    it('should simulate a workflow', async () => {
      const mockSimulation = {
        workflow_id: 'wf-123',
        success: true,
        steps: [
          { node_id: 'n1', status: 'completed', duration_ms: 100, output: { data: 'processed' } },
          { node_id: 'n2', status: 'completed', duration_ms: 200, output: { result: 'success' } },
        ],
        estimated_duration_ms: 300,
        potential_issues: [],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSimulation)),
      });

      const result = await client.workflows.simulate('wf-123', { test_input: 'value' });

      expect(result.success).toBe(true);
      expect(result.steps).toHaveLength(2);
      expect(result.estimated_duration_ms).toBe(300);
    });

    it('should simulate a workflow without inputs', async () => {
      const mockSimulation = {
        workflow_id: 'wf-123',
        success: true,
        steps: [],
        estimated_duration_ms: 100,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSimulation)),
      });

      const result = await client.workflows.simulate('wf-123');

      expect(result.success).toBe(true);
    });

    it('should detect issues during simulation', async () => {
      const mockSimulation = {
        workflow_id: 'wf-123',
        success: false,
        steps: [
          { node_id: 'n1', status: 'completed' },
          { node_id: 'n2', status: 'failed', error: 'Invalid configuration' },
        ],
        potential_issues: [
          { node_id: 'n2', issue: 'Missing required input', severity: 'error' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSimulation)),
      });

      const result = await client.workflows.simulate('wf-123');

      expect(result.success).toBe(false);
      expect(result.potential_issues).toHaveLength(1);
    });
  });

  // ===========================================================================
  // SME Workflows
  // ===========================================================================

  describe('SME Workflows', () => {
    it('should list SME workflows', async () => {
      const mockSMEWorkflows = {
        workflows: [
          { id: 'sme-1', name: 'Invoice Processing', category: 'finance', industry: 'retail' },
          { id: 'sme-2', name: 'Inventory Management', category: 'operations', industry: 'retail' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSMEWorkflows)),
      });

      const result = await client.workflows.listSME({ industry: 'retail' });

      expect(result.workflows).toHaveLength(2);
    });

    it('should list SME workflows by category', async () => {
      const mockSMEWorkflows = {
        workflows: [
          { id: 'sme-1', name: 'Invoice Processing', category: 'finance' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSMEWorkflows)),
      });

      const result = await client.workflows.listSME({ category: 'finance' });

      expect(result.workflows).toHaveLength(1);
      expect(result.workflows[0].category).toBe('finance');
    });

    it('should get an SME workflow', async () => {
      const mockSMEWorkflow = {
        id: 'sme-123',
        name: 'Invoice Processing',
        description: 'Automated invoice processing for SMEs',
        category: 'finance',
        industry: 'retail',
        estimated_savings: '$5000/month',
        implementation_time: '2 weeks',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockSMEWorkflow)),
      });

      const result = await client.workflows.getSME('sme-123');

      expect(result.id).toBe('sme-123');
      expect(result.name).toBe('Invoice Processing');
    });

    it('should execute an SME workflow', async () => {
      const mockExecution = {
        execution_id: 'sme-exec-123',
        status: 'running',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExecution)),
      });

      const result = await client.workflows.executeSME('sme-123', {
        inputs: { invoice_data: 'test' },
        context: { company: 'Test Corp' },
        execute: true,
      });

      expect(result.execution_id).toBe('sme-exec-123');
    });

    it('should execute SME workflow with tenant_id', async () => {
      const mockExecution = {
        execution_id: 'sme-exec-456',
        status: 'running',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExecution)),
      });

      const result = await client.workflows.executeSME('sme-123', {
        tenant_id: 'tenant-abc',
        execute: true,
      });

      expect(result.execution_id).toBe('sme-exec-456');
    });
  });

  // ===========================================================================
  // Error Handling
  // ===========================================================================

  describe('Error Handling', () => {
    it('should handle 404 for non-existent workflow', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({
          error: 'Workflow not found',
          code: 'NOT_FOUND',
        }),
      });

      await expect(client.workflows.get('nonexistent'))
        .rejects.toThrow('Workflow not found');
    });

    it('should handle validation errors on create', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Name is required',
          code: 'MISSING_FIELD',
          field: 'name',
        }),
      });

      await expect(client.workflows.create({ description: 'No name' }))
        .rejects.toThrow('Name is required');
    });

    it('should handle execution errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({
          error: 'Workflow execution failed',
          code: 'INTERNAL_ERROR',
        }),
      });

      await expect(client.workflows.execute('wf-123'))
        .rejects.toThrow('Workflow execution failed');
    });

    it('should handle permission errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: () => Promise.resolve({
          error: 'Access denied to workflow',
          code: 'FORBIDDEN',
        }),
      });

      await expect(client.workflows.get('wf-private'))
        .rejects.toThrow('Access denied');
    });

    it('should handle quota exceeded', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () => Promise.resolve({
          error: 'Workflow execution quota exceeded',
          code: 'QUOTA_EXCEEDED',
          limit: 100,
        }),
      });

      try {
        await client.workflows.execute('wf-123');
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(AragoraError);
        expect((error as AragoraError).code).toBe('QUOTA_EXCEEDED');
      }
    });
  });

  // ===========================================================================
  // Type Safety Tests
  // ===========================================================================

  describe('Type Safety', () => {
    it('should return properly typed workflow', async () => {
      const mockWorkflow = {
        id: 'wf-123',
        name: 'Test Workflow',
        description: 'A test workflow',
        status: 'active',
        nodes: [],
        edges: [],
        created_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockWorkflow)),
      });

      const result = await client.workflows.get('wf-123');

      expect(typeof result.id).toBe('string');
      expect(typeof result.name).toBe('string');
      expect(typeof result.status).toBe('string');
    });

    it('should return properly typed execution', async () => {
      const mockExecution = {
        execution_id: 'exec-123',
        workflow_id: 'wf-123',
        status: 'completed',
        started_at: '2024-01-01T00:00:00Z',
        completed_at: '2024-01-01T00:05:00Z',
        outputs: { result: 'success' },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockExecution)),
      });

      const result = await client.workflows.getExecution('exec-123');

      expect(typeof result.execution_id).toBe('string');
      expect(typeof result.status).toBe('string');
    });

    it('should return properly typed template', async () => {
      const mockTemplate = {
        id: 'tmpl-123',
        name: 'Test Template',
        description: 'A test template',
        category: 'development',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockTemplate)),
      });

      const result = await client.workflows.getTemplate('tmpl-123');

      expect(typeof result.id).toBe('string');
      expect(typeof result.name).toBe('string');
      expect(typeof result.category).toBe('string');
    });

    it('should return properly typed approval', async () => {
      const mockApproval = {
        id: 'apr-123',
        workflow_id: 'wf-123',
        status: 'approved',
        resolved_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockApproval)),
      });

      const result = await client.workflows.resolveApproval('apr-123', { approved: true });

      expect(typeof result.id).toBe('string');
      expect(typeof result.status).toBe('string');
    });
  });

  // ===========================================================================
  // Edge Cases
  // ===========================================================================

  describe('Edge Cases', () => {
    it('should handle empty workflow list', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ workflows: [] })),
      });

      const result = await client.workflows.list();

      expect(result.workflows).toHaveLength(0);
    });

    it('should handle empty executions list', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ executions: [] })),
      });

      const result = await client.workflows.listExecutions();

      expect(result.executions).toHaveLength(0);
    });

    it('should handle empty templates list', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify({ templates: [] })),
      });

      const result = await client.workflows.listTemplates();

      expect(result.templates).toHaveLength(0);
    });

    it('should handle workflow with complex nodes', async () => {
      const mockWorkflow = {
        id: 'wf-complex',
        name: 'Complex Workflow',
        nodes: [
          { id: 'n1', type: 'input', config: { format: 'json' } },
          { id: 'n2', type: 'transform', config: { script: 'data.map(x => x * 2)' } },
          { id: 'n3', type: 'condition', config: { expression: 'data.length > 0' } },
          { id: 'n4', type: 'output', config: { destination: 's3://bucket' } },
        ],
        edges: [
          { source: 'n1', target: 'n2' },
          { source: 'n2', target: 'n3' },
          { source: 'n3', target: 'n4', condition: 'true' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        text: () => Promise.resolve(JSON.stringify(mockWorkflow)),
      });

      const result = await client.workflows.get('wf-complex');

      expect(result.nodes).toHaveLength(4);
      expect(result.edges).toHaveLength(3);
    });
  });
});
