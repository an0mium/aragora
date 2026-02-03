/**
 * SME Namespace Tests
 *
 * Comprehensive tests for the SME namespace API including:
 * - Workflow listing and execution
 * - Onboarding status and completion
 * - Quick start helpers (invoice, inventory, report, followup)
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { SMEAPI } from '../sme';

interface MockClient {
  listSMEWorkflows: Mock;
  getSMEWorkflow: Mock;
  executeSMEWorkflow: Mock;
  getOnboardingStatus: Mock;
  completeOnboarding: Mock;
}

describe('SMEAPI Namespace', () => {
  let api: SMEAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      listSMEWorkflows: vi.fn(),
      getSMEWorkflow: vi.fn(),
      executeSMEWorkflow: vi.fn(),
      getOnboardingStatus: vi.fn(),
      completeOnboarding: vi.fn(),
    };
    api = new SMEAPI(mockClient as any);
  });

  // ===========================================================================
  // SME Workflows
  // ===========================================================================

  describe('SME Workflows', () => {
    it('should list SME workflows', async () => {
      const mockWorkflows = {
        workflows: [
          { id: 'invoice', name: 'Invoice Generator', category: 'billing' },
          { id: 'followup', name: 'Customer Follow-up', category: 'crm' },
          { id: 'inventory', name: 'Inventory Monitor', category: 'operations' },
          { id: 'report', name: 'Business Reports', category: 'analytics' },
        ],
      };
      mockClient.listSMEWorkflows.mockResolvedValue(mockWorkflows);

      const result = await api.listWorkflows();

      expect(mockClient.listSMEWorkflows).toHaveBeenCalledWith(undefined);
      expect(result.workflows).toHaveLength(4);
    });

    it('should list workflows by category', async () => {
      const mockWorkflows = { workflows: [{ id: 'invoice', category: 'billing' }] };
      mockClient.listSMEWorkflows.mockResolvedValue(mockWorkflows);

      await api.listWorkflows({ category: 'billing' });

      expect(mockClient.listSMEWorkflows).toHaveBeenCalledWith({ category: 'billing' });
    });

    it('should list workflows with pagination', async () => {
      const mockWorkflows = { workflows: [] };
      mockClient.listSMEWorkflows.mockResolvedValue(mockWorkflows);

      await api.listWorkflows({ limit: 10, offset: 20 });

      expect(mockClient.listSMEWorkflows).toHaveBeenCalledWith({ limit: 10, offset: 20 });
    });

    it('should get workflow details', async () => {
      const mockWorkflow = {
        id: 'invoice',
        name: 'Invoice Generator',
        description: 'Generate and send invoices automatically',
        category: 'billing',
        inputs: [
          { name: 'customer_email', type: 'string', required: true },
          { name: 'items', type: 'array', required: true },
        ],
      };
      mockClient.getSMEWorkflow.mockResolvedValue(mockWorkflow);

      const result = await api.getWorkflow('invoice');

      expect(mockClient.getSMEWorkflow).toHaveBeenCalledWith('invoice');
      expect(result.inputs).toHaveLength(2);
    });

    it('should execute workflow', async () => {
      const mockResult = { execution_id: 'exec_123' };
      mockClient.executeSMEWorkflow.mockResolvedValue(mockResult);

      const result = await api.executeWorkflow('invoice', {
        inputs: {
          customer_email: 'customer@example.com',
          items: [{ name: 'Service', price: 100 }],
        },
      });

      expect(mockClient.executeSMEWorkflow).toHaveBeenCalledWith('invoice', {
        inputs: {
          customer_email: 'customer@example.com',
          items: [{ name: 'Service', price: 100 }],
        },
      });
      expect(result.execution_id).toBe('exec_123');
    });

    it('should execute workflow with context', async () => {
      const mockResult = { execution_id: 'exec_124' };
      mockClient.executeSMEWorkflow.mockResolvedValue(mockResult);

      await api.executeWorkflow('followup', {
        inputs: { customer_id: 'cust_123' },
        context: { source: 'crm' },
        tenant_id: 't_123',
      });

      expect(mockClient.executeSMEWorkflow).toHaveBeenCalledWith('followup', {
        inputs: { customer_id: 'cust_123' },
        context: { source: 'crm' },
        tenant_id: 't_123',
      });
    });
  });

  // ===========================================================================
  // Onboarding
  // ===========================================================================

  describe('Onboarding', () => {
    it('should get onboarding status', async () => {
      const mockStatus = {
        completed: false,
        current_step: 2,
        total_steps: 5,
        steps: [
          { id: 'account', name: 'Create Account', completed: true },
          { id: 'organization', name: 'Set Up Organization', completed: true },
          { id: 'first_debate', name: 'Run First Debate', completed: false },
          { id: 'invite_team', name: 'Invite Team', completed: false },
          { id: 'explore', name: 'Explore Features', completed: false },
        ],
      };
      mockClient.getOnboardingStatus.mockResolvedValue(mockStatus);

      const result = await api.getOnboardingStatus();

      expect(mockClient.getOnboardingStatus).toHaveBeenCalled();
      expect(result.current_step).toBe(2);
      expect(result.completed).toBe(false);
    });

    it('should complete onboarding', async () => {
      const mockResult = {
        completed: true,
        organization_id: 'org_123',
        completed_at: '2024-01-20T10:00:00Z',
      };
      mockClient.completeOnboarding.mockResolvedValue(mockResult);

      const result = await api.completeOnboarding({
        first_debate_id: 'd_123',
        template_used: 'invoice',
      });

      expect(mockClient.completeOnboarding).toHaveBeenCalledWith({
        first_debate_id: 'd_123',
        template_used: 'invoice',
      });
      expect(result.completed).toBe(true);
    });

    it('should complete onboarding without metadata', async () => {
      const mockResult = {
        completed: true,
        organization_id: 'org_123',
        completed_at: '2024-01-20T10:00:00Z',
      };
      mockClient.completeOnboarding.mockResolvedValue(mockResult);

      await api.completeOnboarding();

      expect(mockClient.completeOnboarding).toHaveBeenCalledWith(undefined);
    });
  });

  // ===========================================================================
  // Quick Start Helpers
  // ===========================================================================

  describe('Quick Start Helpers', () => {
    describe('quickInvoice', () => {
      it('should generate invoice', async () => {
        const mockResult = { execution_id: 'exec_inv_1' };
        mockClient.executeSMEWorkflow.mockResolvedValue(mockResult);

        const result = await api.quickInvoice({
          customerEmail: 'billing@client.com',
          customerName: 'Client Corp',
          items: [{ name: 'Service', price: 1000 }],
        });

        expect(mockClient.executeSMEWorkflow).toHaveBeenCalledWith('invoice', {
          inputs: {
            customer_email: 'billing@client.com',
            customer_name: 'Client Corp',
            items: [{ name: 'Service', unit_price: 1000, quantity: 1 }],
            due_date: undefined,
          },
        });
        expect(result.execution_id).toBe('exec_inv_1');
      });

      it('should generate invoice with multiple items and due date', async () => {
        const mockResult = { execution_id: 'exec_inv_2' };
        mockClient.executeSMEWorkflow.mockResolvedValue(mockResult);

        await api.quickInvoice({
          customerEmail: 'billing@client.com',
          customerName: 'Client Corp',
          items: [
            { name: 'Consulting', price: 150, quantity: 10 },
            { name: 'Support', price: 500, quantity: 1 },
          ],
          dueDate: '2024-02-01',
        });

        expect(mockClient.executeSMEWorkflow).toHaveBeenCalledWith('invoice', {
          inputs: {
            customer_email: 'billing@client.com',
            customer_name: 'Client Corp',
            items: [
              { name: 'Consulting', unit_price: 150, quantity: 10 },
              { name: 'Support', unit_price: 500, quantity: 1 },
            ],
            due_date: '2024-02-01',
          },
        });
      });
    });

    describe('quickInventoryCheck', () => {
      it('should set up inventory monitoring', async () => {
        const mockResult = { execution_id: 'exec_inv_check_1' };
        mockClient.executeSMEWorkflow.mockResolvedValue(mockResult);

        const result = await api.quickInventoryCheck({
          productId: 'SKU-001',
          minThreshold: 10,
          notificationEmail: 'ops@company.com',
        });

        expect(mockClient.executeSMEWorkflow).toHaveBeenCalledWith('inventory', {
          inputs: {
            product_id: 'SKU-001',
            min_threshold: 10,
            notification_email: 'ops@company.com',
          },
        });
        expect(result.execution_id).toBe('exec_inv_check_1');
      });
    });

    describe('quickReport', () => {
      it('should generate report with defaults', async () => {
        const mockResult = { execution_id: 'exec_rpt_1' };
        mockClient.executeSMEWorkflow.mockResolvedValue(mockResult);

        const result = await api.quickReport({
          type: 'sales',
          period: 'weekly',
        });

        expect(mockClient.executeSMEWorkflow).toHaveBeenCalledWith('report', {
          inputs: {
            report_type: 'sales',
            period: 'weekly',
            format: 'pdf',
            delivery_email: undefined,
          },
        });
      });

      it('should generate report with all options', async () => {
        const mockResult = { execution_id: 'exec_rpt_2' };
        mockClient.executeSMEWorkflow.mockResolvedValue(mockResult);

        await api.quickReport({
          type: 'financial',
          period: 'quarterly',
          format: 'excel',
          email: 'cfo@company.com',
        });

        expect(mockClient.executeSMEWorkflow).toHaveBeenCalledWith('report', {
          inputs: {
            report_type: 'financial',
            period: 'quarterly',
            format: 'excel',
            delivery_email: 'cfo@company.com',
          },
        });
      });
    });

    describe('quickFollowup', () => {
      it('should create follow-up with defaults', async () => {
        const mockResult = { execution_id: 'exec_fu_1' };
        mockClient.executeSMEWorkflow.mockResolvedValue(mockResult);

        const result = await api.quickFollowup({
          customerId: 'cust-123',
          type: 'post_sale',
        });

        expect(mockClient.executeSMEWorkflow).toHaveBeenCalledWith('followup', {
          inputs: {
            customer_id: 'cust-123',
            followup_type: 'post_sale',
            custom_message: undefined,
            delay_days: 0,
          },
        });
      });

      it('should create follow-up with custom message and delay', async () => {
        const mockResult = { execution_id: 'exec_fu_2' };
        mockClient.executeSMEWorkflow.mockResolvedValue(mockResult);

        await api.quickFollowup({
          customerId: 'cust-456',
          type: 'renewal',
          message: 'Your subscription is expiring soon!',
          delayDays: 7,
        });

        expect(mockClient.executeSMEWorkflow).toHaveBeenCalledWith('followup', {
          inputs: {
            customer_id: 'cust-456',
            followup_type: 'renewal',
            custom_message: 'Your subscription is expiring soon!',
            delay_days: 7,
          },
        });
      });
    });
  });
});
