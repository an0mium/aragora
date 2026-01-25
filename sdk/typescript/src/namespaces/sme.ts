/**
 * SME Namespace API
 *
 * Provides a namespaced interface for SME (Small/Medium Enterprise) operations
 * including workflows, onboarding, and starter pack features.
 */

import type {
  OnboardingStatus,
  WorkflowTemplate,
} from '../types';

/**
 * SME workflow execution result.
 */
interface SMEWorkflowExecutionResult {
  execution_id: string;
}

/**
 * Interface for the internal client methods used by SMEAPI.
 */
interface SMEClientInterface {
  // SME Workflows
  listSMEWorkflows(params?: { category?: string; limit?: number; offset?: number }): Promise<{ workflows: WorkflowTemplate[] }>;
  getSMEWorkflow(workflowId: string): Promise<WorkflowTemplate>;
  executeSMEWorkflow(workflowId: string, body: { inputs?: Record<string, unknown>; context?: Record<string, unknown> }): Promise<SMEWorkflowExecutionResult>;

  // Onboarding
  getOnboardingStatus(): Promise<OnboardingStatus>;
  completeOnboarding(request?: { first_debate_id?: string; template_used?: string }): Promise<{
    completed: boolean;
    organization_id: string;
    completed_at: string;
  }>;
}

/**
 * SME API namespace.
 *
 * Provides methods for SME (Small/Medium Enterprise) features:
 * - Pre-built SME workflow templates (invoice, followup, inventory, reports)
 * - Onboarding status and completion
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // List available SME workflows
 * const { workflows } = await client.sme.listWorkflows();
 *
 * // Execute an invoice workflow
 * const result = await client.sme.executeWorkflow('invoice', {
 *   inputs: {
 *     customer_email: 'customer@example.com',
 *     items: [{ name: 'Service', price: 100 }]
 *   }
 * });
 *
 * // Check onboarding status
 * const status = await client.sme.getOnboardingStatus();
 *
 * // Complete onboarding
 * await client.sme.completeOnboarding({ template_used: 'invoice' });
 * ```
 */
export class SMEAPI {
  constructor(private client: SMEClientInterface) {}

  // ===========================================================================
  // SME Workflows
  // ===========================================================================

  /**
   * List available SME workflow templates.
   *
   * SME workflows are pre-built templates designed for common small business tasks:
   * - invoice: Generate and send invoices
   * - followup: Customer follow-up campaigns
   * - inventory: Stock level monitoring and alerts
   * - report: Automated business reports
   */
  async listWorkflows(params?: { category?: string; limit?: number; offset?: number }): Promise<{ workflows: WorkflowTemplate[] }> {
    return this.client.listSMEWorkflows(params);
  }

  /**
   * Get details of a specific SME workflow template.
   */
  async getWorkflow(workflowId: string): Promise<WorkflowTemplate> {
    return this.client.getSMEWorkflow(workflowId);
  }

  /**
   * Execute an SME workflow template with inputs.
   *
   * @example
   * ```typescript
   * // Execute invoice workflow
   * const result = await client.sme.executeWorkflow('invoice', {
   *   inputs: {
   *     customer_name: 'Acme Corp',
   *     customer_email: 'billing@acme.com',
   *     items: [
   *       { name: 'Consulting', quantity: 10, unit_price: 150 },
   *       { name: 'Support', quantity: 1, unit_price: 500 }
   *     ],
   *     due_date: '2024-02-01'
   *   }
   * });
   *
   * // Execute inventory alert workflow
   * const result = await client.sme.executeWorkflow('inventory', {
   *   inputs: {
   *     product_id: 'SKU-001',
   *     min_threshold: 10,
   *     notification_email: 'ops@company.com'
   *   }
   * });
   * ```
   */
  async executeWorkflow(
    workflowId: string,
    body: { inputs?: Record<string, unknown>; config?: Record<string, unknown> }
  ): Promise<SMEWorkflowExecutionResult> {
    return this.client.executeSMEWorkflow(workflowId, body);
  }

  // ===========================================================================
  // Onboarding
  // ===========================================================================

  /**
   * Get current onboarding status.
   *
   * Returns the user's progress through the onboarding flow.
   */
  async getOnboardingStatus(): Promise<OnboardingStatus> {
    return this.client.getOnboardingStatus();
  }

  /**
   * Mark onboarding as complete.
   *
   * @param request - Optional metadata about the onboarding completion
   * @param request.first_debate_id - ID of the first debate created during onboarding
   * @param request.template_used - Name of the template used for the first debate
   */
  async completeOnboarding(request?: { first_debate_id?: string; template_used?: string }): Promise<{
    completed: boolean;
    organization_id: string;
    completed_at: string;
  }> {
    return this.client.completeOnboarding(request);
  }

  // ===========================================================================
  // Quick Start Helpers
  // ===========================================================================

  /**
   * Quick invoice generation helper.
   * Creates and executes an invoice workflow with minimal configuration.
   *
   * @example
   * ```typescript
   * const result = await client.sme.quickInvoice({
   *   customerEmail: 'billing@client.com',
   *   customerName: 'Client Corp',
   *   items: [{ name: 'Service', price: 1000 }]
   * });
   * ```
   */
  async quickInvoice(options: {
    customerEmail: string;
    customerName: string;
    items: Array<{ name: string; price: number; quantity?: number }>;
    dueDate?: string;
  }): Promise<SMEWorkflowExecutionResult> {
    return this.executeWorkflow('invoice', {
      inputs: {
        customer_email: options.customerEmail,
        customer_name: options.customerName,
        items: options.items.map(item => ({
          name: item.name,
          unit_price: item.price,
          quantity: item.quantity ?? 1,
        })),
        due_date: options.dueDate,
      },
    });
  }

  /**
   * Quick inventory check helper.
   * Sets up inventory monitoring for a product.
   *
   * @example
   * ```typescript
   * const result = await client.sme.quickInventoryCheck({
   *   productId: 'SKU-001',
   *   minThreshold: 10,
   *   notificationEmail: 'ops@company.com'
   * });
   * ```
   */
  async quickInventoryCheck(options: {
    productId: string;
    minThreshold: number;
    notificationEmail: string;
  }): Promise<SMEWorkflowExecutionResult> {
    return this.executeWorkflow('inventory', {
      inputs: {
        product_id: options.productId,
        min_threshold: options.minThreshold,
        notification_email: options.notificationEmail,
      },
    });
  }

  /**
   * Quick report generation helper.
   * Generates a business report with minimal configuration.
   *
   * @example
   * ```typescript
   * const result = await client.sme.quickReport({
   *   type: 'sales',
   *   period: 'weekly',
   *   format: 'pdf',
   *   email: 'ceo@company.com'
   * });
   * ```
   */
  async quickReport(options: {
    type: 'sales' | 'inventory' | 'customers' | 'financial';
    period: 'daily' | 'weekly' | 'monthly' | 'quarterly';
    format?: 'pdf' | 'excel' | 'html' | 'json';
    email?: string;
  }): Promise<SMEWorkflowExecutionResult> {
    return this.executeWorkflow('report', {
      inputs: {
        report_type: options.type,
        period: options.period,
        format: options.format ?? 'pdf',
        delivery_email: options.email,
      },
    });
  }

  /**
   * Quick customer follow-up helper.
   * Creates a follow-up campaign for a customer.
   *
   * @example
   * ```typescript
   * const result = await client.sme.quickFollowup({
   *   customerId: 'cust-123',
   *   type: 'renewal',
   *   message: 'Your subscription is expiring soon!'
   * });
   * ```
   */
  async quickFollowup(options: {
    customerId: string;
    type: 'post_sale' | 'check_in' | 'renewal' | 'feedback';
    message?: string;
    delayDays?: number;
  }): Promise<SMEWorkflowExecutionResult> {
    return this.executeWorkflow('followup', {
      inputs: {
        customer_id: options.customerId,
        followup_type: options.type,
        custom_message: options.message,
        delay_days: options.delayDays ?? 0,
      },
    });
  }
}
