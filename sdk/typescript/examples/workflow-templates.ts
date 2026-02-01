/**
 * Workflow Templates Example
 *
 * Demonstrates workflow automation capabilities in Aragora:
 * - Listing and running pre-built templates
 * - Creating custom workflows with debate steps
 * - Handling approval workflows
 * - Monitoring execution progress
 *
 * Usage:
 *   npx ts-node examples/workflow-templates.ts
 *
 * Environment:
 *   ARAGORA_API_KEY - Your API key
 *   ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
 */

import { createClient, AragoraError } from '@aragora/sdk';

// Configuration
const API_URL = process.env.ARAGORA_API_URL || 'https://api.aragora.ai';
const API_KEY = process.env.ARAGORA_API_KEY || 'your-api-key-here';

async function main() {
  // Initialize the client
  const client = createClient({
    baseUrl: API_URL,
    apiKey: API_KEY,
    retryEnabled: true,
    maxRetries: 3,
  });

  try {
    // =========================================================================
    // 1. List Available Workflow Templates
    // =========================================================================
    console.log('=== Listing Workflow Templates ===\n');

    const { templates } = await client.workflows.listTemplates({
      limit: 10,
      offset: 0,
    });

    console.log(`Found ${templates.length} templates:\n`);
    for (const template of templates) {
      console.log(`  - ${template.name} (${template.id})`);
      console.log(`    Category: ${template.category || 'General'}`);
      console.log(`    Description: ${template.description}`);
      console.log(`    Steps: ${template.steps?.length || 0}`);
      console.log('');
    }

    // =========================================================================
    // 2. Get Template Details
    // =========================================================================
    console.log('=== Template Details ===\n');

    // Get details of a specific template (using first template or a known ID)
    const templateId = templates[0]?.id || 'code-review-pipeline';
    const templateDetails = await client.workflows.getTemplate(templateId);

    console.log(`Template: ${templateDetails.name}`);
    console.log(`Category: ${templateDetails.category}`);
    console.log('Steps:');
    templateDetails.steps?.forEach((step, index) => {
      console.log(`  ${index + 1}. ${step.name} (${step.type})`);
      if (step.depends_on?.length) {
        console.log(`     Depends on: ${step.depends_on.join(', ')}`);
      }
    });
    console.log('');

    // =========================================================================
    // 3. Run a Template Workflow
    // =========================================================================
    console.log('=== Running Workflow from Template ===\n');

    // Create and execute a workflow from a template
    const execution = await client.workflows.runTemplate(templateId, {
      // Template parameters - these vary by template
      code_content: `
        def calculate_total(items):
            total = 0
            for item in items:
                total += item['price'] * item['quantity']
            return total
      `,
      review_depth: 'standard',
      notify_on_complete: true,
    });

    console.log(`Workflow execution started: ${execution.id}`);
    console.log(`Status: ${execution.status}`);

    // =========================================================================
    // 4. Monitor Execution Progress
    // =========================================================================
    console.log('\n=== Monitoring Execution ===\n');

    // Poll for execution status
    let status = execution;
    let attempts = 0;
    const maxAttempts = 30;

    while (
      (status.status === 'pending' || status.status === 'running') &&
      attempts < maxAttempts
    ) {
      await sleep(2000);
      status = await client.workflows.getExecution(execution.id);

      const progress = status.progress ?? 0;
      const progressBar = createProgressBar(progress);
      console.log(`[${progressBar}] ${progress}% - Step: ${status.current_step || 'initializing'}`);

      attempts++;
    }

    if (status.status === 'completed') {
      console.log('\nWorkflow completed successfully!');
      console.log('Outputs:', JSON.stringify(status.outputs, null, 2));
    } else if (status.status === 'failed') {
      console.log('\nWorkflow failed:', status.error);
    } else if (status.status === 'waiting_approval') {
      console.log('\nWorkflow is waiting for approval');
    }

    // =========================================================================
    // 5. Create a Custom Workflow
    // =========================================================================
    console.log('\n=== Creating Custom Workflow ===\n');

    const customWorkflow = await client.workflows.create({
      name: 'Custom Decision Pipeline',
      description: 'Multi-stage decision process with debates and approval gates',
      steps: [
        {
          id: 'initial-analysis',
          name: 'Initial Analysis',
          type: 'debate',
          config: {
            task: 'Analyze the proposal: {{input.proposal}}',
            agents: ['claude', 'gpt-4'],
            rounds: 2,
            consensus: 'majority',
          },
        },
        {
          id: 'risk-assessment',
          name: 'Risk Assessment',
          type: 'debate',
          config: {
            task: 'Evaluate risks based on: {{steps.initial_analysis.outputs.consensus}}',
            agents: ['claude', 'gemini'],
            rounds: 2,
          },
          depends_on: ['initial-analysis'],
        },
        {
          id: 'approval-gate',
          name: 'Manager Approval',
          type: 'approval',
          config: {
            approvers: ['manager@example.com'],
            timeout_hours: 24,
            auto_approve_if: 'steps.risk_assessment.outputs.risk_level === "low"',
          },
          depends_on: ['risk-assessment'],
        },
        {
          id: 'final-decision',
          name: 'Final Decision',
          type: 'transform',
          config: {
            template: JSON.stringify({
              decision: '{{steps.risk_assessment.outputs.recommendation}}',
              approved_by: '{{steps.approval_gate.outputs.approver}}',
              rationale: '{{steps.initial_analysis.outputs.consensus}}',
            }),
          },
          depends_on: ['approval-gate'],
        },
      ],
    });

    console.log(`Custom workflow created: ${customWorkflow.id}`);
    console.log(`Name: ${customWorkflow.name}`);
    console.log(`Steps: ${customWorkflow.steps?.length || 0}`);

    // =========================================================================
    // 6. Handle Approval Workflows
    // =========================================================================
    console.log('\n=== Handling Approval Workflows ===\n');

    // List pending approvals
    const { approvals } = await client.workflows.listApprovals({
      status: 'pending',
      limit: 10,
    });

    console.log(`Pending approvals: ${approvals.length}`);

    for (const approval of approvals) {
      console.log(`\n  Approval: ${approval.id}`);
      console.log(`  Workflow: ${approval.workflow_id}`);
      console.log(`  Step: ${approval.step_id}`);
      console.log(`  Requested at: ${approval.requested_at}`);

      // To approve (uncomment to use):
      // await client.workflows.approve(approval.id, {
      //   reason: 'Reviewed and approved',
      // });

      // To reject (uncomment to use):
      // await client.workflows.reject(approval.id, {
      //   reason: 'Risk level too high',
      // });
    }

    // =========================================================================
    // 7. Simulate Workflow Execution
    // =========================================================================
    console.log('\n=== Simulating Workflow ===\n');

    const simulation = await client.workflows.simulate(customWorkflow.id, {
      proposal: 'Launch new feature in production',
    });

    console.log(`Simulation valid: ${simulation.valid}`);
    console.log(`Estimated duration: ${simulation.estimated_duration || 'N/A'} seconds`);

    if (simulation.warnings?.length) {
      console.log('Warnings:');
      simulation.warnings.forEach((warning) => console.log(`  - ${warning}`));
    }

    if (simulation.step_previews?.length) {
      console.log('Step previews:');
      simulation.step_previews.forEach((preview) => {
        console.log(`  - ${preview.name}: ${preview.would_execute ? 'Will execute' : 'Will skip'}`);
      });
    }

    // =========================================================================
    // 8. Version Management
    // =========================================================================
    console.log('\n=== Version Management ===\n');

    // Get workflow versions
    const { versions } = await client.workflows.listVersions(customWorkflow.id);
    console.log(`Workflow has ${versions.length} version(s)`);

    for (const version of versions) {
      console.log(`  - Version ${version.version} (${version.created_at})`);
      if (version.changes) {
        console.log(`    Changes: ${version.changes}`);
      }
    }

    // =========================================================================
    // 9. Cleanup and Best Practices
    // =========================================================================
    console.log('\n=== Cleanup ===\n');

    // Cancel any running executions (if needed)
    const { executions: runningExecutions } = await client.workflows.listExecutions({
      status: 'running',
      workflow_id: customWorkflow.id,
    });

    for (const exec of runningExecutions) {
      console.log(`Cancelling execution: ${exec.id}`);
      await client.workflows.cancelExecution(exec.id);
    }

    console.log('Workflow template example completed successfully!');

  } catch (error) {
    handleError(error);
    process.exit(1);
  }
}

// =========================================================================
// Helper Functions
// =========================================================================

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function createProgressBar(percent: number, width: number = 20): string {
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  return '='.repeat(filled) + '-'.repeat(empty);
}

function handleError(error: unknown): void {
  if (error instanceof AragoraError) {
    console.error('\n--- Aragora Error ---');
    console.error(`Message: ${error.message}`);
    console.error(`Code: ${error.code || 'N/A'}`);
    console.error(`Status: ${error.status || 'N/A'}`);
    if (error.traceId) {
      console.error(`Trace ID: ${error.traceId} (provide this for support)`);
    }
    if (error.details) {
      console.error('Details:', JSON.stringify(error.details, null, 2));
    }
  } else if (error instanceof Error) {
    console.error('\n--- Error ---');
    console.error(`Message: ${error.message}`);
    console.error(`Stack: ${error.stack}`);
  } else {
    console.error('\n--- Unknown Error ---');
    console.error(error);
  }
}

// Run the example
main();
