/**
 * Workflow Automation Example
 *
 * Shows how to create automated workflows that chain debates
 * with conditional logic and external integrations.
 *
 * Usage:
 *   npx ts-node examples/workflow-automation.ts
 */

import { createClient } from '@aragora/sdk';

async function main() {
  const client = createClient({
    baseUrl: process.env.ARAGORA_API_URL || 'https://api.aragora.ai',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  console.log('Creating workflow...');

  // Create a workflow with multiple steps
  const workflow = await client.workflows.create({
    name: 'Code Review Pipeline',
    description: 'Multi-stage code review with AI agents',
    triggers: [
      {
        type: 'webhook',
        config: { path: '/webhooks/code-review' },
      },
    ],
    steps: [
      {
        id: 'initial-review',
        type: 'debate',
        config: {
          task: 'Review this code for bugs and security issues: {{input.code}}',
          agents: ['claude', 'gpt-4'],
          rounds: 2,
        },
      },
      {
        id: 'check-severity',
        type: 'condition',
        config: {
          expression: 'steps.initial_review.consensus.confidence > 0.8',
          onTrue: 'deep-analysis',
          onFalse: 'quick-summary',
        },
      },
      {
        id: 'deep-analysis',
        type: 'debate',
        config: {
          task: 'Perform deep security analysis on: {{input.code}}',
          agents: ['claude', 'gemini'],
          rounds: 3,
          context: '{{steps.initial_review.consensus}}',
        },
      },
      {
        id: 'quick-summary',
        type: 'transform',
        config: {
          template: 'Quick review completed. Findings: {{steps.initial_review.consensus.final_answer}}',
        },
      },
      {
        id: 'notify',
        type: 'webhook',
        config: {
          url: '{{env.SLACK_WEBHOOK_URL}}',
          method: 'POST',
          body: {
            text: 'Code review complete: {{workflow.output}}',
          },
        },
      },
    ],
  });

  console.log(`Workflow created: ${workflow.id}`);

  // Execute the workflow with sample input
  const execution = await client.workflows.execute(workflow.id, {
    input: {
      code: `
        def authenticate(username, password):
            query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
            return db.execute(query)
      `,
    },
  });

  console.log(`Execution started: ${execution.execution_id}`);

  // Wait for completion
  let status = execution;
  while (status.status === 'running') {
    await new Promise((r) => setTimeout(r, 3000));
    status = await client.workflows.getExecution(execution.execution_id);
    console.log(`Status: ${status.status}, Step: ${status.current_step}`);
  }

  console.log('\n--- Workflow Results ---');
  console.log(JSON.stringify(status.output, null, 2));
}

main().catch(console.error);
