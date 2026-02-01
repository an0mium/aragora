/**
 * Billing and Usage Metering Example
 *
 * Demonstrates billing and usage management in Aragora:
 * - Getting usage statistics and costs
 * - Budget management and alerts
 * - Invoice retrieval and history
 * - Usage forecasting
 *
 * Usage:
 *   npx ts-node examples/billing-usage.ts
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
  });

  try {
    // =========================================================================
    // 1. View Available Plans
    // =========================================================================
    console.log('=== Available Billing Plans ===\n');

    const { plans } = await client.billing.listPlans();
    console.log('Available plans:\n');

    for (const plan of plans) {
      console.log(`${plan.name} (${plan.tier})`);
      console.log(`  Monthly: $${plan.price_monthly} | Yearly: $${plan.price_yearly}`);
      console.log('  Limits:');
      console.log(`    - ${plan.limits.debates_per_month} debates/month`);
      console.log(`    - ${plan.limits.agents_per_debate} agents/debate`);
      console.log(`    - ${plan.limits.storage_gb}GB storage`);
      console.log(`    - ${plan.limits.api_calls_per_minute} API calls/minute`);
      console.log('  Features:');
      plan.features.slice(0, 3).forEach((feature) => {
        console.log(`    - ${feature}`);
      });
      if (plan.features.length > 3) {
        console.log(`    ... and ${plan.features.length - 3} more`);
      }
      console.log('');
    }

    // =========================================================================
    // 2. Current Subscription
    // =========================================================================
    console.log('=== Current Subscription ===\n');

    const subscription = await client.billing.getSubscription();
    console.log(`Plan: ${subscription.plan_name}`);
    console.log(`Status: ${subscription.status}`);
    console.log(`Period: ${subscription.current_period_start} to ${subscription.current_period_end}`);
    console.log(`Cancel at period end: ${subscription.cancel_at_period_end}`);
    console.log('');

    // =========================================================================
    // 3. Current Period Usage
    // =========================================================================
    console.log('=== Current Period Usage ===\n');

    const usage = await client.billing.getUsage();
    console.log(`Period: ${usage.period_start} to ${usage.period_end}`);
    console.log(`Debates: ${usage.debates_used} / ${usage.debates_limit}`);
    console.log(`Tokens used: ${usage.tokens_used.toLocaleString()}`);
    console.log(`Current cost: $${usage.cost_usd.toFixed(2)}`);
    console.log('');

    console.log('Cost by provider:');
    for (const [provider, data] of Object.entries(usage.by_provider)) {
      console.log(`  ${provider}: ${data.tokens.toLocaleString()} tokens = $${data.cost.toFixed(2)}`);
    }
    console.log('');

    console.log('Cost by feature:');
    for (const [feature, data] of Object.entries(usage.by_feature)) {
      console.log(`  ${feature}: ${data.count} uses = $${data.cost.toFixed(2)}`);
    }
    console.log('');

    // =========================================================================
    // 4. Detailed Usage Metering
    // =========================================================================
    console.log('=== Detailed Usage Breakdown ===\n');

    // Get current month summary
    const { usage: monthlySummary } = await client.usageMetering.getUsage('month');
    console.log('Monthly summary:');
    console.log(`  Total tokens: ${monthlySummary.tokens.total.toLocaleString()}`);
    console.log(`    Input: ${monthlySummary.tokens.input.toLocaleString()}`);
    console.log(`    Output: ${monthlySummary.tokens.output.toLocaleString()}`);
    console.log(`  Total cost: ${monthlySummary.tokens.cost}`);
    console.log(`  Debates: ${monthlySummary.counts.debates}`);
    console.log(`  API calls: ${monthlySummary.counts.api_calls}`);
    console.log('');

    console.log('Usage by model:');
    for (const [model, count] of Object.entries(monthlySummary.by_model)) {
      console.log(`  ${model}: ${count.toLocaleString()} tokens`);
    }
    console.log('');

    console.log('Usage limits:');
    console.log(`  Tokens: ${monthlySummary.usage_percent.tokens.toFixed(1)}% of limit`);
    console.log(`  Debates: ${monthlySummary.usage_percent.debates.toFixed(1)}% of limit`);
    console.log(`  API calls: ${monthlySummary.usage_percent.api_calls.toFixed(1)}% of limit`);
    console.log('');

    // Get detailed breakdown
    const startDate = new Date();
    startDate.setDate(1); // First of month
    const { breakdown } = await client.usageMetering.getBreakdown({
      start: startDate.toISOString(),
      end: new Date().toISOString(),
    });

    console.log('Breakdown by model:');
    for (const modelUsage of breakdown.by_model.slice(0, 5)) {
      console.log(`  ${modelUsage.model}:`);
      console.log(`    Requests: ${modelUsage.requests}`);
      console.log(`    Tokens: ${modelUsage.total_tokens.toLocaleString()}`);
      console.log(`    Cost: ${modelUsage.cost}`);
    }
    console.log('');

    console.log('Daily usage trend:');
    for (const day of breakdown.by_day.slice(-7)) {
      const bar = createUsageBar(day.total_tokens, 1000000); // Scale to 1M
      console.log(`  ${day.day}: ${bar} ${day.cost}`);
    }
    console.log('');

    // =========================================================================
    // 5. Quotas Status
    // =========================================================================
    console.log('=== Quota Status ===\n');

    const quotas = await client.usageMetering.getQuotas();
    const quotaTypes = ['debates', 'api_requests', 'tokens', 'storage_bytes', 'knowledge_bytes'] as const;

    for (const quotaType of quotaTypes) {
      const quota = quotas.quotas[quotaType];
      const statusIcon = quota.is_exceeded ? '[X]' : quota.is_warning ? '[!]' : '[+]';
      console.log(`${statusIcon} ${quotaType}:`);
      console.log(`    ${quota.current.toLocaleString()} / ${quota.limit.toLocaleString()} (${quota.percentage_used.toFixed(1)}%)`);
      console.log(`    Remaining: ${quota.remaining.toLocaleString()}`);
      if (quota.resets_at) {
        console.log(`    Resets at: ${quota.resets_at}`);
      }
    }
    console.log('');

    // =========================================================================
    // 6. Usage Forecast
    // =========================================================================
    console.log('=== Usage Forecast ===\n');

    const forecast = await client.billing.getForecast();
    console.log(`Projected monthly cost: $${forecast.projected_monthly_cost.toFixed(2)}`);
    console.log(`Projected debates: ${forecast.projected_debates}`);
    console.log(`Projected tokens: ${forecast.projected_tokens.toLocaleString()}`);
    console.log(`Recommended tier: ${forecast.recommended_tier}`);

    if (forecast.savings_with_upgrade && forecast.savings_with_upgrade > 0) {
      console.log(`Potential savings with upgrade: $${forecast.savings_with_upgrade.toFixed(2)}/month`);
    }

    if (forecast.days_until_limit && forecast.days_until_limit > 0) {
      console.log(`Days until limit reached: ${forecast.days_until_limit}`);
    }
    console.log('');

    // =========================================================================
    // 7. Budget Management
    // =========================================================================
    console.log('=== Budget Management ===\n');

    // Get budget summary
    const budgetSummary = await client.budgets.getSummary();
    console.log('Budget overview:');
    console.log(`  Total budget: $${budgetSummary.total_budget.toFixed(2)}`);
    console.log(`  Total spent: $${budgetSummary.total_spent.toFixed(2)}`);
    console.log(`  Remaining: $${budgetSummary.total_remaining.toFixed(2)}`);
    console.log(`  Active budgets: ${budgetSummary.active_budgets}`);
    console.log(`  Exceeded: ${budgetSummary.exceeded_budgets}`);
    console.log(`  Warning: ${budgetSummary.warning_budgets}`);
    console.log('');

    // List budgets
    const { budgets } = await client.budgets.list();
    console.log('Budgets:');
    for (const budget of budgets) {
      const percent = (budget.spent_amount / budget.limit_amount * 100).toFixed(1);
      const statusIcon = budget.status === 'exceeded' ? '[X]' :
                        budget.status === 'warning' ? '[!]' : '[+]';
      console.log(`  ${statusIcon} ${budget.name}:`);
      console.log(`      $${budget.spent_amount.toFixed(2)} / $${budget.limit_amount.toFixed(2)} (${percent}%)`);
      console.log(`      Period: ${budget.period}`);
      console.log(`      Alert threshold: ${budget.alert_threshold}%`);
    }
    console.log('');

    // Create a new budget
    console.log('Creating new budget...');
    const newBudget = await client.budgets.create({
      name: 'Q1 Analytics Budget',
      description: 'Budget for analytics team Q1 2025',
      limit_amount: 5000,
      currency: 'USD',
      period: 'quarterly',
      alert_threshold: 80,
    });
    console.log(`Created budget: ${newBudget.name} ($${newBudget.limit_amount})`);
    console.log('');

    // Check budget alerts
    const { alerts } = await client.budgets.listAlerts({ budget_id: newBudget.id });
    if (alerts.length > 0) {
      console.log('Active alerts:');
      for (const alert of alerts) {
        console.log(`  [${alert.severity.toUpperCase()}] ${alert.message}`);
        console.log(`    Type: ${alert.type}`);
        console.log(`    Threshold: ${alert.threshold_pct}%, Current: ${alert.current_pct}%`);
      }
    } else {
      console.log('No active budget alerts');
    }
    console.log('');

    // =========================================================================
    // 8. Invoice History
    // =========================================================================
    console.log('=== Invoice History ===\n');

    const { invoices } = await client.billing.listInvoices({ limit: 5 });
    console.log('Recent invoices:');
    for (const invoice of invoices) {
      const status = invoice.status === 'paid' ? '[Paid]' :
                    invoice.status === 'open' ? '[Open]' : `[${invoice.status}]`;
      console.log(`  ${status} ${invoice.number}`);
      console.log(`    Amount: ${invoice.currency} ${invoice.amount_due.toFixed(2)}`);
      console.log(`    Created: ${invoice.created_at}`);
      if (invoice.paid_at) {
        console.log(`    Paid: ${invoice.paid_at}`);
      }
      if (invoice.pdf_url) {
        console.log(`    PDF: ${invoice.pdf_url}`);
      }
    }
    console.log('');

    // =========================================================================
    // 9. Export Usage Data
    // =========================================================================
    console.log('=== Export Usage Data ===\n');

    // Export as CSV
    const exportResult = await client.billing.exportUsage(
      startDate.toISOString(),
      new Date().toISOString()
    );
    console.log(`Export ready: ${exportResult.download_url}`);
    console.log('');

    // =========================================================================
    // 10. Cost Management Dashboard
    // =========================================================================
    console.log('=== Cost Dashboard ===\n');

    const costDashboard = await client.costManagement.getDashboard();
    console.log('Cost overview:');
    console.log(`  Total cost this period: $${costDashboard.total_cost.toFixed(2)}`);
    console.log(`  Change from last period: ${costDashboard.cost_change_pct >= 0 ? '+' : ''}${costDashboard.cost_change_pct.toFixed(1)}%`);
    console.log(`  Debates run: ${costDashboard.debates_run}`);
    console.log(`  Tokens used: ${costDashboard.tokens_used.toLocaleString()}`);
    console.log(`  Avg cost per debate: $${costDashboard.avg_cost_per_debate.toFixed(2)}`);
    console.log('');

    console.log('Top providers:');
    for (const provider of costDashboard.top_providers.slice(0, 3)) {
      console.log(`  ${provider.provider}: $${provider.cost.toFixed(2)} (${provider.percentage.toFixed(1)}%)`);
    }
    console.log('');

    console.log('Budget status:');
    const bs = costDashboard.budget_status;
    const budgetBar = createUsageBar(bs.pct_used, 100);
    console.log(`  ${budgetBar} $${bs.spent.toFixed(2)} / $${bs.limit.toFixed(2)} (${bs.pct_used.toFixed(1)}%)`);
    console.log(`  Remaining: $${bs.remaining.toFixed(2)}`);
    console.log('');

    // =========================================================================
    // 11. Billing Portal Access
    // =========================================================================
    console.log('=== Billing Portal ===\n');

    const portalResult = await client.billing.getPortalUrl('https://your-app.com/settings/billing');
    console.log('Billing portal URL:', portalResult.url);
    console.log('(Use this URL to manage payment methods, download invoices, etc.)');
    console.log('');

    console.log('Billing and usage example completed successfully!');

  } catch (error) {
    handleError(error);
    process.exit(1);
  }
}

// =========================================================================
// Helper Functions
// =========================================================================

function createUsageBar(current: number, max: number, width: number = 20): string {
  const percent = Math.min(100, (current / max) * 100);
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  return '[' + '='.repeat(filled) + ' '.repeat(empty) + ']';
}

function handleError(error: unknown): void {
  if (error instanceof AragoraError) {
    console.error('\n--- Aragora Error ---');
    console.error(`Message: ${error.message}`);
    console.error(`Code: ${error.code || 'N/A'}`);
    console.error(`Status: ${error.status || 'N/A'}`);

    // Common billing errors
    if (error.code === 'QUOTA_EXCEEDED') {
      console.error('\nNote: Your usage quota has been exceeded.');
      console.error('Consider upgrading your plan or waiting for the next billing period.');
    } else if (error.code === 'RATE_LIMITED') {
      console.error('\nNote: API rate limit exceeded.');
      const retryAfter = error.details?.retry_after || 'shortly';
      console.error(`Please retry after ${retryAfter}`);
    }

    if (error.traceId) {
      console.error(`Trace ID: ${error.traceId}`);
    }
  } else if (error instanceof Error) {
    console.error('\n--- Error ---');
    console.error(`Message: ${error.message}`);
  } else {
    console.error('\n--- Unknown Error ---');
    console.error(error);
  }
}

// Run the example
main();
