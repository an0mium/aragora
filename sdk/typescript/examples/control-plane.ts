/**
 * Control Plane Operations Example
 *
 * Demonstrates control plane capabilities in Aragora:
 * - Agent registry management
 * - Task scheduling and distribution
 * - Health monitoring and probes
 * - Policy management
 *
 * Usage:
 *   npx ts-node examples/control-plane.ts
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
    // 1. Control Plane Health Check
    // =========================================================================
    console.log('=== Control Plane Health ===\n');

    const health = await client.controlPlane.getHealth();
    console.log(`Status: ${health.status}`);
    console.log(`Agents total: ${health.agents_total}`);
    console.log(`Agents active: ${health.agents_active}`);
    console.log(`Tasks pending: ${health.tasks_pending}`);
    console.log(`Tasks running: ${health.tasks_running}`);
    if (health.uptime_seconds) {
      console.log(`Uptime: ${formatUptime(health.uptime_seconds)}`);
    }
    console.log('');

    // =========================================================================
    // 2. Agent Registry
    // =========================================================================
    console.log('=== Agent Registry ===\n');

    // List registered agents
    const { agents } = await client.controlPlane.listAgents();
    console.log(`Registered agents: ${agents.length}`);

    for (const agent of agents) {
      const statusIcon = agent.status === 'idle' ? '[+]' :
                        agent.status === 'busy' ? '[*]' :
                        agent.status === 'draining' ? '[~]' : '[-]';
      console.log(`  ${statusIcon} ${agent.agent_id} (${agent.name || 'Unnamed'})`);
      console.log(`      Status: ${agent.status}`);
      console.log(`      Registered: ${agent.registered_at}`);
      if (agent.last_heartbeat) {
        console.log(`      Last heartbeat: ${agent.last_heartbeat}`);
      }
      if (agent.capabilities?.length) {
        console.log(`      Capabilities: ${agent.capabilities.join(', ')}`);
      }
      if (agent.current_task) {
        console.log(`      Current task: ${agent.current_task}`);
      }
    }
    console.log('');

    // Register a new agent
    console.log('Registering new agent...');
    const newAgent = await client.controlPlane.registerAgent({
      agent_id: `agent-${Date.now()}`,
      name: 'Custom Analysis Agent',
      capabilities: ['code-review', 'security-scan', 'performance-analysis'],
      metadata: {
        version: '1.0.0',
        environment: 'production',
        region: 'us-west-2',
      },
    });
    console.log(`Registered agent: ${newAgent.agent_id}`);
    console.log('');

    // Send heartbeat
    console.log('Sending heartbeat...');
    await client.controlPlane.sendHeartbeat({
      agent_id: newAgent.agent_id,
      status: 'idle',
      metrics: {
        cpu_percent: 45.2,
        memory_mb: 512,
        tasks_completed: 0,
      },
    });
    console.log('Heartbeat sent successfully');
    console.log('');

    // =========================================================================
    // 3. Task Scheduling
    // =========================================================================
    console.log('=== Task Scheduling ===\n');

    // Submit a task
    console.log('Submitting task...');
    const task = await client.controlPlane.submitTask({
      task_type: 'code-review',
      payload: {
        repository: 'acme/web-app',
        branch: 'feature/new-auth',
        files: ['src/auth/login.ts', 'src/auth/session.ts'],
      },
      priority: 'high',
      agent_hint: 'agent-with-security-capability', // Optional hint
      timeout_seconds: 300,
      metadata: {
        requester: 'ci-pipeline',
        pull_request: 'PR-1234',
      },
    });
    console.log(`Task submitted: ${task.task_id}`);
    console.log(`Type: ${task.task_type}`);
    console.log(`Priority: ${task.priority}`);
    console.log(`Status: ${task.status}`);
    console.log('');

    // List tasks
    const { tasks } = await client.controlPlane.listTasks({
      status: 'pending',
      limit: 10,
    });
    console.log(`Pending tasks: ${tasks.length}`);
    for (const t of tasks) {
      console.log(`  - ${t.task_id} (${t.task_type})`);
      console.log(`    Priority: ${t.priority}, Status: ${t.status}`);
      console.log(`    Submitted: ${t.submitted_at}`);
    }
    console.log('');

    // Wait for task completion
    console.log('Waiting for task completion...');
    const completedTask = await waitForTask(client, task.task_id, 60000);
    if (completedTask) {
      console.log(`Task completed: ${completedTask.status}`);
      if (completedTask.result) {
        console.log('Result:', JSON.stringify(completedTask.result, null, 2));
      }
      if (completedTask.error) {
        console.log('Error:', completedTask.error);
      }
    } else {
      console.log('Task did not complete within timeout');
    }
    console.log('');

    // =========================================================================
    // 4. System Health Monitoring
    // =========================================================================
    console.log('=== System Health Monitoring ===\n');

    // Basic health check
    const basicHealth = await client.health.check();
    console.log(`System status: ${basicHealth.status}`);
    console.log(`Version: ${basicHealth.version || 'N/A'}`);
    console.log(`Timestamp: ${basicHealth.timestamp}`);
    console.log('');

    // Detailed health with all checks
    const detailedHealth = await client.health.getDetailed();
    console.log('Detailed health checks:');
    for (const check of detailedHealth.checks) {
      const icon = check.status === 'pass' ? '[+]' :
                  check.status === 'warn' ? '[!]' : '[-]';
      console.log(`  ${icon} ${check.name}: ${check.status}`);
      if (check.latency_ms) {
        console.log(`      Latency: ${check.latency_ms}ms`);
      }
      if (check.message) {
        console.log(`      Message: ${check.message}`);
      }
    }
    console.log('');

    console.log('System metrics:');
    console.log(`  Requests/min: ${detailedHealth.metrics.requests_per_minute}`);
    console.log(`  Avg latency: ${detailedHealth.metrics.average_latency_ms}ms`);
    console.log(`  Error rate: ${(detailedHealth.metrics.error_rate * 100).toFixed(2)}%`);
    console.log(`  Active connections: ${detailedHealth.metrics.active_connections}`);
    console.log(`  Memory usage: ${detailedHealth.metrics.memory_usage_mb}MB`);
    console.log(`  CPU usage: ${detailedHealth.metrics.cpu_usage_percent.toFixed(1)}%`);
    console.log('');

    // Nomic loop health
    const nomicHealth = await client.health.getNomicHealth();
    console.log('Nomic loop health:');
    console.log(`  Status: ${nomicHealth.status}`);
    if (nomicHealth.phase) {
      console.log(`  Current phase: ${nomicHealth.phase}`);
    }
    if (nomicHealth.last_cycle) {
      console.log(`  Last cycle: ${nomicHealth.last_cycle}`);
    }
    console.log('');

    // =========================================================================
    // 5. Monitoring and Anomaly Detection
    // =========================================================================
    console.log('=== Monitoring and Anomaly Detection ===\n');

    // Record a metric
    const metricResult = await client.monitoring.record({
      metric_name: 'api_latency_ms',
      value: 150,
    });
    console.log(`Recorded metric: api_latency_ms = 150`);
    if (metricResult.anomaly_detected) {
      console.log('Anomaly detected!');
      console.log(`  Severity: ${metricResult.anomaly?.severity}`);
      console.log(`  Expected: ${metricResult.anomaly?.expected_value}`);
      console.log(`  Deviation: ${metricResult.anomaly?.deviation}`);
    }
    console.log('');

    // Get all metric trends
    const trends = await client.monitoring.getAllTrends();
    console.log(`Tracking ${trends.count} metrics:`);
    for (const [metricName, trend] of Object.entries(trends.trends)) {
      const arrow = trend.direction === 'up' ? '/' :
                   trend.direction === 'down' ? '\\' : '-';
      console.log(`  ${arrow} ${metricName}: ${trend.current_value} (${trend.change_percent >= 0 ? '+' : ''}${trend.change_percent.toFixed(1)}%)`);
    }
    console.log('');

    // Get specific metric trend
    const { trend: latencyTrend } = await client.monitoring.getTrend('api_latency_ms', {
      period_seconds: 3600, // Last hour
    });
    console.log('API latency trend (last hour):');
    console.log(`  Direction: ${latencyTrend.direction}`);
    console.log(`  Current: ${latencyTrend.current_value}ms`);
    console.log(`  Previous: ${latencyTrend.previous_value}ms`);
    console.log(`  Change: ${latencyTrend.change_percent >= 0 ? '+' : ''}${latencyTrend.change_percent.toFixed(1)}%`);
    console.log(`  Confidence: ${(latencyTrend.confidence * 100).toFixed(1)}%`);
    console.log('');

    // Get recent anomalies
    const { anomalies } = await client.monitoring.getAnomalies({ hours: 24 });
    console.log(`Anomalies in last 24 hours: ${anomalies.length}`);
    for (const anomaly of anomalies.slice(0, 5)) {
      const severityIcon = anomaly.severity === 'critical' ? '[!!]' :
                          anomaly.severity === 'high' ? '[!]' :
                          anomaly.severity === 'medium' ? '[?]' : '[-]';
      console.log(`  ${severityIcon} ${anomaly.metric_name}`);
      console.log(`      Value: ${anomaly.value} (expected: ${anomaly.expected_value})`);
      console.log(`      Deviation: ${(anomaly.deviation * 100).toFixed(1)}%`);
      console.log(`      Time: ${anomaly.timestamp}`);
    }
    console.log('');

    // Get baseline statistics
    const baseline = await client.monitoring.getBaseline('api_latency_ms');
    console.log('API latency baseline:');
    console.log(`  Mean: ${baseline.stats.mean.toFixed(2)}ms`);
    console.log(`  Std Dev: ${baseline.stats.stdev.toFixed(2)}ms`);
    console.log(`  Min: ${baseline.stats.min}ms`);
    console.log(`  Max: ${baseline.stats.max}ms`);
    console.log(`  Median: ${baseline.stats.median}ms`);
    console.log('');

    // =========================================================================
    // 6. Policy Management
    // =========================================================================
    console.log('=== Policy Management ===\n');

    // List policies
    const { policies } = await client.policies.list();
    console.log(`Active policies: ${policies.length}`);
    for (const policy of policies.slice(0, 5)) {
      const p = policy as { id: string; name: string; type: string; enabled: boolean };
      console.log(`  - ${p.name} (${p.id})`);
      console.log(`    Type: ${p.type}`);
      console.log(`    Enabled: ${p.enabled}`);
    }
    console.log('');

    // Create a rate limit policy
    console.log('Creating rate limit policy...');
    const newPolicy = await client.policies.create({
      name: 'API Rate Limit - Standard',
      type: 'rate_limit',
      enabled: true,
      rules: [
        {
          resource: '/api/v1/debates',
          limit: 100,
          window_seconds: 60,
          action: 'throttle',
        },
        {
          resource: '/api/v1/workflows',
          limit: 50,
          window_seconds: 60,
          action: 'reject',
        },
      ],
      metadata: {
        tier: 'standard',
        created_by: 'admin',
      },
    });
    console.log(`Created policy: ${(newPolicy as { name: string }).name}`);
    console.log('');

    // =========================================================================
    // 7. Circuit Breakers (via Admin API)
    // =========================================================================
    console.log('=== Circuit Breakers ===\n');

    const { circuit_breakers } = await client.admin.getCircuitBreakers();
    console.log(`Circuit breakers: ${circuit_breakers.length}`);
    for (const cb of circuit_breakers) {
      const stateIcon = cb.state === 'closed' ? '[+]' :
                       cb.state === 'open' ? '[-]' : '[~]';
      console.log(`  ${stateIcon} ${cb.name}`);
      console.log(`      State: ${cb.state}`);
      console.log(`      Failures: ${cb.failure_count} (threshold: ${cb.threshold})`);
      console.log(`      Successes: ${cb.success_count}`);
      console.log(`      Timeout: ${cb.timeout_seconds}s`);
      if (cb.last_failure) {
        console.log(`      Last failure: ${cb.last_failure}`);
      }
    }
    console.log('');

    // =========================================================================
    // 8. Nomic Loop Control
    // =========================================================================
    console.log('=== Nomic Loop Control ===\n');

    const nomicStatus = await client.admin.getNomicStatus();
    console.log(`Running: ${nomicStatus.running}`);
    console.log(`Health: ${nomicStatus.health}`);
    if (nomicStatus.current_phase) {
      console.log(`Current phase: ${nomicStatus.current_phase}`);
    }
    console.log(`Current cycle: ${nomicStatus.current_cycle} / ${nomicStatus.total_cycles}`);
    if (nomicStatus.last_run) {
      console.log(`Last run: ${nomicStatus.last_run}`);
    }
    if (nomicStatus.next_scheduled) {
      console.log(`Next scheduled: ${nomicStatus.next_scheduled}`);
    }
    console.log('');

    // Control nomic loop (uncomment to use)
    // await client.admin.pauseNomic();
    // console.log('Nomic loop paused');

    // await client.admin.resumeNomic();
    // console.log('Nomic loop resumed');

    // =========================================================================
    // 9. Cleanup
    // =========================================================================
    console.log('=== Cleanup ===\n');

    // Unregister the test agent
    await client.controlPlane.unregisterAgent(newAgent.agent_id);
    console.log(`Unregistered agent: ${newAgent.agent_id}`);

    // Cancel the test task if still pending
    if (task.status === 'pending' || task.status === 'running') {
      await client.controlPlane.cancelTask(task.task_id);
      console.log(`Cancelled task: ${task.task_id}`);
    }

    console.log('\nControl plane example completed successfully!');

  } catch (error) {
    handleError(error);
    process.exit(1);
  }
}

// =========================================================================
// Helper Functions
// =========================================================================

async function waitForTask(
  client: ReturnType<typeof createClient>,
  taskId: string,
  timeoutMs: number
): Promise<typeof import('@aragora/sdk').Task | null> {
  const startTime = Date.now();
  const pollInterval = 2000;

  while (Date.now() - startTime < timeoutMs) {
    const task = await client.controlPlane.getTask(taskId);

    if (task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled') {
      return task;
    }

    await sleep(pollInterval);
  }

  return null;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);

  const parts: string[] = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);

  return parts.join(' ') || '< 1m';
}

function handleError(error: unknown): void {
  if (error instanceof AragoraError) {
    console.error('\n--- Aragora Error ---');
    console.error(`Message: ${error.message}`);
    console.error(`Code: ${error.code || 'N/A'}`);
    console.error(`Status: ${error.status || 'N/A'}`);

    // Common control plane errors
    if (error.code === 'AGENT_TIMEOUT') {
      console.error('\nNote: Agent did not respond within the timeout period.');
      console.error('Check agent health and connectivity.');
    } else if (error.code === 'NOT_FOUND' && error.details?.resource_type === 'task') {
      console.error('\nNote: Task not found. It may have expired or been cleaned up.');
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
