/**
 * Stress Test
 *
 * Tests system behavior under extreme load to find breaking points.
 * Identifies performance degradation patterns and system limits.
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { getEnvConfig, httpOptions, randomQuestion, randomUserId, randomWorkspaceId } from '../config.js';

// Custom metrics
const responseTime = new Trend('response_time');
const errorRate = new Rate('error_rate');
const requestsPerSecond = new Counter('requests_total');
const timeouts = new Counter('timeouts');
const serverErrors = new Counter('server_errors');

export const options = {
  scenarios: {
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },   // Ramp to 100
        { duration: '5m', target: 100 },   // Stay at 100
        { duration: '2m', target: 300 },   // Ramp to 300
        { duration: '5m', target: 300 },   // Stay at 300
        { duration: '2m', target: 500 },   // Ramp to 500
        { duration: '5m', target: 500 },   // Stay at 500
        { duration: '2m', target: 700 },   // Ramp to 700
        { duration: '3m', target: 700 },   // Stay at 700
        { duration: '3m', target: 0 },     // Ramp down
      ],
      gracefulRampDown: '1m',
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<2000', 'p(99)<5000'],
    http_req_failed: ['rate<0.10'],  // Allow up to 10% errors under stress
    error_rate: ['rate<0.10'],
  },
};

const envConfig = getEnvConfig();
const BASE_URL = envConfig.baseUrl;

export default function () {
  const userId = randomUserId();
  const workspaceId = randomWorkspaceId();
  const opts = httpOptions();

  requestsPerSecond.add(1);

  // Mix of different API operations to simulate realistic load

  // 40% - Health checks (lightweight)
  if (Math.random() < 0.4) {
    const res = http.get(`${BASE_URL}/health/ready`, opts);
    responseTime.add(res.timings.duration);

    const success = check(res, {
      'health check success': (r) => r.status === 200,
    });

    if (!success) {
      errorRate.add(1);
      if (res.status >= 500) serverErrors.add(1);
      if (res.timings.duration > 30000) timeouts.add(1);
    }
    sleep(0.1);
    return;
  }

  // 30% - Read operations
  if (Math.random() < 0.5) {
    group('Read Operations', () => {
      // List debates
      const listRes = http.get(`${BASE_URL}/api/debates?limit=20`, opts);
      responseTime.add(listRes.timings.duration);

      check(listRes, {
        'list debates success': (r) => r.status === 200,
      }) || errorRate.add(1);

      sleep(0.2);

      // List agents
      const agentsRes = http.get(`${BASE_URL}/api/agents`, opts);
      responseTime.add(agentsRes.timings.duration);

      check(agentsRes, {
        'list agents success': (r) => r.status === 200,
      }) || errorRate.add(1);
    });
    sleep(0.3);
    return;
  }

  // 30% - Write operations (create debates)
  group('Write Operations', () => {
    const payload = JSON.stringify({
      question: randomQuestion(),
      agents: ['claude', 'gpt4'],
      protocol: { rounds: 1, consensus: 'majority' },
      context: {
        user_id: userId,
        workspace_id: workspaceId,
      },
    });

    const res = http.post(`${BASE_URL}/api/debates`, payload, opts);
    responseTime.add(res.timings.duration);

    const success = check(res, {
      'create debate success': (r) => r.status === 200 || r.status === 201,
    });

    if (!success) {
      errorRate.add(1);
      if (res.status >= 500) serverErrors.add(1);
      if (res.timings.duration > 30000) timeouts.add(1);
    }
  });

  sleep(0.5);
}

export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'stress-test-results.json': JSON.stringify(data, null, 2),
  };
}

function textSummary(data, opts) {
  const metrics = data.metrics;
  return `
Stress Test Summary
===================

Requests:
  Total: ${metrics.http_reqs.values.count}
  Rate: ${metrics.http_reqs.values.rate.toFixed(2)}/s

Response Times:
  p50: ${metrics.http_req_duration.values['p(50)'].toFixed(2)}ms
  p90: ${metrics.http_req_duration.values['p(90)'].toFixed(2)}ms
  p95: ${metrics.http_req_duration.values['p(95)'].toFixed(2)}ms
  p99: ${metrics.http_req_duration.values['p(99)'].toFixed(2)}ms

Errors:
  Failed: ${metrics.http_req_failed.values.passes} (${(metrics.http_req_failed.values.rate * 100).toFixed(2)}%)
  Server Errors: ${metrics.server_errors?.values.count || 0}
  Timeouts: ${metrics.timeouts?.values.count || 0}
`;
}
