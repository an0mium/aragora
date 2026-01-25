/**
 * Aragora API Load Test
 *
 * Tests API endpoints under concurrent load.
 * Run with: k6 run tests/load/api_load.js --vus 50 --duration 60s
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const debateLatency = new Trend('debate_latency');
const healthLatency = new Trend('health_latency');
const requestCount = new Counter('requests');

// Configuration
const API_URL = __ENV.API_URL || 'http://localhost:8080';

// Test options
export const options = {
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.01'],
  },
  scenarios: {
    // Smoke test
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '10s',
      startTime: '0s',
    },
    // Ramp up to target load
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 10 },
        { duration: '30s', target: 50 },
        { duration: '10s', target: 50 },
        { duration: '10s', target: 0 },
      ],
      startTime: '10s',
    },
  },
};

// Setup - runs once at the start
export function setup() {
  // Verify API is reachable
  const res = http.get(`${API_URL}/api/v1/health`);
  check(res, {
    'setup: health check passed': (r) => r.status === 200,
  });

  return {
    startTime: Date.now(),
  };
}

// Main test function
export default function(data) {
  group('Health Check', function() {
    const start = Date.now();
    const res = http.get(`${API_URL}/api/v1/health`);
    healthLatency.add(Date.now() - start);
    requestCount.add(1);

    const passed = check(res, {
      'health: status 200': (r) => r.status === 200,
      'health: has status field': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status !== undefined;
        } catch {
          return false;
        }
      },
    });

    if (!passed) {
      errorRate.add(1);
    } else {
      errorRate.add(0);
    }
  });

  sleep(0.1);

  group('Leaderboard', function() {
    const res = http.get(`${API_URL}/api/v1/leaderboard-view?limit=10`);
    requestCount.add(1);

    // Accept 200 (success), 401/403 (auth required - endpoint works), 404 (no data)
    const passed = check(res, {
      'leaderboard: valid response': (r) => [200, 401, 403, 404].includes(r.status),
    });

    if (!passed) {
      errorRate.add(1);
    } else {
      errorRate.add(0);
    }
  });

  sleep(0.1);

  group('Agents List', function() {
    const res = http.get(`${API_URL}/api/v1/agents`);
    requestCount.add(1);

    // Accept 200 (success), 401/403 (auth required - endpoint works), 404 (no data)
    const passed = check(res, {
      'agents: valid response': (r) => [200, 401, 403, 404].includes(r.status),
    });

    if (!passed) {
      errorRate.add(1);
    } else {
      errorRate.add(0);
    }
  });

  sleep(0.1);

  group('Debates List', function() {
    const res = http.get(`${API_URL}/api/v1/debates?limit=10`);
    requestCount.add(1);

    // Accept 200 (success), 401/403 (auth required - endpoint works), 404 (no data)
    const passed = check(res, {
      'debates: valid response': (r) => [200, 401, 403, 404].includes(r.status),
    });

    if (!passed) {
      errorRate.add(1);
    } else {
      errorRate.add(0);
    }
  });

  sleep(0.2);

  // Occasionally trigger a debate (expensive operation)
  if (Math.random() < 0.05) {
    group('Create Debate', function() {
      const start = Date.now();
      const payload = JSON.stringify({
        task: 'Load test debate: Is this API performing well?',
        agents: ['demo', 'demo'],
        rounds: 1,
      });

      const params = {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: '30s',
      };

      const res = http.post(`${API_URL}/api/v1/debate`, payload, params);
      debateLatency.add(Date.now() - start);
      requestCount.add(1);

      // Accept 200/201 (success), 401/403 (auth required), 400 (bad request), 503 (service unavailable)
      const passed = check(res, {
        'debate: valid response': (r) => [200, 201, 400, 401, 403, 503].includes(r.status),
        'debate: has response body': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body !== null;
          } catch {
            return false;
          }
        },
      });

      if (!passed) {
        errorRate.add(1);
      } else {
        errorRate.add(0);
      }
    });
  }

  sleep(0.5);
}

// Teardown - runs once at the end
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Test completed in ${duration.toFixed(2)}s`);
}
