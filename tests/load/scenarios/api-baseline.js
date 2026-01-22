/**
 * API Baseline Load Test
 *
 * Tests core API endpoints under normal load conditions.
 * Validates response times, error rates, and throughput.
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { config, getEnvConfig, httpOptions, randomQuestion, randomUserId } from '../config.js';

// Custom metrics
const debateCreateDuration = new Trend('debate_creation_duration');
const debateCreateErrors = new Rate('debate_creation_errors');
const apiErrors = new Counter('api_errors');

// Test configuration
export const options = {
  scenarios: {
    api_baseline: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 20 },   // Ramp up
        { duration: '3m', target: 50 },   // Stay at 50
        { duration: '1m', target: 100 },  // Increase
        { duration: '3m', target: 100 },  // Stay at 100
        { duration: '1m', target: 0 },    // Ramp down
      ],
      gracefulRampDown: '30s',
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
    debate_creation_duration: ['p(95)<2000'],
    debate_creation_errors: ['rate<0.02'],
  },
};

const envConfig = getEnvConfig();
const BASE_URL = envConfig.baseUrl;

// Setup function - runs once at the start
export function setup() {
  // Verify API is accessible
  const healthRes = http.get(`${BASE_URL}/health/ready`);
  if (healthRes.status !== 200) {
    throw new Error(`API health check failed: ${healthRes.status}`);
  }
  return { startTime: Date.now() };
}

// Main test function
export default function (data) {
  const userId = randomUserId();
  const opts = httpOptions();

  group('Health Checks', () => {
    const liveRes = http.get(`${BASE_URL}/health/live`, opts);
    check(liveRes, {
      'health/live returns 200': (r) => r.status === 200,
    });

    const readyRes = http.get(`${BASE_URL}/health/ready`, opts);
    check(readyRes, {
      'health/ready returns 200': (r) => r.status === 200,
      'health/ready response time < 100ms': (r) => r.timings.duration < 100,
    });
  });

  sleep(0.5);

  group('API Discovery', () => {
    const capabilitiesRes = http.get(`${BASE_URL}/api/capabilities`, opts);
    check(capabilitiesRes, {
      'capabilities returns 200': (r) => r.status === 200,
      'capabilities has expected fields': (r) => {
        const body = JSON.parse(r.body);
        return body.version && body.features;
      },
    });
  });

  sleep(0.5);

  group('Debate Operations', () => {
    // Create debate
    const createPayload = JSON.stringify({
      question: randomQuestion(),
      agents: ['claude', 'gpt4'],
      protocol: {
        rounds: 2,
        consensus: 'majority',
      },
      context: {
        user_id: userId,
      },
    });

    const startTime = Date.now();
    const createRes = http.post(`${BASE_URL}/api/debates`, createPayload, opts);
    const duration = Date.now() - startTime;

    debateCreateDuration.add(duration);

    const createSuccess = check(createRes, {
      'debate create returns 201 or 200': (r) => r.status === 201 || r.status === 200,
      'debate create has debate_id': (r) => {
        if (r.status === 201 || r.status === 200) {
          const body = JSON.parse(r.body);
          return body.debate_id || body.id;
        }
        return false;
      },
    });

    if (!createSuccess) {
      debateCreateErrors.add(1);
      apiErrors.add(1);
      return;
    }

    const debateBody = JSON.parse(createRes.body);
    const debateId = debateBody.debate_id || debateBody.id;

    sleep(1);

    // Get debate status
    const statusRes = http.get(`${BASE_URL}/api/debates/${debateId}`, opts);
    check(statusRes, {
      'debate status returns 200': (r) => r.status === 200,
      'debate status has expected fields': (r) => {
        const body = JSON.parse(r.body);
        return body.status !== undefined;
      },
    });

    sleep(0.5);

    // List debates
    const listRes = http.get(`${BASE_URL}/api/debates?limit=10`, opts);
    check(listRes, {
      'debate list returns 200': (r) => r.status === 200,
    });
  });

  sleep(1);

  group('Agent Endpoints', () => {
    const agentsRes = http.get(`${BASE_URL}/api/agents`, opts);
    check(agentsRes, {
      'agents list returns 200': (r) => r.status === 200,
      'agents list contains agents': (r) => {
        const body = JSON.parse(r.body);
        return Array.isArray(body.agents) && body.agents.length > 0;
      },
    });
  });

  sleep(0.5);
}

// Teardown function - runs once at the end
export function teardown(data) {
  console.log(`Test completed. Duration: ${(Date.now() - data.startTime) / 1000}s`);
}
