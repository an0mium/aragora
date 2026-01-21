/**
 * Aragora API Load Test
 *
 * Run with: k6 run tests/load/k6_api_test.js
 *
 * Options:
 *   k6 run --vus 10 --duration 30s tests/load/k6_api_test.js
 *   k6 run --vus 50 --duration 5m tests/load/k6_api_test.js
 */

import http from "k6/http";
import { check, sleep } from "k6";
import { Rate, Trend } from "k6/metrics";

// Custom metrics
const errorRate = new Rate("errors");
const debateLatency = new Trend("debate_latency");
const agentLatency = new Trend("agent_latency");

// Configuration
const BASE_URL = __ENV.ARAGORA_URL || "http://localhost:8080";
const API_TOKEN = __ENV.ARAGORA_API_TOKEN || "";

// Test options
export const options = {
  stages: [
    { duration: "30s", target: 10 }, // Ramp up to 10 users
    { duration: "1m", target: 10 }, // Stay at 10 users
    { duration: "30s", target: 50 }, // Ramp up to 50 users
    { duration: "2m", target: 50 }, // Stay at 50 users
    { duration: "30s", target: 0 }, // Ramp down
  ],
  thresholds: {
    http_req_duration: ["p(95)<500"], // 95% of requests under 500ms
    errors: ["rate<0.1"], // Error rate under 10%
    debate_latency: ["p(95)<2000"], // Debate creation under 2s
  },
};

// Headers
const headers = {
  "Content-Type": "application/json",
  ...(API_TOKEN && { Authorization: `Bearer ${API_TOKEN}` }),
};

// Health check
export function healthCheck() {
  const res = http.get(`${BASE_URL}/health`);
  check(res, {
    "health check status is 200": (r) => r.status === 200,
  });
  errorRate.add(res.status !== 200);
}

// List agents
export function listAgents() {
  const res = http.get(`${BASE_URL}/api/v1/agents`, { headers });
  check(res, {
    "agents list status is 200": (r) => r.status === 200,
    "agents list has data": (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.agents && body.agents.length > 0;
      } catch {
        return false;
      }
    },
  });
  agentLatency.add(res.timings.duration);
  errorRate.add(res.status !== 200);
}

// Get agent rankings
export function getAgentRankings() {
  const res = http.get(`${BASE_URL}/api/v1/agents/rankings`, { headers });
  check(res, {
    "rankings status is 200": (r) => r.status === 200,
  });
  errorRate.add(res.status !== 200);
}

// Create debate
export function createDebate() {
  const payload = JSON.stringify({
    task: `Load test debate ${Date.now()}`,
    agents: ["claude", "gpt4"],
    protocol: {
      rounds: 2,
      consensus_mode: "majority",
    },
  });

  const res = http.post(`${BASE_URL}/api/v1/debates`, payload, { headers });
  check(res, {
    "debate creation status is 200 or 201": (r) =>
      r.status === 200 || r.status === 201,
  });
  debateLatency.add(res.timings.duration);
  errorRate.add(res.status !== 200 && res.status !== 201);

  return res;
}

// Get debate history
export function getDebateHistory() {
  const res = http.get(`${BASE_URL}/api/v1/debates?limit=10`, { headers });
  check(res, {
    "debate history status is 200": (r) => r.status === 200,
  });
  errorRate.add(res.status !== 200);
}

// Get consensus memory
export function getConsensusMemory() {
  const res = http.get(`${BASE_URL}/api/v1/consensus`, { headers });
  check(res, {
    "consensus memory status is 200": (r) => r.status === 200,
  });
  errorRate.add(res.status !== 200);
}

// Get metrics
export function getMetrics() {
  const res = http.get(`${BASE_URL}/metrics`);
  check(res, {
    "metrics status is 200": (r) => r.status === 200,
  });
  errorRate.add(res.status !== 200);
}

// Main test scenario
export default function () {
  // Mix of different operations
  const scenario = Math.random();

  if (scenario < 0.3) {
    // 30% health checks
    healthCheck();
  } else if (scenario < 0.5) {
    // 20% list agents
    listAgents();
  } else if (scenario < 0.65) {
    // 15% get rankings
    getAgentRankings();
  } else if (scenario < 0.75) {
    // 10% debate history
    getDebateHistory();
  } else if (scenario < 0.85) {
    // 10% consensus memory
    getConsensusMemory();
  } else if (scenario < 0.95) {
    // 10% metrics
    getMetrics();
  } else {
    // 5% create debate (expensive operation)
    createDebate();
  }

  // Think time between requests
  sleep(Math.random() * 2 + 0.5);
}

// Setup function - runs once before test
export function setup() {
  // Verify server is reachable
  const res = http.get(`${BASE_URL}/health`);
  if (res.status !== 200) {
    throw new Error(`Server not reachable at ${BASE_URL}`);
  }
  console.log(`Testing against: ${BASE_URL}`);
  return { startTime: Date.now() };
}

// Teardown function - runs once after test
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Test completed in ${duration.toFixed(2)}s`);
}
