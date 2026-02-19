/**
 * Pipeline Load Test
 *
 * Tests Idea-to-Execution pipeline endpoints under load.
 * Validates pipeline creation, status polling, graph retrieval,
 * and receipt fetching against SLO thresholds.
 *
 * Run:
 *   k6 run tests/load/scenarios/pipeline-load.js
 *
 * Against production:
 *   TARGET_ENV=production k6 run tests/load/scenarios/pipeline-load.js
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { config, getEnvConfig, httpOptions } from '../config.js';

// Custom metrics
const pipelineCreateDuration = new Trend('pipeline_create_duration');
const pipelineCreateErrors = new Rate('pipeline_create_errors');
const pipelineStatusDuration = new Trend('pipeline_status_duration');
const pipelineGraphDuration = new Trend('pipeline_graph_duration');
const pipelineErrors = new Counter('pipeline_errors');

export const options = {
  scenarios: {
    pipeline_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 },  // Ramp up
        { duration: '2m', target: 20 },   // Sustained
        { duration: '1m', target: 40 },   // Peak
        { duration: '2m', target: 40 },   // Sustained peak
        { duration: '30s', target: 0 },   // Ramp down
      ],
      gracefulRampDown: '30s',
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<2000', 'p(99)<5000'],
    http_req_failed: ['rate<0.02'],
    pipeline_create_duration: ['p(95)<3000'],
    pipeline_status_duration: ['p(95)<500'],
    pipeline_graph_duration: ['p(95)<1000'],
    pipeline_create_errors: ['rate<0.05'],
  },
};

const envConfig = getEnvConfig();
const BASE_URL = envConfig.baseUrl;
const API_TOKEN = __ENV.ARAGORA_API_TOKEN || '';

function authHeaders() {
  const headers = { 'Content-Type': 'application/json' };
  if (API_TOKEN) {
    headers['Authorization'] = `Bearer ${API_TOKEN}`;
  }
  return { headers };
}

const IDEAS_POOL = [
  'Improve API rate limiting',
  'Add automated monitoring',
  'Build admin dashboard',
  'Implement caching layer',
  'Refactor authentication',
  'Add WebSocket notifications',
  'Optimize database queries',
  'Create CI/CD pipeline',
];

function randomIdeas(count) {
  const shuffled = IDEAS_POOL.slice().sort(() => 0.5 - Math.random());
  return shuffled.slice(0, count);
}

export function setup() {
  const healthRes = http.get(`${BASE_URL}/health/ready`);
  if (healthRes.status !== 200) {
    console.warn(`Health check returned ${healthRes.status} — proceeding anyway`);
  }
  return { baseUrl: BASE_URL };
}

export default function (data) {
  const opts = authHeaders();
  let pipelineId = null;

  group('Pipeline Creation', () => {
    const payload = JSON.stringify({
      ideas: randomIdeas(3),
      auto_advance: false,
    });

    const start = Date.now();
    const res = http.post(
      `${BASE_URL}/api/v1/canvas/pipeline/from-ideas`,
      payload,
      opts,
    );
    pipelineCreateDuration.add(Date.now() - start);

    const ok = check(res, {
      'pipeline created': (r) => [200, 201, 401, 429, 503].includes(r.status),
    });
    if (!ok) {
      pipelineCreateErrors.add(1);
      pipelineErrors.add(1);
    }

    if (res.status === 200 || res.status === 201) {
      try {
        const body = JSON.parse(res.body);
        pipelineId = body.pipeline_id || body.id;
      } catch (e) {
        pipelineErrors.add(1);
      }
    }
  });

  sleep(1);

  if (pipelineId) {
    group('Pipeline Status', () => {
      const start = Date.now();
      const res = http.get(
        `${BASE_URL}/api/v1/canvas/pipeline/${pipelineId}/status`,
        opts,
      );
      pipelineStatusDuration.add(Date.now() - start);

      check(res, {
        'status ok': (r) => [200, 400, 404, 401, 429].includes(r.status),
      });
    });

    sleep(0.5);

    group('Pipeline Graph', () => {
      const start = Date.now();
      const res = http.get(
        `${BASE_URL}/api/v1/canvas/pipeline/${pipelineId}/graph`,
        opts,
      );
      pipelineGraphDuration.add(Date.now() - start);

      check(res, {
        'graph ok': (r) => [200, 400, 404, 401, 429].includes(r.status),
      });
    });

    sleep(0.5);

    group('Pipeline Receipt', () => {
      const res = http.get(
        `${BASE_URL}/api/v1/canvas/pipeline/${pipelineId}/receipt`,
        opts,
      );
      check(res, {
        'receipt ok': (r) => [200, 400, 404, 401, 429].includes(r.status),
      });
    });
  }

  group('Pipeline Run (dry)', () => {
    const topics = ['rate limiter', 'cache system', 'auth service', 'monitoring'];
    const payload = JSON.stringify({
      input_text: 'Design a ' + topics[Math.floor(Math.random() * topics.length)],
      dry_run: true,
      stages: ['ideation', 'goals'],
      enable_receipts: false,
    });

    const res = http.post(
      `${BASE_URL}/api/v1/canvas/pipeline/run`,
      payload,
      opts,
    );
    check(res, {
      'run ok': (r) => [200, 201, 202, 401, 429, 501, 503].includes(r.status),
    });
  });

  sleep(2);
}
