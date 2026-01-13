/**
 * Aragora Rate Limit Validation Test
 *
 * Validates that rate limiting is working correctly by:
 * 1. Testing that normal traffic passes through
 * 2. Testing that excessive traffic gets rate limited
 * 3. Verifying rate limit headers are present
 *
 * Run with: k6 run tests/load/rate_limit_test.js --vus 10 --duration 30s
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Counter } from 'k6/metrics';

// Custom metrics
const rateLimitHits = new Counter('rate_limit_hits');
const rateLimitMisses = new Counter('rate_limit_misses');
const rateLimitHeaderPresent = new Rate('rate_limit_header_present');

// Configuration
const API_URL = __ENV.API_URL || 'http://localhost:8080';

// Test options
export const options = {
  thresholds: {
    // Rate limiting should kick in for excessive requests
    rate_limit_hits: ['count>0'],
    // Headers should be present on most responses
    rate_limit_header_present: ['rate>0.9'],
  },
  scenarios: {
    // Normal traffic - should not hit rate limits
    normal: {
      executor: 'constant-arrival-rate',
      rate: 5,           // 5 requests per second
      timeUnit: '1s',
      duration: '10s',
      preAllocatedVUs: 2,
      startTime: '0s',
      exec: 'normalTraffic',
    },
    // Burst traffic - should trigger rate limits
    burst: {
      executor: 'constant-arrival-rate',
      rate: 100,         // 100 requests per second - should exceed limits
      timeUnit: '1s',
      duration: '10s',
      preAllocatedVUs: 20,
      startTime: '15s',
      exec: 'burstTraffic',
    },
  },
};

// Normal traffic test - should pass through
export function normalTraffic() {
  const res = http.get(`${API_URL}/api/health`);

  const hasRateLimitHeader = res.headers['X-RateLimit-Limit'] !== undefined ||
                             res.headers['x-ratelimit-limit'] !== undefined;
  rateLimitHeaderPresent.add(hasRateLimitHeader ? 1 : 0);

  check(res, {
    'normal: status not rate limited': (r) => r.status !== 429,
    'normal: status 200': (r) => r.status === 200,
  });

  if (res.status === 429) {
    rateLimitHits.add(1);
  } else {
    rateLimitMisses.add(1);
  }

  sleep(0.2);
}

// Burst traffic test - should trigger rate limits
export function burstTraffic() {
  group('Burst Traffic', function() {
    // Make rapid requests
    for (let i = 0; i < 5; i++) {
      const res = http.get(`${API_URL}/api/health`);

      const hasRateLimitHeader = res.headers['X-RateLimit-Limit'] !== undefined ||
                                 res.headers['x-ratelimit-limit'] !== undefined;
      rateLimitHeaderPresent.add(hasRateLimitHeader ? 1 : 0);

      if (res.status === 429) {
        rateLimitHits.add(1);

        // Check rate limit response
        check(res, {
          'burst: has Retry-After header': (r) =>
            r.headers['Retry-After'] !== undefined ||
            r.headers['retry-after'] !== undefined,
          'burst: has rate limit error': (r) => {
            try {
              const body = JSON.parse(r.body);
              return body.error && body.error.toLowerCase().includes('rate');
            } catch {
              return false;
            }
          },
        });
      } else {
        rateLimitMisses.add(1);
        check(res, {
          'burst: status 200': (r) => r.status === 200,
        });
      }
    }
  });

  sleep(0.1);
}

// Main default function (not used with scenarios, but required by k6)
export default function() {
  // This won't run when using scenarios
}

// Teardown - summarize results
export function teardown() {
  console.log('Rate limit validation complete.');
  console.log('Note: Some rate limit hits are expected during burst test.');
}
