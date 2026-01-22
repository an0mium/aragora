/**
 * K6 Load Testing Configuration for Aragora
 *
 * This module provides shared configuration and utilities for load tests.
 */

// Environment-specific configuration
export const config = {
  // Target environments
  environments: {
    local: {
      baseUrl: 'http://localhost:8080',
      wsUrl: 'ws://localhost:8080',
    },
    staging: {
      baseUrl: 'https://api.staging.aragora.ai',
      wsUrl: 'wss://api.staging.aragora.ai',
    },
    production: {
      baseUrl: 'https://api.aragora.ai',
      wsUrl: 'wss://api.aragora.ai',
    },
  },

  // Test thresholds (SLO-aligned)
  thresholds: {
    // API response times
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    // Error rate
    http_req_failed: ['rate<0.01'],
    // WebSocket connection time
    ws_connecting: ['p(95)<200'],
    // Custom debate metrics
    debate_creation_duration: ['p(95)<2000'],
    debate_round_duration: ['p(95)<30000'],
  },

  // Load profiles
  profiles: {
    smoke: {
      vus: 5,
      duration: '1m',
    },
    baseline: {
      vus: 50,
      duration: '5m',
    },
    stress: {
      stages: [
        { duration: '2m', target: 50 },
        { duration: '5m', target: 200 },
        { duration: '2m', target: 500 },
        { duration: '5m', target: 500 },
        { duration: '2m', target: 0 },
      ],
    },
    spike: {
      stages: [
        { duration: '1m', target: 50 },
        { duration: '30s', target: 1000 },
        { duration: '1m', target: 1000 },
        { duration: '30s', target: 50 },
        { duration: '2m', target: 50 },
      ],
    },
    soak: {
      vus: 100,
      duration: '1h',
    },
  },
};

// Helper to get environment config
export function getEnvConfig() {
  const env = __ENV.TARGET_ENV || 'local';
  return config.environments[env];
}

// HTTP options with default headers
export function httpOptions(token = null) {
  const headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  return { headers };
}

// Generate random debate question
export function randomQuestion() {
  const topics = [
    'What is the best approach to implement rate limiting?',
    'Should we use microservices or monolith architecture?',
    'What are the trade-offs between SQL and NoSQL databases?',
    'How should we handle authentication in a distributed system?',
    'What is the optimal caching strategy for our use case?',
    'Should we prioritize performance or maintainability?',
    'What testing strategy provides the best ROI?',
    'How should we approach technical debt?',
    'What is the best way to scale our application?',
    'Should we build or buy this component?',
  ];
  return topics[Math.floor(Math.random() * topics.length)];
}

// Generate random user ID
export function randomUserId() {
  return `user_${Math.random().toString(36).substring(2, 15)}`;
}

// Generate random workspace ID
export function randomWorkspaceId() {
  return `ws_${Math.random().toString(36).substring(2, 10)}`;
}
