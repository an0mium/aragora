/**
 * WebSocket Streaming Load Test
 *
 * Tests WebSocket connections and real-time debate streaming
 * under various load conditions.
 */

import ws from 'k6/ws';
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { getEnvConfig, httpOptions, randomQuestion, randomUserId } from '../config.js';

// Custom metrics
const wsConnectDuration = new Trend('ws_connecting');
const wsMessageLatency = new Trend('ws_message_latency');
const wsErrors = new Rate('ws_errors');
const messagesReceived = new Counter('ws_messages_received');

export const options = {
  scenarios: {
    websocket_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '30s', target: 10 },
        { duration: '2m', target: 50 },
        { duration: '2m', target: 100 },
        { duration: '1m', target: 0 },
      ],
    },
  },
  thresholds: {
    ws_connecting: ['p(95)<500'],
    ws_message_latency: ['p(95)<1000'],
    ws_errors: ['rate<0.05'],
  },
};

const envConfig = getEnvConfig();
const BASE_URL = envConfig.baseUrl;
const WS_URL = envConfig.wsUrl;

export default function () {
  const userId = randomUserId();
  const opts = httpOptions();

  // First, create a debate via REST
  const createPayload = JSON.stringify({
    question: randomQuestion(),
    agents: ['claude', 'gpt4'],
    protocol: { rounds: 1 },
    context: { user_id: userId },
  });

  const createRes = http.post(`${BASE_URL}/api/debates`, createPayload, opts);

  if (createRes.status !== 200 && createRes.status !== 201) {
    wsErrors.add(1);
    console.error(`Failed to create debate: ${createRes.status}`);
    return;
  }

  const debateBody = JSON.parse(createRes.body);
  const debateId = debateBody.debate_id || debateBody.id;

  // Connect to WebSocket for streaming
  const wsUrl = `${WS_URL}/ws/debates/${debateId}`;
  const connectStart = Date.now();

  const res = ws.connect(wsUrl, {}, function (socket) {
    const connectDuration = Date.now() - connectStart;
    wsConnectDuration.add(connectDuration);

    socket.on('open', () => {
      check(socket, {
        'WebSocket connected': () => true,
      });

      // Subscribe to debate events
      socket.send(JSON.stringify({
        type: 'subscribe',
        debate_id: debateId,
      }));
    });

    socket.on('message', (msg) => {
      messagesReceived.add(1);

      try {
        const data = JSON.parse(msg);
        const now = Date.now();

        if (data.timestamp) {
          const latency = now - new Date(data.timestamp).getTime();
          wsMessageLatency.add(latency);
        }

        check(data, {
          'message has type': (d) => d.type !== undefined,
        });

        // Close on debate completion
        if (data.type === 'debate_end' || data.type === 'complete') {
          socket.close();
        }
      } catch (e) {
        console.error(`Failed to parse message: ${e}`);
      }
    });

    socket.on('error', (e) => {
      wsErrors.add(1);
      console.error(`WebSocket error: ${e}`);
    });

    socket.on('close', () => {
      // Connection closed
    });

    // Keep connection open for debate duration (max 60s)
    socket.setTimeout(() => {
      socket.close();
    }, 60000);
  });

  check(res, {
    'WebSocket connection successful': (r) => r && r.status === 101,
  });

  sleep(2);
}
