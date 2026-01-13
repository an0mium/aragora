/**
 * Aragora WebSocket Burst Test
 *
 * Tests WebSocket connection handling under burst load.
 * Run with: k6 run tests/load/websocket_burst.js --vus 100 --duration 30s
 */

import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Rate, Counter, Trend } from 'k6/metrics';

// Custom metrics
const connectionErrors = new Rate('ws_connection_errors');
const messageErrors = new Rate('ws_message_errors');
const connectionTime = new Trend('ws_connection_time');
const messagesSent = new Counter('ws_messages_sent');
const messagesReceived = new Counter('ws_messages_received');

// Configuration
const WS_URL = __ENV.WS_URL || 'ws://localhost:8766';

// Test options
export const options = {
  thresholds: {
    ws_connection_errors: ['rate<0.05'],  // Less than 5% connection failures
    ws_message_errors: ['rate<0.01'],     // Less than 1% message failures
    ws_connection_time: ['p(95)<1000'],   // 95% connect in under 1s
  },
  scenarios: {
    burst: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '5s', target: 50 },   // Ramp up
        { duration: '20s', target: 100 }, // Sustain burst
        { duration: '5s', target: 0 },    // Ramp down
      ],
    },
  },
};

export default function() {
  const startTime = Date.now();

  const res = ws.connect(WS_URL, {}, function(socket) {
    connectionTime.add(Date.now() - startTime);

    socket.on('open', function() {
      connectionErrors.add(0);

      // Subscribe to debate updates
      const subscribeMsg = JSON.stringify({
        type: 'subscribe',
        channel: 'debates',
      });
      socket.send(subscribeMsg);
      messagesSent.add(1);

      // Send ping
      socket.send(JSON.stringify({ type: 'ping' }));
      messagesSent.add(1);
    });

    socket.on('message', function(data) {
      messagesReceived.add(1);
      messageErrors.add(0);

      try {
        const msg = JSON.parse(data);
        check(msg, {
          'message has type': (m) => m.type !== undefined,
        });
      } catch (e) {
        messageErrors.add(1);
      }
    });

    socket.on('error', function(e) {
      connectionErrors.add(1);
      console.error(`WebSocket error: ${e.message || e}`);
    });

    socket.on('close', function() {
      // Normal close
    });

    // Keep connection open for a short time
    socket.setTimeout(function() {
      socket.close();
    }, 5000);
  });

  // Check connection result
  check(res, {
    'WebSocket connected': (r) => r && r.status === 101,
  });

  if (!res || res.status !== 101) {
    connectionErrors.add(1);
  }

  sleep(1);
}
