/**
 * Streaming Example
 *
 * Demonstrates real-time debate streaming using WebSocket.
 */

import { AragoraClient, DebateStream, streamDebate } from '../src';

const BASE_URL = process.env.ARAGORA_API_URL || 'http://localhost:8080';

/**
 * Example 1: Using DebateStream class with event handlers
 */
async function streamWithEventHandlers() {
  console.log('=== Event Handler Streaming ===\n');

  const client = new AragoraClient({
    baseUrl: BASE_URL,
    apiKey: process.env.ARAGORA_API_KEY,
  });

  // Create a debate first
  const created = await client.debates.create({
    task: 'What is the best approach to microservices vs monolith?',
    agents: ['claude-sonnet', 'gpt-4'],
    max_rounds: 2,
  });

  console.log(`Debate created: ${created.debate_id}`);
  console.log('Connecting to stream...\n');

  // Create stream with custom options
  const stream = new DebateStream(BASE_URL, created.debate_id, {
    reconnect: true,
    reconnectInterval: 1000,
    maxReconnectAttempts: 3,
  });

  // Set up event handlers
  stream.on('debate_start', (event) => {
    console.log('[START]', event.data);
  });

  stream.on('round_start', (event) => {
    console.log(`\n--- Round ${event.data.round} ---`);
  });

  stream.on('agent_message', (event) => {
    const { agent_id, content } = event.data as { agent_id: string; content: string };
    console.log(`[${agent_id}]: ${content.substring(0, 100)}...`);
  });

  stream.on('critique', (event) => {
    const { critic_id, content } = event.data as { critic_id: string; content: string };
    console.log(`[CRITIQUE by ${critic_id}]: ${content.substring(0, 80)}...`);
  });

  stream.on('consensus', (event) => {
    console.log('\n[CONSENSUS]:', event.data);
  });

  stream.on('debate_end', (event) => {
    console.log('\n[END]:', event.data);
    stream.disconnect();
  });

  stream.onError((error) => {
    console.error('Stream error:', error.message);
  });

  stream.onClose((code, reason) => {
    console.log(`\nConnection closed: ${code} - ${reason}`);
  });

  // Connect and wait
  await stream.connect();
  console.log('Connected! Waiting for events...\n');

  // Keep process alive until debate ends
  await new Promise((resolve) => {
    stream.onClose(() => resolve(undefined));
  });
}

/**
 * Example 2: Using async iterator for simpler consumption
 */
async function streamWithAsyncIterator() {
  console.log('=== Async Iterator Streaming ===\n');

  const client = new AragoraClient({
    baseUrl: BASE_URL,
    apiKey: process.env.ARAGORA_API_KEY,
  });

  // Create a debate
  const created = await client.debates.create({
    task: 'Should AI systems be required to explain their decisions?',
    agents: ['claude-sonnet', 'gpt-4'],
    max_rounds: 2,
  });

  console.log(`Debate created: ${created.debate_id}`);
  console.log('Streaming events...\n');

  // Use async iteration - cleaner for sequential processing
  for await (const event of streamDebate(BASE_URL, created.debate_id)) {
    switch (event.type) {
      case 'debate_start':
        console.log('[START]', event.timestamp);
        break;

      case 'round_start':
        console.log(`\n--- Round ${event.data.round} ---`);
        break;

      case 'agent_message': {
        const { agent_id, content } = event.data as { agent_id: string; content: string };
        console.log(`[${agent_id}]: ${content.substring(0, 100)}...`);
        break;
      }

      case 'consensus':
        console.log('\n[CONSENSUS]:', event.data);
        break;

      case 'debate_end':
        console.log('\n[END] Debate completed');
        break;

      case 'error':
        console.error('[ERROR]:', event.data);
        break;
    }

    // Exit loop when debate ends
    if (event.type === 'debate_end') {
      break;
    }
  }

  console.log('\nStream finished');
}

/**
 * Example 3: Filtering specific event types
 */
async function streamFilteredEvents() {
  console.log('=== Filtered Event Streaming ===\n');

  const client = new AragoraClient({
    baseUrl: BASE_URL,
    apiKey: process.env.ARAGORA_API_KEY,
  });

  const created = await client.debates.create({
    task: 'What are the ethical implications of autonomous vehicles?',
    agents: ['claude-sonnet', 'gpt-4'],
    max_rounds: 2,
  });

  console.log(`Debate: ${created.debate_id}`);
  console.log('Showing only agent messages and consensus...\n');

  // Filter to only agent messages and consensus
  const relevantTypes = new Set(['agent_message', 'consensus', 'debate_end']);

  for await (const event of streamDebate(BASE_URL, created.debate_id)) {
    if (!relevantTypes.has(event.type)) {
      continue;
    }

    if (event.type === 'agent_message') {
      const { agent_id, content, round } = event.data as {
        agent_id: string;
        content: string;
        round: number;
      };
      console.log(`[Round ${round}] ${agent_id}:`);
      console.log(`  ${content.substring(0, 150)}...\n`);
    } else if (event.type === 'consensus') {
      const { reached, conclusion } = event.data as {
        reached: boolean;
        conclusion?: string;
      };
      console.log(`\nConsensus: ${reached ? 'Reached' : 'Not reached'}`);
      if (conclusion) {
        console.log(`Conclusion: ${conclusion}`);
      }
    } else if (event.type === 'debate_end') {
      break;
    }
  }
}

// Run examples
async function main() {
  const example = process.argv[2] || 'iterator';

  switch (example) {
    case 'handlers':
      await streamWithEventHandlers();
      break;
    case 'iterator':
      await streamWithAsyncIterator();
      break;
    case 'filtered':
      await streamFilteredEvents();
      break;
    default:
      console.log('Usage: npx ts-node streaming-example.ts [handlers|iterator|filtered]');
  }
}

main().catch(console.error);
