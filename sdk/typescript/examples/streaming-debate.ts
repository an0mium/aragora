/**
 * Streaming Debate Example
 *
 * Demonstrates real-time debate streaming using WebSockets.
 * Events are received as they happen during the debate.
 *
 * Usage:
 *   npx ts-node examples/streaming-debate.ts
 */

import { createClient } from '@aragora/sdk';

async function main() {
  const client = createClient({
    baseUrl: process.env.ARAGORA_API_URL || 'https://api.aragora.ai',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  // Create WebSocket connection
  const ws = client.createWebSocket();

  // Set up event handlers before connecting
  ws.on('connected', () => {
    console.log('WebSocket connected');
  });

  ws.on('debate_start', (event) => {
    console.log(`\n=== Debate Started: ${event.debate_id} ===`);
    console.log(`Task: ${event.task}`);
    console.log(`Agents: ${event.agents.join(', ')}`);
  });

  ws.on('round_start', (event) => {
    console.log(`\n--- Round ${event.round_number} ---`);
  });

  ws.on('agent_message', (event) => {
    console.log(`\n[${event.agent}]:`);
    console.log(event.content.substring(0, 200) + '...');
  });

  ws.on('critique', (event) => {
    console.log(`\n[${event.critic} critiques ${event.target}]:`);
    console.log(`Score: ${event.score}/10`);
  });

  ws.on('vote', (event) => {
    console.log(`Vote: ${event.voter} -> ${event.choice}`);
  });

  ws.on('consensus', (event) => {
    console.log('\n=== Consensus Reached ===');
    console.log(`Answer: ${event.consensus.final_answer}`);
    console.log(`Confidence: ${(event.consensus.confidence * 100).toFixed(1)}%`);
  });

  ws.on('debate_end', (event) => {
    console.log(`\nDebate ended: ${event.status}`);
    ws.close();
  });

  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });

  // Connect
  await ws.connect();

  // Create debate
  const debate = await client.debates.create({
    task: 'Should AI systems be required to explain their decisions?',
    agents: ['claude', 'gpt-4', 'gemini'],
    protocol: {
      rounds: 2,
      consensus: 'weighted',
    },
  });

  console.log(`Created debate: ${debate.debate_id}`);

  // Subscribe to debate events
  ws.subscribe(debate.debate_id);

  // Keep process running until debate ends
  await new Promise((resolve) => {
    ws.on('debate_end', resolve);
    // Timeout after 5 minutes
    setTimeout(() => {
      console.log('Timeout reached');
      ws.close();
      resolve(undefined);
    }, 5 * 60 * 1000);
  });
}

main().catch(console.error);
