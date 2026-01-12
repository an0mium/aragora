/**
 * Basic Debate Example
 *
 * Demonstrates how to create and run a simple debate using the Aragora SDK.
 */

import { AragoraClient } from '../src';

async function main() {
  // Initialize the client
  const client = new AragoraClient({
    baseUrl: process.env.ARAGORA_API_URL || 'http://localhost:8080',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  // Check server health
  const health = await client.health();
  console.log('Server status:', health.status);

  // Create and run a debate
  console.log('\nStarting debate...');
  const debate = await client.debates.run({
    task: 'Should companies adopt a 4-day work week?',
    agents: ['claude-sonnet', 'gpt-4'],
    rounds: 3,
    protocol: {
      type: 'adversarial',
      consensus_threshold: 0.7,
    },
  });

  // Display results
  console.log('\n=== Debate Results ===');
  console.log(`ID: ${debate.id}`);
  console.log(`Status: ${debate.status}`);
  console.log(`Rounds: ${debate.rounds?.length || 0}`);

  if (debate.consensus) {
    console.log(`\nConsensus reached: ${debate.consensus.position}`);
    console.log(`Confidence: ${(debate.consensus.confidence * 100).toFixed(1)}%`);
  }

  // Display messages
  console.log('\n=== Messages ===');
  for (const round of debate.rounds || []) {
    console.log(`\nRound ${round.number}:`);
    for (const msg of round.messages || []) {
      console.log(`  [${msg.agent}]: ${msg.content.substring(0, 100)}...`);
    }
  }
}

main().catch(console.error);
