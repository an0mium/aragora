/**
 * Basic Debate Example
 *
 * Demonstrates how to create and run a simple debate using the Aragora SDK.
 *
 * Usage:
 *   npx ts-node examples/basic-debate.ts
 *
 * Environment:
 *   ARAGORA_API_KEY - Your API key
 *   ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
 */

import { createClient } from '@aragora/sdk';

async function main() {
  // Create client
  const client = createClient({
    baseUrl: process.env.ARAGORA_API_URL || 'https://api.aragora.ai',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  console.log('Creating debate...');

  // Create a debate
  const debate = await client.debates.create({
    task: 'What is the best programming language for building web APIs?',
    agents: ['claude', 'gpt-4'],
    protocol: {
      rounds: 3,
      consensus: 'majority',
    },
  });

  console.log(`Debate created: ${debate.debate_id}`);
  console.log(`Status: ${debate.status}`);

  // Poll for completion
  let current = debate;
  while (current.status === 'running' || current.status === 'pending') {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    current = await client.debates.get(debate.debate_id);
    console.log(`Status: ${current.status}`);
  }

  // Get results
  if (current.status === 'completed') {
    console.log('\n--- Debate Results ---');
    console.log(`Final Answer: ${current.consensus?.final_answer}`);
    console.log(`Confidence: ${current.consensus?.confidence}`);
    console.log(`Rounds: ${current.rounds?.length}`);
  }
}

main().catch(console.error);
