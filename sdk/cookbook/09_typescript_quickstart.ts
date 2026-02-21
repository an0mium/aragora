#!/usr/bin/env npx ts-node
/**
 * 09_typescript_quickstart.ts - TypeScript SDK quickstart.
 *
 * This is the TypeScript equivalent of 01_simple_debate.py.
 * Shows how to use the Aragora TypeScript SDK for basic debates.
 *
 * Usage:
 *   npx ts-node 09_typescript_quickstart.ts                  # Run actual debate
 *   npx ts-node 09_typescript_quickstart.ts --dry-run        # Test without API calls
 *   npx ts-node 09_typescript_quickstart.ts --topic "Your topic"
 */

import { ArenaClient, DebateConfig, Agent, DebateResult } from 'aragora-sdk';

interface RunOptions {
  topic: string;
  dryRun: boolean;
}

/**
 * Run a basic 3-agent debate on the given topic.
 */
async function runSimpleDebate(options: RunOptions): Promise<DebateResult | object> {
  const { topic, dryRun } = options;

  // Initialize the client (uses ARAGORA_API_URL and ARAGORA_API_TOKEN from env)
  const client = new ArenaClient();

  // Define our three agents - each brings a different perspective
  const agents: Agent[] = [
    { name: 'claude', model: 'claude-sonnet-4-20250514' },
    { name: 'gpt', model: 'gpt-4o' },
    { name: 'gemini', model: 'gemini-3.1-pro-preview' },
  ];

  // Configure the debate
  const config: DebateConfig = {
    topic,
    agents,
    rounds: 3,                    // Number of debate rounds
    consensusThreshold: 0.7,      // 70% agreement needed for consensus
  };

  if (dryRun) {
    // In dry-run mode, return mock result without API calls
    console.log(`[DRY RUN] Would run debate on: ${topic}`);
    console.log(`[DRY RUN] Agents: ${agents.map(a => a.name).join(', ')}`);
    console.log(`[DRY RUN] Rounds: ${config.rounds}`);
    return { status: 'dry_run', topic, agents: agents.map(a => a.name) };
  }

  // Run the debate and wait for result
  const result = await client.runDebate(config);

  // Print the outcome
  console.log('\n=== Debate Results ===');
  console.log(`Topic: ${result.topic}`);
  console.log(`Consensus reached: ${result.consensusReached}`);
  console.log(`Final decision: ${result.decision}`);
  console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);

  return result;
}

/**
 * Parse command line arguments.
 */
function parseArgs(): RunOptions {
  const args = process.argv.slice(2);
  let topic = 'Should AI development prioritize safety over capability?';
  let dryRun = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--dry-run') {
      dryRun = true;
    } else if (args[i] === '--topic' && args[i + 1]) {
      topic = args[i + 1];
      i++;
    }
  }

  return { topic, dryRun };
}

// Main execution
async function main(): Promise<void> {
  const options = parseArgs();
  await runSimpleDebate(options);
}

main().catch(console.error);
