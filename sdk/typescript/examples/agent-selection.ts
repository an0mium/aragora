/**
 * Agent Selection Example
 *
 * Demonstrates how to list available agents, check their
 * capabilities, and select the best agents for a task.
 *
 * Usage:
 *   npx ts-node examples/agent-selection.ts
 */

import { createClient } from '@aragora/sdk';

async function main() {
  const client = createClient({
    baseUrl: process.env.ARAGORA_API_URL || 'https://api.aragora.ai',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  // List all available agents
  console.log('=== Available Agents ===\n');
  const agents = await client.agents.list();

  for (const agent of agents.agents) {
    console.log(`${agent.name} (${agent.id})`);
    console.log(`  Provider: ${agent.provider}`);
    console.log(`  Capabilities: ${agent.capabilities?.join(', ') || 'general'}`);
    console.log(`  ELO Rating: ${agent.elo_rating || 'N/A'}`);
    console.log(`  Status: ${agent.status}`);
    console.log();
  }

  // Get recommendations for a specific task
  console.log('=== Agent Recommendations ===\n');
  const recommendations = await client.agents.recommend({
    task: 'Write and review a Python function for data validation',
    count: 3,
    criteria: {
      capabilities: ['code', 'reasoning'],
      minElo: 1200,
    },
  });

  console.log('Recommended agents for this task:');
  for (const rec of recommendations.recommendations) {
    console.log(`  ${rec.agent_id}: ${rec.reason} (score: ${rec.score})`);
  }

  // Create debate with recommended agents
  const debate = await client.debates.create({
    task: 'Write and review a Python function for data validation',
    agents: recommendations.recommendations.map((r) => r.agent_id),
    protocol: {
      rounds: 2,
      consensus: 'weighted',
    },
  });

  console.log(`\nDebate created with recommended agents: ${debate.debate_id}`);
}

main().catch(console.error);
