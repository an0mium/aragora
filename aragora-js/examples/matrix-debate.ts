/**
 * Matrix Debate Example
 *
 * Demonstrates how to run debates across multiple scenarios
 * to identify universal vs conditional conclusions.
 */

import { AragoraClient } from '../src';

async function main() {
  const client = new AragoraClient({
    baseUrl: process.env.ARAGORA_API_URL || 'http://localhost:8080',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  console.log('=== Matrix Debate: Database Selection ===\n');

  // Define scenarios to test
  const scenarios = [
    {
      name: 'read_heavy',
      parameters: {
        read_ratio: 0.95,
        write_ratio: 0.05,
        data_size_gb: 100,
        concurrent_users: 10000,
      },
      constraints: ['sub-10ms read latency required'],
      is_baseline: true,
    },
    {
      name: 'write_heavy',
      parameters: {
        read_ratio: 0.2,
        write_ratio: 0.8,
        data_size_gb: 100,
        concurrent_users: 5000,
      },
      constraints: ['ACID compliance required'],
    },
    {
      name: 'analytics',
      parameters: {
        read_ratio: 0.99,
        write_ratio: 0.01,
        data_size_gb: 1000,
        concurrent_users: 100,
      },
      constraints: ['complex aggregation queries', 'batch processing'],
    },
    {
      name: 'real_time',
      parameters: {
        read_ratio: 0.7,
        write_ratio: 0.3,
        data_size_gb: 50,
        concurrent_users: 50000,
      },
      constraints: ['sub-5ms latency', 'high availability'],
    },
  ];

  console.log(`Running matrix debate with ${scenarios.length} scenarios...`);
  console.log('Scenarios:', scenarios.map((s) => s.name).join(', '));
  console.log();

  // Create and run matrix debate
  const matrix = await client.matrixDebates.create({
    task: 'Which database technology should we use for our application?',
    agents: ['claude-sonnet', 'gpt-4', 'gemini-pro'],
    scenarios,
    max_rounds: 3,
  });

  console.log(`Matrix debate created: ${matrix.matrix_id}`);
  console.log(`Status: ${matrix.status}\n`);

  // Poll until completion
  let result = await client.matrixDebates.get(matrix.matrix_id);
  while (result.status === 'in_progress') {
    await new Promise((resolve) => setTimeout(resolve, 5000));
    result = await client.matrixDebates.get(matrix.matrix_id);
    console.log(`Status: ${result.status}...`);
  }

  // Display results
  console.log('\n=== Matrix Results ===\n');

  // Per-scenario results
  console.log('--- Scenario Results ---\n');
  for (const scenario of result.scenarios) {
    console.log(`[${scenario.scenario_name}]`);

    if (scenario.consensus?.reached) {
      console.log(`  Consensus: ${scenario.consensus.conclusion}`);
      console.log(`  Confidence: ${(scenario.consensus.confidence * 100).toFixed(1)}%`);
    } else {
      console.log('  No consensus reached');
    }

    console.log('  Key findings:');
    for (const finding of scenario.key_findings) {
      console.log(`    - ${finding}`);
    }

    if (scenario.differences_from_baseline?.length) {
      console.log('  Differences from baseline:');
      for (const diff of scenario.differences_from_baseline) {
        console.log(`    * ${diff}`);
      }
    }
    console.log();
  }

  // Cross-scenario conclusions
  if (result.conclusions) {
    console.log('--- Cross-Scenario Conclusions ---\n');

    if (result.conclusions.universal.length > 0) {
      console.log('Universal (applies to all scenarios):');
      for (const conclusion of result.conclusions.universal) {
        console.log(`  - ${conclusion}`);
      }
      console.log();
    }

    if (Object.keys(result.conclusions.conditional).length > 0) {
      console.log('Conditional (depends on scenario):');
      for (const [condition, conclusions] of Object.entries(result.conclusions.conditional)) {
        console.log(`  When ${condition}:`);
        for (const conclusion of conclusions) {
          console.log(`    - ${conclusion}`);
        }
      }
      console.log();
    }

    if (result.conclusions.contradictions.length > 0) {
      console.log('Contradictions found:');
      for (const contradiction of result.conclusions.contradictions) {
        console.log(`  ! ${contradiction}`);
      }
    }
  }

  // Summary
  console.log('\n--- Summary ---');
  console.log(`Total scenarios: ${result.scenarios.length}`);
  console.log(
    `Consensus reached: ${result.scenarios.filter((s) => s.consensus?.reached).length}/${result.scenarios.length}`
  );
  console.log(`Duration: ${result.completed_at ? 'Complete' : 'In progress'}`);
}

main().catch(console.error);
