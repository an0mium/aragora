/**
 * Consensus Voting Example
 *
 * Demonstrates how to use Aragora for group decision-making:
 * - Running debates to evaluate options
 * - Analyzing agent positions and confidence
 * - Extracting actionable decisions from consensus
 *
 * Use Case: Team or organizational decision support
 */

import { AragoraClient } from '../src';

interface DecisionOption {
  id: string;
  name: string;
  description: string;
}

interface DecisionResult {
  winner: DecisionOption | null;
  confidence: number;
  analysis: string;
  dissent: string[];
}

async function evaluateDecision(
  client: AragoraClient,
  question: string,
  options: DecisionOption[],
): Promise<DecisionResult> {
  // Format the task with clear options
  const formattedOptions = options
    .map((opt, i) => `${i + 1}. ${opt.name}: ${opt.description}`)
    .join('\n');

  const task = `
Decision Question: ${question}

Available Options:
${formattedOptions}

Evaluate each option's merits and drawbacks. Reach consensus on the best choice.
Provide clear reasoning for your recommendation.
`.trim();

  // Run debate with multiple diverse agents
  const debate = await client.debates.run({
    task,
    agents: ['claude-sonnet', 'gpt-4', 'gemini-api'],
    max_rounds: 4,
    consensus_threshold: 0.6,
  });

  // Analyze results
  let winner: DecisionOption | null = null;
  let confidence = 0;
  let analysis = '';
  const dissent: string[] = [];

  if (debate.consensus) {
    // Extract the chosen option from consensus
    const conclusion = (debate.consensus.conclusion || '').toLowerCase();
    for (const option of options) {
      if (conclusion.includes(option.name.toLowerCase()) ||
          conclusion.includes(option.id.toLowerCase())) {
        winner = option;
        break;
      }
    }
    confidence = debate.consensus.confidence;
    analysis = debate.consensus.conclusion || '';
  }

  // Check for dissenting agents
  if (debate.consensus?.dissenting_agents) {
    for (const agent of debate.consensus.dissenting_agents) {
      dissent.push(`[${agent}]: Dissented from consensus`);
    }
  }

  return { winner, confidence, analysis, dissent };
}

async function main() {
  const client = new AragoraClient({
    baseUrl: process.env.ARAGORA_API_URL || 'http://localhost:8080',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  console.log('=== Aragora Consensus Voting ===\n');

  // Example 1: Technology Stack Decision
  console.log('Decision 1: Choosing a Database');
  console.log('-'.repeat(40));

  const dbDecision = await evaluateDecision(
    client,
    'Which database should we use for our new real-time analytics platform?',
    [
      { id: 'postgres', name: 'PostgreSQL', description: 'Relational, ACID compliant, mature ecosystem' },
      { id: 'mongodb', name: 'MongoDB', description: 'Document store, flexible schema, good for rapid iteration' },
      { id: 'clickhouse', name: 'ClickHouse', description: 'Column-oriented, optimized for analytics queries' },
      { id: 'timescale', name: 'TimescaleDB', description: 'Time-series focused, PostgreSQL compatible' },
    ],
  );

  console.log(`\nRecommendation: ${dbDecision.winner?.name || 'No consensus'}`);
  console.log(`Confidence: ${(dbDecision.confidence * 100).toFixed(1)}%`);
  console.log(`\nAnalysis:\n${dbDecision.analysis.substring(0, 500)}...`);

  if (dbDecision.dissent.length > 0) {
    console.log('\nDissenting Views:');
    for (const d of dbDecision.dissent) {
      console.log(`  ${d.substring(0, 200)}...`);
    }
  }

  // Example 2: Strategic Priority Decision
  console.log('\n\nDecision 2: Q2 Priority');
  console.log('-'.repeat(40));

  const priorityDecision = await evaluateDecision(
    client,
    'What should be our top engineering priority for Q2?',
    [
      { id: 'perf', name: 'Performance', description: 'Optimize system performance, reduce latency by 50%' },
      { id: 'scale', name: 'Scalability', description: 'Prepare infrastructure for 10x user growth' },
      { id: 'features', name: 'New Features', description: 'Ship top 5 requested features from roadmap' },
      { id: 'debt', name: 'Tech Debt', description: 'Address critical technical debt, improve maintainability' },
    ],
  );

  console.log(`\nRecommendation: ${priorityDecision.winner?.name || 'No consensus'}`);
  console.log(`Confidence: ${(priorityDecision.confidence * 100).toFixed(1)}%`);
  console.log(`\nAnalysis:\n${priorityDecision.analysis.substring(0, 500)}...`);

  // Example 3: Multi-round weighted voting
  console.log('\n\nDecision 3: Hiring Priority (Weighted)');
  console.log('-'.repeat(40));

  // Run multiple debates and aggregate results
  const roles = [
    { id: 'sre', name: 'SRE', description: 'Site Reliability Engineer for infrastructure' },
    { id: 'frontend', name: 'Frontend Dev', description: 'React specialist for user-facing features' },
    { id: 'ml', name: 'ML Engineer', description: 'Machine learning engineer for AI features' },
  ];

  const votes: Map<string, number> = new Map();
  const iterations = 3;

  for (let i = 0; i < iterations; i++) {
    const result = await evaluateDecision(
      client,
      'Which role should we prioritize hiring for next?',
      roles,
    );

    if (result.winner) {
      const current = votes.get(result.winner.id) || 0;
      votes.set(result.winner.id, current + result.confidence);
    }
  }

  // Calculate final weighted scores
  const sortedVotes = Array.from(votes.entries())
    .map(([id, score]) => ({
      role: roles.find(r => r.id === id)!,
      score: score / iterations,
    }))
    .sort((a, b) => b.score - a.score);

  console.log('\nWeighted Consensus Results:');
  for (const { role, score } of sortedVotes) {
    console.log(`  ${role.name}: ${(score * 100).toFixed(1)}%`);
  }

  if (sortedVotes.length > 0) {
    console.log(`\nFinal Recommendation: ${sortedVotes[0].role.name}`);
  }
}

main().catch(console.error);
