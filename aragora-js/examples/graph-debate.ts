/**
 * Graph Debate Example
 *
 * Demonstrates graph-structured debates where arguments branch and merge.
 * Useful for exploring complex topics with multiple valid perspectives.
 */

import { AragoraClient } from '../src';

async function main() {
  const client = new AragoraClient({
    baseUrl: process.env.ARAGORA_API_URL || 'http://localhost:8080',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  console.log('Creating graph-structured debate...\n');

  // Create a graph debate - allows branching exploration
  const response = await client.graphDebates.create({
    root_claim: 'AI will fundamentally change software development',
    agents: ['claude-sonnet', 'gpt-4', 'gemini-pro'],
    config: {
      max_depth: 4,
      branching_threshold: 0.6, // Branch when disagreement > 60%
      merge_threshold: 0.85, // Merge when agreement > 85%
      exploration_mode: 'breadth-first',
    },
  });

  console.log(`Graph debate created: ${response.debate_id}`);

  // Poll for completion
  let debate = await client.graphDebates.get(response.debate_id);
  while (debate.status === 'running') {
    console.log(`Status: ${debate.status}, Nodes: ${debate.nodes?.length || 0}`);
    await new Promise((r) => setTimeout(r, 2000));
    debate = await client.graphDebates.get(response.debate_id);
  }

  // Get all branches
  const branches = await client.graphDebates.getBranches(response.debate_id);

  console.log('\n=== Graph Debate Results ===');
  console.log(`Total Nodes: ${debate.nodes?.length || 0}`);
  console.log(`Branches: ${branches.length}`);

  // Display branch summaries
  console.log('\n=== Branch Summaries ===');
  for (const branch of branches) {
    console.log(`\n[${branch.id}] ${branch.claim}`);
    console.log(`  Depth: ${branch.depth}`);
    console.log(`  Status: ${branch.status}`);
    console.log(`  Consensus: ${branch.consensus ? 'Yes' : 'No'}`);
    if (branch.conclusion) {
      console.log(`  Conclusion: ${branch.conclusion.substring(0, 100)}...`);
    }
  }

  // Display the argument tree
  console.log('\n=== Argument Tree ===');
  printTree(debate.nodes || [], null, 0);
}

function printTree(nodes: any[], parentId: string | null, depth: number) {
  const children = nodes.filter((n) => n.parent_id === parentId);
  for (const node of children) {
    const indent = '  '.repeat(depth);
    const marker = node.type === 'support' ? '+' : node.type === 'counter' ? '-' : '*';
    console.log(`${indent}${marker} [${node.agent}] ${node.claim?.substring(0, 60)}...`);
    printTree(nodes, node.id, depth + 1);
  }
}

main().catch(console.error);
