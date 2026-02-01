/**
 * Decision Explainability Example
 *
 * Demonstrates explainability capabilities in Aragora:
 * - Getting explanation factors for decisions
 * - Generating counterfactual scenarios
 * - Tracking decision provenance
 * - Batch explanations and comparisons
 *
 * Usage:
 *   npx ts-node examples/explainability.ts
 *
 * Environment:
 *   ARAGORA_API_KEY - Your API key
 *   ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
 */

import { createClient, AragoraError, ExplanationFactor } from '@aragora/sdk';

// Configuration
const API_URL = process.env.ARAGORA_API_URL || 'https://api.aragora.ai';
const API_KEY = process.env.ARAGORA_API_KEY || 'your-api-key-here';

async function main() {
  // Initialize the client
  const client = createClient({
    baseUrl: API_URL,
    apiKey: API_KEY,
  });

  try {
    // =========================================================================
    // 1. Create a Debate for Demonstration
    // =========================================================================
    console.log('=== Creating Sample Debate ===\n');

    const debate = await client.debates.create({
      task: 'Should our company adopt a 4-day work week?',
      agents: ['claude', 'gpt-4', 'gemini'],
      rounds: 3,
      consensus: 'majority',
      context: 'We are a tech company with 150 employees. Current productivity is strong but employee satisfaction scores have declined by 15% this year.',
    });

    console.log(`Debate created: ${debate.debate_id}`);
    console.log('Waiting for completion...');

    // Wait for debate completion
    let currentDebate = await client.debates.get(debate.debate_id);
    while (currentDebate.status === 'pending' || currentDebate.status === 'running') {
      await sleep(3000);
      currentDebate = await client.debates.get(debate.debate_id);
      console.log(`  Status: ${currentDebate.status}`);
    }

    const debateId = debate.debate_id;
    console.log('\nDebate completed!');
    console.log(`Final answer: ${currentDebate.consensus?.final_answer || 'N/A'}`);
    console.log(`Confidence: ${currentDebate.consensus?.confidence || 'N/A'}`);
    console.log('');

    // =========================================================================
    // 2. Get Full Explanation
    // =========================================================================
    console.log('=== Full Decision Explanation ===\n');

    const fullExplanation = await client.explainability.get(debateId, {
      include_factors: true,
      include_counterfactuals: true,
      include_provenance: true,
    });

    console.log(`Debate ID: ${fullExplanation.debate_id}`);
    if (fullExplanation.narrative) {
      console.log(`\nNarrative: ${fullExplanation.narrative}`);
    }
    console.log('');

    // =========================================================================
    // 3. Explanation Factors
    // =========================================================================
    console.log('=== Explanation Factors ===\n');

    const { factors } = await client.explainability.getFactors(debateId, {
      min_contribution: 0.05, // Only factors with >5% contribution
    });

    console.log('Key decision factors:');
    for (const factor of factors) {
      const sign = factor.contribution >= 0 ? '+' : '';
      const bar = createContributionBar(factor.contribution);
      console.log(`\n  ${factor.name}`);
      console.log(`    ${bar} ${sign}${(factor.contribution * 100).toFixed(1)}%`);
      if (factor.description) {
        console.log(`    ${factor.description}`);
      }
      if (factor.evidence?.length) {
        console.log(`    Evidence: ${factor.evidence.slice(0, 2).join('; ')}`);
      }
    }
    console.log('');

    // Get top positive and negative factors using helper methods
    const topPositive = client.explainability.getTopPositiveFactors(fullExplanation, 3);
    const topNegative = client.explainability.getTopNegativeFactors(fullExplanation, 3);

    console.log('Top factors in favor:');
    for (const factor of topPositive) {
      console.log(`  + ${factor.name} (+${(factor.contribution * 100).toFixed(1)}%)`);
    }
    console.log('');

    console.log('Top factors against:');
    for (const factor of topNegative) {
      console.log(`  - ${factor.name} (${(factor.contribution * 100).toFixed(1)}%)`);
    }
    console.log('');

    // =========================================================================
    // 4. Counterfactual Scenarios
    // =========================================================================
    console.log('=== Counterfactual Scenarios ===\n');

    // Get existing counterfactuals
    const { counterfactuals } = await client.explainability.getCounterfactuals(debateId, {
      max_scenarios: 5,
    });

    console.log('Pre-computed counterfactual scenarios:');
    for (const cf of counterfactuals) {
      console.log(`\n  Scenario: ${cf.scenario}`);
      console.log(`    Original outcome: ${cf.original_outcome}`);
      console.log(`    Alternative outcome: ${cf.alternate_outcome}`);
      console.log(`    Probability: ${(cf.probability * 100).toFixed(1)}%`);
      if (cf.changed_factors?.length) {
        console.log(`    Key changes: ${cf.changed_factors.join(', ')}`);
      }
    }
    console.log('');

    // Generate a custom counterfactual
    console.log('Generating custom counterfactual...');
    const customCF = await client.explainability.generateCounterfactual(debateId, {
      hypothesis: 'What if we had unlimited budget for this initiative?',
      affected_agents: ['gpt-4', 'gemini'],
    });

    console.log(`\nCustom Counterfactual:`);
    console.log(`  Hypothesis: ${customCF.counterfactual.scenario}`);
    console.log(`  Original: ${customCF.counterfactual.original_outcome}`);
    console.log(`  If hypothesis were true: ${customCF.counterfactual.alternate_outcome}`);
    console.log(`  Likelihood: ${(customCF.counterfactual.probability * 100).toFixed(1)}%`);
    console.log('');

    // =========================================================================
    // 5. Decision Provenance
    // =========================================================================
    console.log('=== Decision Provenance ===\n');

    const provenance = await client.explainability.getProvenance(debateId);
    console.log(`Total claims tracked: ${provenance.total_claims}`);
    console.log(`Root claims: ${provenance.root_claims}`);
    console.log(`Max reasoning depth: ${provenance.max_depth}`);
    console.log('');

    console.log('Provenance chains:');
    for (const chain of provenance.chains.slice(0, 3)) {
      printProvenanceChain(chain, 0);
    }
    console.log('');

    // =========================================================================
    // 6. Human-Readable Narratives
    // =========================================================================
    console.log('=== Decision Narratives ===\n');

    // Brief summary
    const briefNarrative = await client.explainability.getNarrative(debateId, {
      format: 'brief',
    });
    console.log('Brief summary:');
    console.log(`  ${briefNarrative.summary}`);
    console.log('');

    // Executive summary
    const execNarrative = await client.explainability.getNarrative(debateId, {
      format: 'executive_summary',
    });
    console.log('Executive summary:');
    console.log(`  ${execNarrative.summary}`);
    console.log('');
    console.log('Key points:');
    for (const point of execNarrative.key_points) {
      console.log(`  - ${point}`);
    }
    console.log('');
    console.log(`Decision rationale: ${execNarrative.decision_rationale}`);
    if (execNarrative.dissent_summary) {
      console.log(`Dissenting views: ${execNarrative.dissent_summary}`);
    }
    console.log(`Confidence explanation: ${execNarrative.confidence_explanation}`);
    console.log('');

    // =========================================================================
    // 7. Evidence Chain
    // =========================================================================
    console.log('=== Evidence Chain ===\n');

    const evidence = await client.explainability.getEvidence(debateId, {
      limit: 10,
      min_relevance: 0.5,
    });

    console.log(`Evidence items: ${evidence.evidence_count}`);
    console.log(`Quality score: ${(evidence.evidence_quality_score * 100).toFixed(1)}%`);
    console.log('');

    console.log('Top evidence:');
    for (const item of evidence.evidence.slice(0, 5)) {
      console.log(`\n  Source: ${item.source}`);
      console.log(`  Relevance: ${(item.relevance_score * 100).toFixed(1)}%`);
      console.log(`  Content: ${item.content.slice(0, 150)}...`);
      if (item.cited_by.length > 0) {
        console.log(`  Cited by: ${item.cited_by.join(', ')}`);
      }
    }
    console.log('');

    // =========================================================================
    // 8. Vote Pivots Analysis
    // =========================================================================
    console.log('=== Vote Pivots Analysis ===\n');

    const pivots = await client.explainability.getVotePivots(debateId, {
      min_influence: 0.1,
    });

    console.log(`Total votes: ${pivots.total_votes}`);
    console.log(`Pivotal votes: ${pivots.pivotal_votes}`);
    console.log(`Pivot threshold: ${(pivots.pivot_threshold * 100).toFixed(1)}%`);
    console.log('');

    console.log('Pivotal agent votes:');
    for (const vote of pivots.votes) {
      const influence = (vote.influence_score * 100).toFixed(1);
      console.log(`  ${vote.agent_id}: ${vote.vote} (influence: ${influence}%)`);
      if (vote.reasoning) {
        console.log(`    Reasoning: ${vote.reasoning.slice(0, 100)}...`);
      }
    }
    console.log('');

    // =========================================================================
    // 9. Batch Explanations
    // =========================================================================
    console.log('=== Batch Explanations ===\n');

    // Get some recent debates for comparison
    const { debates: recentDebates } = await client.debates.list({
      limit: 3,
      status: 'completed',
    });

    if (recentDebates.length >= 2) {
      const debateIds = recentDebates.map(d => d.debate_id);

      // Create batch explanation request
      console.log(`Creating batch explanation for ${debateIds.length} debates...`);
      const batchJob = await client.explainability.createBatch({
        debate_ids: debateIds,
        include_factors: true,
        include_narrative: true,
      });

      console.log(`Batch ID: ${batchJob.batch_id}`);
      console.log(`Status URL: ${batchJob.status_url}`);

      // Poll for completion
      let batchStatus = await client.explainability.getBatchStatus(batchJob.batch_id);
      while (batchStatus.status === 'pending' || batchStatus.status === 'processing') {
        console.log(`  Progress: ${batchStatus.progress_pct}%`);
        await sleep(2000);
        batchStatus = await client.explainability.getBatchStatus(batchJob.batch_id);
      }

      console.log(`\nBatch completed: ${batchStatus.status}`);

      // Get results
      const { results } = await client.explainability.getBatchResults(batchJob.batch_id);
      console.log(`\nResults for ${results.length} debates:`);
      for (const result of results) {
        console.log(`  ${result.debate_id}: ${result.status}`);
      }
      console.log('');
    }

    // =========================================================================
    // 10. Compare Explanations
    // =========================================================================
    console.log('=== Explanation Comparison ===\n');

    if (recentDebates.length >= 2) {
      const debateIds = recentDebates.slice(0, 3).map(d => d.debate_id);

      const comparison = await client.explainability.compare({
        debate_ids: debateIds,
        compare_fields: ['confidence', 'evidence_quality'],
      });

      console.log('Comparison summary:');
      console.log(comparison.comparison.summary);
      console.log('');

      console.log('Field comparisons:');
      for (const [field, values] of Object.entries(comparison.comparison.fields)) {
        console.log(`  ${field}:`);
        for (const v of values as Array<{ debate_id: string; value: unknown }>) {
          console.log(`    ${v.debate_id}: ${JSON.stringify(v.value)}`);
        }
      }
      console.log('');
    }

    // =========================================================================
    // 11. Practical Use Cases
    // =========================================================================
    console.log('=== Practical Use Cases ===\n');

    console.log('1. Compliance Report:');
    console.log('   Use provenance + evidence to generate audit trails');
    console.log('');

    console.log('2. Stakeholder Communication:');
    console.log('   Use executive_summary narrative for board presentations');
    console.log('');

    console.log('3. Decision Validation:');
    console.log('   Use counterfactuals to stress-test decisions');
    console.log('');

    console.log('4. Learning from Decisions:');
    console.log('   Compare similar debates to identify patterns');
    console.log('');

    console.log('5. Risk Assessment:');
    console.log('   Analyze negative factors and dissenting views');
    console.log('');

    console.log('Explainability example completed successfully!');

  } catch (error) {
    handleError(error);
    process.exit(1);
  }
}

// =========================================================================
// Helper Functions
// =========================================================================

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function createContributionBar(contribution: number, width: number = 20): string {
  const absContrib = Math.abs(contribution);
  const filled = Math.round(absContrib * width);
  const empty = width - filled;

  if (contribution >= 0) {
    return '[' + '+'.repeat(filled) + ' '.repeat(empty) + ']';
  } else {
    return '[' + '-'.repeat(filled) + ' '.repeat(empty) + ']';
  }
}

interface ProvenanceChainType {
  claim_id: string;
  claim: string;
  source_agent: string;
  round: number;
  confidence: number;
  predecessors: ProvenanceChainType[];
  evidence_ids: string[];
}

function printProvenanceChain(chain: ProvenanceChainType, depth: number): void {
  const indent = '  '.repeat(depth);
  console.log(`${indent}[Round ${chain.round}] ${chain.source_agent}:`);
  console.log(`${indent}  Claim: "${chain.claim.slice(0, 80)}..."`);
  console.log(`${indent}  Confidence: ${(chain.confidence * 100).toFixed(1)}%`);

  if (chain.evidence_ids.length > 0) {
    console.log(`${indent}  Evidence: ${chain.evidence_ids.length} sources`);
  }

  if (chain.predecessors.length > 0 && depth < 2) {
    console.log(`${indent}  Based on:`);
    for (const pred of chain.predecessors.slice(0, 2)) {
      printProvenanceChain(pred, depth + 2);
    }
  }
}

function handleError(error: unknown): void {
  if (error instanceof AragoraError) {
    console.error('\n--- Aragora Error ---');
    console.error(`Message: ${error.message}`);
    console.error(`Code: ${error.code || 'N/A'}`);
    console.error(`Status: ${error.status || 'N/A'}`);

    // Common explainability errors
    if (error.code === 'NOT_FOUND') {
      console.error('\nNote: The debate may not exist or has been archived.');
      console.error('Explanations are only available for completed debates.');
    }

    if (error.traceId) {
      console.error(`Trace ID: ${error.traceId}`);
    }
  } else if (error instanceof Error) {
    console.error('\n--- Error ---');
    console.error(`Message: ${error.message}`);
  } else {
    console.error('\n--- Unknown Error ---');
    console.error(error);
  }
}

// Run the example
main();
