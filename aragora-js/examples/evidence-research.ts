/**
 * Evidence Research Example
 *
 * Demonstrates how to use Aragora's Evidence API for research:
 * - Collecting evidence on topics
 * - Searching existing evidence
 * - Associating evidence with debates
 * - Building research-backed arguments
 *
 * Use Case: Research assistance, fact-checking, citation gathering
 */

import { AragoraClient } from '../src';

interface ResearchSummary {
  topic: string;
  snippetCount: number;
  sources: string[];
  averageReliability: number;
  keyFindings: string[];
}

async function conductResearch(
  client: AragoraClient,
  topic: string,
): Promise<ResearchSummary> {
  console.log(`\nResearching: "${topic}"`);
  console.log('-'.repeat(50));

  // Step 1: Collect evidence from configured sources
  console.log('Collecting evidence from sources...');
  const collection = await client.evidence.collect({
    task: topic,
    connectors: ['duckduckgo', 'wikipedia'], // Specify which connectors to use
  });

  console.log(`Found ${collection.count} evidence snippets`);
  console.log(`Sources searched: ${collection.total_searched}`);

  // Step 2: Analyze collected evidence
  const sources = new Set<string>();
  const keyFindings: string[] = [];

  for (const snippet of collection.snippets.slice(0, 5)) {
    sources.add(snippet.source);
    // Extract key sentences as findings
    const firstSentence = snippet.snippet.split('.')[0];
    if (firstSentence && firstSentence.length > 20) {
      keyFindings.push(`[${snippet.source}] ${firstSentence}.`);
    }
  }

  return {
    topic,
    snippetCount: collection.count,
    sources: Array.from(sources),
    averageReliability: collection.average_reliability,
    keyFindings: keyFindings.slice(0, 5),
  };
}

async function researchBackedDebate(
  client: AragoraClient,
  claim: string,
): Promise<void> {
  console.log(`\n${'='.repeat(60)}`);
  console.log('RESEARCH-BACKED DEBATE');
  console.log(`${'='.repeat(60)}\n`);

  // Step 1: Collect evidence first
  console.log('Phase 1: Gathering Evidence');
  const collection = await client.evidence.collect({
    task: claim,
  });
  console.log(`Collected ${collection.count} evidence snippets`);

  // Step 2: Run debate with evidence context
  console.log('\nPhase 2: Running Debate with Evidence');
  const debate = await client.debates.run({
    task: `Evaluate this claim using the available evidence:\n\n"${claim}"\n\nConsider both supporting and contradicting evidence.`,
    agents: ['claude-sonnet', 'gpt-4'],
    max_rounds: 3,
    consensus_threshold: 0.7,
  });

  // Step 3: Associate evidence with the debate
  const debateId = debate.debate_id || debate.id;
  if (collection.snippets.length > 0 && debateId) {
    console.log('\nPhase 3: Associating Evidence');
    const evidenceIds = collection.snippets
      .slice(0, 10)
      .map(s => s.id)
      .filter(Boolean);

    if (evidenceIds.length > 0) {
      await client.evidence.associateWithDebate(debateId, evidenceIds, {
        relevance_score: 0.8,
      });
      console.log(`Associated ${evidenceIds.length} evidence items with debate`);
    }
  }

  // Step 4: Display results
  console.log('\n' + '='.repeat(60));
  console.log('RESULTS');
  console.log('='.repeat(60));

  console.log(`\nClaim: "${claim}"`);
  console.log(`Evidence Items: ${collection.count}`);
  console.log(`Average Reliability: ${(collection.average_reliability * 100).toFixed(1)}%`);

  if (debate.consensus) {
    console.log(`\nVerdict: ${debate.consensus.conclusion}`);
    console.log(`Confidence: ${(debate.consensus.confidence * 100).toFixed(1)}%`);
  } else {
    console.log('\nNo consensus reached - claim requires further investigation');
  }

  // Show top evidence
  console.log('\nKey Evidence:');
  for (const snippet of collection.snippets.slice(0, 3)) {
    console.log(`  - [${snippet.source}] (reliability: ${(snippet.reliability_score * 100).toFixed(0)}%)`);
    console.log(`    "${snippet.snippet.substring(0, 150)}..."`);
  }
}

async function searchExistingEvidence(
  client: AragoraClient,
  query: string,
): Promise<void> {
  console.log(`\nSearching evidence for: "${query}"`);
  console.log('-'.repeat(50));

  const results = await client.evidence.search({
    query,
    limit: 10,
    min_reliability: 0.5, // Only high-quality sources
  });

  console.log(`Found ${results.count} matching items\n`);

  for (const item of results.results.slice(0, 5)) {
    console.log(`Source: ${item.source}`);
    console.log(`Title: ${item.title}`);
    console.log(`Reliability: ${(item.reliability_score * 100).toFixed(0)}%`);
    console.log(`Snippet: ${item.snippet.substring(0, 200)}...`);
    console.log();
  }
}

async function main() {
  const client = new AragoraClient({
    baseUrl: process.env.ARAGORA_API_URL || 'http://localhost:8080',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  console.log('=== Aragora Evidence Research ===\n');

  // Check if evidence system is available
  try {
    const stats = await client.evidence.statistics();
    console.log('Evidence System Status:');
    console.log(`  Total evidence items: ${stats.total_evidence || 0}`);
    console.log(`  Sources: ${stats.source_count || 0}`);
  } catch (e) {
    console.log('Note: Evidence statistics not available');
  }

  // Example 1: Basic research on a topic
  console.log('\n' + '='.repeat(60));
  console.log('EXAMPLE 1: Topic Research');
  console.log('='.repeat(60));

  const research = await conductResearch(
    client,
    'Benefits and risks of artificial general intelligence',
  );

  console.log('\nResearch Summary:');
  console.log(`  Topic: ${research.topic}`);
  console.log(`  Evidence collected: ${research.snippetCount}`);
  console.log(`  Unique sources: ${research.sources.length}`);
  console.log(`  Avg reliability: ${(research.averageReliability * 100).toFixed(1)}%`);

  console.log('\nKey Findings:');
  for (const finding of research.keyFindings) {
    console.log(`  - ${finding.substring(0, 120)}...`);
  }

  // Example 2: Research-backed debate (fact-checking)
  console.log('\n' + '='.repeat(60));
  console.log('EXAMPLE 2: Fact-Checking with Evidence');
  console.log('='.repeat(60));

  await researchBackedDebate(
    client,
    'Renewable energy is now cheaper than fossil fuels in most markets',
  );

  // Example 3: Search existing evidence
  console.log('\n' + '='.repeat(60));
  console.log('EXAMPLE 3: Evidence Search');
  console.log('='.repeat(60));

  await searchExistingEvidence(client, 'climate change impacts');

  // Example 4: Get evidence for a specific debate
  console.log('\n' + '='.repeat(60));
  console.log('EXAMPLE 4: Multi-topic Research Comparison');
  console.log('='.repeat(60));

  const topics = [
    'quantum computing applications',
    'CRISPR gene editing ethics',
    'autonomous vehicle safety',
  ];

  const researchResults: ResearchSummary[] = [];
  for (const topic of topics) {
    const result = await conductResearch(client, topic);
    researchResults.push(result);
  }

  console.log('\nComparison Summary:');
  console.log('-'.repeat(60));
  for (const r of researchResults) {
    console.log(`${r.topic}`);
    console.log(`  Evidence: ${r.snippetCount} | Sources: ${r.sources.length} | Reliability: ${(r.averageReliability * 100).toFixed(0)}%`);
  }
}

main().catch(console.error);
