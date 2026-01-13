/**
 * Batch Analysis Example
 *
 * Demonstrates how to process multiple topics in batch:
 * - Running debates in parallel using batch API
 * - Monitoring batch progress
 * - Aggregating results across multiple debates
 * - Generating comparative analysis
 *
 * Use Case: Analyzing multiple proposals, comparing options, bulk evaluation
 */

import { AragoraClient } from '../src';

interface AnalysisTask {
  id: string;
  topic: string;
  category: string;
}

interface BatchResult {
  task: AnalysisTask;
  hasConsensus: boolean;
  consensusPosition?: string;
  confidence: number;
  debateId?: string;
}

async function runBatchAnalysis(
  client: AragoraClient,
  tasks: AnalysisTask[],
): Promise<BatchResult[]> {
  console.log(`Starting batch analysis of ${tasks.length} topics...`);

  // Step 1: Submit batch request
  const batchRequest = {
    items: tasks.map(task => ({
      question: task.topic,
      agents: 'claude-sonnet,gpt-4',
      rounds: 3,
      consensus: 'collaborative',
      metadata: {
        task_id: task.id,
        category: task.category,
      },
    })),
  };

  console.log('Submitting batch request...');
  const batch = await client.batchDebates.submit(batchRequest);
  console.log(`Batch ID: ${batch.batch_id}`);
  console.log(`Items queued: ${batch.items_queued}`);

  // Step 2: Poll for completion
  console.log('\nWaiting for completion...');
  let lastStatus = '';

  while (true) {
    const status = await client.batchDebates.status(batch.batch_id);
    const completed = status.completed_items;

    const statusStr = `${completed}/${status.total_items} complete`;
    if (statusStr !== lastStatus) {
      console.log(`  Progress: ${statusStr} (${status.status})`);
      lastStatus = statusStr;
    }

    if (status.status === 'completed' || status.status === 'partial_failure') {
      // Step 3: Collect results
      console.log('\nCollecting results...');
      const results: BatchResult[] = [];

      for (const item of status.items || []) {
        const task = tasks.find(t => t.topic === item.question);

        if (task && item.debate_id) {
          // Fetch full debate for consensus info
          try {
            const debate = await client.debates.get(item.debate_id);
            results.push({
              task,
              hasConsensus: !!debate.consensus,
              consensusPosition: debate.consensus?.conclusion,
              confidence: debate.consensus?.confidence || 0,
              debateId: debate.debate_id,
            });
          } catch {
            results.push({
              task,
              hasConsensus: false,
              confidence: 0,
              debateId: item.debate_id,
            });
          }
        }
      }

      return results;
    }

    // Wait before next poll
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

function generateSummaryReport(results: BatchResult[]): void {
  console.log('\n' + '='.repeat(70));
  console.log('BATCH ANALYSIS SUMMARY REPORT');
  console.log('='.repeat(70));

  // Overall statistics
  const totalWithConsensus = results.filter(r => r.hasConsensus).length;
  const avgConfidence = results
    .filter(r => r.hasConsensus)
    .reduce((sum, r) => sum + r.confidence, 0) / totalWithConsensus || 0;

  console.log('\nOverall Statistics:');
  console.log(`  Total topics analyzed: ${results.length}`);
  console.log(`  Reached consensus: ${totalWithConsensus} (${((totalWithConsensus / results.length) * 100).toFixed(0)}%)`);
  console.log(`  Average confidence: ${(avgConfidence * 100).toFixed(1)}%`);

  // Group by category
  const byCategory = new Map<string, BatchResult[]>();
  for (const result of results) {
    const category = result.task.category;
    if (!byCategory.has(category)) {
      byCategory.set(category, []);
    }
    byCategory.get(category)!.push(result);
  }

  console.log('\nResults by Category:');
  console.log('-'.repeat(70));

  for (const [category, categoryResults] of byCategory) {
    console.log(`\n${category.toUpperCase()}`);
    for (const result of categoryResults) {
      const status = result.hasConsensus
        ? `[CONSENSUS] (${(result.confidence * 100).toFixed(0)}%)`
        : '[NO CONSENSUS]';
      console.log(`  - ${result.task.topic}`);
      console.log(`    ${status}`);
      if (result.consensusPosition) {
        console.log(`    Position: ${result.consensusPosition.substring(0, 100)}...`);
      }
    }
  }

  // High-confidence findings
  const highConfidence = results
    .filter(r => r.hasConsensus && r.confidence > 0.8)
    .sort((a, b) => b.confidence - a.confidence);

  if (highConfidence.length > 0) {
    console.log('\n' + '-'.repeat(70));
    console.log('HIGH-CONFIDENCE FINDINGS (>80%):');
    for (const result of highConfidence) {
      console.log(`\n  ${result.task.topic}`);
      console.log(`  Confidence: ${(result.confidence * 100).toFixed(1)}%`);
      console.log(`  Finding: ${result.consensusPosition?.substring(0, 150)}...`);
    }
  }

  // Contentious topics (no consensus)
  const noConsensus = results.filter(r => !r.hasConsensus);
  if (noConsensus.length > 0) {
    console.log('\n' + '-'.repeat(70));
    console.log('CONTENTIOUS TOPICS (No Consensus):');
    for (const result of noConsensus) {
      console.log(`  - ${result.task.topic}`);
    }
  }
}

async function main() {
  const client = new AragoraClient({
    baseUrl: process.env.ARAGORA_API_URL || 'http://localhost:8080',
    apiKey: process.env.ARAGORA_API_KEY,
  });

  console.log('=== Aragora Batch Analysis ===\n');

  // Check server health
  const health = await client.health();
  console.log(`Server status: ${health.status}\n`);

  // Define analysis tasks
  const tasks: AnalysisTask[] = [
    // Technology decisions
    {
      id: 'tech-1',
      topic: 'Should startups adopt microservices architecture from day one?',
      category: 'Technology',
    },
    {
      id: 'tech-2',
      topic: 'Is TypeScript adoption worth the initial productivity cost?',
      category: 'Technology',
    },
    {
      id: 'tech-3',
      topic: 'Should companies use serverless architecture for production workloads?',
      category: 'Technology',
    },

    // Business strategy
    {
      id: 'biz-1',
      topic: 'Should B2B SaaS companies offer free tiers?',
      category: 'Business',
    },
    {
      id: 'biz-2',
      topic: 'Is remote-first better than hybrid work for engineering teams?',
      category: 'Business',
    },

    // Process & methodology
    {
      id: 'proc-1',
      topic: 'Should teams use story points for estimation?',
      category: 'Process',
    },
    {
      id: 'proc-2',
      topic: 'Is pair programming worth the apparent cost in developer-hours?',
      category: 'Process',
    },

    // AI & Ethics
    {
      id: 'ai-1',
      topic: 'Should AI systems be required to disclose when generating content?',
      category: 'AI Ethics',
    },
    {
      id: 'ai-2',
      topic: 'Is it ethical to use AI for job candidate screening?',
      category: 'AI Ethics',
    },
  ];

  // Run batch analysis
  const results = await runBatchAnalysis(client, tasks);

  // Generate summary report
  generateSummaryReport(results);

  // Export results for further analysis
  console.log('\n' + '='.repeat(70));
  console.log('EXPORTABLE DATA');
  console.log('='.repeat(70));
  console.log('\nJSON Results:');
  console.log(JSON.stringify(results.map(r => ({
    id: r.task.id,
    topic: r.task.topic,
    category: r.task.category,
    hasConsensus: r.hasConsensus,
    confidence: r.confidence,
    position: r.consensusPosition?.substring(0, 200),
  })), null, 2));
}

main().catch(console.error);
