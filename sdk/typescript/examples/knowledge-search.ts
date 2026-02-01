/**
 * Knowledge Search Example
 *
 * Demonstrates Knowledge Mound operations using the Aragora SDK:
 * - Searching knowledge with filters
 * - CRUD operations on knowledge entries
 * - Federation setup and synchronization
 * - Deduplication and pruning
 *
 * Usage:
 *   npx ts-node examples/knowledge-search.ts
 *
 * Environment:
 *   ARAGORA_API_KEY - Your API key
 *   ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
 */

import {
  createClient,
  AragoraError,
  type AragoraConfig,
} from '@aragora/sdk';

// =============================================================================
// Configuration
// =============================================================================

const config: AragoraConfig = {
  baseUrl: process.env.ARAGORA_API_URL || 'https://api.aragora.ai',
  apiKey: process.env.ARAGORA_API_KEY,
  timeout: 30000,
};

// =============================================================================
// Knowledge Search Examples
// =============================================================================

async function demonstrateKnowledgeSearch(): Promise<void> {
  console.log('=== Knowledge Search Examples ===\n');

  const client = createClient(config);

  // -------------------------------------------------------------------------
  // Basic Search
  // -------------------------------------------------------------------------
  console.log('1. Basic Knowledge Search');
  console.log('-'.repeat(40));

  try {
    const searchResults = await client.knowledge.search('machine learning best practices', {
      limit: 10,
      min_confidence: 0.7,
    });

    console.log(`Found ${searchResults.results.length} results:\n`);

    for (const result of searchResults.results) {
      console.log(`  [${result.score.toFixed(2)}] ${result.content.substring(0, 80)}...`);
      if (result.tags?.length) {
        console.log(`         Tags: ${result.tags.join(', ')}`);
      }
    }
  } catch (error) {
    handleError('Search failed', error);
  }

  // -------------------------------------------------------------------------
  // Natural Language Query
  // -------------------------------------------------------------------------
  console.log('\n2. Natural Language Query');
  console.log('-'.repeat(40));

  try {
    const queryResult = await client.knowledge.query(
      'What are the key principles of clean code architecture?'
    );

    console.log(`Answer: ${queryResult.answer}\n`);
    console.log(`Confidence: ${(queryResult.confidence * 100).toFixed(1)}%`);
    console.log(`Sources: ${(queryResult.sources as unknown[]).length} documents`);
  } catch (error) {
    handleError('Query failed', error);
  }

  // -------------------------------------------------------------------------
  // Search with Filters
  // -------------------------------------------------------------------------
  console.log('\n3. Filtered Search');
  console.log('-'.repeat(40));

  try {
    const filteredResults = await client.knowledge.search('API design', {
      type: 'fact',
      tags: ['architecture', 'best-practices'],
      min_confidence: 0.8,
      limit: 5,
    });

    console.log(`Found ${filteredResults.results.length} filtered results`);

    for (const result of filteredResults.results) {
      console.log(`  - ${result.content.substring(0, 60)}...`);
    }
  } catch (error) {
    handleError('Filtered search failed', error);
  }
}

// =============================================================================
// CRUD Operations Examples
// =============================================================================

async function demonstrateCrudOperations(): Promise<void> {
  console.log('\n=== CRUD Operations ===\n');

  const client = createClient(config);

  // -------------------------------------------------------------------------
  // Create Knowledge Entry
  // -------------------------------------------------------------------------
  console.log('1. Create Knowledge Entry');
  console.log('-'.repeat(40));

  let entryId: string | null = null;

  try {
    const entry = await client.knowledge.add({
      content: 'Microservices should be designed around business capabilities, not technical layers.',
      source: 'Architecture Guidelines',
      tags: ['microservices', 'architecture', 'design-patterns'],
      importance: 0.85,
      visibility: 'team',
      metadata: {
        category: 'software-architecture',
        author: 'engineering-team',
        last_reviewed: new Date().toISOString(),
      },
    });

    entryId = entry.id;
    console.log(`Created entry: ${entry.id}`);
    console.log(`Created at: ${entry.created_at}`);
  } catch (error) {
    handleError('Create failed', error);
  }

  // -------------------------------------------------------------------------
  // Read Knowledge Entry
  // -------------------------------------------------------------------------
  console.log('\n2. Read Knowledge Entry');
  console.log('-'.repeat(40));

  if (entryId) {
    try {
      const retrieved = await client.knowledge.get(entryId);

      console.log(`Content: ${retrieved.content}`);
      console.log(`Source: ${retrieved.source}`);
      console.log(`Tags: ${retrieved.tags?.join(', ')}`);
      console.log(`Visibility: ${retrieved.visibility}`);
    } catch (error) {
      handleError('Read failed', error);
    }
  }

  // -------------------------------------------------------------------------
  // Update Knowledge Entry
  // -------------------------------------------------------------------------
  console.log('\n3. Update Knowledge Entry');
  console.log('-'.repeat(40));

  if (entryId) {
    try {
      const updated = await client.knowledge.update(entryId, {
        content: 'Microservices should be designed around business capabilities, enabling independent deployment and scaling.',
        tags: ['microservices', 'architecture', 'design-patterns', 'scalability'],
        importance: 0.9,
      });

      console.log(`Updated content: ${updated.content}`);
      console.log(`Updated tags: ${updated.tags?.join(', ')}`);
    } catch (error) {
      handleError('Update failed', error);
    }
  }

  // -------------------------------------------------------------------------
  // Delete Knowledge Entry
  // -------------------------------------------------------------------------
  console.log('\n4. Delete Knowledge Entry');
  console.log('-'.repeat(40));

  if (entryId) {
    try {
      const deleteResult = await client.knowledge.delete(entryId);
      console.log(`Deleted: ${deleteResult.deleted}`);
    } catch (error) {
      handleError('Delete failed', error);
    }
  }

  // -------------------------------------------------------------------------
  // Bulk Import
  // -------------------------------------------------------------------------
  console.log('\n5. Bulk Import');
  console.log('-'.repeat(40));

  try {
    const importResult = await client.knowledge.bulkImport([
      {
        content: 'REST APIs should use HTTP methods semantically.',
        source: 'API Guidelines',
        tags: ['api', 'rest'],
        importance: 0.8,
      },
      {
        content: 'Use pagination for endpoints returning large collections.',
        source: 'API Guidelines',
        tags: ['api', 'pagination'],
        importance: 0.75,
      },
      {
        content: 'API versioning should be done through URL paths or headers.',
        source: 'API Guidelines',
        tags: ['api', 'versioning'],
        importance: 0.8,
      },
    ]);

    console.log(`Imported: ${importResult.imported}`);
    console.log(`Failed: ${importResult.failed}`);
    if (importResult.errors?.length) {
      console.log('Errors:');
      for (const err of importResult.errors) {
        console.log(`  Index ${err.index}: ${err.error}`);
      }
    }
  } catch (error) {
    handleError('Bulk import failed', error);
  }
}

// =============================================================================
// Knowledge Mound Operations
// =============================================================================

async function demonstrateMoundOperations(): Promise<void> {
  console.log('\n=== Knowledge Mound Operations ===\n');

  const client = createClient(config);

  // -------------------------------------------------------------------------
  // Query the Knowledge Mound Graph
  // -------------------------------------------------------------------------
  console.log('1. Query Knowledge Mound');
  console.log('-'.repeat(40));

  try {
    const moundResult = await client.knowledge.queryMound('software architecture patterns', {
      types: ['fact', 'concept', 'insight'],
      depth: 2,
      include_relationships: true,
      limit: 10,
    });

    console.log(`Nodes found: ${moundResult.nodes.length}`);
    console.log(`Relationships: ${moundResult.relationships.length}`);
    console.log(`Query time: ${moundResult.query_time_ms}ms`);

    console.log('\nSample nodes:');
    for (const node of moundResult.nodes.slice(0, 3)) {
      console.log(`  [${node.node_type}] ${node.content.substring(0, 60)}...`);
      console.log(`    Confidence: ${(node.confidence * 100).toFixed(1)}%`);
    }
  } catch (error) {
    handleError('Mound query failed', error);
  }

  // -------------------------------------------------------------------------
  // Create Knowledge Mound Node
  // -------------------------------------------------------------------------
  console.log('\n2. Create Knowledge Node');
  console.log('-'.repeat(40));

  let nodeId: string | null = null;

  try {
    const node = await client.knowledge.createNode({
      content: 'Event-driven architecture enables loose coupling between services.',
      node_type: 'concept',
      confidence: 0.85,
      source: 'Architecture Team',
      tags: ['architecture', 'event-driven', 'microservices'],
      visibility: 'team',
      metadata: {
        domain: 'software-engineering',
      },
    });

    nodeId = node.id;
    console.log(`Created node: ${node.id}`);
    console.log(`Created at: ${node.created_at}`);
  } catch (error) {
    handleError('Create node failed', error);
  }

  // -------------------------------------------------------------------------
  // Create Relationship
  // -------------------------------------------------------------------------
  console.log('\n3. Create Relationship');
  console.log('-'.repeat(40));

  if (nodeId) {
    try {
      // First, find a related node to connect to
      const relatedNodes = await client.knowledge.listNodes({ type: 'concept', limit: 1 });

      if (relatedNodes.nodes.length > 0) {
        const targetNode = relatedNodes.nodes[0];

        const relationship = await client.knowledge.createRelationship({
          source_id: nodeId,
          target_id: targetNode.id,
          relationship_type: 'related_to',
          strength: 0.7,
          confidence: 0.8,
          metadata: {
            reason: 'Both discuss architectural patterns',
          },
        });

        console.log(`Created relationship: ${relationship.id}`);
        console.log(`  ${nodeId} -> related_to -> ${targetNode.id}`);
      }
    } catch (error) {
      handleError('Create relationship failed', error);
    }
  }

  // -------------------------------------------------------------------------
  // Get Knowledge Mound Statistics
  // -------------------------------------------------------------------------
  console.log('\n4. Knowledge Mound Statistics');
  console.log('-'.repeat(40));

  try {
    const stats = await client.knowledge.moundStats();

    console.log(`Total nodes: ${stats.total_nodes}`);
    console.log(`Total relationships: ${stats.total_relationships}`);
    console.log(`Average confidence: ${(stats.avg_confidence * 100).toFixed(1)}%`);
    console.log(`Last sync: ${stats.last_sync}`);

    console.log('\nNodes by type:');
    for (const [type, count] of Object.entries(stats.nodes_by_type)) {
      console.log(`  ${type}: ${count}`);
    }

    console.log('\nStaleness:');
    console.log(`  Fresh: ${stats.staleness_stats.fresh}`);
    console.log(`  Stale: ${stats.staleness_stats.stale}`);
    console.log(`  Critical: ${stats.staleness_stats.critical}`);
  } catch (error) {
    handleError('Get stats failed', error);
  }

  // -------------------------------------------------------------------------
  // Get Related Knowledge
  // -------------------------------------------------------------------------
  console.log('\n5. Get Related Knowledge');
  console.log('-'.repeat(40));

  if (nodeId) {
    try {
      const related = await client.knowledge.getRelated(nodeId, {
        relationship_types: ['supports', 'elaborates', 'related_to'],
        limit: 5,
      });

      console.log(`Related nodes: ${(related.nodes as unknown[]).length}`);
      console.log(`Relationships: ${(related.relationships as unknown[]).length}`);
    } catch (error) {
      handleError('Get related failed', error);
    }
  }
}

// =============================================================================
// Federation Examples
// =============================================================================

async function demonstrateFederation(): Promise<void> {
  console.log('\n=== Federation Setup ===\n');

  const client = createClient(config);

  // -------------------------------------------------------------------------
  // List Federated Regions
  // -------------------------------------------------------------------------
  console.log('1. List Federated Regions');
  console.log('-'.repeat(40));

  try {
    const regions = await client.knowledge.listRegions();
    console.log(`Configured regions: ${(regions.regions as unknown[]).length}`);

    for (const region of regions.regions as Array<{ name: string; endpoint: string }>) {
      console.log(`  - ${region.name}: ${region.endpoint}`);
    }
  } catch (error) {
    handleError('List regions failed', error);
  }

  // -------------------------------------------------------------------------
  // Get Federation Status
  // -------------------------------------------------------------------------
  console.log('\n2. Federation Status');
  console.log('-'.repeat(40));

  try {
    const status = await client.knowledge.getFederationStatus();

    console.log(`Health: ${status.health}`);
    console.log(`Last sync: ${status.last_sync}`);
    console.log(`Regions: ${(status.regions as unknown[]).length}`);
  } catch (error) {
    handleError('Get federation status failed', error);
  }

  // -------------------------------------------------------------------------
  // Register New Region (Admin only)
  // -------------------------------------------------------------------------
  console.log('\n3. Register Region (Example)');
  console.log('-'.repeat(40));

  console.log('Note: Region registration requires admin privileges.');
  console.log('Example code:');
  console.log(`
  const region = await client.knowledge.registerRegion({
    name: 'eu-west',
    endpoint: 'https://eu-west.aragora.ai',
    api_key: 'region-api-key',
  });
  `);

  // -------------------------------------------------------------------------
  // Sync Operations
  // -------------------------------------------------------------------------
  console.log('4. Sync Operations (Example)');
  console.log('-'.repeat(40));

  console.log('Push to remote region:');
  console.log(`
  const pushResult = await client.knowledge.syncPush('eu-west', {
    scope: 'workspace',
  });
  console.log(\`Synced: \${pushResult.synced}, Failed: \${pushResult.failed}\`);
  `);

  console.log('Pull from remote region:');
  console.log(`
  const pullResult = await client.knowledge.syncPull('eu-west', {
    since: '2024-01-01T00:00:00Z',
    limit: 100,
  });
  console.log(\`Received: \${pullResult.received}, Merged: \${pullResult.merged}\`);
  `);
}

// =============================================================================
// Deduplication and Pruning Examples
// =============================================================================

async function demonstrateDeduplicationAndPruning(): Promise<void> {
  console.log('\n=== Deduplication and Pruning ===\n');

  const client = createClient(config);

  // -------------------------------------------------------------------------
  // Find Duplicate Clusters
  // -------------------------------------------------------------------------
  console.log('1. Find Duplicate Clusters');
  console.log('-'.repeat(40));

  try {
    const clusters = await client.knowledge.getDuplicateClusters({
      threshold: 0.9,
      limit: 10,
    });

    console.log(`Found ${clusters.total} duplicate clusters`);

    for (const cluster of (clusters.clusters as Array<{
      cluster_id: string;
      nodes: Array<{ content: string }>;
      similarity_score: number;
      recommended_action: string;
    }>).slice(0, 3)) {
      console.log(`\n  Cluster ${cluster.cluster_id}:`);
      console.log(`    Similarity: ${(cluster.similarity_score * 100).toFixed(1)}%`);
      console.log(`    Nodes: ${cluster.nodes.length}`);
      console.log(`    Action: ${cluster.recommended_action}`);
    }
  } catch (error) {
    handleError('Get duplicates failed', error);
  }

  // -------------------------------------------------------------------------
  // Deduplication Report
  // -------------------------------------------------------------------------
  console.log('\n2. Deduplication Report');
  console.log('-'.repeat(40));

  try {
    const report = await client.knowledge.getDedupReport();

    console.log(`Total nodes: ${report.total_nodes}`);
    console.log(`Duplicate clusters: ${report.duplicate_clusters}`);
    console.log(`Potential savings: ${report.potential_savings} entries`);

    console.log('\nRecommendations:');
    for (const rec of (report.recommendations as string[]).slice(0, 3)) {
      console.log(`  - ${rec}`);
    }
  } catch (error) {
    handleError('Get dedup report failed', error);
  }

  // -------------------------------------------------------------------------
  // Auto-Merge Duplicates
  // -------------------------------------------------------------------------
  console.log('\n3. Auto-Merge (Dry Run)');
  console.log('-'.repeat(40));

  try {
    const mergeResult = await client.knowledge.autoMergeExactDuplicates({
      dry_run: true,
    });

    console.log(`Would merge: ${mergeResult.merged} entries`);
    console.log(`Clusters to process: ${mergeResult.clusters_processed}`);
  } catch (error) {
    handleError('Auto-merge failed', error);
  }

  // -------------------------------------------------------------------------
  // Get Prunable Items
  // -------------------------------------------------------------------------
  console.log('\n4. Get Prunable Items');
  console.log('-'.repeat(40));

  try {
    const prunable = await client.knowledge.getPrunableItems({
      max_age_days: 180,
      min_staleness: 0.8,
      limit: 20,
    });

    console.log(`Prunable items: ${prunable.total}`);

    for (const item of (prunable.items as Array<{
      node_id: string;
      reason: string;
      confidence: number;
    }>).slice(0, 5)) {
      console.log(`  - ${item.node_id}: ${item.reason} (confidence: ${item.confidence})`);
    }
  } catch (error) {
    handleError('Get prunable items failed', error);
  }

  // -------------------------------------------------------------------------
  // Auto-Prune (Dry Run)
  // -------------------------------------------------------------------------
  console.log('\n5. Auto-Prune (Dry Run)');
  console.log('-'.repeat(40));

  try {
    const pruneResult = await client.knowledge.autoPrune({
      policy: 'conservative',
      dry_run: true,
    });

    console.log(`Would prune: ${pruneResult.pruned} entries`);
    console.log(`Would archive: ${pruneResult.archived} entries`);
  } catch (error) {
    handleError('Auto-prune failed', error);
  }
}

// =============================================================================
// Error Handling
// =============================================================================

function handleError(context: string, error: unknown): void {
  if (error instanceof AragoraError) {
    console.error(`${context}: [${error.code}] ${error.message}`);
  } else if (error instanceof Error) {
    console.error(`${context}: ${error.message}`);
  } else {
    console.error(`${context}:`, error);
  }
}

// =============================================================================
// Run Examples
// =============================================================================

async function main(): Promise<void> {
  if (!process.env.ARAGORA_API_KEY) {
    console.warn('Warning: ARAGORA_API_KEY not set. Some operations may fail.\n');
  }

  try {
    await demonstrateKnowledgeSearch();
    await demonstrateCrudOperations();
    await demonstrateMoundOperations();
    await demonstrateFederation();
    await demonstrateDeduplicationAndPruning();

    console.log('\n=== All examples completed ===');
  } catch (error) {
    if (error instanceof AragoraError) {
      console.error('\nFatal error:', error.message);
      console.error('Code:', error.code);
    } else {
      console.error('\nFatal error:', error);
    }
    process.exit(1);
  }
}

main();
