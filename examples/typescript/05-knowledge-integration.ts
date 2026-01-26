/**
 * Example 05: Knowledge Integration
 *
 * This example demonstrates how to integrate external knowledge sources
 * into debates using documents, memory, and evidence APIs.
 */

import { AragoraClient } from "@aragora/sdk";
import * as fs from "fs";
import * as path from "path";

// Initialize the client
const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL || "http://localhost:8080",
  apiKey: process.env.ARAGORA_API_KEY,
});

// Example 1: Upload and use documents in debate
async function documentIntegration() {
  console.log("=== Example 1: Document Integration ===\n");

  // Check supported formats
  const formats = await client.documents.formats();
  console.log("Supported document formats:", formats.formats.join(", "));
  console.log("");

  // Upload a document (example with mock file)
  // In real usage, you'd read an actual file
  const mockDocument = Buffer.from(`
# System Architecture Overview

## Current State
Our system uses a monolithic architecture with the following components:
- Web server: Node.js Express
- Database: PostgreSQL
- Cache: Redis
- Queue: RabbitMQ

## Known Issues
- Scaling bottlenecks during peak hours
- Long deployment cycles
- Tight coupling between modules

## Proposed Solutions
1. Microservices migration
2. Kubernetes deployment
3. Service mesh for communication
  `);

  const blob = new Blob([mockDocument], { type: "text/markdown" });

  console.log("Uploading architecture document...");
  const doc = await client.documents.upload({
    file: blob,
    name: "architecture-overview.md",
  });
  console.log(`Document uploaded: ${doc.document_id}`);

  // List documents
  const documents = await client.documents.list({ limit: 10 });
  console.log(`Total documents: ${documents.total}`);
  console.log("");

  // Run a debate that references the document
  console.log("Running debate with document context...");
  const debate = await client.debates.run({
    task: "Based on our architecture document, should we proceed with the microservices migration?",
    agents: ["anthropic-api", "openai-api"],
    rounds: 3,
    context: {
      document_ids: [doc.document_id],
    },
  });

  console.log(`Debate completed: ${debate.status}`);
  console.log(`Consensus: ${debate.consensus}`);

  // Get evidence used in the debate
  const evidence = await client.debates.evidence(debate.debate_id);
  console.log(`\nEvidence sources used: ${evidence.sources.length}`);
  evidence.sources.slice(0, 3).forEach((source) => {
    console.log(`  - ${source.type}: ${source.title || source.id}`);
  });
  console.log("");

  // Clean up
  await client.documents.delete(doc.document_id);
  console.log("Document cleaned up");
  console.log("");
}

// Example 2: Memory system integration
async function memoryIntegration() {
  console.log("=== Example 2: Memory Integration ===\n");

  // Get memory analytics
  const analytics = await client.memory.analytics();
  console.log("Memory System Analytics:");
  console.log(`  Total Memories: ${analytics.total_memories}`);
  console.log(`  Total Size: ${(analytics.total_size_bytes / 1024 / 1024).toFixed(2)} MB`);
  console.log(`  Active Debates: ${analytics.active_debates}`);
  console.log("");

  // Get tier statistics
  const tiers = await client.memory.tierStats();
  console.log("Memory Tiers:");
  tiers.tiers.forEach((tier) => {
    console.log(`  ${tier.name}:`);
    console.log(`    Count: ${tier.count}`);
    console.log(`    Size: ${(tier.size_bytes / 1024).toFixed(2)} KB`);
    console.log(`    TTL: ${tier.ttl_seconds}s`);
  });
  console.log("");

  // Retrieve relevant memories
  console.log("Retrieving memories about 'rate limiting'...");
  const memories = await client.memory.retrieve({
    query: "rate limiting implementation",
    limit: 5,
  });

  console.log(`Found ${memories.memories.length} relevant memories:`);
  memories.memories.forEach((memory, idx) => {
    console.log(`  ${idx + 1}. [${memory.tier}] ${memory.content.substring(0, 80)}...`);
    console.log(`     Relevance: ${(memory.relevance_score * 100).toFixed(1)}%`);
  });
  console.log("");

  // Check memory pressure
  const pressure = await client.memory.pressure();
  console.log("Memory Pressure:");
  console.log(`  Level: ${pressure.level}`);
  console.log(`  Usage: ${(pressure.usage_percent * 100).toFixed(1)}%`);
  console.log(`  Recommendation: ${pressure.recommendation}`);
  console.log("");
}

// Example 3: Cross-debate learning
async function crossDebateLearning() {
  console.log("=== Example 3: Cross-Debate Learning ===\n");

  // Run first debate
  console.log("Running first debate on API design...");
  const debate1 = await client.debates.run({
    task: "What's the best approach for API versioning?",
    agents: ["anthropic-api", "openai-api"],
    rounds: 2,
  });
  console.log(`First debate completed: ${debate1.consensus}`);
  console.log("");

  // Get follow-up suggestions
  const followups = await client.debates.followupSuggestions(debate1.debate_id);
  console.log("Suggested follow-up topics:");
  followups.suggestions.slice(0, 3).forEach((suggestion, idx) => {
    console.log(`  ${idx + 1}. ${suggestion}`);
  });
  console.log("");

  // Fork the debate to explore a different angle
  console.log("Forking debate to explore related topic...");
  const forked = await client.debates.fork(debate1.debate_id, {
    task: "How should we handle breaking changes in our API versioning strategy?",
  });
  console.log(`Forked debate ID: ${forked.debate_id}`);

  // Wait for forked debate
  const forkedResult = await client.debates.waitForCompletion(forked.debate_id);
  console.log(`Forked debate completed: ${forkedResult.consensus}`);
  console.log("");

  // Get summary combining insights
  console.log("Getting debate summaries...");
  const summary1 = await client.debates.summary(debate1.debate_id);
  const summary2 = await client.debates.summary(forkedResult.debate_id);

  console.log("\nOriginal debate summary:");
  console.log(`  ${summary1.summary.substring(0, 200)}...`);
  console.log("\nForked debate summary:");
  console.log(`  ${summary2.summary.substring(0, 200)}...`);
  console.log("");
}

// Example 4: Building knowledge over time
async function buildingKnowledge() {
  console.log("=== Example 4: Building Knowledge Over Time ===\n");

  // Define a series of related debates
  const topics = [
    "What are the key principles of good system design?",
    "How should we handle distributed transactions?",
    "What monitoring strategy should we adopt for microservices?",
  ];

  const debateIds: string[] = [];

  console.log("Running series of related debates...\n");

  for (const topic of topics) {
    console.log(`Topic: "${topic.substring(0, 50)}..."`);

    // Include previous debate knowledge in context
    const debate = await client.debates.run({
      task: topic,
      agents: ["anthropic-api", "openai-api"],
      rounds: 2,
      context: {
        // Reference previous debates in the series
        related_debate_ids: debateIds.length > 0 ? debateIds : undefined,
        // Enable cross-debate memory
        use_cross_debate_memory: true,
      },
    });

    debateIds.push(debate.debate_id);
    console.log(`  Result: ${debate.consensus}`);
    console.log(`  Citations: ${(await client.debates.citations(debate.debate_id)).citations.length}`);
    console.log("");
  }

  // Consolidate memories from all debates
  console.log("Consolidating learnings into memory...");
  await client.memory.consolidate();
  console.log("Memory consolidated successfully");

  // Get memory snapshot
  const snapshot = await client.memory.snapshot();
  console.log(`\nMemory Snapshot:`);
  console.log(`  Total Entries: ${snapshot.total_entries}`);
  console.log(`  Latest Update: ${new Date(snapshot.updated_at).toLocaleString()}`);
  console.log("");
}

// Example 5: Export and share knowledge
async function exportKnowledge() {
  console.log("=== Example 5: Export and Share Knowledge ===\n");

  // List recent debates
  const debates = await client.debates.list({
    limit: 5,
    status: "completed",
  });

  if (debates.debates.length === 0) {
    console.log("No completed debates to export");
    return;
  }

  const debateId = debates.debates[0].debate_id;
  console.log(`Exporting debate: ${debateId}\n`);

  // Export in different formats
  const formats = ["markdown", "json", "html"] as const;

  for (const format of formats) {
    const exported = await client.debates.export(debateId, { format });
    console.log(`${format.toUpperCase()} export:`);
    console.log(`  Size: ${exported.content.length} characters`);

    // Show preview
    const preview = exported.content.substring(0, 150).replace(/\n/g, " ");
    console.log(`  Preview: ${preview}...`);
    console.log("");
  }

  // Get citations for bibliography
  const citations = await client.debates.citations(debateId);
  console.log(`Bibliography (${citations.citations.length} sources):`);
  citations.citations.slice(0, 5).forEach((citation, idx) => {
    console.log(`  ${idx + 1}. ${citation.title || citation.source}`);
    console.log(`     URL: ${citation.url || "N/A"}`);
  });
  console.log("");
}

// Run all examples
async function main() {
  console.log("Knowledge Integration Examples\n");
  console.log("==============================\n");

  // Run examples based on what's available
  try {
    await client.health();
  } catch {
    console.log("API not available. Examples require a running Aragora server.");
    return;
  }

  await documentIntegration();
  await memoryIntegration();
  await crossDebateLearning();
  await buildingKnowledge();
  await exportKnowledge();

  console.log("Knowledge integration examples completed!");
}

main().catch(console.error);
