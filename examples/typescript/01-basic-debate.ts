/**
 * Example 01: Basic Debate
 *
 * This example demonstrates how to create and run a basic debate
 * using the Aragora TypeScript SDK.
 */

import { AragoraClient } from "@aragora/sdk";

// Initialize the client
const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL || "http://localhost:8080",
  apiKey: process.env.ARAGORA_API_KEY,
});

async function runBasicDebate() {
  console.log("Starting basic debate example...\n");

  // Method 1: Create and wait for completion in one call
  console.log("Method 1: Using run() for create + wait");
  const result = await client.debates.run({
    task: "Should we adopt TypeScript for our backend services?",
    agents: ["anthropic-api", "openai-api"],
    rounds: 3,
    consensus: "majority",
  });

  console.log("Debate completed!");
  console.log("Debate ID:", result.debate_id);
  console.log("Status:", result.status);
  console.log("Consensus:", result.consensus);
  console.log("Final Answer:", result.final_answer);
  console.log("\n---\n");

  // Method 2: Create, then poll for completion
  console.log("Method 2: Using create() + waitForCompletion()");
  const created = await client.debates.create({
    task: "What's the best approach for handling API rate limiting?",
    agents: ["anthropic-api", "openai-api", "gemini"],
    rounds: 2,
  });

  console.log("Debate created with ID:", created.debate_id);
  console.log("Waiting for completion...");

  const completed = await client.debates.waitForCompletion(created.debate_id, {
    timeout: 300000, // 5 minutes
    pollInterval: 3000, // Poll every 3 seconds
  });

  console.log("Debate completed!");
  console.log("Status:", completed.status);
  console.log("Consensus:", completed.consensus);
  console.log("\n---\n");

  // Get additional debate information
  console.log("Getting additional information...");

  // Get debate messages
  const messages = await client.debates.messages(completed.debate_id, {
    limit: 10,
  });
  console.log(`Retrieved ${messages.messages.length} messages`);

  // Get summary
  const summary = await client.debates.summary(completed.debate_id);
  console.log("Summary:", summary.summary);

  // Get citations
  const citations = await client.debates.citations(completed.debate_id);
  console.log(`Found ${citations.citations.length} citations`);

  // Export debate
  const exported = await client.debates.export(completed.debate_id, {
    format: "markdown",
  });
  console.log("Exported debate length:", exported.content.length, "characters");

  console.log("\nBasic debate example completed!");
}

// Run the example
runBasicDebate().catch(console.error);
