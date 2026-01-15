/**
 * Aragora TypeScript SDK Example
 *
 * Demonstrates how to use the Aragora JavaScript/TypeScript SDK
 * to run debates, tournaments, and query results programmatically.
 *
 * Prerequisites:
 *   npm install aragora-js
 *   # or: yarn add aragora-js
 *
 * Run:
 *   npx ts-node examples/05_typescript_sdk.ts
 *   # or: npx tsx examples/05_typescript_sdk.ts
 *
 * Environment:
 *   ARAGORA_API_URL - API endpoint (default: http://localhost:8080)
 *   ARAGORA_API_TOKEN - Optional auth token
 */

import { AragoraClient } from "aragora-js";

const API_URL = process.env.ARAGORA_API_URL || "http://localhost:8080";

async function main() {
  // Initialize client
  const client = new AragoraClient({
    baseUrl: API_URL,
    // token: process.env.ARAGORA_API_TOKEN, // Optional auth
  });

  console.log("=== Aragora TypeScript SDK Demo ===\n");

  // 1. Check server health
  console.log("1. Checking server health...");
  try {
    const health = await client.health.check();
    console.log(`   Server status: ${health.status}`);
    console.log(`   Version: ${health.version || "unknown"}\n`);
  } catch (error) {
    console.error("   Server not reachable. Start with: python -m aragora.server.unified_server");
    process.exit(1);
  }

  // 2. List available agents
  console.log("2. Available agents:");
  const agents = await client.agents.list();
  const agentNames = agents.slice(0, 5).map((a) => a.name);
  console.log(`   ${agentNames.join(", ")}${agents.length > 5 ? "..." : ""}\n`);

  // 3. Create a debate
  console.log("3. Creating debate...");
  const debate = await client.debates.create({
    topic: "What are the pros and cons of serverless architecture?",
    agents: ["anthropic-api", "openai-api"],
    rounds: 2,
    consensus: "majority",
  });
  console.log(`   Debate ID: ${debate.id}`);
  console.log(`   Status: ${debate.status}\n`);

  // 4. Wait for completion (poll status)
  console.log("4. Waiting for debate to complete...");
  let result = await client.debates.get(debate.id);
  let attempts = 0;
  const maxAttempts = 60;

  while (result.status !== "completed" && attempts < maxAttempts) {
    await sleep(2000);
    result = await client.debates.get(debate.id);
    process.stdout.write(".");
    attempts++;
  }
  console.log("\n");

  if (result.status !== "completed") {
    console.log("   Debate timed out. Check server logs.");
    return;
  }

  // 5. Display results
  console.log("5. Debate Results:");
  console.log(`   Consensus: ${result.consensus ? "Reached" : "Not reached"}`);
  console.log(`   Confidence: ${((result.confidence || 0) * 100).toFixed(1)}%`);
  console.log(`   Rounds completed: ${result.rounds_completed || 0}`);

  if (result.synthesis) {
    console.log("\n   Final Synthesis:");
    console.log("   " + result.synthesis.slice(0, 500) + "...\n");
  }

  // 6. Query agent rankings
  console.log("6. Agent Rankings (Top 5):");
  const rankings = await client.agents.rankings();
  rankings.slice(0, 5).forEach((agent, i) => {
    console.log(`   ${i + 1}. ${agent.name}: ${agent.elo?.toFixed(0) || "N/A"} ELO`);
  });
  console.log("");

  // 7. Get debate history
  console.log("7. Recent Debates:");
  const history = await client.debates.list({ limit: 3 });
  history.forEach((d) => {
    const status = d.status === "completed" ? "✓" : "○";
    console.log(`   ${status} ${d.id.slice(0, 8)}... - ${d.topic?.slice(0, 40) || "No topic"}...`);
  });

  console.log("\n=== Demo Complete ===");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Run
main().catch((error) => {
  console.error("Error:", error.message);
  process.exit(1);
});
