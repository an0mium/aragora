/**
 * Quickstart: run a multi-agent debate via the Aragora API.
 *
 * Usage:
 *   npx ts-node examples/quickstart.ts
 *
 * Environment:
 *   ARAGORA_API_URL - API URL (default: http://localhost:8080)
 *   ARAGORA_API_KEY - Your API key (optional for demo mode)
 */

import { AragoraClient } from "@aragora/sdk";

const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL || "http://localhost:8080",
  apiKey: process.env.ARAGORA_API_KEY,
});

async function main() {
  const result = await client.debates.create({
    task: "Should we use microservices or a monolith?",
    agents: ["claude", "openai"],
    rounds: 3,
  });
  console.log(result.summary);
}

main().catch(console.error);
