/**
 * Example 04: Working with Agents
 *
 * This example demonstrates how to work with AI agents in debates,
 * including viewing profiles, comparing agents, and understanding rankings.
 */

import { AragoraClient } from "@aragora/sdk";

// Initialize the client
const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL || "http://localhost:8080",
  apiKey: process.env.ARAGORA_API_KEY,
});

// Example 1: Get agent profiles
async function getAgentProfiles() {
  console.log("=== Example 1: Agent Profiles ===\n");

  const agents = ["anthropic-api", "openai-api", "gemini"];

  for (const agentId of agents) {
    try {
      const profile = await client.agents.profile(agentId);
      console.log(`Agent: ${profile.name}`);
      console.log(`  ID: ${profile.id}`);
      console.log(`  Provider: ${profile.provider}`);
      console.log(`  ELO Rating: ${profile.elo_rating}`);
      console.log(`  Win Rate: ${(profile.win_rate * 100).toFixed(1)}%`);
      console.log(`  Debates Participated: ${profile.total_debates}`);
      console.log(`  Specialties: ${profile.specialties?.join(", ") || "N/A"}`);
      console.log("");
    } catch (error) {
      console.log(`Failed to get profile for ${agentId}:`, (error as Error).message);
    }
  }
}

// Example 2: View leaderboard
async function viewLeaderboard() {
  console.log("=== Example 2: Agent Leaderboard ===\n");

  // Get top agents by ELO
  const leaderboard = await client.agents.leaderboard({
    limit: 10,
    sort: "elo",
  });

  console.log("Top 10 Agents by ELO Rating:");
  console.log("-".repeat(60));
  console.log(
    "Rank | Agent".padEnd(30) +
      "| ELO".padEnd(10) +
      "| Win Rate".padEnd(12) +
      "| Debates"
  );
  console.log("-".repeat(60));

  leaderboard.entries.forEach((entry, index) => {
    const rank = `${index + 1}`.padStart(4);
    const name = entry.name.padEnd(24);
    const elo = entry.elo_rating.toString().padEnd(8);
    const winRate = `${(entry.win_rate * 100).toFixed(1)}%`.padEnd(10);
    const debates = entry.total_debates.toString();

    console.log(`${rank} | ${name} | ${elo} | ${winRate} | ${debates}`);
  });
  console.log("");
}

// Example 3: Compare agents
async function compareAgents() {
  console.log("=== Example 3: Agent Comparison ===\n");

  const comparison = await client.agents.compare([
    "anthropic-api",
    "openai-api",
  ]);

  console.log("Agent Comparison: anthropic-api vs openai-api");
  console.log("-".repeat(50));

  console.log("\nHead-to-Head Stats:");
  console.log(`  Total Debates: ${comparison.head_to_head.total_debates}`);
  console.log(`  anthropic-api Wins: ${comparison.head_to_head.agent1_wins}`);
  console.log(`  openai-api Wins: ${comparison.head_to_head.agent2_wins}`);
  console.log(`  Ties: ${comparison.head_to_head.ties}`);

  console.log("\nStrength Comparison:");
  comparison.strengths.forEach(({ category, agent1_score, agent2_score }) => {
    const bar1 = "█".repeat(Math.round(agent1_score * 10));
    const bar2 = "█".repeat(Math.round(agent2_score * 10));
    console.log(`  ${category.padEnd(20)}`);
    console.log(`    anthropic-api: ${bar1} ${agent1_score.toFixed(2)}`);
    console.log(`    openai-api:    ${bar2} ${agent2_score.toFixed(2)}`);
  });
  console.log("");
}

// Example 4: Agent network (who debates with whom)
async function viewAgentNetwork() {
  console.log("=== Example 4: Agent Network ===\n");

  const network = await client.agents.network("anthropic-api");

  console.log("anthropic-api's Debate Network:");
  console.log(`  Total Connections: ${network.connections.length}`);
  console.log("\n  Most Frequent Debate Partners:");

  network.connections
    .slice(0, 5)
    .forEach(({ agent_id, debate_count, win_rate }) => {
      console.log(
        `    - ${agent_id}: ${debate_count} debates, ${(win_rate * 100).toFixed(1)}% win rate`
      );
    });
  console.log("");
}

// Example 5: Agent consistency metrics
async function viewAgentConsistency() {
  console.log("=== Example 5: Agent Consistency ===\n");

  const consistency = await client.agents.consistency("anthropic-api");

  console.log("anthropic-api Consistency Metrics:");
  console.log(`  Overall Consistency Score: ${(consistency.overall_score * 100).toFixed(1)}%`);
  console.log(`  Position Stability: ${(consistency.position_stability * 100).toFixed(1)}%`);
  console.log(`  Argument Quality Variance: ${consistency.quality_variance.toFixed(3)}`);

  console.log("\n  Consistency by Topic:");
  consistency.by_topic?.slice(0, 5).forEach(({ topic, score }) => {
    const bar = "█".repeat(Math.round(score * 10));
    console.log(`    ${topic.padEnd(25)} ${bar} ${(score * 100).toFixed(1)}%`);
  });
  console.log("");
}

// Example 6: Agent debate history
async function viewAgentHistory() {
  console.log("=== Example 6: Agent History ===\n");

  const history = await client.agents.history("anthropic-api", {
    limit: 5,
  });

  console.log("anthropic-api's Recent Debates:");
  console.log("-".repeat(70));

  history.debates.forEach((debate) => {
    console.log(`  Debate ID: ${debate.debate_id}`);
    console.log(`    Task: ${debate.task.substring(0, 50)}...`);
    console.log(`    Result: ${debate.result}`);
    console.log(`    ELO Change: ${debate.elo_change > 0 ? "+" : ""}${debate.elo_change}`);
    console.log(`    Date: ${new Date(debate.completed_at).toLocaleDateString()}`);
    console.log("");
  });
}

// Example 7: Select optimal agent team for a task
async function selectAgentTeam() {
  console.log("=== Example 7: Selecting Optimal Agent Team ===\n");

  // Get leaderboard to find top performers
  const leaderboard = await client.agents.leaderboard({
    limit: 20,
    sort: "elo",
  });

  // Define task requirements
  const task = "Design a fault-tolerant distributed system";
  const requiredCapabilities = ["technical", "architecture", "critical-thinking"];

  console.log(`Task: "${task}"`);
  console.log(`Required Capabilities: ${requiredCapabilities.join(", ")}`);
  console.log("\nRecommended Team:");

  // Simple selection: top 3 by ELO with diverse providers
  const selectedAgents: string[] = [];
  const usedProviders = new Set<string>();

  for (const entry of leaderboard.entries) {
    if (selectedAgents.length >= 3) break;

    // Ensure diversity by checking provider
    const provider = entry.id.split("-")[0];
    if (!usedProviders.has(provider)) {
      selectedAgents.push(entry.id);
      usedProviders.add(provider);
      console.log(`  - ${entry.name} (ELO: ${entry.elo_rating}, Provider: ${provider})`);
    }
  }

  // Run debate with selected team
  console.log("\nStarting debate with selected team...");
  const result = await client.debates.run({
    task,
    agents: selectedAgents,
    rounds: 3,
  });

  console.log(`\nDebate completed!`);
  console.log(`  Consensus: ${result.consensus}`);
  console.log(`  Status: ${result.status}`);
  console.log("");
}

// Run all examples
async function main() {
  console.log("Working with Agents Examples\n");
  console.log("============================\n");

  await getAgentProfiles();
  await viewLeaderboard();
  await compareAgents();
  await viewAgentNetwork();
  await viewAgentConsistency();
  await viewAgentHistory();
  await selectAgentTeam();

  console.log("Agent examples completed!");
}

main().catch(console.error);
