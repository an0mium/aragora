/**
 * Example 02: Streaming Debate
 *
 * This example demonstrates how to use WebSocket streaming to receive
 * real-time updates during a debate.
 */

import { AragoraClient } from "@aragora/sdk";

// Initialize the client
const client = new AragoraClient({
  baseUrl: process.env.ARAGORA_API_URL || "http://localhost:8080",
  apiKey: process.env.ARAGORA_API_KEY,
});

// WebSocket message types
interface WSMessage {
  type: string;
  loop_id?: string;
  agent?: string;
  data?: {
    task?: string;
    content?: string;
    answer?: string;
    debate_id?: string;
    loop_id?: string;
    agent?: string;
    round?: number;
    phase?: string;
    critique?: string;
    score?: number;
    method?: string;
    [key: string]: unknown;
  };
}

async function runStreamingDebate() {
  console.log("Starting streaming debate example...\n");

  // Create the debate first
  const created = await client.debates.create({
    task: "How should we design a distributed caching system?",
    agents: ["anthropic-api", "openai-api"],
    rounds: 3,
  });

  const debateId = created.debate_id;
  console.log("Debate created with ID:", debateId);

  // Connect to WebSocket for streaming updates
  const wsUrl = process.env.ARAGORA_WS_URL || "ws://localhost:8765/ws";
  const ws = new WebSocket(wsUrl);

  return new Promise<void>((resolve, reject) => {
    ws.onopen = () => {
      console.log("WebSocket connected, listening for updates...\n");
    };

    ws.onmessage = (event) => {
      const message: WSMessage = JSON.parse(event.data);

      // Skip connection/sync messages
      if (["connection_info", "loop_list", "sync"].includes(message.type)) {
        return;
      }

      // Filter to only our debate
      const eventLoopId =
        message.loop_id ||
        message.data?.debate_id ||
        message.data?.loop_id;
      if (eventLoopId && eventLoopId !== debateId) {
        return;
      }

      // Handle different event types
      switch (message.type) {
        case "debate_start":
          console.log("🚀 DEBATE STARTED");
          console.log(`   Task: ${message.data?.task}`);
          console.log("");
          break;

        case "round_start":
          console.log(`📍 ROUND ${message.data?.round} - ${message.data?.phase}`);
          console.log("");
          break;

        case "agent_message":
          console.log(`💬 ${message.agent || message.data?.agent}:`);
          const content = message.data?.content || "";
          // Truncate long messages for display
          const displayContent =
            content.length > 200
              ? content.substring(0, 200) + "..."
              : content;
          console.log(`   ${displayContent}`);
          console.log("");
          break;

        case "critique":
          console.log(`🔍 CRITIQUE from ${message.data?.agent}:`);
          const critique = message.data?.critique || "";
          const displayCritique =
            critique.length > 150
              ? critique.substring(0, 150) + "..."
              : critique;
          console.log(`   ${displayCritique}`);
          if (message.data?.score !== undefined) {
            console.log(`   Score: ${message.data.score}/10`);
          }
          console.log("");
          break;

        case "vote":
          console.log(`🗳️  VOTE from ${message.data?.agent}`);
          break;

        case "consensus":
          console.log("✅ CONSENSUS REACHED");
          console.log(`   Method: ${message.data?.method}`);
          console.log(`   Answer: ${message.data?.answer}`);
          console.log("");
          break;

        case "debate_end":
          console.log("🏁 DEBATE ENDED");
          ws.close();
          resolve();
          break;

        case "error":
          console.error("❌ ERROR:", message.data);
          ws.close();
          reject(new Error(String(message.data)));
          break;

        default:
          // Log unknown events for debugging
          console.log(`📌 ${message.type}:`, message.data);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      reject(error);
    };

    ws.onclose = () => {
      console.log("\nWebSocket connection closed");
    };

    // Set a timeout in case debate doesn't complete
    setTimeout(
      () => {
        if (ws.readyState === WebSocket.OPEN) {
          console.log("\nTimeout reached, closing connection");
          ws.close();
          resolve();
        }
      },
      5 * 60 * 1000
    ); // 5 minutes
  });
}

// Run the example
runStreamingDebate()
  .then(() => {
    console.log("\nStreaming debate example completed!");
    process.exit(0);
  })
  .catch((error) => {
    console.error("Error:", error);
    process.exit(1);
  });
