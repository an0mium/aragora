/**
 * Debates Quickstart Example
 *
 * Demonstrates the complete debate lifecycle using the Aragora SDK:
 * - Creating a client and initializing a debate
 * - Streaming real-time events via WebSocket
 * - Handling different event types
 * - Retrieving final results
 *
 * Usage:
 *   npx ts-node examples/debates-quickstart.ts
 *
 * Environment:
 *   ARAGORA_API_KEY - Your API key
 *   ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
 */

import {
  createClient,
  streamDebate,
  AragoraError,
  type Debate,
  type ConsensusResult,
  type WebSocketEvent,
} from '@aragora/sdk';

// =============================================================================
// Configuration
// =============================================================================

const config = {
  baseUrl: process.env.ARAGORA_API_URL || 'https://api.aragora.ai',
  apiKey: process.env.ARAGORA_API_KEY,
  timeout: 60000,
  retryEnabled: true,
  maxRetries: 3,
};

// =============================================================================
// Main Example
// =============================================================================

async function runDebateWithStreaming(): Promise<void> {
  console.log('=== Aragora Debates Quickstart ===\n');

  // Create the client
  const client = createClient(config);

  try {
    // -------------------------------------------------------------------------
    // Step 1: Create a debate
    // -------------------------------------------------------------------------
    console.log('Step 1: Creating debate...');

    const debate = await client.debates.create({
      task: 'What is the most effective approach to reducing technical debt in a large codebase?',
      agents: ['claude', 'gpt-4', 'gemini'],
      rounds: 3,
      consensus: 'weighted',
      context: 'Focus on practical strategies for engineering teams.',
    });

    console.log(`  Debate ID: ${debate.debate_id}`);
    console.log(`  Status: ${debate.status}`);
    console.log(`  Agents: ${debate.agents.join(', ')}`);
    if (debate.websocket_url) {
      console.log(`  WebSocket URL: ${debate.websocket_url}`);
    }

    // -------------------------------------------------------------------------
    // Step 2: Stream events using async iterator (recommended approach)
    // -------------------------------------------------------------------------
    console.log('\nStep 2: Streaming debate events...\n');

    const eventCounts: Record<string, number> = {};
    let consensusResult: ConsensusResult | null = null;

    // Stream events for this specific debate
    const stream = streamDebate(config, { debateId: debate.debate_id });

    for await (const event of stream) {
      // Track event counts
      eventCounts[event.type] = (eventCounts[event.type] || 0) + 1;

      // Handle different event types
      switch (event.type) {
        case 'debate_start':
          console.log('--- Debate Started ---');
          const startData = event.data as { task: string; agents: string[]; total_rounds: number };
          console.log(`  Task: ${startData.task}`);
          console.log(`  Agents: ${startData.agents.join(', ')}`);
          console.log(`  Rounds: ${startData.total_rounds}`);
          break;

        case 'round_start':
          const roundData = event.data as { round_number: number };
          console.log(`\n=== Round ${roundData.round_number} ===`);
          break;

        case 'agent_message':
          const msgData = event.data as { agent: string; content: string; confidence?: number };
          console.log(`\n[${msgData.agent}]:`);
          // Truncate long messages for display
          const preview = msgData.content.length > 200
            ? msgData.content.substring(0, 200) + '...'
            : msgData.content;
          console.log(`  ${preview}`);
          if (msgData.confidence !== undefined) {
            console.log(`  Confidence: ${(msgData.confidence * 100).toFixed(1)}%`);
          }
          break;

        case 'critique':
          const critiqueData = event.data as {
            critic: string;
            target: string;
            critique: string;
            severity?: string;
          };
          console.log(`\n[${critiqueData.critic} critiques ${critiqueData.target}]:`);
          console.log(`  ${critiqueData.critique.substring(0, 150)}...`);
          if (critiqueData.severity) {
            console.log(`  Severity: ${critiqueData.severity}`);
          }
          break;

        case 'vote':
          const voteData = event.data as { agent: string; vote: string; confidence?: number };
          console.log(`  Vote: ${voteData.agent} -> ${voteData.vote}`);
          break;

        case 'consensus':
        case 'consensus_reached':
          const consData = event.data as { consensus: ConsensusResult };
          consensusResult = consData.consensus;
          console.log('\n=== Consensus Reached ===');
          console.log(`  Answer: ${consensusResult.final_answer || consensusResult.conclusion}`);
          console.log(`  Confidence: ${((consensusResult.confidence || 0) * 100).toFixed(1)}%`);
          if (consensusResult.supporting_agents) {
            console.log(`  Supporting: ${consensusResult.supporting_agents.join(', ')}`);
          }
          if (consensusResult.dissenting_agents?.length) {
            console.log(`  Dissenting: ${consensusResult.dissenting_agents.join(', ')}`);
          }
          break;

        case 'debate_end':
          const endData = event.data as { status: string };
          console.log(`\n--- Debate Ended (${endData.status}) ---`);
          break;

        case 'error':
          const errData = event.data as { code: string; message: string };
          console.error(`Error: [${errData.code}] ${errData.message}`);
          break;

        case 'warning':
          const warnData = event.data as { code: string; message: string };
          console.warn(`Warning: [${warnData.code}] ${warnData.message}`);
          break;

        default:
          // Log other event types for debugging
          // console.log(`Event: ${event.type}`);
          break;
      }

      // Exit the stream when debate ends
      if (event.type === 'debate_end') {
        break;
      }
    }

    // -------------------------------------------------------------------------
    // Step 3: Retrieve final debate results
    // -------------------------------------------------------------------------
    console.log('\nStep 3: Retrieving final results...');

    const finalDebate = await client.debates.get(debate.debate_id);

    console.log(`\n=== Final Results ===`);
    console.log(`  Status: ${finalDebate.status}`);
    console.log(`  Rounds Used: ${finalDebate.rounds_used || finalDebate.rounds?.length || 0}`);
    if (finalDebate.duration_seconds) {
      console.log(`  Duration: ${finalDebate.duration_seconds}s`);
    }

    if (finalDebate.consensus) {
      console.log(`\n  Consensus:`);
      console.log(`    Answer: ${finalDebate.consensus.final_answer || finalDebate.consensus.conclusion}`);
      console.log(`    Confidence: ${((finalDebate.consensus.confidence || 0) * 100).toFixed(1)}%`);
      console.log(`    Reached: ${finalDebate.consensus.reached ? 'Yes' : 'No'}`);
    }

    // -------------------------------------------------------------------------
    // Step 4: Event summary
    // -------------------------------------------------------------------------
    console.log('\n=== Event Summary ===');
    for (const [eventType, count] of Object.entries(eventCounts).sort()) {
      console.log(`  ${eventType}: ${count}`);
    }

  } catch (error) {
    handleError(error);
  }
}

// =============================================================================
// Alternative: WebSocket Callback-based Approach
// =============================================================================

async function runDebateWithCallbacks(): Promise<void> {
  console.log('\n=== Alternative: Callback-based WebSocket ===\n');

  const client = createClient(config);

  try {
    // Create WebSocket connection
    const ws = client.createWebSocket({
      autoReconnect: true,
      maxReconnectAttempts: 3,
      heartbeatInterval: 30000,
    });

    // Set up event handlers before connecting
    ws.on('connected', () => {
      console.log('WebSocket connected');
    });

    ws.on('error', (error) => {
      console.error('WebSocket error:', error.message);
    });

    ws.on('debate_start', (event) => {
      console.log(`Debate started: ${event.debate_id}`);
    });

    ws.on('agent_message', (event) => {
      console.log(`[${event.agent}]: ${event.content.substring(0, 100)}...`);
    });

    ws.on('consensus', (event) => {
      console.log('Consensus reached:', event.consensus.final_answer);
    });

    // Connect to the WebSocket server
    await ws.connect();

    // Create a debate
    const debate = await client.debates.create({
      task: 'What are the best practices for API versioning?',
      agents: ['claude', 'gpt-4'],
      rounds: 2,
    });

    // Subscribe to the debate's events
    ws.subscribe(debate.debate_id);
    console.log(`Subscribed to debate: ${debate.debate_id}`);

    // Wait for the debate to end (with timeout)
    await Promise.race([
      ws.once('debate_end'),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Debate timeout')), 5 * 60 * 1000)
      ),
    ]);

    // Cleanup
    ws.disconnect();
    console.log('WebSocket disconnected');

  } catch (error) {
    handleError(error);
  }
}

// =============================================================================
// Error Handling
// =============================================================================

function handleError(error: unknown): void {
  if (error instanceof AragoraError) {
    console.error('\nAragora Error:');
    console.error(`  Message: ${error.message}`);
    console.error(`  Code: ${error.code}`);
    console.error(`  Status: ${error.status}`);
    if (error.traceId) {
      console.error(`  Trace ID: ${error.traceId}`);
    }
    if (error.details) {
      console.error(`  Details:`, error.details);
    }
  } else if (error instanceof Error) {
    console.error('\nError:', error.message);
  } else {
    console.error('\nUnknown error:', error);
  }
  process.exit(1);
}

// =============================================================================
// Run the example
// =============================================================================

async function main(): Promise<void> {
  // Check for API key
  if (!process.env.ARAGORA_API_KEY) {
    console.warn('Warning: ARAGORA_API_KEY not set. Some operations may fail.\n');
  }

  // Run the streaming example
  await runDebateWithStreaming();

  // Optionally run the callback-based example
  // await runDebateWithCallbacks();
}

main().catch(handleError);
