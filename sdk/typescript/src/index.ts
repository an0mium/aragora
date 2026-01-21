/**
 * Aragora SDK for TypeScript/JavaScript
 *
 * Official SDK for the Aragora multi-agent debate platform.
 * Works in both browser and Node.js environments.
 *
 * @example
 * ```typescript
 * import { createClient } from '@aragora/sdk';
 *
 * const client = createClient({
 *   baseUrl: 'https://api.aragora.ai',
 *   apiKey: 'your-api-key'
 * });
 *
 * // Create a debate
 * const debate = await client.createDebate({
 *   task: 'Should we use TypeScript or JavaScript?',
 *   agents: ['claude', 'gpt-4', 'gemini'],
 *   rounds: 3
 * });
 *
 * // Stream debate events
 * const ws = client.createWebSocket();
 * await ws.connect();
 * ws.subscribe(debate.debate_id);
 *
 * ws.on('agent_message', (event) => {
 *   console.log(`${event.agent}: ${event.content}`);
 * });
 *
 * ws.on('consensus', (event) => {
 *   console.log('Consensus reached:', event.consensus.final_answer);
 * });
 * ```
 *
 * @packageDocumentation
 */

// Re-export all types
export type {
  // Configuration
  AragoraConfig,

  // Errors
  ErrorCode,
  ApiError,

  // Common
  PaginatedResponse,
  PaginationParams,

  // Agents
  Agent,
  AgentProfile,

  // Debates
  DebateStatus,
  ConsensusType,
  ConsensusResult,
  Message,
  Round,
  Debate,
  DebateCreateRequest,
  DebateCreateResponse,

  // Workflows
  StepDefinition,
  TransitionRule,
  Workflow,
  WorkflowTemplate,

  // Gauntlet
  DecisionReceipt,
  RiskHeatmap,

  // Explainability
  ExplanationFactor,
  CounterfactualScenario,
  ExplainabilityResult,

  // WebSocket
  WebSocketEventType,
  WebSocketEvent,
  DebateStartEvent,
  RoundStartEvent,
  AgentMessageEvent,
  CritiqueEvent,
  VoteEvent,
  ConsensusEvent,
  DebateEndEvent,

  // Health
  HealthCheck,

  // Marketplace
  MarketplaceTemplate,
  TemplateReview,
} from './types';

export { AragoraError } from './types';

// Re-export client
export { AragoraClient, createClient } from './client';

// Re-export WebSocket
export type { WebSocketState, WebSocketOptions } from './websocket';
export { AragoraWebSocket, createWebSocket } from './websocket';

// Default export for convenience
export { createClient as default } from './client';
