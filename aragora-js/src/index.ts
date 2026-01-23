/**
 * Aragora TypeScript SDK
 *
 * A TypeScript client for the Aragora control plane for multi-agent deliberation.
 *
 * @example
 * ```typescript
 * import { AragoraClient } from '@aragora/client';
 *
 * const client = new AragoraClient('http://localhost:8080');
 *
 * // Run a debate
 * const debate = await client.debates.run('Should we use microservices?');
 * console.log(debate.consensus?.conclusion);
 *
 * // Use control plane
 * const agents = await client.controlPlane.listAgents();
 * const taskId = await client.controlPlane.submitTask('debate', { topic: 'test' });
 * const task = await client.controlPlane.waitForTask(taskId);
 * ```
 */

// Client
export { AragoraClient, AragoraError, DebatesAPI, AgentsAPI } from './client';
export type { AragoraClientOptions } from './client';

// Control Plane
export { ControlPlaneAPI } from './control-plane';

// WebSocket
export { DebateStream, streamDebate } from './websocket';

// Types
export type {
  // Common types
  DebateStatus,
  AgentStatus,
  TaskStatus,
  TaskPriority,
  ConsensusResult,
  AgentMessage,
  Debate,
  AgentProfile,
  HealthStatus,
  // Control plane types
  RegisteredAgent,
  AgentHealth,
  Task,
  ControlPlaneStatus,
  ResourceUtilization,
  // Request types
  CreateDebateRequest,
  SubmitTaskRequest,
  RegisterAgentRequest,
  // WebSocket types
  DebateEvent,
  DebateEventType,
  WebSocketOptions,
} from './types';
