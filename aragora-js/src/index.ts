/**
 * Aragora TypeScript SDK
 *
 * A TypeScript client for the Aragora control plane for multi-agent vetted decisionmaking.
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
export {
  AragoraClient,
  AragoraError,
  DebatesAPI,
  AgentsAPI,
  GraphDebatesAPI,
  MatrixDebatesAPI,
  VerificationAPI,
  GauntletAPI,
  TeamSelectionAPI,
  AnalyticsAPI,
  MemoryAPI,
  KnowledgeAPI,
  WorkflowsAPI,
  TournamentsAPI,
  RBACAPI,
  AuthAPI,
} from './client';
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
  // Graph debate types
  GraphDebate,
  GraphBranch,
  CreateGraphDebateRequest,
  // Matrix debate types
  MatrixDebate,
  MatrixScenario,
  MatrixCell,
  MatrixConclusion,
  CreateMatrixDebateRequest,
  // Verification types
  VerificationStatus,
  VerificationResult,
  VerifyClaimRequest,
  // Gauntlet types
  GauntletChallenge,
  GauntletResult,
  GauntletReceipt,
  RunGauntletRequest,
  // Memory types
  MemoryTierStats,
  MemoryAnalytics,
  // Team selection types
  AgentScore,
  TeamSelection,
  SelectionPlugins,
  SelectTeamRequest,
  // Knowledge types
  KnowledgeEntry,
  KnowledgeSearchResult,
  KnowledgeStats,
  Fact,
  KnowledgeQueryResponse,
  // Workflow types
  Workflow,
  WorkflowStatus,
  WorkflowStep,
  WorkflowTrigger,
  WorkflowExecution,
  ExecutionStatus,
  WorkflowTemplate,
  // Tournament types
  Tournament,
  TournamentStatus,
  TournamentFormat,
  TournamentStandings,
  TournamentBracket,
  TournamentMatch,
  // RBAC types
  Role,
  Permission,
  RoleAssignment,
  // Auth types
  AuthToken,
  User,
  MFASetupResponse,
  MFAVerifyResponse,
} from './types';
