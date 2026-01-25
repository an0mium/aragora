/**
 * Aragora TypeScript SDK
 *
 * @deprecated This package (@aragora/client) is deprecated. Please migrate to @aragora/sdk.
 *
 * Migration guide:
 * ```typescript
 * // Before (deprecated)
 * import { AragoraClient } from '@aragora/client';
 * const client = new AragoraClient('http://localhost:8080');
 *
 * // After (recommended)
 * import { createClient } from '@aragora/sdk';
 * const client = createClient({ baseUrl: 'http://localhost:8080' });
 * ```
 *
 * The @aragora/sdk package provides:
 * - More comprehensive API coverage
 * - Better TypeScript types
 * - WebSocket streaming with async iterators
 * - Control Plane APIs
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

// Log deprecation warning at runtime (once per session)
const DEPRECATION_WARNING_KEY = '__aragora_client_deprecation_warned__';

/* eslint-disable @typescript-eslint/no-explicit-any */
const globalStore = (typeof globalThis !== 'undefined' ? globalThis : {}) as any;
if (!globalStore[DEPRECATION_WARNING_KEY]) {
  console.warn(
    '\n\x1b[33m[DEPRECATION WARNING]\x1b[0m @aragora/client is deprecated and will be removed in v3.0.0.\n' +
    'Please migrate to @aragora/sdk:\n\n' +
    '  npm install @aragora/sdk\n\n' +
    'Migration:\n' +
    '  // Before\n' +
    "  import { AragoraClient } from '@aragora/client';\n" +
    "  const client = new AragoraClient('http://localhost:8080');\n\n" +
    '  // After\n' +
    "  import { createClient } from '@aragora/sdk';\n" +
    "  const client = createClient({ baseUrl: 'http://localhost:8080' });\n\n" +
    'See: https://github.com/aragora/aragora/blob/main/docs/SDK_CONSOLIDATION.md\n'
  );
  globalStore[DEPRECATION_WARNING_KEY] = true;
}
/* eslint-enable @typescript-eslint/no-explicit-any */

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
