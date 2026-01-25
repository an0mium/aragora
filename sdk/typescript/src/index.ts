/**
 * Aragora SDK for TypeScript/JavaScript
 *
 * Official SDK for the Aragora multi-agent debate platform.
 * Works in both browser and Node.js environments.
 *
 * @example
 * ```typescript
 * import { createClient, streamDebate } from '@aragora/sdk';
 *
 * const config = {
 *   baseUrl: 'https://api.aragora.ai',
 *   apiKey: 'your-api-key'
 * };
 *
 * const client = createClient(config);
 *
 * // Create a debate
 * const debate = await client.createDebate({
 *   task: 'Should we use TypeScript or JavaScript?',
 *   agents: ['claude', 'gpt-4', 'gemini'],
 *   rounds: 3
 * });
 *
 * // Stream events using async iterator (recommended)
 * for await (const event of streamDebate(config, { debateId: debate.debate_id })) {
 *   console.log(`${event.type}:`, event.data);
 *
 *   if (event.type === 'debate_end') {
 *     break;
 *   }
 * }
 *
 * // Or use the callback-based WebSocket API
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
  WorkflowExecution,
  WorkflowApproval,
  WorkflowVersion,
  WorkflowSimulationResult,

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
  SynthesisEvent,
  RevisionEvent,
  PhaseChangeEvent,
  AudienceSuggestionEvent,
  UserVoteEvent,
  ErrorEvent,
  WarningEvent,

  // Health
  HealthCheck,

  // Control Plane
  AgentStatus,
  TaskStatus,
  TaskPriority,
  RegisteredAgent,
  Task,
  TaskSubmitRequest,
  AgentRegisterRequest,
  HeartbeatRequest,
  ControlPlaneHealth,

  // Marketplace
  MarketplaceTemplate,
  TemplateReview,

  // Matrix Debates
  MatrixDebateCreateRequest,
  MatrixDebate,
  MatrixConclusion,

  // Graph Debates
  GraphDebateCreateRequest,
  GraphDebate,
  GraphBranch,
  GraphNode,
  GraphStats,
  MatrixScenarioResult,

  // Verification
  VerificationBackend,
  VerifyClaimRequest,
  VerificationResult,
  VerificationStatus,

  // Selection
  SelectionPlugin,
  AgentScore,
  ScoreAgentsRequest,
  TeamSelectionRequest,
  TeamSelection,

  // Replays
  ReplayFormat,
  Replay,

  // Memory Analytics
  MemoryTier,
  MemoryAnalytics,
  MemoryTierStats,

  // Auth
  RegisterRequest,
  RegisterResponse,
  LoginRequest,
  AuthToken,
  RefreshRequest,
  VerifyEmailRequest,
  VerifyResponse,
  User,
  UpdateProfileRequest,
  UpdateProfileResponse,
  ChangePasswordRequest,
  ForgotPasswordRequest,
  ResetPasswordRequest,
  OAuthUrlParams,
  OAuthUrl,
  OAuthCallbackRequest,
  MFASetupRequest,
  MFASetupResponse,
  MFAVerifyRequest,
  MFAVerifyResponse,

  // Tenancy
  Tenant,
  CreateTenantRequest,
  UpdateTenantRequest,
  TenantList,
  QuotaStatus,
  QuotaUpdate,
  TenantMember,
  MemberList,
  AddMemberRequest,

  // Organizations
  Organization,
  OrganizationMember,
  OrganizationInvitation,
  UserOrganization,

  // RBAC
  Permission,
  Role,
  RoleList,
  CreateRoleRequest,
  UpdateRoleRequest,
  PermissionList,
  PermissionCheck,
  RoleAssignment,
  AssignmentList,
  BulkAssignRequest,
  BulkAssignResponse,

  // Tournaments
  Tournament,
  CreateTournamentRequest,
  TournamentStandings,
  TournamentBracket,
  TournamentMatch,

  // Audit
  AuditEvent,
  AuditStats,
  AuditSession,
  CreateAuditSessionRequest,
  AuditFinding,

  // Onboarding
  OnboardingStatus,

  // Billing
  BillingPlan,
  BillingPlanList,
  BillingUsage,
  Subscription,
  Invoice,
  InvoiceList,
  UsageForecast,

  // Notifications
  NotificationStatus,
  EmailNotificationConfig,
  TelegramNotificationConfig,
  NotificationRecipient,

  // Budgets
  Budget,
  BudgetList,
  CreateBudgetRequest,
  UpdateBudgetRequest,
  BudgetAlert,
  BudgetAlertList,
  BudgetSummary,

  // Costs
  CostDashboard,
  CostBreakdown,
  CostTimeline,
  CostAlert,

  // Audit Trails
  AuditTrail,
  AuditTrailList,
} from './types';

export { AragoraError } from './types';

// Re-export client
export { AragoraClient, createClient } from './client';

// Re-export sync client
export { AragoraClientSync, createSyncClient } from './sync';

// Re-export WebSocket
export type { WebSocketState, WebSocketOptions, StreamOptions } from './websocket';
export {
  AragoraWebSocket,
  createWebSocket,
  streamDebate,
  streamDebateById,
} from './websocket';

// Default export for convenience
export { createClient as default } from './client';
