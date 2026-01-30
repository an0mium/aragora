/**
 * Aragora SDK API Modules
 *
 * Barrel export for all API classes and types.
 */

// Base classes and utilities
export { HttpClient, BaseAPI, AragoraError, type AragoraClientConfig } from './base';

// Debates API
export { DebatesAPI } from './debates';
export type {
  Debate,
  DebateMessage,
  DebateRound,
  DebateCreateRequest,
  DebateCreateResponse,
  DebateListParams,
  DebateListResponse,
  ConsensusResult,
} from './debates';

// Agents API
export { AgentsAPI } from './agents';
export type {
  AgentProfile,
  AgentCreateRequest,
  LeaderboardEntry,
  LeaderboardResponse,
  AgentStatsResponse,
} from './agents';

// Analytics API
export { AnalyticsAPI } from './analytics';
export type {
  AnalyticsSummary,
  AnalyticsOverview,
  FindingsTrend,
  RemediationMetrics,
  AgentMetrics,
  CostAnalysis,
  ComplianceScore,
  HeatmapData,
  DisagreementStats,
} from './analytics';

// Workflows API
export { WorkflowsAPI } from './workflows';
export type {
  Workflow,
  WorkflowNode,
  WorkflowTemplate,
  WorkflowExecution,
  WorkflowCreateRequest,
} from './workflows';

// Admin API
export { AdminAPI } from './admin';
export type {
  RevenueData,
  RevenueResponse,
  AdminStats,
  AdminStatsResponse,
  Organization,
  OrganizationsResponse,
  User,
  UsersResponse,
  SystemHealth,
} from './admin';

// Training API
export { TrainingAPI } from './training';
export type {
  TrainingStats,
  TrainingStatsResponse,
  SFTExample,
  DPOExample,
  GauntletExample,
  TrainingExportOptions,
  TrainingExportResponse,
  TrainingJob,
  TrainingJobsResponse,
} from './training';

// WebSocket
export { AragoraWebSocket, createWebSocket } from './websocket';
export type {
  WebSocketOptions,
  WebSocketState,
  DebateEvent,
  EventHandler,
  StateHandler,
  ErrorHandler,
} from './websocket';

// Control Plane API
export { ControlPlaneAPI } from './control-plane';
export type {
  Agent,
  AgentRegistration,
  AgentHealth,
  Task,
  TaskSubmission,
  TaskResult,
  Deliberation,
  DeliberationRequest,
  DeliberationResult,
  SystemHealth as ControlPlaneHealth,
  ControlPlaneStats,
  QueueStatus,
  DashboardMetrics,
  PolicyViolation,
  ViolationStats,
} from './control-plane';

// Graph Debates API
export { GraphDebatesAPI } from './graph-debates';
export type {
  GraphDebateRequest,
  GraphDebateResponse,
  GraphNode,
  Branch,
  MergeResult,
  DebateGraph,
  GraphDebate,
  BranchPolicy,
} from './graph-debates';

// Audit API
export { AuditAPI } from './audit';
export type {
  AuditSession,
  AuditConfig,
  Finding,
  FindingsResponse,
  AuditEvent,
  AuditSummary,
  AuditReport,
  DeepAuditRequest,
  DeepAuditResponse,
} from './audit';

// Knowledge Mound API
export { KnowledgeAPI } from './knowledge';
export type {
  KnowledgeNode,
  CreateNodeRequest,
  Relationship,
  CreateRelationshipRequest,
  QueryOptions,
  QueryResponse,
  GraphTraversalOptions,
  GraphResponse,
  VisibilityInfo,
  MoundStats,
  FederationRegion,
  Checkpoint,
  ContradictionResult,
  AnalyticsSummary as KnowledgeAnalyticsSummary,
} from './knowledge';

// Connectors API
export { ConnectorsAPI } from './connectors';
export type {
  Connector,
  ConnectorType,
  ConnectorCreateRequest,
  SyncOperation,
  SyncSchedule,
  ConnectorStats,
  ConnectorHealth,
  Integration,
  IntegrationType,
} from './connectors';
