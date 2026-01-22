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
