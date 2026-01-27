/**
 * SDK Namespace APIs
 *
 * Re-exports all namespace APIs for the Aragora SDK.
 */

export { DebatesAPI } from './debates';
export { AgentsAPI } from './agents';
export { WorkflowsAPI } from './workflows';
export { SMEAPI } from './sme';
export { ControlPlaneAPI } from './control-plane';
export { GauntletAPI } from './gauntlet';
export type { GauntletRunRequest } from '../types';
export { AnalyticsAPI, type AnalyticsPeriodOptions } from './analytics';
export { MemoryAPI, type MemoryStoreOptions, type MemoryRetrieveOptions } from './memory';
export { RBACAPI, type CreateRoleRequest, type UpdateRoleRequest } from './rbac';
export { KnowledgeAPI, type KnowledgeSearchOptions, type KnowledgeMoundQueryOptions } from './knowledge';
export { TournamentsAPI, type MatchResultSubmission } from './tournaments';
export { AuthAPI, type SessionInfo, type ApiKeyInfo, type OAuthProviderInfo } from './auth';
export { VerificationAPI, type DebateConclusionVerifyOptions } from './verification';
export { AuditAPI, type AuditEventFilterOptions, type AuditExportOptions } from './audit';
export { TenantsAPI } from './tenants';
export { OrganizationsAPI } from './organizations';

// SME Starter Pack APIs
export { BillingAPI, type BillingPlan, type BillingUsage, type Subscription, type UsageForecast } from './billing';
export { BudgetsAPI, type Budget, type BudgetAlert, type BudgetSummary, type CreateBudgetRequest, type UpdateBudgetRequest } from './budgets';
export { ReceiptsAPI, type DecisionReceipt, type GauntletReceiptExport } from './receipts';
export { ExplainabilityAPI, type ExplainabilityResult, type ExplanationFactor, type CounterfactualScenario, type Provenance, type Narrative } from './explainability';

// Enterprise APIs
export { WebhooksAPI, type Webhook, type WebhookEvent, type WebhookDelivery, type WebhookSLOStatus, type CreateWebhookRequest, type UpdateWebhookRequest } from './webhooks';
export { PluginsAPI, type Plugin, type PluginListing, type PluginSubmission, type PluginConfigSchema, type InstallPluginRequest, type PluginQueryRequest, type PluginValidateRequest } from './plugins';
export { WorkspacesAPI, type Workspace, type WorkspaceSettings, type WorkspaceMember, type WorkspaceProfile, type CreateWorkspaceRequest, type UpdateWorkspaceRequest, type AddMemberRequest, type UpdateMemberRequest } from './workspaces';
export { IntegrationsAPI, type Integration, type AvailableIntegration, type IntegrationConfigSchema, type IntegrationSyncStatus, type CreateIntegrationRequest, type UpdateIntegrationRequest } from './integrations';

// Marketplace
export {
  MarketplaceAPI,
  type DeploymentStatus,
  type MarketplaceListParams,
  type TemplateDeployment,
  type TemplateRatings,
} from './marketplace';

// Codebase Analysis
export { CodebaseAPI } from './codebase';

// Consensus & Detection
export {
  ConsensusAPI,
  type SimilarDebate,
  type SimilarDebatesResponse,
  type SettledTopic,
  type SettledTopicsResponse,
  type ConsensusStats,
  type DissentView,
  type DissentsResponse,
  type ContrarianView,
  type ContrariansResponse,
  type RiskWarning,
  type WarningsResponse,
  type DomainHistoryEntry,
  type DomainHistoryResponse,
  type FindSimilarOptions,
  type SettledOptions,
  type FilterOptions,
} from './consensus';

// OAuth Authentication
export {
  OAuthAPI,
  type OAuthProvider,
  type OAuthProviderConfig,
  type OAuthCallbackResponse,
  type LinkedOAuthAccount,
  type ProvidersResponse,
  type LinkedProvidersResponse,
  type LinkAccountRequest,
  type LinkAccountResponse,
  type UnlinkAccountRequest,
  type UnlinkAccountResponse,
  type AuthorizationUrlResponse,
} from './oauth';

// Monitoring & Anomaly Detection
export {
  MonitoringAPI,
  type TrendDirection,
  type AnomalySeverity,
  type MetricTrend,
  type TrendSummary,
  type Anomaly,
  type BaselineStats,
  type RecordMetricResponse,
  type GetTrendResponse,
  type GetAllTrendsResponse,
  type GetAnomaliesResponse,
  type GetBaselineResponse,
  type RecordMetricOptions,
  type GetAnomaliesOptions,
} from './monitoring';

// System Administration
export {
  SystemAPI,
  type MaintenanceTask,
  type CircuitBreakerStatus,
  type CycleEntry,
  type EventEntry,
  type DebateHistoryEntry,
  type HistorySummary,
  type CircuitBreakerMetrics,
  type AuthStats,
  type MaintenanceResult,
  type DebugTestResponse,
  type CyclesResponse,
  type EventsResponse,
  type DebateHistoryResponse,
  type CircuitBreakersResponse,
  type RevokeTokenResponse,
  type HistoryOptions,
} from './system';

// Nomic Loop Control
export {
  NomicAPI,
  type NomicHealthStatus,
  type NomicPhase,
  type NomicState,
  type NomicHealth,
  type NomicMetrics,
  type NomicLogResponse,
  type RiskEntry,
  type RiskRegisterResponse,
  type Proposal,
  type ProposalsResponse,
  type StartLoopRequest,
  type StopLoopRequest,
  type ApproveProposalRequest,
  type RejectProposalRequest,
  type OperationalMode,
  type ModesResponse,
} from './nomic';

// Cross-Pollination
export {
  CrossPollinationAPI,
  type CrossPollinationStats,
  type CrossPollinationSubscriber,
  type CrossPollinationBridge,
  type CrossPollinationMetrics,
  type KMIntegrationStatus,
  type KMSyncResult,
  type KMStalenessResult,
  type CulturePattern,
  type WorkspaceCulture,
} from './cross-pollination';

// Unified Decisions
export {
  DecisionsAPI,
  type DecisionType,
  type DecisionPriority,
  type DecisionStatus,
  type ResponseChannel,
  type DecisionContext,
  type DecisionConfig,
  type DecisionRequest,
  type DecisionResult,
  type DecisionStatusResponse,
  type DecisionSummary,
  type DecisionListResponse,
} from './decisions';
