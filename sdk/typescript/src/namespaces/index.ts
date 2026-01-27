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

// Critiques
export {
  CritiquesAPI,
  type Critique,
  type CritiquePattern,
  type ArchivedCritique,
} from './critiques';

// Belief Network
export {
  BeliefNetworkAPI,
  type Crux,
  type LoadBearingClaim,
  type BeliefNode,
  type BeliefEdge,
} from './belief-network';

// Moments
export {
  MomentsAPI,
  type Moment,
  type MomentsSummary,
  type MomentsTimeline,
  type TrendingMoments,
} from './moments';

// Introspection
export {
  IntrospectionAPI,
  type AgentInfo,
  type LeaderboardEntry,
  type SystemIntrospection,
} from './introspection';

// Documents
export {
  DocumentsAPI,
  type Document,
  type DocumentFormat,
  type UploadResult,
} from './documents';

// Retention
export {
  RetentionAPI,
  type RetentionPolicy,
  type ExpiringItem,
  type ExecutionResult,
} from './retention';

// Notifications
export {
  NotificationsAPI,
  type NotificationChannel,
  type NotificationEventType,
  type IntegrationStatus,
  type EmailConfig,
  type TelegramConfig,
  type EmailRecipient,
  type NotificationDelivery,
} from './notifications';

// Pulse (Trending Topics)
export {
  PulseAPI,
  type PulseSource,
  type TrendingTopic,
  type DebateSuggestion,
  type PulseAnalytics,
  type SchedulerStatus,
  type SchedulerConfig,
  type ScheduledDebate,
} from './pulse';

// Leaderboard
export {
  LeaderboardAPI,
  type RankingEntry,
  type AgentPerformance,
  type HeadToHead,
  type DomainLeaderboard,
  type LeaderboardView,
} from './leaderboard';

// Policies
export {
  PoliciesAPI,
  type PolicyType,
  type PolicySeverity,
  type PolicyAction,
  type Policy,
  type PolicyRule,
  type PolicyViolation,
  type ComplianceSummary,
  type CreatePolicyRequest,
  type UpdatePolicyRequest,
} from './policies';

// Replays
export {
  ReplaysAPI,
  type ReplayEventType,
  type ReplaySummary,
  type ReplayEvent,
  type Replay,
  type EvolutionEntry,
  type ReplayFork,
} from './replays';

// Cost Management
export {
  CostManagementAPI,
  type CostBreakdownItem,
  type DailyCost,
  type CostAlert,
  type CostSummary,
  type CostBudget,
  type CostRecommendation,
  type EfficiencyMetrics,
  type CostForecast,
  type TimeRange,
  type GroupBy,
} from './cost-management';

// Onboarding
export {
  OnboardingAPI,
  type StarterTemplate,
  type OnboardingFlow,
  type QuickStartConfig,
  type OnboardingAnalytics,
  type QuickStartProfile,
  type FlowAction,
} from './onboarding';

// Deliberations
export {
  DeliberationsAPI,
  type Deliberation,
  type DeliberationStatus,
  type DeliberationStats,
  type ActiveDeliberationsResponse,
  type DeliberationStreamConfig,
} from './deliberations';

// Genesis (Evolution Visibility)
export {
  GenesisAPI,
  type GenesisEventType,
  type GenesisEvent,
  type GenesisStats,
  type Genome,
  type LineageNode,
  type DescendantNode,
  type Population,
  type DebateTree,
  type ListEventsOptions,
  type ListGenomesOptions,
} from './genesis';

// Laboratory (Persona Lab)
export {
  LaboratoryAPI,
  type EmergentTrait,
  type CrossPollinationSuggestion,
  type EmergentTraitsResponse,
  type CrossPollinationResponse,
  type EmergentTraitsOptions,
  type CrossPollinationRequest,
} from './laboratory';

// Teams Bot Integration
export {
  TeamsAPI,
  type TeamsBotStatus,
} from './teams';

// Learning (Meta-Learning Analytics)
export {
  LearningAPI,
  type LearningSession,
  type MetaLearningStats,
  type LearningPattern,
  type LearningEfficiency,
  type ListSessionsOptions as LearningSessionsOptions,
  type ListPatternsOptions as LearningPatternsOptions,
} from './learning';

// Batch Operations
export {
  BatchAPI,
  type BatchItem,
  type BatchItemStatus,
  type BatchStatus,
  type BatchSubmitRequest,
  type BatchSubmitResponse,
  type BatchItemResult,
  type BatchStatusResponse,
  type BatchSummary,
  type QueueStatus,
  type ListBatchesOptions,
} from './batch';

// Email Integrations
export {
  GmailAPI,
  type GmailConnection,
  type EmailTriageRule,
  type EmailDebateConfig,
  type ProcessedEmail,
  type GmailStats,
  type OAuthInitResponse,
  type SyncResult,
  type CreateTriageRuleRequest,
  type UpdateTriageRuleRequest,
  type CreateDebateConfigRequest,
} from './gmail';

export {
  OutlookAPI,
  type OutlookOAuthResponse,
  type OutlookAuthResult,
  type OutlookStatus,
  type MailFolder,
  type MessageSummary,
  type Message,
  type Attachment,
  type Conversation,
  type SendMessageRequest,
  type ReplyMessageRequest,
  type ListMessagesParams,
} from './outlook';

// Email Priority
export {
  EmailPriorityAPI,
  type ScoringTier,
  type UserAction,
  type EmailCategory,
  type GmailScopes,
  type EmailMessage,
  type PriorityFactor,
  type PriorityResult,
  type CategoryResult,
  type CategoryStats,
  type EmailConfig as EmailPriorityConfig,
  type FeedbackItem,
  type InboxItem,
  type InboxParams,
  type CrossChannelContext,
  type ContextBoost,
} from './email-priority';

// Accounting & Payroll
export {
  AccountingAPI,
  type Company,
  type FinancialStats,
  type Customer,
  type Transaction,
  type AccountingStatus,
  type FinancialReport,
  type ReportRequest,
  type Employee,
  type PayrollRun,
  type PayrollDetails,
  type JournalEntry,
  type ListCustomersParams,
  type ListTransactionsParams,
} from './accounting';

// Unified Inbox (Multi-Account Email)
export {
  UnifiedInboxAPI,
  type EmailProvider,
  type AccountStatus,
  type TriageAction,
  type PriorityTier,
  type ConnectedAccount,
  type UnifiedMessage,
  type TriageResult,
  type InboxStats,
  type InboxTrends,
  type OAuthUrlResponse,
  type ConnectAccountRequest,
  type ListMessagesParams as UnifiedInboxListParams,
  type TriageRequest,
  type BulkAction,
  type BulkActionRequest,
} from './unified-inbox';

// Expenses
export {
  ExpensesAPI,
  type ExpenseCategory,
  type ExpenseStatus,
  type PaymentMethod,
  type Expense,
  type UploadReceiptRequest,
  type CreateExpenseRequest,
  type UpdateExpenseRequest,
  type ListExpensesParams,
  type ExpenseStats,
  type SyncResult as ExpenseSyncResult,
  type CategorizeResult,
} from './expenses';

// Threat Intelligence
export {
  ThreatIntelAPI,
  type ThreatSeverity,
  type ThreatType,
  type HashType,
  type URLCheckResult,
  type URLBatchSummary,
  type IPReputationResult,
  type IPBatchSummary,
  type HashCheckResult,
  type HashBatchSummary,
  type EmailScanResult,
  type ThreatIntelStatus,
  type CheckURLRequest,
  type CheckURLsBatchRequest,
  type CheckIPsBatchRequest,
  type CheckHashesBatchRequest,
  type ScanEmailRequest,
} from './threat-intel';

// Devices & Push Notifications
export {
  DevicesAPI,
  type DeviceType,
  type NotificationStatus,
  type DeviceRegistration,
  type Device,
  type NotificationMessage,
  type NotificationResult,
  type UserNotificationResult,
  type ConnectorHealth,
  type AlexaRequest,
  type AlexaResponse,
  type GoogleActionsRequest,
  type GoogleActionsResponse,
} from './devices';

// Privacy (GDPR/CCPA)
export {
  PrivacyAPI,
  type DataCategory,
  type ThirdPartySharing,
  type DataInventory,
  type PrivacyPreferences,
  type ExportMetadata,
  type DataExport,
  type AccountDeletionResponse,
  type AccountDeletionRequest,
} from './privacy';

// Feedback (NPS & User Feedback)
export {
  FeedbackAPI,
  type FeedbackType,
  type NPSSubmission,
  type FeedbackSubmission,
  type FeedbackResponse,
  type NPSSummary,
  type FeedbackPrompt,
} from './feedback';

// Code Review
export {
  CodeReviewAPI,
  type FindingSeverity,
  type FindingCategory,
  type ReviewFinding,
  type ReviewResult,
  type CodeReviewRequest,
  type DiffReviewRequest,
  type PRReviewRequest,
  type SecurityScanResult,
  type ReviewHistoryResponse,
} from './code-review';

// RLM (Recursive Language Models)
export {
  RLMAPI,
  type RLMStrategy,
  type SourceType,
  type StreamMode,
  type StrategyInfo,
  type CompressionResult,
  type QueryResult,
  type ContextSummary,
  type ContextDetails,
  type StreamChunk,
  type RLMStats,
} from './rlm';

// Backups (Disaster Recovery)
export {
  BackupsAPI,
  type BackupType,
  type BackupStatus,
  type Backup,
  type VerificationResult,
  type ComprehensiveVerificationResult,
  type RetentionPolicy as BackupRetentionPolicy,
  type BackupStats,
} from './backups';

// Dashboard
export {
  DashboardAPI,
  type StatCard,
  type DashboardOverview,
  type TeamPerformance,
  type TopSender,
  type DashboardStats,
  type ActivityItem,
  type LabelInfo,
  type UrgentEmail,
  type PendingAction,
  type InboxSummary,
  type QuickAction,
  type QuickActionResult,
} from './dashboard';

// Payments
export {
  PaymentsAPI,
  type PaymentProvider,
  type PaymentStatus,
  type SubscriptionInterval,
  type BillingAddress,
  type PaymentMethodDetails,
  type ChargeRequest,
  type AuthorizeRequest,
  type RefundRequest,
  type PaymentResult,
  type TransactionDetails,
  type CustomerProfile,
  type PaymentMethodSummary,
  type CreateCustomerRequest,
  type UpdateCustomerRequest,
  type Subscription as PaymentSubscription,
  type CreateSubscriptionRequest,
  type UpdateSubscriptionRequest,
} from './payments';

// Training (Model Fine-tuning)
export {
  TrainingAPI,
  type ExportFormat,
  type ExportType,
  type JobStatus,
  type GauntletPersona,
  type SFTExportParams,
  type DPOExportParams,
  type GauntletExportParams,
  type SFTRecord,
  type DPORecord,
  type TrainingExportResult,
  type TrainingStats,
  type TrainingFormats,
  type TrainingJob,
  type TrainingJobDetails,
  type ListJobsParams,
  type CompleteJobData,
  type TrainingMetrics,
  type TrainingArtifacts,
} from './training';

// Email Priority (Inbox Intelligence)
export {
  EmailPriorityAPI,
  type ScoringTier,
  type UserAction,
  type EmailCategory,
  type GmailScopes,
  type EmailMessage,
  type PriorityFactor,
  type PriorityResult,
  type CategoryResult,
  type CategoryStats,
  type EmailConfig as EmailPriorityConfig,
  type FeedbackItem,
  type InboxItem,
  type InboxParams,
  type CrossChannelContext,
  type ContextBoost,
} from './email-priority';

// AP Automation (Accounts Payable)
export {
  APAutomationAPI,
  type PaymentPriority,
  type APPaymentMethod,
  type APInvoiceStatus,
  type APInvoice,
  type AddAPInvoiceRequest,
  type RecordAPPaymentRequest,
  type ListAPInvoicesParams,
  type OptimizePaymentsRequest,
  type PaymentScheduleEntry,
  type PaymentSchedule,
  type BatchPaymentRequest,
  type BatchPayment,
  type CashFlowEntry,
  type CashFlowForecast,
  type DiscountOpportunity,
} from './ap-automation';

// AR Automation (Accounts Receivable)
export {
  ARAutomationAPI,
  type ARInvoiceStatus,
  type ReminderLevel,
  type ARLineItem,
  type ARInvoice,
  type CreateARInvoiceRequest,
  type ListARInvoicesParams,
  type RecordARPaymentRequest,
  type AgingBucket,
  type AgingReport,
  type CollectionActionType,
  type CollectionSuggestion,
  type AddARCustomerRequest,
  type CustomerBalance,
} from './ar-automation';

// Invoice Processing (OCR & Approval Workflows)
export {
  InvoiceProcessingAPI,
  type InvoiceProcessingStatus,
  type AnomalySeverity as InvoiceAnomalySeverity,
  type InvoiceLineItem,
  type ProcessedInvoice,
  type InvoiceAnomaly,
  type CreateInvoiceRequest,
  type ListInvoicesParams,
  type POMatch,
  type SchedulePaymentRequest,
  type ScheduledPayment,
  type InvoiceStats,
  type PurchaseOrder,
  type CreatePORequest,
} from './invoice-processing';
