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
export { GauntletAPI, type GauntletRunRequest } from './gauntlet';
export { AnalyticsAPI, type AnalyticsDateRange } from './analytics';
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
