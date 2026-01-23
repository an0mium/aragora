/**
 * Control Plane Components
 *
 * Enterprise Multi-Agent Control Plane UI components for Aragora.
 * Provides workflow building, agent management, knowledge exploration,
 * and execution monitoring.
 */

// Agent Catalog
export {
  AgentCatalog,
  AgentCard,
  type AgentCatalogProps,
  type AgentCardProps,
  type AgentInfo,
  type AgentStatus,
  type AgentFilter,
  type AgentSort,
} from './AgentCatalog';

// Workflow Builder
export {
  WorkflowBuilder,
  WorkflowCanvas,
  WorkflowToolbar,
  NodePalette,
  type WorkflowBuilderProps,
  type WorkflowCanvasProps,
  type WorkflowToolbarProps,
  type NodePaletteProps,
} from './WorkflowBuilder';

// Knowledge Explorer
export {
  KnowledgeExplorer,
  QueryInterface,
  NodeBrowser,
  GraphViewer,
  type KnowledgeExplorerProps,
  type QueryInterfaceProps,
  type NodeBrowserProps,
  type GraphViewerProps,
  type ExplorerTab,
} from './KnowledgeExplorer';

// Execution Monitor
export {
  ExecutionMonitor,
  ExecutionTimeline,
  ApprovalQueue,
  type ExecutionMonitorProps,
  type ExecutionTimelineProps,
  type ApprovalQueueProps,
  type MonitorTab,
} from './ExecutionMonitor';

// Policy Dashboard
export {
  PolicyDashboard,
  ComplianceFrameworkList,
  ViolationTracker,
  RiskOverview,
  type PolicyDashboardProps,
  type PolicyTab,
  type ComplianceFrameworkListProps,
  type ComplianceFramework,
  type ViolationTrackerProps,
  type ComplianceViolation,
  type RiskOverviewProps,
} from './PolicyDashboard';

// Workspace Manager
export {
  WorkspaceManager,
  WorkspaceSettings,
  TeamAccessPanel,
  type WorkspaceManagerProps,
  type Workspace,
  type WorkspaceMember,
  type WorkspaceSettingsProps,
  type TeamAccessPanelProps,
} from './WorkspaceManager';

// Fine-tuning Pipeline
export {
  FineTuningDashboard,
  ModelSelector,
  TrainingConfig,
  JobMonitor,
  type FineTuningDashboardProps,
  type FineTuningJob,
  type ModelSelectorProps,
  type AvailableModel,
  type TrainingConfigProps,
  type TrainingParameters,
  type JobMonitorProps,
} from './FineTuning';

// Connector Dashboard
export {
  ConnectorDashboard,
  ConnectorCard,
  ConnectorConfigModal,
  SyncStatusWidget,
  type ConnectorDashboardProps,
  type ConnectorFilter,
  type DashboardTab,
  type ConnectorCardProps,
  type ConnectorInfo,
  type ConnectorType,
  type ConnectorStatus,
  type ConnectorConfigModalProps,
  type ConnectorConfigField,
  type SyncStatusWidgetProps,
  type SyncHistoryItem,
} from './ConnectorDashboard';

// Template Gallery
export {
  TemplateGallery,
  TemplateCard,
  TemplatePreview,
  type TemplateGalleryProps,
  type TemplateFilter as GalleryTemplateFilter,
  type TemplateSort,
  type TemplateCardProps,
  type WorkflowTemplate,
  type WorkflowStep,
  type WorkflowCategory,
  type TemplatePreviewProps,
} from './TemplateGallery';

// Fleet Status Widget
export {
  FleetStatusWidget,
  FleetHealthGauge,
  type FleetStatusWidgetProps,
  type FleetAgent,
  type AgentStatus as FleetAgentStatus,
  type FleetHealthGaugeProps,
} from './FleetStatusWidget';

// Activity Feed
export {
  ActivityFeed,
  ActivityEventItem,
  type ActivityFeedProps,
  type ActivityEventItemProps,
  type ActivityEvent,
  type ActivityEventType,
} from './ActivityFeed';

// Deliberation Tracker
export {
  DeliberationTracker,
  DeliberationCard,
  type DeliberationTrackerProps,
  type DeliberationCardProps,
  type Deliberation,
  type DeliberationStatus,
  type DeliberationAgent,
} from './DeliberationTracker';
