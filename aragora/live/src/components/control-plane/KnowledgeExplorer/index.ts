export { QueryInterface, type QueryInterfaceProps } from './QueryInterface';
export { NodeBrowser, type NodeBrowserProps } from './NodeBrowser';
export { GraphViewer, type GraphViewerProps } from './GraphViewer';
export { StaleKnowledgeTab, type StaleNode } from './StaleKnowledgeTab';
export { KnowledgeExplorer, type KnowledgeExplorerProps, type ExplorerTab } from './KnowledgeExplorer';
export { default } from './KnowledgeExplorer';

// Visibility and sharing components
export { VisibilitySelector, type VisibilityLevel, type VisibilitySelectorProps } from './VisibilitySelector';
export { ShareDialog, type ShareDialogProps, type ShareGrant } from './ShareDialog';
export { SharedWithMeTab, type SharedWithMeTabProps, type SharedItem } from './SharedWithMeTab';
export { AccessGrantsList, type AccessGrantsListProps, type AccessGrant } from './AccessGrantsList';

// Federation components
export { FederationStatus, type FederationStatusProps, type FederatedRegion, type SyncMode, type SyncScope, type RegionHealth } from './FederationStatus';
