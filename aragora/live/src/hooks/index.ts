/**
 * Aragora React Hooks
 *
 * Central export for all Aragora React hooks.
 * Import from '@/hooks' instead of individual files.
 *
 * @example
 * ```typescript
 * import {
 *   useAragoraClient,
 *   useDebateWebSocket,
 *   useAuth,
 *   useWorkspaces
 * } from '@/hooks';
 *
 * function MyComponent() {
 *   const client = useAragoraClient();
 *   const { debates } = await client.debates.list();
 * }
 * ```
 */

// =============================================================================
// Core Client & Auth
// =============================================================================

export { useAragoraClient, useClientCleanup, useClientAuth } from './useAragoraClient';
export { useAuth } from '@/context/AuthContext';
export { useAuthenticatedFetch } from './useAuthenticatedFetch';
export { useSession } from './useSession';

// =============================================================================
// API & Data Fetching
// =============================================================================

export { useApi } from './useApi';
export { useFetch } from './useFetch';
export { useSWRFetch } from './useSWRFetch';
export { useAsyncData } from './useAsyncData';

// =============================================================================
// Debate Hooks
// =============================================================================

export { useDebateWebSocket } from './debate-websocket/useDebateWebSocket';
export { useDebateFork } from './useDebateFork';
export { useBatchDebate } from './useBatchDebate';
export { useGraphDebateWebSocket } from './useGraphDebateWebSocket';
export { useDebateWebSocketStore } from './useDebateWebSocketStore';

// Re-export debate WebSocket types
export type {
  TranscriptMessage,
  StreamingMessage,
  DebateConnectionStatus,
  UseDebateWebSocketOptions,
  UseDebateWebSocketReturn,
} from './debate-websocket/types';

// =============================================================================
// Workspace & Organization
// =============================================================================

export { useWorkspaces } from './useWorkspaces';
export { useWorkspaceInvites } from './useWorkspaceInvites';

// =============================================================================
// Knowledge & Evidence
// =============================================================================

export { useKnowledgeQuery } from './useKnowledgeQuery';
export { useEvidence } from './useEvidence';
export { useGlobalKnowledge } from './useGlobalKnowledge';

// =============================================================================
// Explainability
// =============================================================================

export { useExplanation } from './useExplanation';
export type {
  ExplanationData,
  EvidenceLink as ExplanationEvidenceLink,
  VotePivot,
  BeliefChange,
  ConfidenceAttribution,
  Counterfactual,
} from './useExplanation';

// =============================================================================
// Control Plane & Policies
// =============================================================================

export { useControlPlane } from './useControlPlane';
export { useControlPlaneWebSocket } from './useControlPlaneWebSocket';
export { usePolicies } from './usePolicies';

// =============================================================================
// Workflow & Automation
// =============================================================================

export { useWorkflowBuilder } from './useWorkflowBuilder';
export { useWorkflowExecution } from './useWorkflowExecution';
export { useWorkflowWebSocket } from './useWorkflowWebSocket';
export { usePipeline } from './usePipeline';
export { usePipelineCanvas } from './usePipelineCanvas';
export { usePipelineWebSocket } from './usePipelineWebSocket';
export { useFractalNavigation } from './useFractalNavigation';
export type { NavigationLevel, FractalNavigationResult } from './useFractalNavigation';
export { usePlaybooks } from './usePlaybooks';

// =============================================================================
// Monitoring & Analytics
// =============================================================================

export { useCosts } from './useCosts';
export { useUsageDashboard, useUsageTrend, useCostBreakdown } from './useUsageDashboard';
export { useQueueMonitoring } from './useQueueMonitoring';

// =============================================================================
// Integrations & Connectors
// =============================================================================

export { useConnectorWebSocket } from './useConnectorWebSocket';
export { useBroadcast } from './useBroadcast';
export { useAgentRouting } from './useAgentRouting';

// =============================================================================
// Spectate (Real-Time Debate Observation)
// =============================================================================

export { useSpectate } from './useSpectate';
export type { SpectateEvent } from './useSpectate';

// =============================================================================
// Gauntlet & Testing
// =============================================================================

export { useGauntletWebSocket } from './useGauntletWebSocket';

// =============================================================================
// Nomic (Self-Improvement)
// =============================================================================

export { useNomicLoopWebSocket } from './useNomicLoopWebSocket';
export { useNomicStream } from './useNomicStream';

// =============================================================================
// Features & Capabilities
// =============================================================================

export { useFeatures } from './useFeatures';
export { useFineTuning } from './useFineTuning';

// =============================================================================
// UI & UX Helpers
// =============================================================================

export { useToast } from './useToast';
export { useFocusTrap } from './useFocusTrap';
export { useMediaQuery } from './useMediaQuery';
export { useSwipeGesture } from './useSwipeGesture';
export { useCommandPaletteSearch } from './useCommandPaletteSearch';
export { useDashboardPreferences } from './useDashboardPreferences';
export { useErrorHandler } from './useErrorHandler';
export { usePWA } from './usePWA';
export { useTimeout, useTimeoutEffect, useInterval, useIntervalEffect, useDebounce, useThrottle } from './useTimers';

// =============================================================================
// Data Management
// =============================================================================

export { useLocalHistory } from './useLocalHistory';
export { useSupabaseHistory } from './useSupabaseHistory';
export { useBlocklist } from './useBlocklist';
export { useDedup } from './useDedup';
export { usePruning } from './usePruning';
export { useVisibility } from './useVisibility';
export { useSharing } from './useSharing';
export { useFederation } from './useFederation';

// =============================================================================
// Pulse (Trending Topics)
// =============================================================================

export { usePulseScheduler } from './usePulseScheduler';

// =============================================================================
// Inbox
// =============================================================================

export { useInboxSync } from './useInboxSync';

// =============================================================================
// Base WebSocket
// =============================================================================

export { useWebSocketBase } from './useWebSocketBase';
