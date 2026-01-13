/**
 * Aragora Live - Zustand State Management
 *
 * This module provides centralized state management using Zustand.
 * All stores use devtools for debugging and subscribeWithSelector for
 * optimized re-renders.
 *
 * Usage:
 * ```tsx
 * import { useDebateStore, useSettingsStore, useUIStore } from '@/store';
 *
 * function MyComponent() {
 *   // Get full store access
 *   const messages = useDebateStore((state) => state.current.messages);
 *
 *   // Or use selectors for optimized subscriptions
 *   const theme = useSettingsStore(selectTheme);
 *
 *   // Actions are stable and can be called directly
 *   const addToast = useUIStore((state) => state.addToast);
 * }
 * ```
 *
 * Store overview:
 * - debateStore: Live debate sessions, messages, streaming, artifacts
 * - settingsStore: User preferences, feature config, API keys
 * - uiStore: Panel visibility, modals, toasts, loading states
 */

// Debate store - live debate sessions and artifacts
export {
  useDebateStore,
  // Types
  type TranscriptMessage,
  type StreamingMessage,
  type DebateConnectionStatus,
  type DebateArtifact,
  // Selectors
  selectDebateStatus,
  selectDebateMessages,
  selectStreamingMessages,
  selectDebateAgents,
  selectDebateTask,
  selectStreamEvents,
  selectHasCitations,
  selectDebateUI,
  selectArtifact,
} from './debateStore';

// Settings store - user preferences and feature configuration
export {
  useSettingsStore,
  // Types
  type NotificationPreferences,
  type DisplayPreferences,
  type IntegrationConfig,
  type UserPreferences,
  type FeatureConfig,
  type APIKey,
  type BackendConfig,
  // Selectors
  selectTheme,
  selectNotifications,
  selectDisplay,
  selectIntegrations,
  selectFeatureConfig,
  selectFeatureLoading,
  selectApiKeys,
  selectBackend,
  selectSaveStatus,
} from './settingsStore';

// UI store - global UI state, panels, modals, toasts
export {
  useUIStore,
  // Types
  type ViewMode,
  type MobileViewMode,
  type PanelVisibility,
  type PanelPosition,
  type ToastMessage,
  // Selectors
  selectViewMode,
  selectMobileViewMode,
  selectIsMobile,
  selectPanels,
  selectPanelPositions,
  selectActiveModal,
  selectModalData,
  selectToasts,
  selectGlobalLoading,
  selectKeyboardShortcutsEnabled,
} from './uiStore';
