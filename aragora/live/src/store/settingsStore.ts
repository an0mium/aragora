'use client';

import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export interface NotificationPreferences {
  debateComplete: boolean;
  consensusReached: boolean;
  mentions: boolean;
  systemAlerts: boolean;
}

export interface DisplayPreferences {
  compactMode: boolean;
  showTimestamps: boolean;
  showRounds: boolean;
  animationsEnabled: boolean;
}

export interface IntegrationConfig {
  slackWebhook: string;
  discordWebhook: string;
  slackNotifications: {
    debateComplete: boolean;
    consensusReached: boolean;
  };
}

export interface UserPreferences {
  theme: 'dark' | 'light' | 'system';
  notifications: NotificationPreferences;
  display: DisplayPreferences;
  integrations: IntegrationConfig;
}

export interface FeatureConfig {
  calibration: boolean;
  trickster: boolean;
  rhetorical: boolean;
  streaming: boolean;
  audience: boolean;
  citations: boolean;
  memory: boolean;
  evidenceCollection: boolean;
  [key: string]: boolean;
}

export interface APIKey {
  id: string;
  name: string;
  prefix: string;
  createdAt: string;
  lastUsed?: string;
}

export interface BackendConfig {
  apiUrl: string;
  wsUrl: string;
  defaultAgents: string[];
  defaultRounds: number;
}

// ============================================================================
// Store State
// ============================================================================

interface SettingsState {
  // User preferences (persisted)
  preferences: UserPreferences;

  // Feature configuration (from server)
  featureConfig: FeatureConfig;
  featureLoading: boolean;

  // API keys
  apiKeys: APIKey[];

  // Backend configuration
  backend: BackendConfig;

  // Save status
  saveStatus: 'idle' | 'saving' | 'saved' | 'error';
  saveError: string | null;
}

interface SettingsActions {
  // Preference actions
  setTheme: (theme: 'dark' | 'light' | 'system') => void;
  updateNotifications: (notifications: Partial<NotificationPreferences>) => void;
  updateDisplay: (display: Partial<DisplayPreferences>) => void;
  updateIntegrations: (integrations: Partial<IntegrationConfig>) => void;
  updateSlackNotifications: (notifications: Partial<IntegrationConfig['slackNotifications']>) => void;

  // Feature config actions
  setFeatureConfig: (config: FeatureConfig) => void;
  setFeatureLoading: (loading: boolean) => void;
  toggleFeature: (feature: string) => void;

  // API key actions
  setApiKeys: (keys: APIKey[]) => void;
  addApiKey: (key: APIKey) => void;
  removeApiKey: (id: string) => void;

  // Backend config actions
  setBackendConfig: (config: Partial<BackendConfig>) => void;

  // Save status
  setSaveStatus: (status: 'idle' | 'saving' | 'saved' | 'error', error?: string) => void;

  // Reset
  resetPreferences: () => void;
  resetAll: () => void;
}

type SettingsStore = SettingsState & SettingsActions;

// ============================================================================
// Defaults
// ============================================================================

const defaultNotifications: NotificationPreferences = {
  debateComplete: true,
  consensusReached: true,
  mentions: true,
  systemAlerts: true,
};

const defaultDisplay: DisplayPreferences = {
  compactMode: false,
  showTimestamps: true,
  showRounds: true,
  animationsEnabled: true,
};

const defaultIntegrations: IntegrationConfig = {
  slackWebhook: '',
  discordWebhook: '',
  slackNotifications: {
    debateComplete: false,
    consensusReached: false,
  },
};

const defaultPreferences: UserPreferences = {
  theme: 'dark',
  notifications: defaultNotifications,
  display: defaultDisplay,
  integrations: defaultIntegrations,
};

const defaultFeatureConfig: FeatureConfig = {
  calibration: false,
  trickster: false,
  rhetorical: false,
  streaming: true,
  audience: true,
  citations: false,
  memory: true,
  evidenceCollection: false,
};

const defaultBackend: BackendConfig = {
  apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080',
  wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8765/ws',
  defaultAgents: (process.env.NEXT_PUBLIC_DEFAULT_AGENTS || 'anthropic-api,openai-api').split(','),
  defaultRounds: parseInt(process.env.NEXT_PUBLIC_DEFAULT_ROUNDS || '3', 10),
};

// ============================================================================
// Store Implementation
// ============================================================================

export const useSettingsStore = create<SettingsStore>()(
  devtools(
    persist(
      subscribeWithSelector((set) => ({
        // Initial state
        preferences: { ...defaultPreferences },
        featureConfig: { ...defaultFeatureConfig },
        featureLoading: false,
        apiKeys: [],
        backend: { ...defaultBackend },
        saveStatus: 'idle',
        saveError: null,

        // Preference actions
        setTheme: (theme) => set(
          (state) => ({
            preferences: { ...state.preferences, theme },
          }),
          false,
          'setTheme'
        ),

        updateNotifications: (notifications) => set(
          (state) => ({
            preferences: {
              ...state.preferences,
              notifications: { ...state.preferences.notifications, ...notifications },
            },
          }),
          false,
          'updateNotifications'
        ),

        updateDisplay: (display) => set(
          (state) => ({
            preferences: {
              ...state.preferences,
              display: { ...state.preferences.display, ...display },
            },
          }),
          false,
          'updateDisplay'
        ),

        updateIntegrations: (integrations) => set(
          (state) => ({
            preferences: {
              ...state.preferences,
              integrations: { ...state.preferences.integrations, ...integrations },
            },
          }),
          false,
          'updateIntegrations'
        ),

        updateSlackNotifications: (notifications) => set(
          (state) => ({
            preferences: {
              ...state.preferences,
              integrations: {
                ...state.preferences.integrations,
                slackNotifications: {
                  ...state.preferences.integrations.slackNotifications,
                  ...notifications,
                },
              },
            },
          }),
          false,
          'updateSlackNotifications'
        ),

        // Feature config actions
        setFeatureConfig: (config) => set(
          { featureConfig: config },
          false,
          'setFeatureConfig'
        ),

        setFeatureLoading: (loading) => set(
          { featureLoading: loading },
          false,
          'setFeatureLoading'
        ),

        toggleFeature: (feature) => set(
          (state) => ({
            featureConfig: {
              ...state.featureConfig,
              [feature]: !state.featureConfig[feature],
            },
          }),
          false,
          'toggleFeature'
        ),

        // API key actions
        setApiKeys: (keys) => set({ apiKeys: keys }, false, 'setApiKeys'),

        addApiKey: (key) => set(
          (state) => ({ apiKeys: [...state.apiKeys, key] }),
          false,
          'addApiKey'
        ),

        removeApiKey: (id) => set(
          (state) => ({ apiKeys: state.apiKeys.filter((k) => k.id !== id) }),
          false,
          'removeApiKey'
        ),

        // Backend config actions
        setBackendConfig: (config) => set(
          (state) => ({ backend: { ...state.backend, ...config } }),
          false,
          'setBackendConfig'
        ),

        // Save status
        setSaveStatus: (status, error) => set(
          { saveStatus: status, saveError: error || null },
          false,
          'setSaveStatus'
        ),

        // Reset
        resetPreferences: () => set(
          { preferences: { ...defaultPreferences } },
          false,
          'resetPreferences'
        ),

        resetAll: () => set(
          {
            preferences: { ...defaultPreferences },
            featureConfig: { ...defaultFeatureConfig },
            featureLoading: false,
            apiKeys: [],
            backend: { ...defaultBackend },
            saveStatus: 'idle',
            saveError: null,
          },
          false,
          'resetAll'
        ),
      })),
      {
        name: 'aragora-settings',
        partialize: (state) => ({
          // Only persist user preferences, not server-side config
          preferences: state.preferences,
          backend: state.backend,
        }),
      }
    ),
    { name: 'settings-store' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectTheme = (state: SettingsStore) => state.preferences.theme;
export const selectNotifications = (state: SettingsStore) => state.preferences.notifications;
export const selectDisplay = (state: SettingsStore) => state.preferences.display;
export const selectIntegrations = (state: SettingsStore) => state.preferences.integrations;
export const selectFeatureConfig = (state: SettingsStore) => state.featureConfig;
export const selectFeatureLoading = (state: SettingsStore) => state.featureLoading;
export const selectApiKeys = (state: SettingsStore) => state.apiKeys;
export const selectBackend = (state: SettingsStore) => state.backend;
export const selectSaveStatus = (state: SettingsStore) => ({
  status: state.saveStatus,
  error: state.saveError,
});
