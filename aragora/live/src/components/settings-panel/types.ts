/**
 * Types for Settings Panel components.
 */

export interface FeatureConfig {
  // Feature toggles
  calibration: boolean;
  trickster: boolean;
  rhetorical: boolean;
  insights: boolean;
  moments: boolean;
  crux: boolean;
  evolution: boolean;
  continuum_memory: boolean;
  consensus_memory: boolean;
  laboratory: boolean;
  // Display
  show_advanced_metrics: boolean;
  compact_mode: boolean;
  theme: string;
  // Debate defaults
  default_mode: string;
  default_rounds: number;
  default_agents: string;
  // Notifications
  telegram_enabled: boolean;
  email_digest: string;
  consensus_alert_threshold: number;
}

export const DEFAULT_FEATURE_CONFIG: FeatureConfig = {
  calibration: true,
  trickster: false,
  rhetorical: true,
  insights: true,
  moments: true,
  crux: true,
  evolution: true,
  continuum_memory: true,
  consensus_memory: true,
  laboratory: true,
  show_advanced_metrics: false,
  compact_mode: false,
  theme: 'system',
  default_mode: 'standard',
  default_rounds: 3,
  default_agents: 'claude,gemini,gpt4',
  telegram_enabled: false,
  email_digest: 'none',
  consensus_alert_threshold: 0.7,
};

export interface UserPreferences {
  theme: 'dark' | 'light' | 'system';
  notifications: {
    email_digest: boolean;
    debate_completed: boolean;
    weekly_summary: boolean;
  };
  display: {
    compact_mode: boolean;
    show_agent_icons: boolean;
    auto_scroll_messages: boolean;
  };
  api_keys: {
    name: string;
    prefix: string;
    created_at: string;
    last_used: string | null;
  }[];
  integrations: {
    slack_webhook: string | null;
    discord_webhook: string | null;
  };
}

export const DEFAULT_PREFERENCES: UserPreferences = {
  theme: 'dark',
  notifications: {
    email_digest: true,
    debate_completed: true,
    weekly_summary: false,
  },
  display: {
    compact_mode: false,
    show_agent_icons: true,
    auto_scroll_messages: true,
  },
  api_keys: [],
  integrations: {
    slack_webhook: null,
    discord_webhook: null,
  },
};

export type SettingsTab = 'features' | 'debate' | 'appearance' | 'notifications' | 'api' | 'integrations' | 'account';

export interface SlackNotifications {
  notify_on_consensus: boolean;
  notify_on_debate_end: boolean;
  notify_on_error: boolean;
  notify_on_leaderboard: boolean;
}

export const PREFERENCES_KEY = 'aragora_preferences';

export function getStoredPreferences(): Partial<UserPreferences> {
  if (typeof window === 'undefined') return {};
  const stored = localStorage.getItem(PREFERENCES_KEY);
  if (!stored) return {};
  try {
    return JSON.parse(stored);
  } catch {
    return {};
  }
}

export function storePreferences(prefs: Partial<UserPreferences>): void {
  const current = getStoredPreferences();
  const merged = { ...current, ...prefs };
  localStorage.setItem(PREFERENCES_KEY, JSON.stringify(merged));
}
