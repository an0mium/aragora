'use client';

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useTheme, type ThemePreference } from '@/context/ThemeContext';
import { useBackend } from '@/components/BackendSelector';

import type { FeatureConfig, UserPreferences, SettingsTab } from './types';
import { DEFAULT_FEATURE_CONFIG, DEFAULT_PREFERENCES, getStoredPreferences, storePreferences } from './types';
import { FeaturesTab } from './FeaturesTab';
import { DebateTab } from './DebateTab';
import { AppearanceTab } from './AppearanceTab';
import { NotificationsTab } from './NotificationsTab';
import { ApiKeysTab } from './ApiKeysTab';
import { IntegrationsTab } from './IntegrationsTab';
import { AccountTab } from './AccountTab';

const TABS = [
  { id: 'features', label: 'FEATURES' },
  { id: 'debate', label: 'DEBATE' },
  { id: 'appearance', label: 'APPEARANCE' },
  { id: 'notifications', label: 'NOTIFICATIONS' },
  { id: 'api', label: 'API KEYS' },
  { id: 'integrations', label: 'INTEGRATIONS' },
  { id: 'account', label: 'ACCOUNT' },
] as const;

export function SettingsPanel() {
  const { user, isAuthenticated } = useAuth();
  const { preference: themePreference, setTheme } = useTheme();
  const { config: backendConfig } = useBackend();
  const [activeTab, setActiveTab] = useState<SettingsTab>('features');
  const [featureConfig, setFeatureConfig] = useState<FeatureConfig>(DEFAULT_FEATURE_CONFIG);
  const [featureLoading, setFeatureLoading] = useState(true);
  const [featureSaveStatus, setFeatureSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [preferences, setPreferences] = useState<UserPreferences>(DEFAULT_PREFERENCES);
  const [slackWebhook, setSlackWebhook] = useState('');
  const [discordWebhook, setDiscordWebhook] = useState('');
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  // Load preferences and sync theme from context
  useEffect(() => {
    const stored = getStoredPreferences();

    setPreferences(prev => ({
      ...prev,
      ...stored,
      theme: themePreference, // Use theme from context
    }));

    if (stored.integrations?.slack_webhook) {
      setSlackWebhook(stored.integrations.slack_webhook);
    }
    if (stored.integrations?.discord_webhook) {
      setDiscordWebhook(stored.integrations.discord_webhook);
    }
  }, [themePreference]);

  // Fetch feature config from backend
  useEffect(() => {
    async function fetchFeatureConfig() {
      try {
        setFeatureLoading(true);
        const response = await fetch(`${backendConfig.api}/api/features/config`);
        if (response.ok) {
          const data = await response.json();
          if (data.preferences) {
            setFeatureConfig(prev => ({ ...prev, ...data.preferences }));
          }
        }
      } catch (error) {
        console.warn('Failed to fetch feature config:', error);
      } finally {
        setFeatureLoading(false);
      }
    }
    fetchFeatureConfig();
  }, [backendConfig.api]);

  const updateFeatureConfig = useCallback(async (key: keyof FeatureConfig, value: boolean | string | number) => {
    const newConfig = { ...featureConfig, [key]: value };
    setFeatureConfig(newConfig);

    setFeatureSaveStatus('saving');
    try {
      const response = await fetch(`${backendConfig.api}/api/features/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value }),
      });

      if (response.ok) {
        setFeatureSaveStatus('saved');
        setTimeout(() => setFeatureSaveStatus('idle'), 1500);
      } else {
        setFeatureSaveStatus('error');
        setTimeout(() => setFeatureSaveStatus('idle'), 2000);
      }
    } catch {
      setFeatureSaveStatus('error');
      setTimeout(() => setFeatureSaveStatus('idle'), 2000);
    }
  }, [featureConfig, backendConfig.api]);

  const updateThemePreference = useCallback((theme: 'dark' | 'light' | 'system') => {
    // Update context (handles localStorage, DOM, and everything)
    setTheme(theme);
    // Update local preferences state for UI sync
    setPreferences(prev => ({ ...prev, theme }));
    // Persist to preferences storage
    storePreferences({ theme });
  }, [setTheme]);

  const updateNotification = useCallback((key: keyof UserPreferences['notifications'], value: boolean) => {
    setPreferences(prev => {
      const newPrefs = {
        ...prev,
        notifications: { ...prev.notifications, [key]: value },
      };
      storePreferences({ notifications: newPrefs.notifications });
      return newPrefs;
    });
  }, []);

  const updateDisplay = useCallback((key: keyof UserPreferences['display'], value: boolean) => {
    setPreferences(prev => {
      const newPrefs = {
        ...prev,
        display: { ...prev.display, [key]: value },
      };
      storePreferences({ display: newPrefs.display });
      return newPrefs;
    });
  }, []);

  const generateApiKey = useCallback(async (name: string): Promise<string> => {
    const key = `ara_${Array.from({ length: 32 }, () =>
      'abcdefghijklmnopqrstuvwxyz0123456789'[Math.floor(Math.random() * 36)]
    ).join('')}`;

    const newKey = {
      name,
      prefix: key.slice(0, 8) + '...',
      created_at: new Date().toISOString(),
      last_used: null,
    };

    setPreferences(prev => {
      const newPrefs = {
        ...prev,
        api_keys: [...prev.api_keys, newKey],
      };
      storePreferences({ api_keys: newPrefs.api_keys });
      return newPrefs;
    });

    return key;
  }, []);

  const revokeApiKey = useCallback((prefix: string) => {
    setPreferences(prev => {
      const newPrefs = {
        ...prev,
        api_keys: prev.api_keys.filter(k => k.prefix !== prefix),
      };
      storePreferences({ api_keys: newPrefs.api_keys });
      return newPrefs;
    });
  }, []);

  const saveIntegrations = useCallback(() => {
    setSaveStatus('saving');

    const integrations = {
      slack_webhook: slackWebhook || null,
      discord_webhook: discordWebhook || null,
    };

    setPreferences(prev => ({ ...prev, integrations }));
    storePreferences({ integrations });

    setTimeout(() => {
      setSaveStatus('saved');
      setTimeout(() => setSaveStatus('idle'), 2000);
    }, 500);
  }, [slackWebhook, discordWebhook]);

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="relative">
        <div className="absolute left-0 top-0 bottom-0 w-4 bg-gradient-to-r from-bg to-transparent pointer-events-none z-10 md:hidden" />
        <div className="absolute right-0 top-0 bottom-0 w-4 bg-gradient-to-l from-bg to-transparent pointer-events-none z-10 md:hidden" />

        <div
          className="flex gap-1 md:gap-2 border-b border-acid-green/20 pb-2 overflow-x-auto scrollbar-hide snap-x snap-mandatory"
          role="tablist"
          aria-label="Settings sections"
        >
          {TABS.map((tab) => (
            <button
              key={tab.id}
              id={`tab-${tab.id}`}
              onClick={() => setActiveTab(tab.id)}
              className={`px-2 md:px-4 py-2 font-mono text-xs md:text-sm whitespace-nowrap transition-colors snap-start ${
                activeTab === tab.id
                  ? 'text-acid-green border-b-2 border-acid-green'
                  : 'text-text-muted hover:text-text'
              }`}
              aria-selected={activeTab === tab.id}
              aria-controls={`panel-${tab.id}`}
              role="tab"
            >
              {tab.label}
            </button>
          ))}
          {featureSaveStatus !== 'idle' && (
            <span className={`ml-2 text-xs font-mono self-center ${
              featureSaveStatus === 'saving' ? 'text-acid-cyan' :
              featureSaveStatus === 'saved' ? 'text-acid-green' :
              'text-acid-red'
            }`}>
              {featureSaveStatus === 'saving' ? '...' :
               featureSaveStatus === 'saved' ? '\u2713' : '\u2717'}
            </span>
          )}
        </div>
      </div>

      {/* Tab Panels */}
      {activeTab === 'features' && (
        <FeaturesTab
          featureConfig={featureConfig}
          featureLoading={featureLoading}
          updateFeatureConfig={updateFeatureConfig}
        />
      )}

      {activeTab === 'debate' && (
        <DebateTab
          featureConfig={featureConfig}
          updateFeatureConfig={updateFeatureConfig}
        />
      )}

      {activeTab === 'appearance' && (
        <AppearanceTab
          preferences={preferences}
          updateTheme={updateThemePreference}
          updateDisplay={updateDisplay}
        />
      )}

      {activeTab === 'notifications' && (
        <NotificationsTab
          preferences={preferences}
          updateNotification={updateNotification}
        />
      )}

      {activeTab === 'api' && (
        <ApiKeysTab
          preferences={preferences}
          onGenerateKey={generateApiKey}
          onRevokeKey={revokeApiKey}
        />
      )}

      {activeTab === 'integrations' && (
        <IntegrationsTab
          backendApi={backendConfig.api}
          slackWebhook={slackWebhook}
          discordWebhook={discordWebhook}
          onSlackWebhookChange={setSlackWebhook}
          onDiscordWebhookChange={setDiscordWebhook}
          onSave={saveIntegrations}
          saveStatus={saveStatus}
        />
      )}

      {activeTab === 'account' && (
        <AccountTab
          user={user}
          isAuthenticated={isAuthenticated}
          backendApi={backendConfig.api}
        />
      )}
    </div>
  );
}

// Re-export types and sub-components
export type { FeatureConfig, UserPreferences, SettingsTab } from './types';
export { ToggleSwitch } from './ToggleSwitch';
export { FeaturesTab } from './FeaturesTab';
export { DebateTab } from './DebateTab';
export { AppearanceTab } from './AppearanceTab';
export { NotificationsTab } from './NotificationsTab';
export { ApiKeysTab } from './ApiKeysTab';
export { IntegrationsTab } from './IntegrationsTab';
export { AccountTab } from './AccountTab';

export default SettingsPanel;
