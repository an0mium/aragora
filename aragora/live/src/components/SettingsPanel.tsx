'use client';

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';

interface UserPreferences {
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

const PREFERENCES_KEY = 'aragora_preferences';

function getStoredPreferences(): Partial<UserPreferences> {
  if (typeof window === 'undefined') return {};
  const stored = localStorage.getItem(PREFERENCES_KEY);
  if (!stored) return {};
  try {
    return JSON.parse(stored);
  } catch {
    return {};
  }
}

function storePreferences(prefs: Partial<UserPreferences>): void {
  const current = getStoredPreferences();
  const merged = { ...current, ...prefs };
  localStorage.setItem(PREFERENCES_KEY, JSON.stringify(merged));
}

export function SettingsPanel() {
  const { user, isAuthenticated } = useAuth();
  const [activeTab, setActiveTab] = useState<'appearance' | 'notifications' | 'api' | 'integrations' | 'account'>('appearance');
  const [preferences, setPreferences] = useState<UserPreferences>({
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
  });
  const [newKeyName, setNewKeyName] = useState('');
  const [generatedKey, setGeneratedKey] = useState<string | null>(null);
  const [slackWebhook, setSlackWebhook] = useState('');
  const [discordWebhook, setDiscordWebhook] = useState('');
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  // Load preferences
  useEffect(() => {
    const stored = getStoredPreferences();
    const theme = localStorage.getItem('aragora-theme') as 'dark' | 'light' | null;

    setPreferences(prev => ({
      ...prev,
      ...stored,
      theme: theme || stored.theme || 'dark',
    }));

    if (stored.integrations?.slack_webhook) {
      setSlackWebhook(stored.integrations.slack_webhook);
    }
    if (stored.integrations?.discord_webhook) {
      setDiscordWebhook(stored.integrations.discord_webhook);
    }
  }, []);

  const updateTheme = useCallback((theme: 'dark' | 'light' | 'system') => {
    setPreferences(prev => ({ ...prev, theme }));

    let effectiveTheme: 'dark' | 'light' = 'dark';
    if (theme === 'system') {
      effectiveTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      localStorage.removeItem('aragora-theme');
    } else {
      effectiveTheme = theme;
      localStorage.setItem('aragora-theme', theme);
    }

    if (effectiveTheme === 'light') {
      document.body.setAttribute('data-theme', 'light');
    } else {
      document.body.removeAttribute('data-theme');
    }

    storePreferences({ theme });
  }, []);

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

  const generateApiKey = useCallback(async () => {
    if (!newKeyName.trim()) return;

    // Simulate API key generation (would be backend call in production)
    const key = `ara_${Array.from({ length: 32 }, () =>
      'abcdefghijklmnopqrstuvwxyz0123456789'[Math.floor(Math.random() * 36)]
    ).join('')}`;

    const newKey = {
      name: newKeyName,
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

    setGeneratedKey(key);
    setNewKeyName('');
  }, [newKeyName]);

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

  const tabs = [
    { id: 'appearance', label: 'APPEARANCE' },
    { id: 'notifications', label: 'NOTIFICATIONS' },
    { id: 'api', label: 'API KEYS' },
    { id: 'integrations', label: 'INTEGRATIONS' },
    { id: 'account', label: 'ACCOUNT' },
  ] as const;

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-acid-green/20 pb-2 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 font-mono text-sm whitespace-nowrap transition-colors ${
              activeTab === tab.id
                ? 'text-acid-green border-b-2 border-acid-green'
                : 'text-text-muted hover:text-text'
            }`}
            aria-selected={activeTab === tab.id}
            role="tab"
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Appearance Tab */}
      {activeTab === 'appearance' && (
        <div className="space-y-6">
          <div className="card p-6">
            <h3 className="font-mono text-acid-green mb-4">Theme</h3>
            <div className="space-y-3">
              {(['dark', 'light', 'system'] as const).map((theme) => (
                <label
                  key={theme}
                  className={`flex items-center gap-3 p-3 rounded border cursor-pointer transition-colors ${
                    preferences.theme === theme
                      ? 'border-acid-green bg-acid-green/10'
                      : 'border-acid-green/30 hover:border-acid-green/60'
                  }`}
                >
                  <input
                    type="radio"
                    name="theme"
                    value={theme}
                    checked={preferences.theme === theme}
                    onChange={() => updateTheme(theme)}
                    className="sr-only"
                  />
                  <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                    preferences.theme === theme
                      ? 'border-acid-green'
                      : 'border-text-muted'
                  }`}>
                    {preferences.theme === theme && (
                      <div className="w-2 h-2 rounded-full bg-acid-green" />
                    )}
                  </div>
                  <div>
                    <div className="font-mono text-sm text-text capitalize">{theme}</div>
                    <div className="font-mono text-xs text-text-muted">
                      {theme === 'dark' && 'Default dark theme with acid green accents'}
                      {theme === 'light' && 'Light theme for bright environments'}
                      {theme === 'system' && 'Match your system preference'}
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className="card p-6">
            <h3 className="font-mono text-acid-green mb-4">Display Options</h3>
            <div className="space-y-4">
              <label className="flex items-center justify-between cursor-pointer">
                <div>
                  <div className="font-mono text-sm text-text">Compact Mode</div>
                  <div className="font-mono text-xs text-text-muted">Reduce spacing in lists and panels</div>
                </div>
                <button
                  role="switch"
                  aria-checked={preferences.display.compact_mode}
                  onClick={() => updateDisplay('compact_mode', !preferences.display.compact_mode)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    preferences.display.compact_mode ? 'bg-acid-green' : 'bg-surface'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                    preferences.display.compact_mode ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </label>

              <label className="flex items-center justify-between cursor-pointer">
                <div>
                  <div className="font-mono text-sm text-text">Show Agent Icons</div>
                  <div className="font-mono text-xs text-text-muted">Display model icons next to agent names</div>
                </div>
                <button
                  role="switch"
                  aria-checked={preferences.display.show_agent_icons}
                  onClick={() => updateDisplay('show_agent_icons', !preferences.display.show_agent_icons)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    preferences.display.show_agent_icons ? 'bg-acid-green' : 'bg-surface'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                    preferences.display.show_agent_icons ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </label>

              <label className="flex items-center justify-between cursor-pointer">
                <div>
                  <div className="font-mono text-sm text-text">Auto-scroll Messages</div>
                  <div className="font-mono text-xs text-text-muted">Automatically scroll to new messages in debates</div>
                </div>
                <button
                  role="switch"
                  aria-checked={preferences.display.auto_scroll_messages}
                  onClick={() => updateDisplay('auto_scroll_messages', !preferences.display.auto_scroll_messages)}
                  className={`w-12 h-6 rounded-full transition-colors ${
                    preferences.display.auto_scroll_messages ? 'bg-acid-green' : 'bg-surface'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                    preferences.display.auto_scroll_messages ? 'translate-x-6' : 'translate-x-0.5'
                  }`} />
                </button>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Notifications Tab */}
      {activeTab === 'notifications' && (
        <div className="card p-6">
          <h3 className="font-mono text-acid-green mb-4">Email Notifications</h3>
          <div className="space-y-4">
            <label className="flex items-center justify-between cursor-pointer">
              <div>
                <div className="font-mono text-sm text-text">Debate Completed</div>
                <div className="font-mono text-xs text-text-muted">Notify when a debate finishes</div>
              </div>
              <button
                role="switch"
                aria-checked={preferences.notifications.debate_completed}
                onClick={() => updateNotification('debate_completed', !preferences.notifications.debate_completed)}
                className={`w-12 h-6 rounded-full transition-colors ${
                  preferences.notifications.debate_completed ? 'bg-acid-green' : 'bg-surface'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                  preferences.notifications.debate_completed ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </label>

            <label className="flex items-center justify-between cursor-pointer">
              <div>
                <div className="font-mono text-sm text-text">Daily Digest</div>
                <div className="font-mono text-xs text-text-muted">Summary of your debate activity</div>
              </div>
              <button
                role="switch"
                aria-checked={preferences.notifications.email_digest}
                onClick={() => updateNotification('email_digest', !preferences.notifications.email_digest)}
                className={`w-12 h-6 rounded-full transition-colors ${
                  preferences.notifications.email_digest ? 'bg-acid-green' : 'bg-surface'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                  preferences.notifications.email_digest ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </label>

            <label className="flex items-center justify-between cursor-pointer">
              <div>
                <div className="font-mono text-sm text-text">Weekly Summary</div>
                <div className="font-mono text-xs text-text-muted">Weekly insights and trends</div>
              </div>
              <button
                role="switch"
                aria-checked={preferences.notifications.weekly_summary}
                onClick={() => updateNotification('weekly_summary', !preferences.notifications.weekly_summary)}
                className={`w-12 h-6 rounded-full transition-colors ${
                  preferences.notifications.weekly_summary ? 'bg-acid-green' : 'bg-surface'
                }`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                  preferences.notifications.weekly_summary ? 'translate-x-6' : 'translate-x-0.5'
                }`} />
              </button>
            </label>
          </div>
        </div>
      )}

      {/* API Keys Tab */}
      {activeTab === 'api' && (
        <div className="space-y-6">
          <div className="card p-6">
            <h3 className="font-mono text-acid-green mb-4">Generate API Key</h3>
            <div className="flex gap-3">
              <input
                type="text"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                placeholder="Key name (e.g., CI/CD, Development)"
                className="flex-1 bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
                aria-label="API key name"
              />
              <button
                onClick={generateApiKey}
                disabled={!newKeyName.trim()}
                className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Generate
              </button>
            </div>

            {generatedKey && (
              <div className="mt-4 p-4 bg-acid-yellow/10 border border-acid-yellow/30 rounded">
                <div className="font-mono text-xs text-acid-yellow mb-2">
                  Copy this key now - it won&apos;t be shown again!
                </div>
                <div className="flex gap-2">
                  <code className="flex-1 bg-surface p-2 rounded font-mono text-sm text-text break-all">
                    {generatedKey}
                  </code>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(generatedKey);
                      setGeneratedKey(null);
                    }}
                    className="px-3 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30"
                  >
                    Copy
                  </button>
                </div>
              </div>
            )}
          </div>

          <div className="card p-6">
            <h3 className="font-mono text-acid-green mb-4">
              Your API Keys ({preferences.api_keys.length})
            </h3>
            {preferences.api_keys.length === 0 ? (
              <p className="font-mono text-sm text-text-muted">
                No API keys generated yet. Create one to access the Aragora API programmatically.
              </p>
            ) : (
              <div className="space-y-3">
                {preferences.api_keys.map((key) => (
                  <div
                    key={key.prefix}
                    className="flex items-center justify-between p-3 bg-surface rounded border border-acid-green/20"
                  >
                    <div>
                      <div className="font-mono text-sm text-text">{key.name}</div>
                      <div className="font-mono text-xs text-text-muted">
                        {key.prefix} &middot; Created {new Date(key.created_at).toLocaleDateString()}
                      </div>
                    </div>
                    <button
                      onClick={() => revokeApiKey(key.prefix)}
                      className="px-3 py-1 text-acid-red font-mono text-xs hover:bg-acid-red/10 rounded transition-colors"
                    >
                      Revoke
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="card p-6">
            <h3 className="font-mono text-acid-green mb-2">API Documentation</h3>
            <p className="font-mono text-sm text-text-muted mb-4">
              Use your API key to authenticate requests to the Aragora API.
            </p>
            <pre className="bg-surface p-4 rounded font-mono text-xs text-text overflow-x-auto">
{`curl -X POST https://api.aragora.ai/api/v1/debates \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"task": "Evaluate rate limiting strategies"}'`}
            </pre>
          </div>
        </div>
      )}

      {/* Integrations Tab */}
      {activeTab === 'integrations' && (
        <div className="space-y-6">
          <div className="card p-6">
            <h3 className="font-mono text-acid-green mb-4">Slack Integration</h3>
            <p className="font-mono text-xs text-text-muted mb-4">
              Receive debate notifications in your Slack workspace.
            </p>
            <input
              type="url"
              value={slackWebhook}
              onChange={(e) => setSlackWebhook(e.target.value)}
              placeholder="https://hooks.slack.com/services/..."
              className="w-full bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
              aria-label="Slack webhook URL"
            />
          </div>

          <div className="card p-6">
            <h3 className="font-mono text-acid-green mb-4">Discord Integration</h3>
            <p className="font-mono text-xs text-text-muted mb-4">
              Post debate results to your Discord server.
            </p>
            <input
              type="url"
              value={discordWebhook}
              onChange={(e) => setDiscordWebhook(e.target.value)}
              placeholder="https://discord.com/api/webhooks/..."
              className="w-full bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
              aria-label="Discord webhook URL"
            />
          </div>

          <button
            onClick={saveIntegrations}
            disabled={saveStatus === 'saving'}
            className="px-6 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50"
          >
            {saveStatus === 'saving' ? 'Saving...' : saveStatus === 'saved' ? 'Saved!' : 'Save Integrations'}
          </button>
        </div>
      )}

      {/* Account Tab */}
      {activeTab === 'account' && (
        <div className="space-y-6">
          {isAuthenticated && user ? (
            <>
              <div className="card p-6">
                <h3 className="font-mono text-acid-green mb-4">Account Information</h3>
                <div className="space-y-4">
                  <div>
                    <label className="font-mono text-xs text-text-muted">Email</label>
                    <div className="font-mono text-sm text-text">{user.email}</div>
                  </div>
                  <div>
                    <label className="font-mono text-xs text-text-muted">Name</label>
                    <div className="font-mono text-sm text-text">{user.name || 'Not set'}</div>
                  </div>
                  <div>
                    <label className="font-mono text-xs text-text-muted">Role</label>
                    <div className="font-mono text-sm text-text capitalize">{user.role}</div>
                  </div>
                  <div>
                    <label className="font-mono text-xs text-text-muted">Member Since</label>
                    <div className="font-mono text-sm text-text">
                      {new Date(user.created_at).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              </div>

              <div className="card p-6 border-acid-red/30">
                <h3 className="font-mono text-acid-red mb-4">Danger Zone</h3>
                <p className="font-mono text-xs text-text-muted mb-4">
                  These actions are irreversible. Please proceed with caution.
                </p>
                <div className="space-y-3">
                  <button
                    className="w-full px-4 py-2 border border-acid-yellow/40 text-acid-yellow font-mono text-sm rounded hover:bg-acid-yellow/10 transition-colors text-left"
                  >
                    Export All Data
                  </button>
                  <button
                    className="w-full px-4 py-2 border border-acid-red/40 text-acid-red font-mono text-sm rounded hover:bg-acid-red/10 transition-colors text-left"
                  >
                    Delete Account
                  </button>
                </div>
              </div>
            </>
          ) : (
            <div className="card p-6 text-center">
              <h3 className="font-mono text-acid-green mb-4">Not Signed In</h3>
              <p className="font-mono text-sm text-text-muted mb-4">
                Sign in to manage your account settings and access personalized features.
              </p>
              <a
                href="/auth/login"
                className="inline-block px-6 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors"
              >
                Sign In
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
