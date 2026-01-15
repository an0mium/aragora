'use client';

import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';
import { useBackend } from '@/components/BackendSelector';
import { API_BASE_URL } from '@/config';

interface FeatureConfig {
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

const DEFAULT_FEATURE_CONFIG: FeatureConfig = {
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
  const { config: backendConfig } = useBackend();
  const [activeTab, setActiveTab] = useState<'features' | 'debate' | 'appearance' | 'notifications' | 'api' | 'integrations' | 'account'>('features');
  const [featureConfig, setFeatureConfig] = useState<FeatureConfig>(DEFAULT_FEATURE_CONFIG);
  const [featureLoading, setFeatureLoading] = useState(true);
  const [featureSaveStatus, setFeatureSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
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
  const [slackNotifications, setSlackNotifications] = useState({
    notify_on_consensus: true,
    notify_on_debate_end: true,
    notify_on_error: true,
    notify_on_leaderboard: false,
  });
  const [slackTestStatus, setSlackTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [logoutAllStatus, setLogoutAllStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  // OAuth/Connected Accounts state
  const [oauthProviders, setOauthProviders] = useState<{ id: string; name: string; enabled: boolean; auth_url: string }[]>([]);
  const [linkedProviders, setLinkedProviders] = useState<{ provider: string; email: string | null; linked_at: string }[]>([]);
  const [oauthLoading, setOauthLoading] = useState(true);
  const [oauthLinkStatus, setOauthLinkStatus] = useState<Record<string, 'idle' | 'linking' | 'unlinking' | 'error'>>({});

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

  // Logout from all devices (revoke all tokens)
  const handleLogoutAllDevices = useCallback(async () => {
    if (logoutAllStatus === 'loading') return;

    const confirmed = window.confirm(
      'This will log you out from all devices and sessions. You will need to sign in again. Continue?'
    );
    if (!confirmed) return;

    setLogoutAllStatus('loading');
    try {
      const response = await fetch(`${backendConfig.api}/api/auth/logout-all`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
      });

      if (response.ok) {
        setLogoutAllStatus('success');
        // Redirect to login after short delay
        setTimeout(() => {
          window.location.href = '/auth/login?reason=logout_all';
        }, 1500);
      } else {
        const data = await response.json().catch(() => ({}));
        console.error('Logout all failed:', data);
        setLogoutAllStatus('error');
        setTimeout(() => setLogoutAllStatus('idle'), 3000);
      }
    } catch (error) {
      console.error('Logout all error:', error);
      setLogoutAllStatus('error');
      setTimeout(() => setLogoutAllStatus('idle'), 3000);
    }
  }, [backendConfig.api, logoutAllStatus]);

  // Fetch OAuth providers and user's linked accounts
  useEffect(() => {
    if (!isAuthenticated) {
      setOauthLoading(false);
      return;
    }

    async function fetchOAuthData() {
      setOauthLoading(true);
      try {
        // Fetch available providers
        const providersRes = await fetch(`${backendConfig.api}/api/auth/oauth/providers`);
        if (providersRes.ok) {
          const data = await providersRes.json();
          setOauthProviders(data.providers || []);
        }

        // Fetch user's linked providers
        const linkedRes = await fetch(`${backendConfig.api}/api/user/oauth-providers`, {
          credentials: 'include',
        });
        if (linkedRes.ok) {
          const data = await linkedRes.json();
          setLinkedProviders(data.providers || []);
        }
      } catch (error) {
        console.warn('Failed to fetch OAuth data:', error);
      } finally {
        setOauthLoading(false);
      }
    }

    fetchOAuthData();
  }, [isAuthenticated, backendConfig.api]);

  // Handle OAuth link - redirect to provider's auth URL
  const handleOAuthLink = useCallback(async (providerId: string) => {
    setOauthLinkStatus(prev => ({ ...prev, [providerId]: 'linking' }));
    try {
      const response = await fetch(`${backendConfig.api}/api/auth/oauth/link`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: providerId,
          redirect_url: `${window.location.origin}/settings?linked=${providerId}`,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.auth_url) {
          // Redirect to OAuth provider
          window.location.href = data.auth_url;
        }
      } else {
        setOauthLinkStatus(prev => ({ ...prev, [providerId]: 'error' }));
        setTimeout(() => setOauthLinkStatus(prev => ({ ...prev, [providerId]: 'idle' })), 3000);
      }
    } catch (error) {
      console.error('OAuth link error:', error);
      setOauthLinkStatus(prev => ({ ...prev, [providerId]: 'error' }));
      setTimeout(() => setOauthLinkStatus(prev => ({ ...prev, [providerId]: 'idle' })), 3000);
    }
  }, [backendConfig.api]);

  // Handle OAuth unlink
  const handleOAuthUnlink = useCallback(async (providerId: string) => {
    const confirmed = window.confirm(
      `Unlink your ${providerId} account? You can still sign in with email/password.`
    );
    if (!confirmed) return;

    setOauthLinkStatus(prev => ({ ...prev, [providerId]: 'unlinking' }));
    try {
      const response = await fetch(`${backendConfig.api}/api/auth/oauth/unlink`, {
        method: 'DELETE',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: providerId }),
      });

      if (response.ok) {
        // Remove from linked providers
        setLinkedProviders(prev => prev.filter(p => p.provider !== providerId));
        setOauthLinkStatus(prev => ({ ...prev, [providerId]: 'idle' }));
      } else {
        const data = await response.json().catch(() => ({}));
        alert(data.error || 'Failed to unlink account');
        setOauthLinkStatus(prev => ({ ...prev, [providerId]: 'error' }));
        setTimeout(() => setOauthLinkStatus(prev => ({ ...prev, [providerId]: 'idle' })), 3000);
      }
    } catch (error) {
      console.error('OAuth unlink error:', error);
      setOauthLinkStatus(prev => ({ ...prev, [providerId]: 'error' }));
      setTimeout(() => setOauthLinkStatus(prev => ({ ...prev, [providerId]: 'idle' })), 3000);
    }
  }, [backendConfig.api]);

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

    // Save to backend
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

  const tabs = [
    { id: 'features', label: 'FEATURES' },
    { id: 'debate', label: 'DEBATE' },
    { id: 'appearance', label: 'APPEARANCE' },
    { id: 'notifications', label: 'NOTIFICATIONS' },
    { id: 'api', label: 'API KEYS' },
    { id: 'integrations', label: 'INTEGRATIONS' },
    { id: 'account', label: 'ACCOUNT' },
  ] as const;

  return (
    <div className="space-y-6">
      {/* Tab Navigation - Mobile-optimized with horizontal scroll */}
      <div className="relative">
        {/* Scroll shadow indicators */}
        <div className="absolute left-0 top-0 bottom-0 w-4 bg-gradient-to-r from-bg to-transparent pointer-events-none z-10 md:hidden" />
        <div className="absolute right-0 top-0 bottom-0 w-4 bg-gradient-to-l from-bg to-transparent pointer-events-none z-10 md:hidden" />

        <div
          className="flex gap-1 md:gap-2 border-b border-acid-green/20 pb-2 overflow-x-auto scrollbar-hide snap-x snap-mandatory"
          role="tablist"
          aria-label="Settings sections"
        >
          {tabs.map((tab) => (
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
          {/* Save Status Indicator */}
          {featureSaveStatus !== 'idle' && (
            <span className={`ml-2 text-xs font-mono self-center ${
              featureSaveStatus === 'saving' ? 'text-acid-cyan' :
              featureSaveStatus === 'saved' ? 'text-acid-green' :
              'text-acid-red'
            }`}>
              {featureSaveStatus === 'saving' ? '...' :
               featureSaveStatus === 'saved' ? '✓' : '✗'}
            </span>
          )}
        </div>
      </div>

      {/* Features Tab */}
      {activeTab === 'features' && (
        <div className="space-y-6" role="tabpanel" id="panel-features" aria-labelledby="tab-features">
          {featureLoading ? (
            <div className="card p-6 animate-pulse">
              <div className="h-32 bg-surface rounded" />
            </div>
          ) : (
            <>
              <div className="card p-6">
                <h3 className="font-mono text-acid-green mb-4">Analysis Features</h3>
                <div className="space-y-4">
                  <label className="flex items-center justify-between cursor-pointer">
                    <div>
                      <div className="font-mono text-sm text-text">Calibration Tracking</div>
                      <div className="font-mono text-xs text-text-muted">Track agent prediction accuracy over time</div>
                    </div>
                    <button
                      role="switch"
                      aria-checked={featureConfig.calibration}
                      onClick={() => updateFeatureConfig('calibration', !featureConfig.calibration)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        featureConfig.calibration ? 'bg-acid-green' : 'bg-surface'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        featureConfig.calibration ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </label>

                  <label className="flex items-center justify-between cursor-pointer">
                    <div>
                      <div className="font-mono text-sm text-text">Trickster (Hollow Consensus)</div>
                      <div className="font-mono text-xs text-text-muted">Detect and challenge artificial agreement</div>
                    </div>
                    <button
                      role="switch"
                      aria-checked={featureConfig.trickster}
                      onClick={() => updateFeatureConfig('trickster', !featureConfig.trickster)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        featureConfig.trickster ? 'bg-acid-green' : 'bg-surface'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        featureConfig.trickster ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </label>

                  <label className="flex items-center justify-between cursor-pointer">
                    <div>
                      <div className="font-mono text-sm text-text">Rhetorical Observer</div>
                      <div className="font-mono text-xs text-text-muted">Detect rhetorical patterns like concession and rebuttal</div>
                    </div>
                    <button
                      role="switch"
                      aria-checked={featureConfig.rhetorical}
                      onClick={() => updateFeatureConfig('rhetorical', !featureConfig.rhetorical)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        featureConfig.rhetorical ? 'bg-acid-green' : 'bg-surface'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        featureConfig.rhetorical ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </label>

                  <label className="flex items-center justify-between cursor-pointer">
                    <div>
                      <div className="font-mono text-sm text-text">Crux Analysis</div>
                      <div className="font-mono text-xs text-text-muted">Identify key points of disagreement</div>
                    </div>
                    <button
                      role="switch"
                      aria-checked={featureConfig.crux}
                      onClick={() => updateFeatureConfig('crux', !featureConfig.crux)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        featureConfig.crux ? 'bg-acid-green' : 'bg-surface'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        featureConfig.crux ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </label>
                </div>
              </div>

              <div className="card p-6">
                <h3 className="font-mono text-acid-green mb-4">Learning & Memory</h3>
                <div className="space-y-4">
                  <label className="flex items-center justify-between cursor-pointer">
                    <div>
                      <div className="font-mono text-sm text-text">Continuum Memory</div>
                      <div className="font-mono text-xs text-text-muted">Multi-tier memory with surprise-based consolidation</div>
                    </div>
                    <button
                      role="switch"
                      aria-checked={featureConfig.continuum_memory}
                      onClick={() => updateFeatureConfig('continuum_memory', !featureConfig.continuum_memory)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        featureConfig.continuum_memory ? 'bg-acid-green' : 'bg-surface'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        featureConfig.continuum_memory ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </label>

                  <label className="flex items-center justify-between cursor-pointer">
                    <div>
                      <div className="font-mono text-sm text-text">Consensus Memory</div>
                      <div className="font-mono text-xs text-text-muted">Store historical debate outcomes</div>
                    </div>
                    <button
                      role="switch"
                      aria-checked={featureConfig.consensus_memory}
                      onClick={() => updateFeatureConfig('consensus_memory', !featureConfig.consensus_memory)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        featureConfig.consensus_memory ? 'bg-acid-green' : 'bg-surface'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        featureConfig.consensus_memory ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </label>

                  <label className="flex items-center justify-between cursor-pointer">
                    <div>
                      <div className="font-mono text-sm text-text">Prompt Evolution</div>
                      <div className="font-mono text-xs text-text-muted">Learn from debates to improve agent prompts</div>
                    </div>
                    <button
                      role="switch"
                      aria-checked={featureConfig.evolution}
                      onClick={() => updateFeatureConfig('evolution', !featureConfig.evolution)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        featureConfig.evolution ? 'bg-acid-green' : 'bg-surface'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        featureConfig.evolution ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </label>
                </div>
              </div>

              <div className="card p-6">
                <h3 className="font-mono text-acid-green mb-4">Panels & UI</h3>
                <div className="space-y-4">
                  <label className="flex items-center justify-between cursor-pointer">
                    <div>
                      <div className="font-mono text-sm text-text">Insights Panel</div>
                      <div className="font-mono text-xs text-text-muted">Show extracted learnings and patterns</div>
                    </div>
                    <button
                      role="switch"
                      aria-checked={featureConfig.insights}
                      onClick={() => updateFeatureConfig('insights', !featureConfig.insights)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        featureConfig.insights ? 'bg-acid-green' : 'bg-surface'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        featureConfig.insights ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </label>

                  <label className="flex items-center justify-between cursor-pointer">
                    <div>
                      <div className="font-mono text-sm text-text">Moments Timeline</div>
                      <div className="font-mono text-xs text-text-muted">Detect significant narrative moments</div>
                    </div>
                    <button
                      role="switch"
                      aria-checked={featureConfig.moments}
                      onClick={() => updateFeatureConfig('moments', !featureConfig.moments)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        featureConfig.moments ? 'bg-acid-green' : 'bg-surface'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        featureConfig.moments ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </label>

                  <label className="flex items-center justify-between cursor-pointer">
                    <div>
                      <div className="font-mono text-sm text-text">Persona Laboratory</div>
                      <div className="font-mono text-xs text-text-muted">Agent personality trait detection</div>
                    </div>
                    <button
                      role="switch"
                      aria-checked={featureConfig.laboratory}
                      onClick={() => updateFeatureConfig('laboratory', !featureConfig.laboratory)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        featureConfig.laboratory ? 'bg-acid-green' : 'bg-surface'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        featureConfig.laboratory ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </label>

                  <label className="flex items-center justify-between cursor-pointer">
                    <div>
                      <div className="font-mono text-sm text-text">Show Advanced Metrics</div>
                      <div className="font-mono text-xs text-text-muted">Display detailed telemetry in panels</div>
                    </div>
                    <button
                      role="switch"
                      aria-checked={featureConfig.show_advanced_metrics}
                      onClick={() => updateFeatureConfig('show_advanced_metrics', !featureConfig.show_advanced_metrics)}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        featureConfig.show_advanced_metrics ? 'bg-acid-green' : 'bg-surface'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                        featureConfig.show_advanced_metrics ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </label>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Debate Tab */}
      {activeTab === 'debate' && (
        <div className="space-y-6" role="tabpanel" id="panel-debate" aria-labelledby="tab-debate">
          <div className="card p-6">
            <h3 className="font-mono text-acid-green mb-4">Default Debate Settings</h3>
            <div className="space-y-4">
              <div>
                <label htmlFor="default-mode-select" className="font-mono text-sm text-text block mb-2">Default Mode</label>
                <select
                  id="default-mode-select"
                  value={featureConfig.default_mode}
                  onChange={(e) => updateFeatureConfig('default_mode', e.target.value)}
                  className="w-full bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
                >
                  <option value="standard">Standard</option>
                  <option value="graph">Graph</option>
                  <option value="matrix">Matrix</option>
                </select>
              </div>

              <div>
                <label htmlFor="default-rounds-input" className="font-mono text-sm text-text block mb-2">Default Rounds</label>
                <input
                  id="default-rounds-input"
                  type="number"
                  min={1}
                  max={10}
                  value={featureConfig.default_rounds}
                  onChange={(e) => updateFeatureConfig('default_rounds', parseInt(e.target.value) || 3)}
                  className="w-full bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
                />
              </div>

              <div>
                <label htmlFor="default-agents-input" className="font-mono text-sm text-text block mb-2">Default Agents</label>
                <input
                  id="default-agents-input"
                  type="text"
                  value={featureConfig.default_agents}
                  onChange={(e) => updateFeatureConfig('default_agents', e.target.value)}
                  placeholder="claude,gemini,gpt4,grok"
                  className="w-full bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
                />
                <p className="font-mono text-xs text-text-muted mt-1">Comma-separated list of agents</p>
              </div>
            </div>
          </div>

          <div className="card p-6">
            <h3 className="font-mono text-acid-green mb-4">Alert Thresholds</h3>
            <div className="space-y-4">
              <div>
                <label htmlFor="consensus-threshold-range" className="font-mono text-sm text-text block mb-2">
                  Consensus Alert Threshold: {(featureConfig.consensus_alert_threshold * 100).toFixed(0)}%
                </label>
                <input
                  id="consensus-threshold-range"
                  type="range"
                  min={0.5}
                  max={1.0}
                  step={0.05}
                  value={featureConfig.consensus_alert_threshold}
                  onChange={(e) => updateFeatureConfig('consensus_alert_threshold', parseFloat(e.target.value))}
                  className="w-full accent-acid-green"
                  aria-label={`Consensus alert threshold: ${(featureConfig.consensus_alert_threshold * 100).toFixed(0)}%`}
                />
                <p className="font-mono text-xs text-text-muted mt-1">
                  Notify when consensus confidence exceeds this threshold
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Appearance Tab */}
      {activeTab === 'appearance' && (
        <div className="space-y-6" role="tabpanel" id="panel-appearance" aria-labelledby="tab-appearance">
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
        <div className="card p-6" role="tabpanel" id="panel-notifications" aria-labelledby="tab-notifications">
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
        <div className="space-y-6" role="tabpanel" id="panel-api" aria-labelledby="tab-api">
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
        <div className="space-y-6" role="tabpanel" id="panel-integrations" aria-labelledby="tab-integrations">
          <div className="card p-6">
            <h3 className="font-mono text-acid-green mb-4">Slack Integration</h3>
            <p className="font-mono text-xs text-text-muted mb-4">
              Receive debate notifications in your Slack workspace.
            </p>
            <div className="space-y-4">
              <div>
                <label className="font-mono text-xs text-text-muted block mb-2">Webhook URL</label>
                <div className="flex gap-2">
                  <input
                    type="url"
                    value={slackWebhook}
                    onChange={(e) => setSlackWebhook(e.target.value)}
                    placeholder="https://hooks.slack.com/services/..."
                    className="flex-1 bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
                    aria-label="Slack webhook URL"
                  />
                  <button
                    onClick={async () => {
                      if (!slackWebhook) return;
                      setSlackTestStatus('testing');
                      try {
                        const response = await fetch(`${backendConfig.api}/api/integrations/slack/test`, {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ webhook_url: slackWebhook }),
                        });
                        setSlackTestStatus(response.ok ? 'success' : 'error');
                      } catch {
                        setSlackTestStatus('error');
                      }
                      setTimeout(() => setSlackTestStatus('idle'), 3000);
                    }}
                    disabled={!slackWebhook || slackTestStatus === 'testing'}
                    className={`px-4 py-2 font-mono text-sm rounded transition-colors disabled:opacity-50 ${
                      slackTestStatus === 'success' ? 'bg-acid-green/20 border border-acid-green/40 text-acid-green' :
                      slackTestStatus === 'error' ? 'bg-acid-red/20 border border-acid-red/40 text-acid-red' :
                      'bg-surface border border-acid-green/30 text-text hover:border-acid-green/50'
                    }`}
                  >
                    {slackTestStatus === 'testing' ? '...' :
                     slackTestStatus === 'success' ? 'Sent!' :
                     slackTestStatus === 'error' ? 'Failed' : 'Test'}
                  </button>
                </div>
              </div>

              {slackWebhook && (
                <div className="pt-4 border-t border-acid-green/20">
                  <h4 className="font-mono text-xs text-acid-cyan mb-3">NOTIFICATION SETTINGS</h4>
                  <div className="space-y-3">
                    <label className="flex items-center justify-between cursor-pointer">
                      <div>
                        <div className="font-mono text-sm text-text">Consensus Reached</div>
                        <div className="font-mono text-xs text-text-muted">Alert when debates reach consensus</div>
                      </div>
                      <button
                        role="switch"
                        aria-checked={slackNotifications.notify_on_consensus}
                        onClick={() => setSlackNotifications(prev => ({ ...prev, notify_on_consensus: !prev.notify_on_consensus }))}
                        className={`w-12 h-6 rounded-full transition-colors ${
                          slackNotifications.notify_on_consensus ? 'bg-acid-green' : 'bg-surface'
                        }`}
                      >
                        <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                          slackNotifications.notify_on_consensus ? 'translate-x-6' : 'translate-x-0.5'
                        }`} />
                      </button>
                    </label>

                    <label className="flex items-center justify-between cursor-pointer">
                      <div>
                        <div className="font-mono text-sm text-text">Debate Completed</div>
                        <div className="font-mono text-xs text-text-muted">Post summaries when debates end</div>
                      </div>
                      <button
                        role="switch"
                        aria-checked={slackNotifications.notify_on_debate_end}
                        onClick={() => setSlackNotifications(prev => ({ ...prev, notify_on_debate_end: !prev.notify_on_debate_end }))}
                        className={`w-12 h-6 rounded-full transition-colors ${
                          slackNotifications.notify_on_debate_end ? 'bg-acid-green' : 'bg-surface'
                        }`}
                      >
                        <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                          slackNotifications.notify_on_debate_end ? 'translate-x-6' : 'translate-x-0.5'
                        }`} />
                      </button>
                    </label>

                    <label className="flex items-center justify-between cursor-pointer">
                      <div>
                        <div className="font-mono text-sm text-text">Error Alerts</div>
                        <div className="font-mono text-xs text-text-muted">Notify on debate errors</div>
                      </div>
                      <button
                        role="switch"
                        aria-checked={slackNotifications.notify_on_error}
                        onClick={() => setSlackNotifications(prev => ({ ...prev, notify_on_error: !prev.notify_on_error }))}
                        className={`w-12 h-6 rounded-full transition-colors ${
                          slackNotifications.notify_on_error ? 'bg-acid-green' : 'bg-surface'
                        }`}
                      >
                        <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                          slackNotifications.notify_on_error ? 'translate-x-6' : 'translate-x-0.5'
                        }`} />
                      </button>
                    </label>

                    <label className="flex items-center justify-between cursor-pointer">
                      <div>
                        <div className="font-mono text-sm text-text">Leaderboard Updates</div>
                        <div className="font-mono text-xs text-text-muted">Post agent ranking changes</div>
                      </div>
                      <button
                        role="switch"
                        aria-checked={slackNotifications.notify_on_leaderboard}
                        onClick={() => setSlackNotifications(prev => ({ ...prev, notify_on_leaderboard: !prev.notify_on_leaderboard }))}
                        className={`w-12 h-6 rounded-full transition-colors ${
                          slackNotifications.notify_on_leaderboard ? 'bg-acid-green' : 'bg-surface'
                        }`}
                      >
                        <div className={`w-5 h-5 rounded-full bg-white transition-transform ${
                          slackNotifications.notify_on_leaderboard ? 'translate-x-6' : 'translate-x-0.5'
                        }`} />
                      </button>
                    </label>
                  </div>
                </div>
              )}
            </div>
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
        <div className="space-y-6" role="tabpanel" id="panel-account" aria-labelledby="tab-account">
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

              {/* Connected Accounts (OAuth) */}
              <div className="card p-6">
                <h3 className="font-mono text-acid-green mb-4">Connected Accounts</h3>
                <p className="font-mono text-xs text-text-muted mb-4">
                  Link your social accounts for quick sign-in. You can always use email/password.
                </p>

                {oauthLoading ? (
                  <div className="animate-pulse space-y-3">
                    <div className="h-12 bg-surface rounded" />
                  </div>
                ) : oauthProviders.length === 0 ? (
                  <p className="font-mono text-xs text-text-muted">
                    No OAuth providers configured.
                  </p>
                ) : (
                  <div className="space-y-3">
                    {oauthProviders.map((provider) => {
                      const linked = linkedProviders.find(p => p.provider === provider.id);
                      const status = oauthLinkStatus[provider.id] || 'idle';

                      return (
                        <div
                          key={provider.id}
                          className={`flex items-center justify-between p-3 rounded border ${
                            linked
                              ? 'border-acid-green/30 bg-acid-green/5'
                              : 'border-acid-green/20 bg-surface/50'
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            {/* Provider icon */}
                            <div className="w-8 h-8 flex items-center justify-center bg-surface rounded">
                              {provider.id === 'google' && (
                                <svg className="w-5 h-5" viewBox="0 0 24 24" aria-hidden="true">
                                  <path
                                    fill="#4285F4"
                                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                                  />
                                  <path
                                    fill="#34A853"
                                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                                  />
                                  <path
                                    fill="#FBBC05"
                                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                                  />
                                  <path
                                    fill="#EA4335"
                                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                                  />
                                </svg>
                              )}
                            </div>

                            {/* Provider info */}
                            <div>
                              <div className="font-mono text-sm text-text">{provider.name}</div>
                              {linked ? (
                                <div className="font-mono text-xs text-acid-green">
                                  Connected {linked.email ? `as ${linked.email}` : ''}
                                </div>
                              ) : (
                                <div className="font-mono text-xs text-text-muted">
                                  Not connected
                                </div>
                              )}
                            </div>
                          </div>

                          {/* Link/Unlink button */}
                          {linked ? (
                            <button
                              onClick={() => handleOAuthUnlink(provider.id)}
                              disabled={status === 'unlinking'}
                              className={`px-3 py-1 font-mono text-xs rounded transition-colors ${
                                status === 'unlinking'
                                  ? 'text-text-muted cursor-wait'
                                  : status === 'error'
                                  ? 'text-acid-red'
                                  : 'text-acid-red/70 hover:text-acid-red hover:bg-acid-red/10'
                              }`}
                            >
                              {status === 'unlinking' ? 'Unlinking...' : status === 'error' ? 'Error' : 'Unlink'}
                            </button>
                          ) : (
                            <button
                              onClick={() => handleOAuthLink(provider.id)}
                              disabled={status === 'linking'}
                              className={`px-3 py-1 font-mono text-xs rounded transition-colors ${
                                status === 'linking'
                                  ? 'text-text-muted cursor-wait'
                                  : status === 'error'
                                  ? 'text-acid-red'
                                  : 'text-acid-cyan hover:text-acid-green hover:bg-acid-green/10'
                              }`}
                            >
                              {status === 'linking' ? 'Linking...' : status === 'error' ? 'Error' : 'Link'}
                            </button>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="card p-6 border-acid-yellow/30">
                <h3 className="font-mono text-acid-yellow mb-4">Security</h3>
                <p className="font-mono text-xs text-text-muted mb-4">
                  Manage your session security and active logins.
                </p>
                <button
                  onClick={handleLogoutAllDevices}
                  disabled={logoutAllStatus === 'loading'}
                  className={`w-full px-4 py-2 border font-mono text-sm rounded transition-colors text-left ${
                    logoutAllStatus === 'success'
                      ? 'border-acid-green/40 text-acid-green bg-acid-green/10'
                      : logoutAllStatus === 'error'
                      ? 'border-acid-red/40 text-acid-red bg-acid-red/10'
                      : 'border-acid-yellow/40 text-acid-yellow hover:bg-acid-yellow/10'
                  } disabled:opacity-50`}
                >
                  {logoutAllStatus === 'loading'
                    ? 'Logging out...'
                    : logoutAllStatus === 'success'
                    ? 'Logged out! Redirecting...'
                    : logoutAllStatus === 'error'
                    ? 'Failed - try again'
                    : 'Logout All Devices'}
                </button>
                <p className="font-mono text-xs text-text-muted mt-2">
                  Invalidates all sessions and tokens. You will be signed out everywhere.
                </p>
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
