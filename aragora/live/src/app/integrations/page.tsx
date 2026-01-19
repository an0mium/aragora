'use client';

import { useState, useCallback, useEffect } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import {
  IntegrationSetupWizard,
  IntegrationStatusDashboard,
  IntegrationType,
  INTEGRATION_CONFIGS,
} from '@/components/integrations';

interface BotStatus {
  platform: string;
  label: string;
  icon: string;
  status: 'online' | 'offline' | 'error' | 'loading';
  endpoint: string;
  features: string[];
  lastPing?: string;
  errorMessage?: string;
}

interface SystemIntegration {
  title: string;
  description: string;
  href: string;
  icon: string;
  status: 'stable' | 'beta' | 'alpha';
  features: string[];
}

const systemIntegrations: SystemIntegration[] = [
  {
    title: 'Webhooks',
    description: 'Receive real-time HTTP callbacks for debate events. Integrate with any system.',
    href: '/webhooks',
    icon: '>>',
    status: 'stable',
    features: ['Lifecycle events', 'Consensus alerts', 'HMAC signatures', 'Retry logic'],
  },
  {
    title: 'Plugin Marketplace',
    description: 'Extend Aragora with custom plugins for selection, scoring, and behaviors.',
    href: '/plugins',
    icon: '+>',
    status: 'stable',
    features: ['Agent scorers', 'Team selectors', 'Custom behaviors'],
  },
  {
    title: 'Training Export',
    description: 'Export debate data for fine-tuning language models.',
    href: '/training',
    icon: '[]',
    status: 'stable',
    features: ['SFT format', 'DPO pairs', 'JSONL export'],
  },
  {
    title: 'Evidence Connectors',
    description: 'Connect external knowledge sources for fact-grounded debates.',
    href: '/evidence',
    icon: '??',
    status: 'beta',
    features: ['Web search', 'Document upload', 'Citations'],
  },
  {
    title: 'API Explorer',
    description: 'Interactive documentation and testing for all API endpoints.',
    href: '/api-explorer',
    icon: '{}',
    status: 'stable',
    features: ['OpenAPI spec', 'Try requests', 'Authentication'],
  },
  {
    title: 'MCP Server',
    description: 'Model Context Protocol server for AI assistant integrations.',
    href: '/developer',
    icon: '<>',
    status: 'beta',
    features: ['Claude integration', '17 MCP tools', 'Context streaming'],
  },
];

// Chat platform integrations
const chatPlatforms: IntegrationType[] = ['slack', 'discord', 'telegram', 'email', 'teams', 'whatsapp', 'matrix'];

// Bot integrations configuration
const BOT_CONFIGS: Omit<BotStatus, 'status' | 'lastPing' | 'errorMessage'>[] = [
  {
    platform: 'slack',
    label: 'Slack',
    icon: '#',
    endpoint: '/api/integrations/slack/status',
    features: ['Slash commands', 'Interactive components', 'Event subscriptions'],
  },
  {
    platform: 'discord',
    label: 'Discord',
    icon: '>',
    endpoint: '/api/bots/discord/status',
    features: ['Bot interactions', 'Slash commands', 'Message components'],
  },
  {
    platform: 'teams',
    label: 'Microsoft Teams',
    icon: 'T',
    endpoint: '/api/bots/teams/status',
    features: ['Bot messages', 'Adaptive cards', 'Activity handler'],
  },
  {
    platform: 'zoom',
    label: 'Zoom',
    icon: 'Z',
    endpoint: '/api/bots/zoom/status',
    features: ['Meeting events', 'Webhooks', 'Chat integration'],
  },
];

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    stable: 'bg-acid-green/20 text-acid-green border-acid-green/30',
    beta: 'bg-acid-cyan/20 text-acid-cyan border-acid-cyan/30',
    alpha: 'bg-warning/20 text-warning border-warning/30',
  };
  return (
    <span className={`px-2 py-0.5 text-xs font-mono rounded border ${colors[status]}`}>
      {status.toUpperCase()}
    </span>
  );
}

function BotStatusBadge({ status }: { status: BotStatus['status'] }) {
  const styles: Record<BotStatus['status'], string> = {
    online: 'bg-acid-green/20 text-acid-green',
    offline: 'bg-text-muted/20 text-text-muted',
    error: 'bg-warning/20 text-warning',
    loading: 'bg-acid-cyan/20 text-acid-cyan animate-pulse',
  };
  return (
    <span className={`px-2 py-0.5 text-xs font-mono rounded ${styles[status]}`}>
      {status.toUpperCase()}
    </span>
  );
}

export default function IntegrationsPage() {
  const { config: backendConfig } = useBackend();
  const [activeTab, setActiveTab] = useState<'notifications' | 'bots' | 'system' | 'docs'>('notifications');
  const [wizardOpen, setWizardOpen] = useState<IntegrationType | null>(null);
  const [editingConfig, setEditingConfig] = useState<Record<string, unknown> | undefined>();
  const [botStatuses, setBotStatuses] = useState<BotStatus[]>(
    BOT_CONFIGS.map(cfg => ({ ...cfg, status: 'loading' as const }))
  );
  const [botsLoading, setBotsLoading] = useState(false);

  // Fetch bot statuses
  const fetchBotStatuses = useCallback(async () => {
    setBotsLoading(true);
    const newStatuses = await Promise.all(
      BOT_CONFIGS.map(async (cfg) => {
        try {
          const res = await fetch(`${backendConfig.api}${cfg.endpoint}`);
          if (res.ok) {
            const data = await res.json();
            return {
              ...cfg,
              status: (data.online || data.status === 'online' || data.configured) ? 'online' : 'offline',
              lastPing: data.last_activity ? new Date(data.last_activity * 1000).toLocaleString() : undefined,
            } as BotStatus;
          }
          return { ...cfg, status: 'offline' as const };
        } catch (err) {
          return {
            ...cfg,
            status: 'error' as const,
            errorMessage: err instanceof Error ? err.message : 'Failed to connect',
          } as BotStatus;
        }
      })
    );
    setBotStatuses(newStatuses);
    setBotsLoading(false);
  }, [backendConfig.api]);

  // Load bot statuses when tab is active
  useEffect(() => {
    if (activeTab === 'bots') {
      fetchBotStatuses();
    }
  }, [activeTab, fetchBotStatuses]);

  const handleConfigure = useCallback((type: IntegrationType) => {
    setEditingConfig(undefined);
    setWizardOpen(type);
  }, []);

  const handleEdit = useCallback((type: IntegrationType, config: Record<string, unknown>) => {
    setEditingConfig(config);
    setWizardOpen(type);
  }, []);

  const handleSaveIntegration = async (config: Record<string, unknown>) => {
    const res = await fetch(`${backendConfig.api}/api/integrations/${config.type}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });

    if (!res.ok) {
      const data = await res.json();
      throw new Error(data.error || 'Failed to save integration');
    }
  };

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-3">
              <Link
                href="/"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <Link
                href="/api-explorer"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [API]
              </Link>
              <Link
                href="/settings"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [SETTINGS]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} INTEGRATIONS
            </h1>
            <p className="text-text-muted font-mono text-sm max-w-2xl">
              Connect Aragora to chat platforms for notifications, export data for ML training,
              and extend functionality with plugins and webhooks.
            </p>
          </div>

          {/* SDK Installation */}
          <div className="mb-6 p-4 border border-acid-cyan/30 bg-surface/30 rounded">
            <h3 className="font-mono text-acid-cyan text-sm mb-3">SDK Installation</h3>
            <div className="flex items-center gap-4">
              <code className="flex-1 bg-bg px-3 py-2 font-mono text-sm text-text border border-acid-green/20 rounded">
                npm install @aragora/sdk
              </code>
              <Link
                href="https://www.npmjs.com/package/@aragora/sdk"
                target="_blank"
                className="px-3 py-2 border border-acid-green/30 text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [NPM]
              </Link>
              <Link
                href="/api-explorer"
                className="px-3 py-2 border border-acid-green/30 text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DOCS]
              </Link>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => setActiveTab('notifications')}
              className={`px-4 py-2 font-mono text-sm border transition-colors ${
                activeTab === 'notifications'
                  ? 'border-acid-green bg-acid-green/10 text-acid-green'
                  : 'border-acid-green/30 text-text-muted hover:text-text'
              }`}
            >
              [NOTIFICATIONS]
            </button>
            <button
              onClick={() => setActiveTab('bots')}
              className={`px-4 py-2 font-mono text-sm border transition-colors ${
                activeTab === 'bots'
                  ? 'border-acid-green bg-acid-green/10 text-acid-green'
                  : 'border-acid-green/30 text-text-muted hover:text-text'
              }`}
            >
              [BOTS]
            </button>
            <button
              onClick={() => setActiveTab('system')}
              className={`px-4 py-2 font-mono text-sm border transition-colors ${
                activeTab === 'system'
                  ? 'border-acid-green bg-acid-green/10 text-acid-green'
                  : 'border-acid-green/30 text-text-muted hover:text-text'
              }`}
            >
              [SYSTEM]
            </button>
            <button
              onClick={() => setActiveTab('docs')}
              className={`px-4 py-2 font-mono text-sm border transition-colors ${
                activeTab === 'docs'
                  ? 'border-acid-green bg-acid-green/10 text-acid-green'
                  : 'border-acid-green/30 text-text-muted hover:text-text'
              }`}
            >
              [DOCUMENTATION]
            </button>
          </div>

          {/* Tab Content */}
          {activeTab === 'notifications' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="font-mono text-text">Chat & Notification Platforms</h2>
                <div className="flex gap-2">
                  {chatPlatforms.slice(0, 3).map(type => (
                    <button
                      key={type}
                      onClick={() => handleConfigure(type)}
                      className="px-3 py-1 text-xs font-mono border border-acid-cyan/30 text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
                    >
                      [+ {INTEGRATION_CONFIGS[type].title.toUpperCase()}]
                    </button>
                  ))}
                </div>
              </div>

              <IntegrationStatusDashboard
                onConfigure={handleConfigure}
                onEdit={handleEdit}
              />
            </div>
          )}

          {activeTab === 'bots' && (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="font-mono text-text">Chat Platform Bots</h2>
                <button
                  onClick={fetchBotStatuses}
                  disabled={botsLoading}
                  className="px-3 py-1 text-xs font-mono border border-acid-green/30 text-text-muted hover:text-acid-green transition-colors disabled:opacity-50"
                >
                  {botsLoading ? '[CHECKING...]' : '[REFRESH STATUS]'}
                </button>
              </div>

              <p className="text-text-muted font-mono text-sm">
                Bots allow Aragora to interact directly with chat platforms, handling commands and events in real-time.
              </p>

              {/* Bot Status Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {botStatuses.map((bot) => (
                  <div
                    key={bot.platform}
                    className={`p-4 border rounded bg-surface/30 ${
                      bot.status === 'online' ? 'border-acid-green/40' :
                      bot.status === 'error' ? 'border-warning/40' :
                      'border-acid-green/20'
                    }`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <span className="w-10 h-10 flex items-center justify-center bg-surface rounded font-mono text-lg text-acid-cyan">
                          {bot.icon}
                        </span>
                        <div>
                          <h3 className="font-mono text-text">{bot.label}</h3>
                          <div className="text-xs font-mono text-text-muted">{bot.endpoint}</div>
                        </div>
                      </div>
                      <BotStatusBadge status={bot.status} />
                    </div>

                    {bot.errorMessage && (
                      <div className="mb-3 p-2 bg-warning/10 border border-warning/30 rounded text-xs font-mono text-warning">
                        {bot.errorMessage}
                      </div>
                    )}

                    <div className="flex flex-wrap gap-1 mb-3">
                      {bot.features.map((feature) => (
                        <span
                          key={feature}
                          className="px-2 py-0.5 text-xs font-mono bg-acid-green/10 text-acid-green/70 rounded"
                        >
                          {feature}
                        </span>
                      ))}
                    </div>

                    {bot.lastPing && (
                      <div className="text-xs font-mono text-text-muted">
                        Last activity: {bot.lastPing}
                      </div>
                    )}

                    <div className="flex gap-2 mt-3">
                      {bot.platform !== 'zoom' && (
                        <button
                          onClick={() => handleConfigure(bot.platform as IntegrationType)}
                          className="px-3 py-1 text-xs font-mono border border-acid-cyan/30 text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
                        >
                          [CONFIGURE]
                        </button>
                      )}
                      <Link
                        href={`/api-explorer?path=${encodeURIComponent(bot.endpoint)}`}
                        className="px-3 py-1 text-xs font-mono border border-acid-green/30 text-text-muted hover:text-acid-green transition-colors"
                      >
                        [API DOCS]
                      </Link>
                    </div>
                  </div>
                ))}
              </div>

              {/* Bot Setup Guide */}
              <div className="p-4 border border-acid-cyan/30 rounded bg-surface/20">
                <h3 className="font-mono text-acid-cyan text-sm mb-3">Bot Setup Guide</h3>
                <div className="space-y-2 text-xs font-mono text-text-muted">
                  <p>1. Configure your platform credentials in the Notifications tab or environment variables</p>
                  <p>2. Set up webhook URLs to point to your Aragora server endpoints</p>
                  <p>3. The bot will automatically handle incoming events and commands</p>
                </div>
                <div className="mt-3 p-3 bg-bg/50 rounded border border-acid-green/20">
                  <div className="text-xs font-mono text-text-muted mb-1">Example Slack command:</div>
                  <code className="text-xs font-mono text-acid-cyan">/aragora debate &quot;Should we use microservices?&quot;</code>
                </div>
              </div>

              {/* Environment Variables */}
              <div className="p-4 border border-acid-green/20 rounded bg-bg/50">
                <h3 className="font-mono text-text text-sm mb-3">Required Environment Variables</h3>
                <pre className="font-mono text-xs text-text-muted whitespace-pre overflow-x-auto">
{`# Slack Bot
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...

# Discord Bot
DISCORD_BOT_TOKEN=...
DISCORD_PUBLIC_KEY=...

# Microsoft Teams
TEAMS_APP_ID=...
TEAMS_APP_PASSWORD=...

# Zoom
ZOOM_CLIENT_ID=...
ZOOM_CLIENT_SECRET=...
ZOOM_WEBHOOK_SECRET=...`}
                </pre>
              </div>
            </div>
          )}

          {activeTab === 'system' && (
            <div className="space-y-4">
              <h2 className="font-mono text-text mb-4">System Integrations</h2>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {systemIntegrations.map((integration) => (
                  <Link
                    key={integration.href}
                    href={integration.href}
                    className="group p-4 border border-acid-green/20 hover:border-acid-green/50 bg-surface/30 hover:bg-surface/50 rounded transition-all"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-acid-cyan text-lg">{integration.icon}</span>
                        <h3 className="font-mono text-text group-hover:text-acid-green transition-colors">
                          {integration.title}
                        </h3>
                      </div>
                      <StatusBadge status={integration.status} />
                    </div>
                    <p className="text-text-muted font-mono text-xs mb-3 line-clamp-2">
                      {integration.description}
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {integration.features.map((feature) => (
                        <span
                          key={feature}
                          className="px-2 py-0.5 text-xs font-mono bg-acid-green/10 text-acid-green/70 rounded"
                        >
                          {feature}
                        </span>
                      ))}
                    </div>
                  </Link>
                ))}
              </div>

              {/* Quick Links */}
              <div className="mt-6 p-4 border border-acid-green/20 rounded bg-surface/20">
                <h3 className="font-mono text-text mb-3 text-sm">Quick Actions</h3>
                <div className="flex flex-wrap gap-2">
                  <Link
                    href="/webhooks"
                    className="px-3 py-2 border border-acid-cyan/30 text-xs font-mono text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
                  >
                    [+ NEW WEBHOOK]
                  </Link>
                  <Link
                    href="/plugins"
                    className="px-3 py-2 border border-acid-cyan/30 text-xs font-mono text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
                  >
                    [BROWSE PLUGINS]
                  </Link>
                  <Link
                    href="/training"
                    className="px-3 py-2 border border-acid-cyan/30 text-xs font-mono text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
                  >
                    [EXPORT DATA]
                  </Link>
                  <Link
                    href="/developer"
                    className="px-3 py-2 border border-acid-cyan/30 text-xs font-mono text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
                  >
                    [MCP SERVER]
                  </Link>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'docs' && (
            <div className="space-y-6">
              {/* Webhook Event Types */}
              <div>
                <h3 className="font-mono text-text mb-4 text-sm">Webhook Event Types</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                  {[
                    { event: 'debate_start', desc: 'Debate begins' },
                    { event: 'debate_end', desc: 'Debate completes' },
                    { event: 'consensus', desc: 'Consensus reached' },
                    { event: 'round_start', desc: 'Round begins' },
                    { event: 'agent_message', desc: 'Agent responds' },
                    { event: 'vote', desc: 'Vote cast' },
                    { event: 'insight_extracted', desc: 'Insight found' },
                    { event: 'claim_verification_result', desc: 'Claim verified' },
                    { event: 'gauntlet_complete', desc: 'Gauntlet done' },
                    { event: 'graph_branch_created', desc: 'Branch created' },
                    { event: 'breakpoint', desc: 'Human intervention' },
                    { event: 'genesis_evolution', desc: 'Population evolved' },
                  ].map(({ event, desc }) => (
                    <div
                      key={event}
                      className="p-2 border border-acid-green/10 rounded bg-surface/20 flex items-center justify-between"
                    >
                      <code className="font-mono text-xs text-acid-cyan">{event}</code>
                      <span className="font-mono text-xs text-text-muted">{desc}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Environment Variables */}
              <div>
                <h3 className="font-mono text-text mb-4 text-sm">Environment Variables</h3>
                <div className="p-4 border border-acid-green/20 rounded bg-bg/50 overflow-x-auto">
                  <pre className="font-mono text-xs text-text-muted whitespace-pre">
{`# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_BOT_TOKEN=xoxb-...

# Discord
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=-1001234567890

# Email
SENDGRID_API_KEY=SG.xxxxxxxxx
# or AWS SES credentials

# Microsoft Teams
TEAMS_WEBHOOK_URL=https://xxx.webhook.office.com/webhookb2/...

# WhatsApp (Meta)
WHATSAPP_PHONE_NUMBER_ID=1234567890
WHATSAPP_ACCESS_TOKEN=EAAxxxxxxxxx
# or Twilio credentials

# Matrix
MATRIX_HOMESERVER_URL=https://matrix.org
MATRIX_ACCESS_TOKEN=syt_xxxxx
MATRIX_ROOM_ID=!abc123:matrix.org`}
                  </pre>
                </div>
              </div>

              {/* API Example */}
              <div>
                <h3 className="font-mono text-text mb-4 text-sm">SDK Example</h3>
                <div className="p-4 border border-acid-green/20 rounded bg-bg/50 overflow-x-auto">
                  <pre className="font-mono text-xs text-acid-cyan whitespace-pre">
{`import { Aragora } from '@aragora/sdk';

const client = new Aragora({ apiKey: 'your-api-key' });

// Configure Slack integration
await client.integrations.configure('slack', {
  webhook_url: 'https://hooks.slack.com/...',
  notify_on_consensus: true,
  notify_on_debate_end: true,
});

// List configured integrations
const integrations = await client.integrations.list();

// Test connection
const result = await client.integrations.test('slack');
console.log(result.success); // true`}
                  </pre>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // INTEGRATIONS HUB
          </p>
        </footer>
      </main>

      {/* Setup Wizard Modal */}
      {wizardOpen && (
        <IntegrationSetupWizard
          type={wizardOpen}
          onClose={() => {
            setWizardOpen(null);
            setEditingConfig(undefined);
          }}
          onSave={handleSaveIntegration}
          existingConfig={editingConfig}
        />
      )}
    </>
  );
}
