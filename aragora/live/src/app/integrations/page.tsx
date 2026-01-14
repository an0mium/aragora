'use client';

import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector } from '@/components/BackendSelector';

interface IntegrationCard {
  title: string;
  description: string;
  href: string;
  icon: string;
  status: 'stable' | 'beta' | 'alpha';
  features: string[];
}

const integrations: IntegrationCard[] = [
  {
    title: 'Webhooks',
    description: 'Receive real-time HTTP callbacks for debate events. Integrate with Slack, Discord, or custom systems.',
    href: '/webhooks',
    icon: '>>',
    status: 'stable',
    features: ['Debate lifecycle events', 'Consensus notifications', 'Agent message hooks', 'HMAC signatures'],
  },
  {
    title: 'Plugin Marketplace',
    description: 'Extend Aragora with custom plugins for selection, scoring, and debate behaviors.',
    href: '/plugins',
    icon: '+>',
    status: 'stable',
    features: ['Agent scorers', 'Team selectors', 'Role assigners', 'Custom behaviors'],
  },
  {
    title: 'Training Export',
    description: 'Export debate data for fine-tuning language models. SFT, DPO, and adversarial formats.',
    href: '/training',
    icon: '[]',
    status: 'stable',
    features: ['SFT format', 'DPO pairs', 'Gauntlet adversarial', 'JSONL export'],
  },
  {
    title: 'Evidence Connectors',
    description: 'Connect external knowledge sources for fact-grounded debates.',
    href: '/evidence',
    icon: '??',
    status: 'beta',
    features: ['Web search', 'Document upload', 'API sources', 'Citation tracking'],
  },
  {
    title: 'API Explorer',
    description: 'Interactive documentation and testing for all 247 API endpoints.',
    href: '/api-explorer',
    icon: '{}',
    status: 'stable',
    features: ['OpenAPI spec', 'Try requests', 'Response examples', 'Authentication'],
  },
  {
    title: 'MCP Server',
    description: 'Model Context Protocol server for AI assistant integrations.',
    href: '/developer',
    icon: '<>',
    status: 'beta',
    features: ['Claude integration', 'Tool discovery', '17 MCP tools', 'Context streaming'],
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

export default function IntegrationsPage() {
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
          <div className="mb-8">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} INTEGRATIONS
            </h1>
            <p className="text-text-muted font-mono text-sm max-w-2xl">
              Connect Aragora with external systems, export data for ML training,
              and extend functionality with plugins. All integrations support both
              REST API and SDK access.
            </p>
          </div>

          {/* SDK Installation */}
          <div className="mb-8 p-4 border border-acid-cyan/30 bg-surface/30 rounded">
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

          {/* Integration Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {integrations.map((integration) => (
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
          <div className="mt-8 p-4 border border-acid-green/20 rounded bg-surface/20">
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

          {/* Event Types Reference */}
          <div className="mt-8">
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
    </>
  );
}
