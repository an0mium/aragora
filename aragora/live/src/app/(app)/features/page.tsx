'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';

type FeatureStatus = 'active' | 'available' | 'beta';

interface Feature {
  name: string;
  description: string;
  status: FeatureStatus;
  href?: string;
}

interface FeatureCategory {
  id: string;
  title: string;
  description: string;
  features: Feature[];
}

const STATUS_STYLES: Record<FeatureStatus, { bg: string; text: string; label: string }> = {
  active: { bg: 'bg-emerald-500/10', text: 'text-emerald-400', label: 'ACTIVE' },
  available: { bg: 'bg-blue-500/10', text: 'text-blue-400', label: 'AVAILABLE' },
  beta: { bg: 'bg-amber-500/10', text: 'text-amber-400', label: 'BETA' },
};

const CATEGORIES: FeatureCategory[] = [
  {
    id: 'debate',
    title: 'Debate Engine',
    description: 'Multi-agent adversarial debate orchestration with consensus detection',
    features: [
      { name: 'Arena Orchestrator', description: 'Run structured debates with configurable rounds, agents, and consensus methods', status: 'active', href: '/' },
      { name: 'Consensus Detection', description: 'Automatic convergence and majority-vote consensus with confidence scoring', status: 'active' },
      { name: 'ELO Rankings', description: 'Agent skill tracking via ELO rating system with tournament support', status: 'active', href: '/leaderboard' },
      { name: 'Decision Receipts', description: 'Cryptographic audit trails with SHA-256 hashing for every decision', status: 'active', href: '/receipts' },
      { name: 'Trickster Detection', description: 'Hollow consensus detection to prevent groupthink among agents', status: 'active' },
      { name: 'Live Explainability', description: 'Real-time factor tracking during debates via EventBus', status: 'active', href: '/spectate' },
    ],
  },
  {
    id: 'knowledge',
    title: 'Knowledge Management',
    description: 'Unified knowledge superstructure with 34 adapters across all memory systems',
    features: [
      { name: 'Knowledge Mound', description: '34 adapters integrating memory, consensus, evidence, and debate knowledge', status: 'active', href: '/knowledge' },
      { name: 'Semantic Search', description: 'Vector-based knowledge retrieval with confidence scoring', status: 'active', href: '/knowledge' },
      { name: 'Contradiction Detection', description: 'Automatic identification of conflicting knowledge nodes', status: 'active', href: '/knowledge' },
      { name: 'Multi-Tier Memory', description: 'Fast/Medium/Slow/Glacial memory tiers with TTL-based promotion', status: 'active', href: '/memory' },
      { name: 'Supermemory', description: 'Cross-session external memory for persistent organizational learning', status: 'active' },
      { name: 'Unified Memory Gateway', description: 'Fan-out queries across ContinuumMemory, KM, Supermemory, and claude-mem', status: 'active' },
    ],
  },
  {
    id: 'agents',
    title: 'Agent Types',
    description: '42 agent types from 10+ providers with automatic fallback and resilience',
    features: [
      { name: 'Anthropic (Opus 4.5)', description: 'Claude API agent with structured output support', status: 'active', href: '/agents' },
      { name: 'OpenAI (GPT 5.2)', description: 'GPT API agent with function calling', status: 'active', href: '/agents' },
      { name: 'Gemini (3.1 Pro)', description: 'Google Gemini with multimodal capabilities', status: 'active', href: '/agents' },
      { name: 'Grok (4)', description: 'xAI Grok agent with real-time knowledge', status: 'active', href: '/agents' },
      { name: 'DeepSeek / R1', description: 'Reasoning-optimized models via OpenRouter', status: 'active', href: '/agents' },
      { name: 'Qwen / Kimi / Llama / Mistral', description: 'Broad model coverage via OpenRouter fallback', status: 'active', href: '/agents' },
      { name: 'Agent Calibration', description: 'Brier score tracking and domain-specific performance assessment', status: 'active', href: '/calibration' },
      { name: 'Circuit Breaker', description: 'Automatic failure handling with OpenRouter fallback on quota errors', status: 'active' },
    ],
  },
  {
    id: 'analytics',
    title: 'Analytics',
    description: 'Comprehensive debate metrics, agent performance, usage trends, and cost analysis',
    features: [
      { name: 'Analytics Dashboard', description: 'Debate metrics, agent performance, usage trends, and cost analysis', status: 'active', href: '/analytics' },
      { name: 'Cost Tracking', description: 'Per-model and per-provider cost breakdown with budget alerts', status: 'active', href: '/costs' },
      { name: 'Observability', description: 'Prometheus metrics, Grafana dashboards, OpenTelemetry tracing', status: 'active', href: '/observability' },
      { name: 'Usage Dashboard', description: 'Token usage, API call volume, and trend analysis', status: 'active', href: '/usage' },
      { name: 'Pulse (Trending Topics)', description: 'HackerNews, Reddit, Twitter ingestors with quality filtering', status: 'active', href: '/pulse' },
    ],
  },
  {
    id: 'security',
    title: 'Security',
    description: 'Enterprise-grade security with encryption, RBAC, and anomaly detection',
    features: [
      { name: 'AES-256-GCM Encryption', description: 'Data encryption at rest and in transit', status: 'active', href: '/security' },
      { name: 'RBAC v2', description: '50+ fine-grained permissions with role hierarchy and middleware', status: 'active' },
      { name: 'Key Rotation', description: 'Automated cryptographic key rotation pipeline', status: 'active', href: '/security' },
      { name: 'SSRF Protection', description: 'Safe HTTP wrapper preventing server-side request forgery', status: 'active' },
      { name: 'Anomaly Detection', description: 'Real-time security anomaly detection and alerting', status: 'active', href: '/security-scan' },
      { name: 'OIDC/SAML SSO', description: 'Enterprise single sign-on with MFA support', status: 'active', href: '/auth' },
    ],
  },
  {
    id: 'compliance',
    title: 'Compliance',
    description: 'SOC 2 controls, GDPR support, EU AI Act artifact generation',
    features: [
      { name: 'EU AI Act Compliance', description: 'Risk classification, conformity assessment, and artifact bundle generation', status: 'active', href: '/compliance' },
      { name: 'Audit Trails', description: 'Complete audit logging of all decisions and actions', status: 'active', href: '/audit' },
      { name: 'Privacy Controls', description: 'GDPR anonymization, consent management, data deletion, retention policies', status: 'active', href: '/privacy' },
      { name: 'Policy Governance', description: 'Conflict detection, distributed cache, and sync scheduling', status: 'active', href: '/policy' },
    ],
  },
  {
    id: 'self-improvement',
    title: 'Self-Improvement',
    description: 'Autonomous Nomic Loop for self-directing codebase improvement',
    features: [
      { name: 'Nomic Loop', description: 'Autonomous cycle: debate improvements, design, implement, verify', status: 'active', href: '/self-improve' },
      { name: 'Meta-Planner', description: 'Debate-driven goal prioritization with cross-cycle learning', status: 'active' },
      { name: 'Branch Coordinator', description: 'Parallel git worktree management for safe changes', status: 'active' },
      { name: 'Strategic Scanner', description: '12-signal codebase analysis for self-directed goal generation', status: 'active' },
      { name: 'Pipeline (Idea-to-Execution)', description: '4-stage pipeline: Ideas, Goals, Workflows, Orchestration', status: 'active', href: '/command' },
      { name: 'Spectator Mode', description: 'Real-time observation of autonomous improvement cycles', status: 'active', href: '/spectate' },
    ],
  },
  {
    id: 'integrations',
    title: 'Integrations & Connectors',
    description: 'Broad connector ecosystem for enterprise and consumer platforms',
    features: [
      { name: 'Slack / Discord / Teams', description: 'Chat platform connectors for debate interfaces', status: 'active', href: '/integrations' },
      { name: 'Telegram / WhatsApp', description: 'Mobile messaging connectors with bidirectional routing', status: 'active', href: '/social' },
      { name: 'Kafka / RabbitMQ', description: 'Enterprise event stream ingestion for real-time data', status: 'active', href: '/connectors' },
      { name: 'Zapier / Webhooks', description: 'No-code automation and custom webhook delivery', status: 'active', href: '/webhooks' },
      { name: 'LangChain Integration', description: 'LangChain tools and chains for debate workflows', status: 'available' },
      { name: 'MCP Server', description: 'Model Context Protocol server for tool-use AI agents', status: 'active', href: '/mcp' },
    ],
  },
];

export default function FeaturesPage() {
  const [filter, setFilter] = useState<FeatureStatus | 'all'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const filteredCategories = CATEGORIES.map(cat => ({
    ...cat,
    features: cat.features.filter(f => {
      const matchesFilter = filter === 'all' || f.status === filter;
      const matchesSearch = !searchQuery.trim() ||
        f.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        f.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        cat.title.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesFilter && matchesSearch;
    }),
  })).filter(cat => cat.features.length > 0);

  const totalFeatures = CATEGORIES.reduce((sum, cat) => sum + cat.features.length, 0);
  const activeCount = CATEGORIES.reduce((sum, cat) => sum + cat.features.filter(f => f.status === 'active').length, 0);
  const betaCount = CATEGORIES.reduce((sum, cat) => sum + cat.features.filter(f => f.status === 'beta').length, 0);

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <div className="container mx-auto px-4 py-6 max-w-5xl">
          {/* Header */}
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} FEATURE DISCOVERY
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Explore the full capabilities of the Aragora Decision Integrity Platform.
            </p>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 md:grid-cols-4 gap-3 mb-6">
            <div className="p-3 bg-surface border border-border text-center">
              <div className="text-xl font-mono text-acid-green">{totalFeatures}</div>
              <div className="text-[10px] font-mono text-text-muted">Total Features</div>
            </div>
            <div className="p-3 bg-surface border border-border text-center">
              <div className="text-xl font-mono text-emerald-400">{activeCount}</div>
              <div className="text-[10px] font-mono text-text-muted">Active</div>
            </div>
            <div className="p-3 bg-surface border border-border text-center">
              <div className="text-xl font-mono text-amber-400">{betaCount}</div>
              <div className="text-[10px] font-mono text-text-muted">Beta</div>
            </div>
            <div className="hidden md:block p-3 bg-surface border border-border text-center">
              <div className="text-xl font-mono text-blue-400">{CATEGORIES.length}</div>
              <div className="text-[10px] font-mono text-text-muted">Categories</div>
            </div>
          </div>

          {/* Search + Filter */}
          <div className="flex flex-wrap gap-3 mb-6">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search features..."
              className="flex-1 min-w-[200px] px-3 py-2 bg-surface border border-border text-text font-mono text-sm focus:border-acid-green focus:outline-none"
            />
            <div className="flex gap-1">
              {(['all', 'active', 'available', 'beta'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setFilter(s)}
                  className={`px-3 py-2 text-xs font-mono transition-colors ${
                    filter === s
                      ? 'bg-acid-green text-bg'
                      : 'text-text-muted hover:text-text border border-border'
                  }`}
                >
                  {s.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Feature Categories */}
          <div className="space-y-6">
            {filteredCategories.map((category) => (
              <section key={category.id} className="bg-surface border border-border">
                <div className="p-4 border-b border-border">
                  <h2 className="text-sm font-mono text-acid-green font-bold uppercase">
                    {'>'} {category.title}
                  </h2>
                  <p className="text-xs font-mono text-text-muted mt-1">{category.description}</p>
                </div>
                <div className="divide-y divide-border">
                  {category.features.map((feature) => {
                    const style = STATUS_STYLES[feature.status];
                    const content = (
                      <div className="p-3 flex items-center justify-between hover:bg-bg/50 transition-colors">
                        <div className="flex-1 min-w-0 mr-3">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-mono text-text">{feature.name}</span>
                            <span className={`px-1.5 py-0.5 text-[10px] font-mono ${style.bg} ${style.text} border border-current/20`}>
                              {style.label}
                            </span>
                          </div>
                          <p className="text-xs font-mono text-text-muted mt-0.5 truncate">{feature.description}</p>
                        </div>
                        {feature.href && (
                          <span className="text-xs font-mono text-acid-green/60 flex-shrink-0">
                            {'->'}
                          </span>
                        )}
                      </div>
                    );

                    if (feature.href) {
                      return (
                        <Link key={feature.name} href={feature.href} className="block">
                          {content}
                        </Link>
                      );
                    }
                    return <div key={feature.name}>{content}</div>;
                  })}
                </div>
              </section>
            ))}
          </div>

          {filteredCategories.length === 0 && (
            <div className="text-center py-12 text-text-muted font-mono">
              No features match your search.
            </div>
          )}
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <p className="text-text-muted">{'>'} ARAGORA // FEATURE DISCOVERY</p>
        </footer>
      </main>
    </>
  );
}
