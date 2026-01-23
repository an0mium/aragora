'use client';

import { SectionHeader } from './SectionHeader';

const DATA_SOURCES = [
  { icon: '[]', label: 'Documents', count: '25+ formats' },
  { icon: '{}', label: 'APIs', count: 'REST/GraphQL' },
  { icon: '()', label: 'Databases', count: 'SQL/NoSQL' },
  { icon: '<>', label: 'Cloud', count: 'S3/GCS/Azure' },
];

const MODELS = [
  { name: 'Claude', color: 'text-orange-400' },
  { name: 'GPT-4', color: 'text-green-400' },
  { name: 'Gemini', color: 'text-blue-400' },
  { name: 'Mistral', color: 'text-purple-400' },
  { name: '+11 more', color: 'text-text-muted' },
];

const CHANNELS = [
  { icon: '#', label: 'Slack' },
  { icon: '@', label: 'Teams' },
  { icon: '!', label: 'Discord' },
  { icon: '>', label: 'Email' },
  { icon: '+', label: '20 more' },
];

/**
 * ControlPlaneVisualization - Visual diagram showing the control plane architecture.
 *
 * Displays the data flow: Sources → Control Plane → Vetted Decisionmaking → Channels
 * to reinforce the orchestration and governance positioning.
 */
export function ControlPlaneVisualization() {
  return (
    <section className="py-12 border-t border-acid-green/20">
      <div className="container mx-auto px-4">
        <SectionHeader title="CONTROL PLANE ARCHITECTURE" />
        <p className="text-text-muted font-mono text-xs text-center mb-8 max-w-xl mx-auto">
          Orchestrating AI vetted decisionmaking across your organization&apos;s knowledge and communication channels.
        </p>

        {/* Main Flow Visualization */}
        <div className="max-w-5xl mx-auto">
          {/* Desktop Layout */}
          <div className="hidden md:flex items-stretch justify-center gap-2">
            {/* Data Sources */}
            <div className="flex-1 max-w-[180px]">
              <div className="border border-acid-cyan/40 rounded-lg p-4 bg-surface/30 h-full">
                <div className="text-acid-cyan font-mono text-xs font-bold mb-3 text-center">
                  [SOURCES]
                </div>
                <div className="space-y-2">
                  {DATA_SOURCES.map((source) => (
                    <div key={source.label} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-acid-cyan font-mono text-xs">{source.icon}</span>
                        <span className="text-text text-xs font-mono">{source.label}</span>
                      </div>
                      <span className="text-text-muted text-[10px] font-mono">{source.count}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-3 pt-3 border-t border-border/50 text-center">
                  <span className="text-acid-cyan font-mono text-lg font-bold">25+</span>
                  <div className="text-text-muted text-[10px] font-mono">formats</div>
                </div>
              </div>
            </div>

            {/* Arrow */}
            <div className="flex items-center px-2">
              <div className="text-acid-green font-mono text-xl animate-pulse">→</div>
            </div>

            {/* Control Plane Core */}
            <div className="flex-1 max-w-[220px]">
              <div className="border-2 border-acid-green/60 rounded-lg p-4 bg-acid-green/5 h-full">
                <div className="text-acid-green font-mono text-xs font-bold mb-3 text-center">
                  [CONTROL PLANE]
                </div>

                {/* Orchestration */}
                <div className="bg-surface/50 rounded p-2 mb-2">
                  <div className="text-acid-green font-mono text-[10px] mb-1">ORCHESTRATION</div>
                  <div className="text-text-muted text-[10px] font-mono">
                    Route tasks to optimal models
                  </div>
                </div>

                {/* Governance */}
                <div className="bg-surface/50 rounded p-2 mb-2">
                  <div className="text-acid-yellow font-mono text-[10px] mb-1">GOVERNANCE</div>
                  <div className="text-text-muted text-[10px] font-mono">
                    Enforce policies &amp; compliance
                  </div>
                </div>

                {/* Memory */}
                <div className="bg-surface/50 rounded p-2">
                  <div className="text-acid-cyan font-mono text-[10px] mb-1">MEMORY</div>
                  <div className="text-text-muted text-[10px] font-mono">
                    4-tier learning continuum
                  </div>
                </div>
              </div>
            </div>

            {/* Arrow */}
            <div className="flex items-center px-2">
              <div className="text-acid-green font-mono text-xl animate-pulse">→</div>
            </div>

            {/* Vetted Decisionmaking Engine */}
            <div className="flex-1 max-w-[180px]">
              <div className="border border-acid-yellow/40 rounded-lg p-4 bg-surface/30 h-full">
                <div className="text-acid-yellow font-mono text-xs font-bold mb-3 text-center">
                  [ROBUST DECISIONMAKING]
                </div>
                <div className="space-y-1 mb-3">
                  {MODELS.map((model) => (
                    <div key={model.name} className="flex items-center gap-2">
                      <span className={`w-1.5 h-1.5 rounded-full ${model.color.replace('text-', 'bg-')}`} />
                      <span className={`text-xs font-mono ${model.color}`}>{model.name}</span>
                    </div>
                  ))}
                </div>
                <div className="pt-3 border-t border-border/50 text-center">
                  <span className="text-acid-yellow font-mono text-lg font-bold">15+</span>
                  <div className="text-text-muted text-[10px] font-mono">models</div>
                </div>
              </div>
            </div>

            {/* Arrow */}
            <div className="flex items-center px-2">
              <div className="text-acid-green font-mono text-xl animate-pulse">→</div>
            </div>

            {/* Channels */}
            <div className="flex-1 max-w-[180px]">
              <div className="border border-acid-magenta/40 rounded-lg p-4 bg-surface/30 h-full">
                <div className="text-acid-magenta font-mono text-xs font-bold mb-3 text-center">
                  [CHANNELS]
                </div>
                <div className="space-y-2">
                  {CHANNELS.map((channel) => (
                    <div key={channel.label} className="flex items-center gap-2">
                      <span className="text-acid-magenta font-mono text-xs">{channel.icon}</span>
                      <span className="text-text text-xs font-mono">{channel.label}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-3 pt-3 border-t border-border/50 text-center">
                  <span className="text-acid-magenta font-mono text-lg font-bold">24+</span>
                  <div className="text-text-muted text-[10px] font-mono">integrations</div>
                </div>
              </div>
            </div>
          </div>

          {/* Mobile Layout - Vertical */}
          <div className="md:hidden space-y-4">
            {/* Sources */}
            <div className="border border-acid-cyan/40 rounded-lg p-4 bg-surface/30">
              <div className="text-acid-cyan font-mono text-xs font-bold mb-2">[SOURCES] 25+ formats</div>
              <div className="flex flex-wrap gap-2">
                {DATA_SOURCES.map((s) => (
                  <span key={s.label} className="text-xs font-mono text-text-muted">{s.label}</span>
                ))}
              </div>
            </div>

            <div className="text-center text-acid-green font-mono">↓</div>

            {/* Control Plane */}
            <div className="border-2 border-acid-green/60 rounded-lg p-4 bg-acid-green/5">
              <div className="text-acid-green font-mono text-xs font-bold mb-2">[CONTROL PLANE]</div>
              <div className="flex flex-wrap gap-2 text-xs font-mono text-text-muted">
                <span>Orchestration</span>
                <span>|</span>
                <span>Governance</span>
                <span>|</span>
                <span>Memory</span>
              </div>
            </div>

            <div className="text-center text-acid-green font-mono">↓</div>

            {/* Vetted Decisionmaking */}
            <div className="border border-acid-yellow/40 rounded-lg p-4 bg-surface/30">
              <div className="text-acid-yellow font-mono text-xs font-bold mb-2">[ROBUST DECISIONMAKING] 15+ models</div>
              <div className="flex flex-wrap gap-2">
                {MODELS.slice(0, 4).map((m) => (
                  <span key={m.name} className={`text-xs font-mono ${m.color}`}>{m.name}</span>
                ))}
              </div>
            </div>

            <div className="text-center text-acid-green font-mono">↓</div>

            {/* Channels */}
            <div className="border border-acid-magenta/40 rounded-lg p-4 bg-surface/30">
              <div className="text-acid-magenta font-mono text-xs font-bold mb-2">[CHANNELS] 24+ integrations</div>
              <div className="flex flex-wrap gap-2">
                {CHANNELS.slice(0, 4).map((c) => (
                  <span key={c.label} className="text-xs font-mono text-text-muted">{c.label}</span>
                ))}
              </div>
            </div>
          </div>

          {/* Output: Decision Receipt */}
          <div className="mt-6 border border-acid-green/40 rounded-lg p-4 bg-surface/30 max-w-md mx-auto">
            <div className="flex items-center justify-between mb-2">
              <div className="text-acid-green font-mono text-xs font-bold">[OUTPUT] DECISION RECEIPT</div>
              <span className="px-2 py-0.5 text-[10px] font-mono bg-acid-green/20 text-acid-green rounded">
                AUDIT-READY
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <div className="text-acid-cyan text-xs font-mono">Verdict</div>
                <div className="text-text-muted text-[10px] font-mono">+ Confidence</div>
              </div>
              <div>
                <div className="text-acid-yellow text-xs font-mono">Findings</div>
                <div className="text-text-muted text-[10px] font-mono">+ Evidence</div>
              </div>
              <div>
                <div className="text-acid-magenta text-xs font-mono">Dissent</div>
                <div className="text-text-muted text-[10px] font-mono">+ Audit Trail</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default ControlPlaneVisualization;
