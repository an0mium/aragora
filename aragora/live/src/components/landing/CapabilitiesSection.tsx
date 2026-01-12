'use client';

import { SectionHeader } from './SectionHeader';

const NOMIC_LOOP = `CONTEXT → DEBATE → DESIGN → IMPLEMENT → VERIFY → COMMIT
                      ↑__________________________|`;

export function CapabilitiesSection() {
  return (
    <section className="py-12 border-t border-acid-green/20">
      <div className="container mx-auto px-4">
        <SectionHeader title="UNIQUE CAPABILITIES" />
        <p className="text-text-muted font-mono text-xs text-center mb-8 max-w-xl mx-auto">
          These capabilities power the stress-test engine and its audit-ready outputs.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-6xl mx-auto">
          {/* ELO Rankings */}
          <div className="border border-acid-green/40 p-5 bg-surface/50">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-acid-yellow text-lg">{'#'}</span>
              <h3 className="text-acid-green font-mono text-sm">ELO RANKINGS</h3>
            </div>
            <p className="text-text-muted text-xs font-mono leading-relaxed mb-3">
              Agents earn reputation through stress-test performance. Domain-specific ratings: security,
              architecture, testing.
            </p>
            <p className="text-acid-green/60 text-xs font-mono">
              See who&apos;s actually good at what &mdash; backed by data.
            </p>
          </div>

          {/* Continuum Memory */}
          <div className="border border-acid-cyan/40 p-5 bg-surface/50">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-acid-cyan text-lg">{'~'}</span>
              <h3 className="text-acid-cyan font-mono text-sm">CONTINUUM MEMORY</h3>
            </div>
            <p className="text-text-muted text-xs font-mono leading-relaxed mb-2">
              4-tier memory system inspired by cognitive science:
            </p>
            <div className="text-xs font-mono space-y-1 mb-2">
              <div className="flex justify-between text-text-muted">
                <span>FAST</span>
                <span className="text-acid-green">1 hour</span>
              </div>
              <div className="flex justify-between text-text-muted">
                <span>MEDIUM</span>
                <span className="text-acid-green">1 day</span>
              </div>
              <div className="flex justify-between text-text-muted">
                <span>SLOW</span>
                <span className="text-acid-green">1 week</span>
              </div>
              <div className="flex justify-between text-text-muted">
                <span>GLACIAL</span>
                <span className="text-acid-green">1 month</span>
              </div>
            </div>
            <p className="text-acid-cyan/60 text-xs font-mono">
              Surprise-based promotion: unexpected outcomes remembered.
            </p>
          </div>

          {/* Calibration Tracking */}
          <div className="border border-acid-green/40 p-5 bg-surface/50">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-acid-green text-lg">{'%'}</span>
              <h3 className="text-acid-green font-mono text-sm">CALIBRATION TRACKING</h3>
            </div>
            <p className="text-text-muted text-xs font-mono leading-relaxed mb-2">
              Beyond win/loss: we measure prediction confidence.
            </p>
            <ul className="text-text-muted text-xs font-mono space-y-1 mb-2">
              <li>&bull; Brier scores for prediction accuracy</li>
              <li>&bull; Over/underconfidence detection</li>
              <li>&bull; Domain-specific calibration curves</li>
            </ul>
            <p className="text-acid-green/60 text-xs font-mono">
              Identify agents that are confidently wrong (dangerous).
            </p>
          </div>

          {/* Formal Verification */}
          <div className="border border-acid-cyan/40 p-5 bg-surface/50">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-acid-magenta text-lg">{'!'}</span>
              <h3 className="text-acid-cyan font-mono text-sm">FORMAL VERIFICATION</h3>
            </div>
            <p className="text-text-muted text-xs font-mono leading-relaxed mb-2">
              Z3 and Lean backends for provable correctness. When persuasion isn&apos;t enough, demand
              proof.
            </p>
            <p className="text-acid-cyan/60 text-xs font-mono">
              Machine-verified consensus for high-stakes decisions.
            </p>
          </div>

          {/* The Nomic Loop */}
          <div className="border border-acid-green/40 p-5 bg-surface/50 md:col-span-2 lg:col-span-2">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-acid-green text-lg">{'@'}</span>
              <h3 className="text-acid-green font-mono text-sm">THE NOMIC LOOP</h3>
            </div>
            <p className="text-text-muted text-xs font-mono leading-relaxed mb-3">
              Aragora improves itself through autonomous red-team cycles:
            </p>
            <pre className="text-acid-green/80 text-xs font-mono mb-3 overflow-x-auto">
              {NOMIC_LOOP}
            </pre>
            <p className="text-text-muted text-xs font-mono mb-2">
              Protected files checksummed. Automatic rollback on failure.
            </p>
            <p className="text-acid-green/80 text-xs font-mono font-bold">
              The only AI red-team system that evolves its own code.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
