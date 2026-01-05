'use client';

import { useState, useCallback } from 'react';
import { DebateInput } from './DebateInput';
import { AsciiBannerCompact } from './AsciiBanner';
import { Scanlines, CRTVignette } from './MatrixRain';
import { ThemeToggle } from './ThemeToggle';
import { BackendSelector } from './BackendSelector';

interface LandingPageProps {
  apiBase: string;
  wsUrl: string;
  onDebateStarted: (debateId: string) => void;
}

export function LandingPage({ apiBase, wsUrl, onDebateStarted }: LandingPageProps) {
  const [error, setError] = useState<string | null>(null);
  const [activeDebateId, setActiveDebateId] = useState<string | null>(null);
  const [activeQuestion, setActiveQuestion] = useState<string | null>(null);

  const handleDebateStarted = useCallback((debateId: string, question: string) => {
    setActiveDebateId(debateId);
    setActiveQuestion(question);
    onDebateStarted(debateId);
  }, [onDebateStarted]);

  const handleError = useCallback((err: string) => {
    setError(err);
    setTimeout(() => setError(null), 5000);
  }, []);

  return (
    <>
      {/* CRT Effects */}
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10 flex flex-col">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <AsciiBannerCompact connected={true} />
            <div className="flex items-center gap-4">
              <a
                href="/about"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [ABOUT]
              </a>
              <a
                href="https://live.aragora.ai"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [LIVE DASHBOARD]
              </a>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <div className="flex flex-col items-center justify-center px-4 py-12 sm:py-16">
          {/* ASCII Art Title */}
          <pre className="text-acid-green text-[6px] sm:text-[7px] font-mono text-center mb-6 hidden sm:block leading-tight">
{`    ▄▄▄       ██▀███   ▄▄▄        ▄████  ▒█████   ██▀███   ▄▄▄
   ▒████▄    ▓██ ▒ ██▒▒████▄     ██▒ ▀█▒▒██▒  ██▒▓██ ▒ ██▒▒████▄
   ▒██  ▀█▄  ▓██ ░▄█ ▒▒██  ▀█▄  ▒██░▄▄▄░▒██░  ██▒▓██ ░▄█ ▒▒██  ▀█▄
   ░██▄▄▄▄██ ▒██▀▀█▄  ░██▄▄▄▄██ ░▓█  ██▓▒██   ██░▒██▀▀█▄  ░██▄▄▄▄██
    ▓█   ▓██▒░██▓ ▒██▒ ▓█   ▓██▒░▒▓███▀▒░ ████▓▒░░██▓ ▒██▒ ▓█   ▓██▒
    ▒▒   ▓▒█░░ ▒▓ ░▒▓░ ▒▒   ▓▒█░ ░▒   ▒ ░ ▒░▒░▒░ ░ ▒▓ ░▒▓░ ▒▒   ▓▒█░
     ▒   ▒▒ ░  ░▒ ░ ▒░  ▒   ▒▒ ░  ░   ░   ░ ▒ ▒░   ░▒ ░ ▒░  ▒   ▒▒ ░
     ░   ▒     ░░   ░   ░   ▒   ░ ░   ░ ░ ░ ░ ▒    ░░   ░   ░   ▒
         ░  ░   ░           ░  ░      ░     ░ ░     ░           ░  ░`}
          </pre>

          {/* New Tagline */}
          <h1 className="text-2xl sm:text-3xl font-mono text-center mb-3">
            <span className="text-acid-green">The Self-Evolving</span>{' '}
            <span className="text-text">Debate Arena</span>
          </h1>

          {/* Model Names */}
          <p className="text-acid-cyan font-mono text-sm sm:text-base text-center mb-2">
            Claude vs GPT vs Gemini vs Grok vs DeepSeek
          </p>

          {/* Subtitle */}
          <p className="text-text-muted font-mono text-xs sm:text-sm text-center mb-10 max-w-lg">
            Real AI models. Adversarial reasoning. Verified consensus.
          </p>

          {/* Error Banner */}
          {error && (
            <div className="w-full max-w-3xl mb-6 bg-warning/10 border border-warning/30 p-4 flex items-center justify-between">
              <span className="text-warning font-mono text-sm">{error}</span>
              <button
                onClick={() => setError(null)}
                className="text-warning hover:text-warning/80"
              >
                ✕
              </button>
            </div>
          )}

          {/* Debate Started Indicator */}
          {activeDebateId && (
            <div className="w-full max-w-3xl mb-6 bg-acid-green/10 border border-acid-green/30 p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-2 h-2 bg-acid-green rounded-full animate-pulse"></span>
                <span className="text-acid-green font-mono text-sm font-bold">DEBATE IN PROGRESS</span>
              </div>
              <p className="text-text font-mono text-sm truncate">{activeQuestion}</p>
              <p className="text-text-muted font-mono text-xs mt-2">
                ID: {activeDebateId} | Events streaming via WebSocket
              </p>
            </div>
          )}

          {/* Debate Input */}
          <DebateInput
            apiBase={apiBase}
            onDebateStarted={handleDebateStarted}
            onError={handleError}
          />
        </div>

        {/* WHY ARAGORA Section */}
        <section className="py-12 border-t border-acid-green/20">
          <div className="container mx-auto px-4">
            <div className="text-center mb-8">
              <p className="text-acid-green/50 font-mono text-xs mb-2">{'═'.repeat(30)}</p>
              <h2 className="text-acid-green font-mono text-lg">{'>'} WHY ARAGORA?</h2>
              <p className="text-acid-green/50 font-mono text-xs mt-2">{'═'.repeat(30)}</p>
            </div>

            <p className="text-text-muted font-mono text-xs text-center mb-8 max-w-xl mx-auto">
              Most &quot;multi-agent&quot; systems run copies of the same model talking to itself. Aragora is different:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-5xl mx-auto">
              {/* Heterogeneous Arena */}
              <div className="border border-acid-green/30 p-4 bg-surface/30">
                <h3 className="text-acid-green font-mono text-sm mb-3 flex items-center gap-2">
                  <span className="text-acid-cyan">{'>'}</span> HETEROGENEOUS ARENA
                </h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  7+ distinct AI providers compete in the same debate.
                  Claude&apos;s caution vs GPT&apos;s creativity vs Gemini&apos;s speed.
                  Real diversity. Real disagreement. Real signal.
                </p>
              </div>

              {/* Self-Improving Framework */}
              <div className="border border-acid-cyan/30 p-4 bg-surface/30">
                <h3 className="text-acid-cyan font-mono text-sm mb-3 flex items-center gap-2">
                  <span className="text-acid-green">{'>'}</span> SELF-IMPROVING FRAMEWORK
                </h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  Aragora runs the &quot;Nomic Loop&quot; &mdash; agents debate improvements
                  to their own framework, implement code, verify changes.
                  The arena evolves through its own debates.
                </p>
              </div>

              {/* Calibrated Trust */}
              <div className="border border-acid-green/30 p-4 bg-surface/30">
                <h3 className="text-acid-green font-mono text-sm mb-3 flex items-center gap-2">
                  <span className="text-acid-cyan">{'>'}</span> CALIBRATED TRUST
                </h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  We track prediction accuracy over time. Know which agents
                  are confidently wrong vs genuinely uncertain.
                  Trust earned through track record, not marketing.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* THE DEBATE PROTOCOL Section */}
        <section className="py-12 border-t border-acid-green/20">
          <div className="container mx-auto px-4">
            <div className="text-center mb-8">
              <p className="text-acid-green/50 font-mono text-xs mb-2">{'═'.repeat(30)}</p>
              <h2 className="text-acid-green font-mono text-lg">{'>'} THE DEBATE PROTOCOL</h2>
              <p className="text-acid-green/50 font-mono text-xs mt-2">{'═'.repeat(30)}</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 max-w-6xl mx-auto">
              {/* Step 01: Propose */}
              <div className="border border-acid-green/20 p-4 bg-surface/30">
                <h3 className="text-acid-green font-mono text-sm mb-2">[01] PROPOSE</h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  Agents from different providers generate diverse solutions.
                  Claude might suggest caution. GPT might go bold.
                </p>
              </div>

              {/* Step 02: Critique */}
              <div className="border border-acid-cyan/20 p-4 bg-surface/30">
                <h3 className="text-acid-cyan font-mono text-sm mb-2">[02] CRITIQUE</h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  Each proposal faces adversarial review from rival models.
                  Agents rotate through cognitive roles: Analyst, Skeptic, Devil&apos;s Advocate.
                </p>
              </div>

              {/* Step 03: Adapt */}
              <div className="border border-acid-green/20 p-4 bg-surface/30">
                <h3 className="text-acid-green font-mono text-sm mb-2">[03] ADAPT</h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  Agents update positions based on valid critiques.
                  We track who changed their mind and why.
                </p>
              </div>

              {/* Step 04: Converge */}
              <div className="border border-acid-cyan/20 p-4 bg-surface/30">
                <h3 className="text-acid-cyan font-mono text-sm mb-2">[04] CONVERGE</h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  Semantic similarity detection finds natural consensus.
                  Dissenting views preserved with full reasoning chains.
                </p>
              </div>

              {/* Step 05: Learn */}
              <div className="border border-acid-green/20 p-4 bg-surface/30">
                <h3 className="text-acid-green font-mono text-sm mb-2">[05] LEARN</h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  Outcomes feed into agent memory and ELO ratings.
                  The arena gets smarter with every debate.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* UNIQUE CAPABILITIES Section */}
        <section className="py-12 border-t border-acid-green/20">
          <div className="container mx-auto px-4">
            <div className="text-center mb-8">
              <p className="text-acid-green/50 font-mono text-xs mb-2">{'═'.repeat(30)}</p>
              <h2 className="text-acid-green font-mono text-lg">{'>'} UNIQUE CAPABILITIES</h2>
              <p className="text-acid-green/50 font-mono text-xs mt-2">{'═'.repeat(30)}</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-6xl mx-auto">
              {/* ELO Rankings */}
              <div className="border border-acid-green/40 p-5 bg-surface/50">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-acid-yellow text-lg">{'#'}</span>
                  <h3 className="text-acid-green font-mono text-sm">ELO RANKINGS</h3>
                </div>
                <p className="text-text-muted text-xs font-mono leading-relaxed mb-3">
                  Agents earn reputation through debate performance.
                  Domain-specific ratings: security, architecture, testing.
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
                    <span>FAST</span><span className="text-acid-green">1 hour</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>MEDIUM</span><span className="text-acid-green">1 day</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>SLOW</span><span className="text-acid-green">1 week</span>
                  </div>
                  <div className="flex justify-between text-text-muted">
                    <span>GLACIAL</span><span className="text-acid-green">1 month</span>
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
                  Z3 and Lean backends for provable correctness.
                  When persuasion isn&apos;t enough, demand proof.
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
                  Aragora improves itself through autonomous debate cycles:
                </p>
                <pre className="text-acid-green/80 text-xs font-mono mb-3 overflow-x-auto">
{`CONTEXT → DEBATE → DESIGN → IMPLEMENT → VERIFY → COMMIT
                      ↑__________________________|`}
                </pre>
                <p className="text-text-muted text-xs font-mono mb-2">
                  Protected files checksummed. Automatic rollback on failure.
                </p>
                <p className="text-acid-green/80 text-xs font-mono font-bold">
                  The only AI debate system that evolves its own code.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* WHY TRUST ARAGORA Section */}
        <section className="py-12 border-t border-acid-green/20">
          <div className="container mx-auto px-4">
            <div className="text-center mb-8">
              <p className="text-acid-green/50 font-mono text-xs mb-2">{'═'.repeat(30)}</p>
              <h2 className="text-acid-green font-mono text-lg">{'>'} WHY TRUST ARAGORA?</h2>
              <p className="text-acid-green/50 font-mono text-xs mt-2">{'═'.repeat(30)}</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 max-w-5xl mx-auto">
              {/* Transparency */}
              <div className="border-l-2 border-acid-green pl-4 py-2">
                <h3 className="text-acid-green font-mono text-xs mb-2">TRANSPARENCY</h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  Every debate is fully auditable. See exact prompts, responses, critiques, and votes. No black boxes.
                </p>
              </div>

              {/* Dissent Preserved */}
              <div className="border-l-2 border-acid-cyan pl-4 py-2">
                <h3 className="text-acid-cyan font-mono text-xs mb-2">DISSENT PRESERVED</h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  Minority opinions recorded with full reasoning chains. Consensus doesn&apos;t mean unanimity was forced.
                </p>
              </div>

              {/* Evidence Chains */}
              <div className="border-l-2 border-acid-green pl-4 py-2">
                <h3 className="text-acid-green font-mono text-xs mb-2">EVIDENCE CHAINS</h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  Claims linked to supporting/refuting evidence. Citation grounding with scholarly rigor.
                </p>
              </div>

              {/* Track Records */}
              <div className="border-l-2 border-acid-cyan pl-4 py-2">
                <h3 className="text-acid-cyan font-mono text-xs mb-2">TRACK RECORDS</h3>
                <p className="text-text-muted text-xs font-mono leading-relaxed">
                  Agent personas built from verified debate outcomes. Not self-reported traits &mdash; empirical performance.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-auto">
          <div className="text-acid-green/50 mb-2">
            {'═'.repeat(40)}
          </div>
          <p className="text-acid-green font-bold mb-1">
            {'>'} ARAGORA // THE SELF-EVOLVING MULTI-AGENT DEBATE FRAMEWORK
          </p>
          <p className="text-text-muted/70 text-[11px] italic max-w-md mx-auto mb-4">
            &quot;The only AI system that debates its own improvements and implements them autonomously.&quot;
          </p>
          <div className="flex justify-center gap-4 text-text-muted/50 mb-4">
            <a
              href="https://live.aragora.ai"
              className="hover:text-acid-green transition-colors"
            >
              Live Dashboard
            </a>
            <span>|</span>
            <a
              href="https://github.com/aragora"
              className="hover:text-acid-green transition-colors"
            >
              GitHub
            </a>
            <span>|</span>
            <a
              href="/about"
              className="hover:text-acid-green transition-colors"
            >
              API Docs
            </a>
          </div>
          <div className="text-acid-green/50">
            {'═'.repeat(40)}
          </div>
        </footer>
      </main>
    </>
  );
}
