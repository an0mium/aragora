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
  const [showLearnMore, setShowLearnMore] = useState(false);

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
            <AsciiBannerCompact connected={true} showAsciiArt={false} />
            <div className="flex items-center gap-4">
              <a
                href="/debates"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors hidden sm:block"
              >
                [DEBATES]
              </a>
              <a
                href="/agents"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors hidden sm:block"
              >
                [AGENTS]
              </a>
              <a
                href="/about"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [ABOUT]
              </a>
              <a
                href="https://api.aragora.ai/docs"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors hidden sm:block"
              >
                [API]
              </a>
              <a
                href="https://live.aragora.ai"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [LIVE]
              </a>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Hero Section - Clean and Minimal */}
        <div className="flex flex-col items-center justify-center px-4 py-16 sm:py-20">
          {/* ASCII Art Title */}
          <pre className="text-acid-green text-[6px] sm:text-[8px] font-mono text-center mb-8 hidden sm:block leading-tight select-none">
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

          {/* Mobile title */}
          <h1 className="sm:hidden text-2xl font-mono text-acid-green mb-4">ARAGORA</h1>

          {/* Value Proposition - Minimal */}
          <div className="text-center mb-8 max-w-2xl">
            <p className="text-sm font-mono text-text-muted/80 mb-4">
              Multi-agent AI platform for decision makers
            </p>
            <div className="flex flex-wrap justify-center gap-4 text-xs font-mono text-text-muted/70">
              <span className="flex items-center gap-1">
                <span className="text-acid-green">+</span> Diverse Frontier Models Forge Consensus
              </span>
              <span className="flex items-center gap-1">
                <span className="text-acid-green">+</span> Audit-ready receipts
              </span>
              <span className="flex items-center gap-1">
                <span className="text-acid-green">+</span> 5-15 min turnaround
              </span>
            </div>
          </div>

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
                ID: {activeDebateId}
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

        {/* Learn More Section - Single Parent Collapsible */}
        <section className="border-t border-acid-green/20 mt-auto">
          <div className="container mx-auto px-4">
            {/* Main Toggle */}
            <button
              onClick={() => setShowLearnMore(!showLearnMore)}
              className="w-full py-6 text-center group cursor-pointer"
              aria-expanded={showLearnMore}
              aria-controls="learn-more-content"
            >
              <div className="flex items-center justify-center gap-3">
                <span className="text-acid-green/40 font-mono text-xs">{'─'.repeat(12)}</span>
                <span className={`text-acid-green/60 text-xs transition-transform duration-200 ${showLearnMore ? 'rotate-90' : ''}`}>
                  {'>'}
                </span>
                <span className="text-acid-green/80 font-mono text-xs tracking-wider group-hover:text-acid-green transition-colors">
                  LEARN MORE
                </span>
                <span className="text-text-muted/50 text-[10px] font-mono">
                  {showLearnMore ? '[−]' : '[+]'}
                </span>
                <span className="text-acid-green/40 font-mono text-xs">{'─'.repeat(12)}</span>
              </div>
            </button>

            {/* Collapsible Content */}
            <div
              id="learn-more-content"
              className={`transition-all duration-300 ease-in-out overflow-hidden ${
                showLearnMore ? 'max-h-[8000px] opacity-100 pb-8' : 'max-h-0 opacity-0'
              }`}
            >

              {/* Use Case Highlight */}
              <div className="max-w-2xl mx-auto mb-8 text-center">
                <div className="border border-acid-yellow/30 bg-acid-yellow/5 p-4 rounded">
                  <p className="text-text-muted/50 font-mono text-[10px] tracking-wide mb-2">
                    <span className="text-acid-green/40">{'>'}</span>
                    {' '}Multiple AI models will adversarially debate your question
                    {' '}<span className="text-acid-green/40">{'<'}</span>
                  </p>
                  <p className="text-acid-yellow/80 font-mono text-[10px]">
                    <span className="text-acid-yellow">USE CASE:</span> Find the $500K flaw before launch — AI stress-tests your architecture in 30 minutes.
                  </p>
                </div>
              </div>

              {/* Intro Section */}
              <div className="max-w-4xl mx-auto mb-8 text-center">
                <h2 className="text-acid-green font-mono text-sm mb-2">
                  The Self-Evolving Red Team Arena
                </h2>
                <p className="text-acid-cyan/80 font-mono text-[11px] mb-3">
                  Claude vs GPT vs Gemini vs Mistral vs Grok vs DeepSeek vs Qwen vs Kimi
                </p>
                <p className="text-text-muted/70 font-mono text-[10px] max-w-lg mx-auto">
                  Real AI models. Adversarial reasoning. Decision receipts.
                </p>
              </div>

              {/* Why Aragora Grid */}
              <div className="max-w-4xl mx-auto mb-8">
                <p className="text-text-muted/60 font-mono text-[10px] text-center mb-4">
                  Most multi-agent systems run copies of the same model talking to itself. Aragora is different.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div className="border border-acid-green/20 p-3 bg-surface/20">
                    <h3 className="text-acid-green/90 font-mono text-[10px] mb-2 flex items-center gap-1">
                      <span className="text-acid-cyan/60">{'>'}</span> HETEROGENEOUS ARENA
                    </h3>
                    <p className="text-text-muted/60 text-[9px] font-mono leading-relaxed">
                      7+ distinct AI providers compete. Claude&apos;s caution vs GPT&apos;s creativity vs Gemini&apos;s speed. Real diversity. Real signal.
                    </p>
                  </div>
                  <div className="border border-acid-cyan/20 p-3 bg-surface/20">
                    <h3 className="text-acid-cyan/90 font-mono text-[10px] mb-2 flex items-center gap-1">
                      <span className="text-acid-green/60">{'>'}</span> SELF-IMPROVING
                    </h3>
                    <p className="text-text-muted/60 text-[9px] font-mono leading-relaxed">
                      Aragora runs the &quot;Nomic Loop&quot; — agents red-team improvements to their own framework. The arena evolves through its own debates.
                    </p>
                  </div>
                  <div className="border border-acid-green/20 p-3 bg-surface/20">
                    <h3 className="text-acid-green/90 font-mono text-[10px] mb-2 flex items-center gap-1">
                      <span className="text-acid-cyan/60">{'>'}</span> CALIBRATED TRUST
                    </h3>
                    <p className="text-text-muted/60 text-[9px] font-mono leading-relaxed">
                      Track prediction accuracy over time. Know which agents are confidently wrong vs genuinely uncertain.
                    </p>
                  </div>
                </div>
              </div>

              {/* Protocol Steps */}
              <div className="max-w-4xl mx-auto mb-8">
                <h3 className="text-acid-green/60 font-mono text-[10px] text-center mb-4 tracking-wider">
                  THE STRESS-TEST PROTOCOL
                </h3>
                <div className="flex flex-wrap justify-center gap-2">
                  {[
                    { num: '01', name: 'PROPOSE', desc: 'Generate diverse solutions' },
                    { num: '02', name: 'ATTACK', desc: 'Red-team each proposal' },
                    { num: '03', name: 'ADAPT', desc: 'Update on valid critiques' },
                    { num: '04', name: 'CONVERGE', desc: 'Find natural consensus' },
                    { num: '05', name: 'RECORD', desc: 'Save decision receipts' },
                  ].map((step, i) => (
                    <div key={step.num} className="flex items-center gap-2">
                      <div className="border border-acid-green/20 px-2 py-1 bg-surface/20">
                        <span className="text-acid-green/60 font-mono text-[9px]">[{step.num}]</span>
                        <span className="text-text-muted/70 font-mono text-[9px] ml-1">{step.name}</span>
                      </div>
                      {i < 4 && <span className="text-acid-green/30 text-[9px]">→</span>}
                    </div>
                  ))}
                </div>
              </div>

              {/* Capabilities Grid - Compact */}
              <div className="max-w-4xl mx-auto mb-8">
                <h3 className="text-acid-cyan/60 font-mono text-[10px] text-center mb-4 tracking-wider">
                  CAPABILITIES
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {[
                    { icon: '#', title: 'ELO RANKINGS', desc: 'Domain-specific ratings backed by data' },
                    { icon: '~', title: 'CONTINUUM MEMORY', desc: '4-tier cognitive memory system' },
                    { icon: '%', title: 'CALIBRATION', desc: 'Brier scores & confidence tracking' },
                    { icon: '!', title: 'FORMAL VERIFICATION', desc: 'Z3/Lean proofs for high stakes' },
                  ].map((cap) => (
                    <div key={cap.title} className="border border-acid-green/15 p-2 bg-surface/10">
                      <div className="flex items-center gap-1 mb-1">
                        <span className="text-acid-yellow/60 text-[10px]">{cap.icon}</span>
                        <span className="text-acid-green/70 font-mono text-[9px]">{cap.title}</span>
                      </div>
                      <p className="text-text-muted/50 text-[8px] font-mono leading-tight">{cap.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* The Nomic Loop - Compact */}
              <div className="max-w-3xl mx-auto mb-8">
                <div className="border border-acid-green/20 p-3 bg-surface/10">
                  <div className="flex items-center justify-center gap-2 mb-2">
                    <span className="text-acid-green/60 text-[10px]">@</span>
                    <span className="text-acid-green/80 font-mono text-[10px]">THE NOMIC LOOP</span>
                  </div>
                  <pre className="text-acid-green/50 text-[8px] font-mono text-center mb-2">
{`CONTEXT → DEBATE → DESIGN → IMPLEMENT → VERIFY → COMMIT
              ↑_______________________________|`}
                  </pre>
                  <p className="text-text-muted/50 text-[8px] font-mono text-center">
                    Protected files checksummed. Automatic rollback on failure.
                  </p>
                </div>
              </div>

              {/* Trust Section - Horizontal */}
              <div className="max-w-4xl mx-auto">
                <h3 className="text-acid-cyan/60 font-mono text-[10px] text-center mb-3 tracking-wider">
                  WHY TRUST ARAGORA
                </h3>
                <div className="flex flex-wrap justify-center gap-4">
                  {[
                    { title: 'TRANSPARENCY', desc: 'Fully auditable debates' },
                    { title: 'DISSENT PRESERVED', desc: 'Minority opinions recorded' },
                    { title: 'EVIDENCE CHAINS', desc: 'Citation grounding' },
                    { title: 'TRACK RECORDS', desc: 'Empirical performance' },
                  ].map((item, i) => (
                    <div key={item.title} className="flex items-center gap-2">
                      <span className={`w-px h-4 ${i % 2 === 0 ? 'bg-acid-green/30' : 'bg-acid-cyan/30'}`}></span>
                      <div>
                        <span className={`font-mono text-[9px] ${i % 2 === 0 ? 'text-acid-green/70' : 'text-acid-cyan/70'}`}>
                          {item.title}
                        </span>
                        <span className="text-text-muted/40 font-mono text-[8px] ml-1">{item.desc}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="text-center font-mono py-4 border-t border-acid-green/10">
          <div className="flex items-center justify-center gap-3 text-[10px] mb-2">
            <span className="text-acid-green/40">ARAGORA</span>
            <span className="text-acid-green/20">|</span>
            <a href="https://aragora.ai" className="text-text-muted/40 hover:text-acid-green/60 transition-colors">
              Dashboard
            </a>
            <span className="text-acid-green/20">|</span>
            <a href="https://github.com/an0mium/aragora" className="text-text-muted/40 hover:text-acid-green/60 transition-colors">
              GitHub
            </a>
            <span className="text-acid-green/20">|</span>
            <a href="/about" className="text-text-muted/40 hover:text-acid-green/60 transition-colors">
              Docs
            </a>
          </div>
        </footer>
      </main>
    </>
  );
}
