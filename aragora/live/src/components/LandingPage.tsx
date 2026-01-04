'use client';

import { useState, useCallback } from 'react';
import { DebateInput } from './DebateInput';
import { AsciiBannerCompact } from './AsciiBanner';
import { Scanlines, CRTVignette } from './MatrixRain';
import { ThemeToggle } from './ThemeToggle';

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
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <div className="flex-1 flex flex-col items-center justify-center px-4 py-16">
          {/* ASCII Art Title - compact size */}
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

          {/* Tagline */}
          <h1 className="text-2xl sm:text-3xl font-mono text-center mb-2">
            <span className="text-acid-green">Multi-Agent</span>{' '}
            <span className="text-text">Debate Arena</span>
          </h1>
          <p className="text-text-muted font-mono text-sm sm:text-base text-center mb-12 max-w-xl">
            AI agents debate your questions, critique each other, and reach consensus.
            Get answers validated through adversarial reasoning.
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

          {/* Features */}
          <div className="mt-16 grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-4xl">
            <div className="border border-acid-green/20 p-4 bg-surface/30">
              <h3 className="text-acid-green font-mono text-sm mb-2">[01] PROPOSE</h3>
              <p className="text-text-muted text-xs font-mono">
                Agents generate diverse solutions to your question
              </p>
            </div>
            <div className="border border-acid-green/20 p-4 bg-surface/30">
              <h3 className="text-acid-cyan font-mono text-sm mb-2">[02] CRITIQUE</h3>
              <p className="text-text-muted text-xs font-mono">
                Each proposal faces rigorous adversarial review
              </p>
            </div>
            <div className="border border-acid-green/20 p-4 bg-surface/30">
              <h3 className="text-acid-green font-mono text-sm mb-2">[03] CONSENSUS</h3>
              <p className="text-text-muted text-xs font-mono">
                Agents converge on the strongest answer
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20">
          <div className="text-acid-green/50 mb-2">
            {'═'.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // MULTI-AGENT DEBATE FRAMEWORK
          </p>
          <div className="mt-4 flex justify-center gap-4 text-text-muted/50">
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
          </div>
          <div className="text-acid-green/50 mt-4">
            {'═'.repeat(40)}
          </div>
        </footer>
      </main>
    </>
  );
}
