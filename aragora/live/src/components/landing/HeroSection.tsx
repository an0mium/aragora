'use client';

import { DebateInput } from '../DebateInput';
import type { HeroSectionProps } from './types';

const ASCII_BANNER = `    ▄▄▄       ██▀███   ▄▄▄        ▄████  ▒█████   ██▀███   ▄▄▄
   ▒████▄    ▓██ ▒ ██▒▒████▄     ██▒ ▀█▒▒██▒  ██▒▓██ ▒ ██▒▒████▄
   ▒██  ▀█▄  ▓██ ░▄█ ▒▒██  ▀█▄  ▒██░▄▄▄░▒██░  ██▒▓██ ░▄█ ▒▒██  ▀█▄
   ░██▄▄▄▄██ ▒██▀▀█▄  ░██▄▄▄▄██ ░▓█  ██▓▒██   ██░▒██▀▀█▄  ░██▄▄▄▄██
    ▓█   ▓██▒░██▓ ▒██▒ ▓█   ▓██▒░▒▓███▀▒░ ████▓▒░░██▓ ▒██▒ ▓█   ▓██▒
    ▒▒   ▓▒█░░ ▒▓ ░▒▓░ ▒▒   ▓▒█░ ░▒   ▒ ░ ▒░▒░▒░ ░ ▒▓ ░▒▓░ ▒▒   ▓▒█░
     ▒   ▒▒ ░  ░▒ ░ ▒░  ▒   ▒▒ ░  ░   ░   ░ ▒ ▒░   ░▒ ░ ▒░  ▒   ▒▒ ░
     ░   ▒     ░░   ░   ░   ▒   ░ ░   ░ ░ ░ ░ ▒    ░░   ░   ░   ▒
         ░  ░   ░           ░  ░      ░     ░ ░     ░           ░  ░`;

export function HeroSection({
  error,
  activeDebateId,
  activeQuestion,
  apiBase,
  onDismissError,
  onDebateStarted,
  onError,
}: HeroSectionProps) {
  return (
    <div className="flex flex-col items-center justify-center px-4 py-12 sm:py-16">
      <pre className="text-acid-green text-[6px] sm:text-[7px] font-mono text-center mb-6 hidden sm:block leading-tight">
        {ASCII_BANNER}
      </pre>

      <h1 className="text-2xl sm:text-4xl font-mono text-center mb-4 text-text">
        What should AI agents debate?
      </h1>

      <p className="text-acid-cyan font-mono text-sm sm:text-base text-center mb-10 max-w-xl">
        Harness Claude, GPT, Gemini, Grok & DeepSeek for collaborative decision making
      </p>

      {error && (
        <div className="w-full max-w-3xl mb-6 bg-warning/10 border border-warning/30 p-4 flex items-center justify-between">
          <span className="text-warning font-mono text-sm">{error}</span>
          <button
            onClick={onDismissError}
            className="text-warning hover:text-warning/80"
            aria-label="Dismiss error"
          >
            ✕
          </button>
        </div>
      )}

      {activeDebateId && (
        <div className="w-full max-w-3xl mb-6 bg-acid-green/10 border border-acid-green/30 p-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2 h-2 bg-acid-green rounded-full animate-pulse"></span>
            <span className="text-acid-green font-mono text-sm font-bold">DECISION IN PROGRESS</span>
          </div>
          <p className="text-text font-mono text-sm truncate">{activeQuestion}</p>
          <p className="text-text-muted font-mono text-xs mt-2">
            ID: {activeDebateId} | Events streaming via WebSocket
          </p>
        </div>
      )}

      <DebateInput apiBase={apiBase} onDebateStarted={onDebateStarted} onError={onError} />
    </div>
  );
}
