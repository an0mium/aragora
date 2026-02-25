'use client';

import Link from 'next/link';
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

/**
 * HeroSection supports two modes:
 * - Landing mode (no props): Shows dual CTA buttons for "Try a Debate" and "Sign In"
 * - Dashboard mode (with props): Shows the full DebateInput with auth-gated functionality
 */
export function HeroSection(props: Partial<HeroSectionProps> & Record<string, unknown>) {
  const isDashboardMode = 'apiBase' in props && props.apiBase;

  return (
    <div className="flex flex-col items-center justify-center px-4 py-12 sm:py-16">
      <pre className="text-acid-green text-[6px] sm:text-[7px] font-mono text-center mb-6 hidden sm:block leading-tight">
        {ASCII_BANNER}
      </pre>

      <h1 className="text-base sm:text-2xl font-mono text-center mb-4 text-text">
        What decision should AI debate for you?
      </h1>

      <p className="text-acid-cyan font-mono text-xs sm:text-sm text-center mb-10 max-w-xl">
        Ask any question. Multiple AI models will argue every angle and deliver a verdict with confidence scores.
      </p>

      {isDashboardMode ? (
        /* Dashboard mode: full DebateInput with error handling and active debate status */
        <>
          {props.error && (
            <div className="w-full max-w-3xl mb-6 bg-warning/10 border border-warning/30 p-4 flex items-center justify-between">
              <span className="text-warning font-mono text-sm">
                {(props.error as string).toLowerCase().includes('authentication') || (props.error as string).toLowerCase().includes('unauthorized') ? (
                  <>
                    Please{' '}
                    <a
                      href="/login"
                      className="underline hover:text-warning/80 font-bold"
                    >
                      Log In
                    </a>
                    {' '}to start vetted decisionmaking.
                  </>
                ) : (
                  props.error
                )}
              </span>
              <button
                onClick={props.onDismissError}
                className="text-warning hover:text-warning/80"
                aria-label="Dismiss error"
              >
                x
              </button>
            </div>
          )}

          {props.activeDebateId && (
            <div className="w-full max-w-3xl mb-6 bg-acid-green/10 border border-acid-green/30 p-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="w-2 h-2 bg-acid-green rounded-full animate-pulse"></span>
                <span className="text-acid-green font-mono text-sm font-bold">DECISION IN PROGRESS</span>
              </div>
              <p className="text-text font-mono text-sm truncate">{props.activeQuestion}</p>
              <p className="text-text-muted font-mono text-xs mt-2">
                ID: {props.activeDebateId} | Events streaming via WebSocket
              </p>
            </div>
          )}

          <DebateInput
            apiBase={props.apiBase as string}
            onDebateStarted={props.onDebateStarted}
            onError={props.onError}
          />
        </>
      ) : (
        /* Landing mode: dual CTA buttons */
        <>
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <Link
              href="/playground"
              className="px-8 py-4 bg-acid-green text-bg font-mono font-bold text-lg
                         hover:bg-acid-green/80 transition-colors
                         flex items-center gap-3"
            >
              <span>[&gt;]</span>
              <span>TRY A FREE DEBATE</span>
            </Link>

            <Link
              href="/login"
              className="px-8 py-4 border-2 border-acid-cyan/60 text-acid-cyan font-mono font-bold text-sm
                         hover:border-acid-cyan hover:bg-acid-cyan/10 transition-colors
                         flex items-center gap-2"
            >
              <span>SIGN IN</span>
              <span className="text-acid-cyan/60">&rarr;</span>
            </Link>
          </div>

          <p className="text-text-muted/50 font-mono text-[10px] mt-4">
            No signup required to try a debate
          </p>
        </>
      )}
    </div>
  );
}
