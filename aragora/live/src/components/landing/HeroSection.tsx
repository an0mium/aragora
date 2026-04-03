'use client';

import { useState, useRef, useEffect, useCallback, type FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { useTheme } from '@/context/ThemeContext';
import { DebateResultPreview, RETURN_URL_KEY, PENDING_DEBATE_KEY, type DebateResponse } from '../DebateResultPreview';
import { getCurrentReturnUrl, normalizeReturnUrl } from '@/utils/returnUrl';
import { useBackend, BACKENDS } from '../BackendSelector';
import { DebateInput } from '../DebateInput';
import { ConnectOpenRouterButton } from '../openrouter/ConnectOpenRouterButton';
import type { HeroSectionProps } from './types';
import {
  prepareLandingDebate,
  type LandingDebatePreflight,
  type LandingPreparedDebateOption,
} from './landingPreflight';
import { trackLandingEvent } from './landingTelemetry';

const ASCII_BANNER = `    \u2584\u2584\u2584       \u2588\u2588\u2580\u2588\u2588\u2588   \u2584\u2584\u2584        \u2584\u2588\u2588\u2588\u2588  \u2592\u2588\u2588\u2588\u2588\u2588   \u2588\u2588\u2580\u2588\u2588\u2588   \u2584\u2584\u2584
   \u2592\u2588\u2588\u2588\u2588\u2584    \u2593\u2588\u2588 \u2592 \u2588\u2588\u2592\u2592\u2588\u2588\u2588\u2588\u2584     \u2588\u2588\u2592 \u2580\u2588\u2592\u2592\u2588\u2588\u2592  \u2588\u2588\u2592\u2593\u2588\u2588 \u2592 \u2588\u2588\u2592\u2592\u2588\u2588\u2588\u2588\u2584
   \u2592\u2588\u2588  \u2580\u2588\u2584  \u2593\u2588\u2588 \u2591\u2584\u2588 \u2592\u2592\u2588\u2588  \u2580\u2588\u2584  \u2592\u2588\u2588\u2591\u2584\u2584\u2584\u2591\u2592\u2588\u2588\u2591  \u2588\u2588\u2592\u2593\u2588\u2588 \u2591\u2584\u2588 \u2592\u2592\u2588\u2588  \u2580\u2588\u2584
   \u2591\u2588\u2588\u2584\u2584\u2584\u2584\u2588\u2588 \u2592\u2588\u2588\u2580\u2580\u2588\u2584  \u2591\u2588\u2588\u2584\u2584\u2584\u2584\u2588\u2588 \u2591\u2593\u2588  \u2588\u2588\u2593\u2592\u2588\u2588   \u2588\u2588\u2591\u2592\u2588\u2588\u2580\u2580\u2588\u2584  \u2591\u2588\u2588\u2584\u2584\u2584\u2584\u2588\u2588
    \u2593\u2588   \u2593\u2588\u2588\u2592\u2591\u2588\u2588\u2593 \u2592\u2588\u2588\u2592 \u2593\u2588   \u2593\u2588\u2588\u2592\u2591\u2592\u2593\u2588\u2588\u2588\u2580\u2592\u2591 \u2588\u2588\u2588\u2588\u2593\u2592\u2591\u2591\u2588\u2588\u2593 \u2592\u2588\u2588\u2592 \u2593\u2588   \u2593\u2588\u2588\u2592
    \u2592\u2592   \u2593\u2592\u2588\u2591\u2591 \u2592\u2593 \u2591\u2592\u2593\u2591 \u2592\u2592   \u2593\u2592\u2588\u2591 \u2591\u2592   \u2592 \u2591 \u2592\u2591\u2592\u2591\u2592\u2591 \u2591 \u2592\u2593 \u2591\u2592\u2593\u2591 \u2592\u2592   \u2593\u2592\u2588\u2591
     \u2592   \u2592\u2592 \u2591  \u2591\u2592 \u2591 \u2592\u2591  \u2592   \u2592\u2592 \u2591  \u2591   \u2591   \u2591 \u2592 \u2592\u2591   \u2591\u2592 \u2591 \u2592\u2591  \u2592   \u2592\u2592 \u2591
     \u2591   \u2592     \u2591\u2591   \u2591   \u2591   \u2592   \u2591 \u2591   \u2591 \u2591 \u2591 \u2591 \u2592    \u2591\u2591   \u2591   \u2591   \u2592
         \u2591  \u2591   \u2591           \u2591  \u2591      \u2591     \u2591 \u2591     \u2591           \u2591  \u2591`;

const DEBATE_PHASES = [
  { label: 'Assembling panel', agents: ['Claude', 'GPT-4', 'Gemini'], duration: 3000 },
  { label: 'Opening arguments', agents: ['Claude', 'GPT-4', 'Gemini'], duration: 5000 },
  { label: 'Cross-examination', agents: ['GPT-4', 'Claude'], duration: 4000 },
  { label: 'Building consensus', agents: ['Gemini', 'Claude', 'GPT-4'], duration: 4000 },
  { label: 'Rendering verdict', agents: [], duration: 3000 },
];

const AGENT_DOT_COLORS: Record<string, string> = {
  'Claude': 'var(--acid-cyan, #00e5ff)',
  'GPT-4': 'var(--acid-green, #39ff14)',
  'Gemini': 'var(--acid-magenta, #ff00ff)',
  'Mistral': 'var(--acid-yellow, #ffd700)',
};

function parseRetryAfterSeconds(retryAfter: string | null): number {
  if (!retryAfter) return 60;

  const deltaSeconds = Number.parseInt(retryAfter, 10);
  if (Number.isFinite(deltaSeconds) && deltaSeconds >= 0) {
    return deltaSeconds;
  }

  const retryTime = Date.parse(retryAfter);
  if (Number.isNaN(retryTime)) return 60;

  return Math.max(1, Math.ceil((retryTime - Date.now()) / 1000));
}

function buildLandingErrorMessage(status: number, data: Record<string, unknown> | null): string {
  const message = typeof data?.message === 'string' ? data.message.trim() : '';
  const error = typeof data?.error === 'string' ? data.error.trim() : '';
  const code = typeof data?.code === 'string' ? data.code : '';
  const timeoutSeconds =
    typeof data?.timeout_seconds === 'number' ? Math.round(data.timeout_seconds) : null;

  if (code === 'request_timeout' && timeoutSeconds !== null) {
    return `Request timed out after ${timeoutSeconds}s. The landing page works best with one focused question. Shorten the prompt, pick one interpretation, or retry with a narrower scope.`;
  }

  if (code === 'landing_preview_timeout') {
    if (message) return message;
    if (timeoutSeconds !== null) {
      return `The landing preview timed out after ${timeoutSeconds}s. Shorten the prompt or pick one interpretation first.`;
    }
    return error || 'The landing preview timed out before the models returned a clean result.';
  }

  if (code === 'landing_preview_needs_clarification') {
    return (
      message
      || error
      || 'The fast preview drifted away from your question. Tighten the wording or pick one interpretation first.'
    );
  }

  if (code === 'timeout') {
    return message || error || 'The live debate timed out. Try a shorter, more focused question.';
  }

  if (message) return message;
  if (error) return error;
  return `Something went wrong (${status}). Please try again.`;
}

/**
 * HeroSection supports two modes:
 * - Landing mode (no props): self-contained debate form with tri-theme styling
 * - Dashboard mode (with apiBase prop): full DebateInput with auth-gated functionality
 */
const DEMO_TOPIC = 'Should we migrate our monolithic app to microservices?';

export function HeroSection(props: Partial<HeroSectionProps> & Record<string, unknown> = {}) {
  const isDashboardMode = 'apiBase' in props && props.apiBase;
  const { theme } = useTheme();
  const isDark = theme === 'dark';
  const router = useRouter();

  // All hooks must be called before any early return (Rules of Hooks)
  const [question, setQuestion] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [isDemoRunning, setIsDemoRunning] = useState(false);
  const [result, setResult] = useState<DebateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editorNotice, setEditorNotice] = useState<string | null>(null);
  const [lastTopic, setLastTopic] = useState('');
  const [lastPreparedOption, setLastPreparedOption] = useState<LandingPreparedDebateOption | null>(null);
  const [pendingPreflight, setPendingPreflight] = useState<LandingDebatePreflight | null>(null);
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [shareCopied, setShareCopied] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const startTimeRef = useRef<number>(0);
  const resultRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Phase progression during debate
  useEffect(() => {
    if (!isRunning) {
      setPhaseIndex(0);
      setElapsed(0);
      return;
    }
    startTimeRef.current = Date.now();
    let cumulative = 0;
    const timeouts: ReturnType<typeof setTimeout>[] = [];
    DEBATE_PHASES.forEach((phase, i) => {
      if (i > 0) {
        cumulative += DEBATE_PHASES[i - 1].duration;
        timeouts.push(setTimeout(() => setPhaseIndex(i), cumulative));
      }
    });
    const ticker = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
    return () => {
      timeouts.forEach(clearTimeout);
      clearInterval(ticker);
    };
  }, [isRunning]);
  // Cycling placeholder examples
  const PLACEHOLDER_EXAMPLES = [
    'Should we migrate to microservices or keep our monolith?',
    'Is this contract clause a liability risk?',
    'Should we raise prices 15% or expand to a new market?',
    'What are the security risks in our OAuth implementation?',
    'Should we build or buy our analytics platform?',
  ];
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const cycleTimer = useRef<ReturnType<typeof setInterval>>(null);
  const cyclePlaceholder = useCallback(() => {
    setPlaceholderIdx((i) => (i + 1) % PLACEHOLDER_EXAMPLES.length);
  }, [PLACEHOLDER_EXAMPLES.length]);

  useEffect(() => {
    if (question || isRunning) return; // stop cycling when user types
    cycleTimer.current = setInterval(cyclePlaceholder, 3500);
    return () => { if (cycleTimer.current) clearInterval(cycleTimer.current); };
  }, [question, isRunning, cyclePlaceholder]);

  // Scroll to results when they appear
  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [result]);

  const { config: backendConfig } = useBackend();
  const apiBase =
    isDashboardMode
      ? (props.apiBase as string | undefined) ?? BACKENDS.production.api
      : backendConfig.api;
  const playgroundDebateUrl =
    apiBase === ''
      ? '/api/v1/playground/debate/'
      : `${apiBase}/api/v1/playground/debate`;
  const trackEvent = useCallback((
    eventType: Parameters<typeof trackLandingEvent>[1],
    data: Parameters<typeof trackLandingEvent>[2] = {},
  ) => {
    trackLandingEvent(apiBase, eventType, data);
  }, [apiBase]);
  const focusComposer = useCallback(() => {
    const focus = () => {
      textareaRef.current?.focus();
      textareaRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'center' });
    };
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(focus);
      return;
    }
    setTimeout(focus, 0);
  }, []);

  // Dashboard mode — preserves original behavior from old HeroSection
  if (isDashboardMode) {
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

        {props.error && (
          <div className="w-full max-w-3xl mb-6 bg-warning/10 border border-warning/30 p-4 flex items-center justify-between">
            <span className="text-warning font-mono text-sm">
              {(props.error as string).toLowerCase().includes('authentication') || (props.error as string).toLowerCase().includes('unauthorized') ? (
                <>
                  Please{' '}
                  <a href="/login" className="underline hover:text-warning/80 font-bold">
                    Log In
                  </a>
                  {' '}to start debating with real AI models.
                </>
              ) : (
                props.error as string
              )}
            </span>
            <button
              onClick={props.onDismissError as (() => void) | undefined}
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
            <p className="text-text font-mono text-sm truncate">{props.activeQuestion as string}</p>
            <p className="text-text-muted font-mono text-xs mt-2">
              ID: {props.activeDebateId as string} | Events streaming via WebSocket
            </p>
          </div>
        )}

        <DebateInput
          apiBase={props.apiBase as string}
          onDebateStarted={props.onDebateStarted as ((debateId: string, question: string) => void) | undefined}
          onError={props.onError as ((err: string) => void) | undefined}
        />
      </div>
    );
  }

  // Landing mode — self-contained debate form with tri-theme styling

  function saveDebateBeforeLogin() {
    if (result) {
      sessionStorage.setItem(PENDING_DEBATE_KEY, JSON.stringify(result));
      const debateDestination = result.id ? `/debates/${encodeURIComponent(result.id)}` : getCurrentReturnUrl();
      sessionStorage.setItem(RETURN_URL_KEY, normalizeReturnUrl(debateDestination));
    }
  }

  async function executeDebate(option: LandingPreparedDebateOption) {
    setIsRunning(true);
    setError(null);
    setEditorNotice(null);
    setResult(null);
    setPendingPreflight(null);
    setLastTopic(option.originalQuestion);
    setLastPreparedOption(option);

    trackEvent('preflight_selected', {
      option_id: option.id,
      recommended: Boolean(option.recommended),
      rewritten: option.interpretedQuestion !== option.originalQuestion,
      agents: option.agents,
      rounds: option.rounds,
      question_length: option.originalQuestion.length,
    });

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(playgroundDebateUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: option.debatePrompt,
          question: option.debatePrompt,
          original_question: option.originalQuestion,
          interpreted_question: option.interpretedQuestion,
          rounds: option.rounds,
          agents: option.agents,
          source: 'landing',
        }),
        signal: controller.signal,
      });

      if (res.status === 429) {
        const retryAfter = parseRetryAfterSeconds(res.headers.get('Retry-After'));
        const waitText = retryAfter > 60 ? `${Math.ceil(retryAfter / 60)} minutes` : `${retryAfter} seconds`;
        setError(`Rate limit reached. Please try again in ${waitText}.`);
        return;
      }

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        const code = typeof data?.code === 'string' ? data.code : '';
        if (code === 'landing_preview_timeout') {
          trackEvent('preview_timeout', {
            timeout_seconds:
              typeof data?.timeout_seconds === 'number' ? Math.round(data.timeout_seconds) : null,
            rewritten: option.interpretedQuestion !== option.originalQuestion,
            question_length: option.originalQuestion.length,
          });
        } else if (code === 'landing_preview_needs_clarification') {
          trackEvent('preview_clarification_requested', {
            rewritten: option.interpretedQuestion !== option.originalQuestion,
            question_length: option.originalQuestion.length,
          });
        }
        setError(buildLandingErrorMessage(res.status, data));
        return;
      }

      const data: DebateResponse = await res.json();
      const nextResult: DebateResponse = {
        ...data,
        original_question: option.originalQuestion,
        interpreted_question: option.interpretedQuestion,
        result_warning:
          data.result_warning
          || (option.interpretedQuestion !== option.originalQuestion
            ? 'Aragora debated the focused interpretation you chose before opening the full transcript.'
            : undefined),
      };
      trackEvent('preview_rendered', {
        result_mode: nextResult.result_mode || 'full',
        rewritten: option.interpretedQuestion !== option.originalQuestion,
        participant_count: nextResult.participants.length,
        has_warning: Boolean(nextResult.result_warning),
      });
      setResult(nextResult);
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      setError('Could not connect to the server. Check your connection and try again.');
    } finally {
      setIsRunning(false);
    }
  }

  function runDebate(rawQuestion: string) {
    const preflight = prepareLandingDebate(rawQuestion);
    setError(null);
    setEditorNotice(null);
    setResult(null);
    setLastTopic(rawQuestion);

    if (preflight.type === 'confirm') {
      setPendingPreflight(preflight.preflight);
      trackEvent('preflight_shown', {
        option_count: preflight.preflight.options.length,
        recommended_count: preflight.preflight.options.filter((option) => option.recommended).length,
        has_warning: Boolean(preflight.preflight.warning),
        question_length: rawQuestion.length,
      });
      return;
    }

    setPendingPreflight(null);
    void executeDebate(preflight.option);
  }

  async function runDemoDebate() {
    setIsDemoRunning(true);
    setIsRunning(true);
    setError(null);
    setEditorNotice(null);
    setResult(null);
    setPendingPreflight(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(playgroundDebateUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: DEMO_TOPIC, question: DEMO_TOPIC, rounds: 2, agents: 3, source: 'demo' }),
        signal: controller.signal,
      });

      if (res.status === 429) {
        const data = await res.json().catch(() => null);
        const retryAfter = data?.retry_after || 60;
        setError(`Rate limit reached. Please try again in ${retryAfter} seconds.`);
        return;
      }

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        setError(data?.error || 'Something went wrong. Please try again.');
        return;
      }

      const data: DebateResponse = await res.json();
      if (data.id) {
        router.push(`/debate/${data.id}`);
      } else {
        // Fallback: show inline if no ID returned
        setResult(data);
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      setError('Could not connect to the server. Check your connection and try again.');
    } finally {
      setIsRunning(false);
      setIsDemoRunning(false);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (question.trim()) {
      runDebate(question.trim());
    }
  }

  const handleWrongAnswer = useCallback((currentResult: DebateResponse) => {
    const sourceQuestion =
      currentResult.original_question
      || question
      || lastTopic
      || currentResult.topic;
    const preflight = prepareLandingDebate(sourceQuestion);

    setQuestion(sourceQuestion);
    setResult(null);
    setError(null);
    setLastTopic(sourceQuestion);
    setLastPreparedOption(null);

    if (preflight.type === 'confirm') {
      setPendingPreflight(preflight.preflight);
      setEditorNotice('Pick a narrower interpretation or edit the wording below before rerunning.');
      trackEvent('preflight_shown', {
        option_count: preflight.preflight.options.length,
        recommended_count: preflight.preflight.options.filter((option) => option.recommended).length,
        has_warning: Boolean(preflight.preflight.warning),
        question_length: sourceQuestion.length,
      });
    } else {
      setPendingPreflight(null);
      setEditorNotice('Edit the wording below and rerun the debate with one more specific detail.');
    }

    trackEvent('wrong_answer_clicked', {
      result_mode: currentResult.result_mode || 'full',
      rewritten:
        Boolean(currentResult.interpreted_question)
        && currentResult.interpreted_question !== (currentResult.original_question || currentResult.topic),
    });
    focusComposer();
  }, [focusComposer, lastTopic, question, trackEvent]);

  // Keep the saveDebateBeforeLogin available for external use (not currently wired but preserving)
  void saveDebateBeforeLogin;

  return (
    <section
      className="relative px-4 flex flex-col items-center justify-center"
      style={{
        minHeight: 'calc(100vh - 52px)',
        fontFamily: 'var(--font-landing)',
      }}
    >
      {/* CRT scanline overlay — dark theme only */}
      {isDark && (
        <div
          className="pointer-events-none fixed inset-0 z-[9999]"
          style={{
            background: 'var(--scanline)',
            opacity: 0.03,
          }}
        />
      )}

      <div className="max-w-xl mx-auto text-center w-full">
        {/* Mobile-only brand text (ASCII banner is hidden on small screens) */}
        <div className="block sm:hidden text-center mb-4">
          <span className="text-[var(--acid-green)] font-mono font-bold text-2xl tracking-[0.3em]">ARAGORA</span>
        </div>

        {/* ASCII banner — dark theme only, desktop */}
        {isDark && (
          <pre
            className="text-[6px] sm:text-[7px] text-center mb-10 hidden sm:block leading-tight"
            style={{ color: 'var(--accent)', fontFamily: "'JetBrains Mono', monospace" }}
          >
            {ASCII_BANNER}
          </pre>
        )}

        {/* Headline */}
        <h1
          className="leading-tight"
          style={{
            fontSize: isDark ? '38px' : '44px',
            fontWeight: isDark ? 700 : 400,
            color: 'var(--text)',
            fontFamily: 'var(--font-display, var(--font-landing))',
            marginBottom: '16px',
            letterSpacing: isDark ? '0' : '-0.02em',
          }}
        >
          Don&apos;t trust one AI.
          <br />
          <span
            style={{
              color: 'var(--accent)',
              textShadow: isDark ? '0 0 10px var(--accent), 0 0 20px var(--accent)' : 'none',
            }}
          >
            Make them compete.
          </span>
        </h1>

        {/* Subtitle */}
        <p
          className="mx-auto leading-relaxed"
          style={{
            fontSize: '14px',
            color: 'var(--text-muted)',
            fontFamily: 'var(--font-landing)',
            marginBottom: '48px',
          }}
        >
          Multiple AI models debate your question and deliver an audit-ready verdict.
        </p>

        {/* Debate input form */}
        <form onSubmit={handleSubmit} className="text-left">
          <div className="relative">
            {isDark && (
              <span
                className="absolute left-4 top-5 text-base select-none"
                style={{ color: 'var(--accent)', fontFamily: "'JetBrains Mono', monospace" }}
              >
                &gt;
              </span>
            )}
            <textarea
              ref={textareaRef}
              value={question}
              onChange={(e) => {
                setQuestion(e.target.value);
                if (editorNotice) setEditorNotice(null);
              }}
              placeholder={PLACEHOLDER_EXAMPLES[placeholderIdx]}
              disabled={isRunning}
              rows={3}
              className="w-full placeholder:opacity-40 focus:outline-none transition-all resize-none disabled:opacity-50"
              style={{
                backgroundColor: 'var(--surface)',
                border: '2px solid var(--border)',
                color: 'var(--text)',
                fontFamily: 'var(--font-landing)',
                fontSize: '16px',
                lineHeight: '1.6',
                borderRadius: 'var(--radius-input)',
                padding: isDark ? '18px 20px 18px 36px' : '18px 20px',
                boxShadow: isDark ? 'none' : 'var(--shadow-card-hover)',
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = 'var(--accent)';
                e.currentTarget.style.boxShadow = isDark
                  ? '0 0 0 1px var(--accent), 0 0 20px var(--accent-glow)'
                  : '0 0 0 3px var(--accent-glow), var(--shadow-card-hover)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = 'var(--border)';
                e.currentTarget.style.boxShadow = isDark ? 'none' : 'var(--shadow-card-hover)';
              }}
            />
          </div>
          <button
            type="submit"
            disabled={isRunning || !question.trim()}
            className="w-full text-sm font-bold transition-all hover:scale-[1.01] active:scale-[0.99] disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            style={{
              backgroundColor: 'var(--accent)',
              color: 'var(--bg)',
              fontFamily: 'var(--font-landing)',
              fontSize: '15px',
              borderRadius: 'var(--radius-button)',
              padding: '16px 32px',
              marginTop: '12px',
              boxShadow: isDark ? '0 0 20px var(--accent-glow)' : '0 2px 8px var(--accent-glow)',
            }}
          >
            {isRunning && !isDemoRunning ? 'Agents debating...' : isDark ? '> Start Debate' : 'Start Debate'}
          </button>
        </form>

        {editorNotice && (
          <div
            className="mt-4 text-left"
            style={{
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-landing)',
              fontSize: '13px',
            }}
          >
            {editorNotice}
          </div>
        )}

        {pendingPreflight && !isRunning && !result && (
          <div
            className="mt-6 text-left"
            style={{
              backgroundColor: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-card)',
              padding: '20px',
            }}
          >
            <h2
              className="mb-2"
              style={{
                color: 'var(--accent)',
                fontFamily: 'var(--font-display, var(--font-landing))',
                fontSize: '18px',
                fontWeight: 600,
              }}
            >
              {pendingPreflight.title}
            </h2>
            <p
              className="mb-4"
              style={{
                color: 'var(--text)',
                fontFamily: 'var(--font-landing)',
                fontSize: '14px',
                lineHeight: 1.6,
              }}
            >
              {pendingPreflight.prompt}
            </p>
            {pendingPreflight.warning && (
              <p
                className="mb-4"
                style={{
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-landing)',
                  fontSize: '12px',
                  lineHeight: 1.5,
                }}
              >
                {pendingPreflight.warning}
              </p>
            )}
            <div className="space-y-3">
              {pendingPreflight.options.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => { void executeDebate(option); }}
                  className="w-full text-left transition-all hover:opacity-90 cursor-pointer"
                  style={{
                    backgroundColor: 'transparent',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-button)',
                    padding: '16px',
                  }}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span
                      style={{
                        color: 'var(--text)',
                        fontFamily: 'var(--font-landing)',
                        fontSize: '14px',
                        fontWeight: 600,
                      }}
                    >
                      {option.label}
                    </span>
                    {option.recommended && (
                      <span
                        style={{
                          color: 'var(--accent)',
                          fontFamily: 'var(--font-landing)',
                          fontSize: '11px',
                          fontWeight: 700,
                          letterSpacing: '0.08em',
                          textTransform: 'uppercase',
                        }}
                      >
                        Recommended
                      </span>
                    )}
                  </div>
                  <p
                    className="mt-2"
                    style={{
                      color: 'var(--text-muted)',
                      fontFamily: 'var(--font-landing)',
                      fontSize: '13px',
                      lineHeight: 1.5,
                    }}
                  >
                    {option.description}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Demo CTA — runs a preset topic against the real backend, no typing needed */}
        {!isRunning && !result && !pendingPreflight && (
          <button
            onClick={runDemoDebate}
            className="w-full text-sm font-bold transition-all hover:scale-[1.01] active:scale-[0.99] cursor-pointer"
            style={{
              backgroundColor: 'transparent',
              color: 'var(--accent)',
              fontFamily: 'var(--font-landing)',
              fontSize: '14px',
              borderRadius: 'var(--radius-button)',
              padding: '14px 32px',
              marginTop: '8px',
              border: '1px solid var(--accent)',
            }}
          >
            {isDark ? '> Try a Demo Debate' : 'Try a Demo Debate'}
          </button>
        )}
        {!isRunning && !result && !pendingPreflight && (
          <p
            className="text-center"
            style={{
              fontSize: '12px',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-landing)',
              marginTop: '8px',
              opacity: 0.6,
            }}
          >
            No account needed -- watch AI agents debate a real question
          </p>
        )}
        {!isRunning && !result && !pendingPreflight && (
          <div className="text-center mt-4">
            <ConnectOpenRouterButton compact />
          </div>
        )}

        {/* Loading state — phased progress */}
        {isRunning && (
          <div className="mt-8 max-w-xl mx-auto">
            {isDemoRunning && (
              <p
                className="text-center mb-3"
                style={{
                  fontSize: '13px',
                  color: 'var(--accent)',
                  fontFamily: 'var(--font-landing)',
                }}
              >
                Debating: &quot;{DEMO_TOPIC}&quot;
              </p>
            )}
            <div
              className="p-5 text-left"
              style={{
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-card, 8px)',
                backgroundColor: 'var(--surface)',
              }}
            >
              {/* Phase steps */}
              <div className="space-y-3 mb-4">
                {DEBATE_PHASES.map((phase, i) => {
                  const isActive = i === phaseIndex;
                  const isDone = i < phaseIndex;
                  return (
                    <div
                      key={i}
                      className="flex items-center gap-3 transition-opacity duration-300"
                      style={{ opacity: isDone ? 0.4 : isActive ? 1 : 0.25 }}
                    >
                      {/* Step indicator */}
                      <div
                        className="w-6 h-6 flex items-center justify-center shrink-0 text-xs font-bold"
                        style={{
                          borderRadius: '50%',
                          border: `2px solid ${isActive ? 'var(--accent)' : isDone ? 'var(--accent)' : 'var(--border)'}`,
                          color: isActive || isDone ? 'var(--accent)' : 'var(--text-muted)',
                          backgroundColor: isDone ? 'var(--accent)' : 'transparent',
                          ...(isDone ? { color: 'var(--bg)' } : {}),
                          fontFamily: 'var(--font-landing)',
                        }}
                      >
                        {isDone ? '\u2713' : i + 1}
                      </div>
                      {/* Label + agents */}
                      <div className="flex-1 min-w-0">
                        <span
                          className="text-sm font-medium"
                          style={{
                            color: isActive ? 'var(--text)' : 'var(--text-muted)',
                            fontFamily: 'var(--font-landing)',
                          }}
                        >
                          {phase.label}
                        </span>
                        {isActive && phase.agents.length > 0 && (
                          <div className="flex items-center gap-2 mt-1">
                            {phase.agents.map((agent) => (
                              <span
                                key={agent}
                                className="flex items-center gap-1 text-xs"
                                style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-landing)' }}
                              >
                                <span
                                  className="w-2 h-2 rounded-full inline-block animate-pulse"
                                  style={{ backgroundColor: AGENT_DOT_COLORS[agent] || 'var(--accent)' }}
                                />
                                {agent}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      {/* Active spinner */}
                      {isActive && (
                        <svg className="animate-spin h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" style={{ color: 'var(--accent)' }}>
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                      )}
                    </div>
                  );
                })}
              </div>
              {/* Progress bar */}
              <div
                className="h-1 rounded-full overflow-hidden"
                style={{ backgroundColor: 'var(--border)' }}
              >
                <div
                  className="h-full rounded-full transition-all duration-1000 ease-out"
                  style={{
                    backgroundColor: 'var(--accent)',
                    width: `${Math.min(((phaseIndex + 1) / DEBATE_PHASES.length) * 100, 100)}%`,
                    boxShadow: isDark ? '0 0 8px var(--accent-glow)' : 'none',
                  }}
                />
              </div>
              <div
                className="flex justify-between mt-2 text-xs"
                style={{ color: 'var(--text-muted)', opacity: 0.6, fontFamily: 'var(--font-landing)' }}
              >
                <span>{elapsed}s elapsed</span>
                <span>~15s remaining</span>
              </div>
            </div>
          </div>
        )}

        {/* Error state */}
        {error && (
          <div
            className="mt-6 text-left max-w-xl mx-auto"
            style={{
              padding: '20px 24px',
              border: '1px solid var(--crimson)',
              borderRadius: 'var(--radius-card)',
              backgroundColor: isDark ? 'rgba(255,0,64,0.05)' : 'rgba(163,59,59,0.05)',
            }}
          >
            <p className="text-sm mb-4" style={{ color: 'var(--crimson)', fontFamily: 'var(--font-landing)' }}>
              {error}
            </p>
            <button
              onClick={() => {
                trackEvent('retry_clicked', {
                  has_prepared_option: Boolean(lastPreparedOption),
                  has_question: Boolean(lastTopic || question.trim()),
                });
                setError(null);
                if (lastPreparedOption) {
                  void executeDebate(lastPreparedOption);
                  return;
                }
                if (lastTopic) {
                  runDebate(lastTopic);
                  return;
                }
                if (question.trim()) runDebate(question.trim());
              }}
              className="text-xs px-5 py-2.5 transition-colors hover:opacity-80 cursor-pointer"
              style={{
                fontFamily: 'var(--font-landing)',
                border: '1px solid var(--crimson)',
                borderRadius: 'var(--radius-button)',
                color: 'var(--crimson)',
                backgroundColor: 'transparent',
              }}
            >
              Try again
            </button>
          </div>
        )}

        {/* Result preview */}
        {result && (
          <div ref={resultRef}>
            <DebateResultPreview
              result={result}
              condensed
              onFlagWrongAnswer={handleWrongAnswer}
              onOpenFullDebate={(debateResult, surface) => {
                trackEvent('open_full_debate_clicked', {
                  result_mode: debateResult.result_mode || 'full',
                  surface,
                });
              }}
              onShare={(debateResult) => {
                trackEvent('share_clicked', {
                  result_mode: debateResult.result_mode || 'full',
                });
              }}
            />
          </div>
        )}

        {/* Post-debate CTAs */}
        {result && (
          <div className="mt-6 max-w-xl mx-auto space-y-3">
            {/* Primary: View full debate page */}
            {result.id && (
              <button
                onClick={() => {
                  trackEvent('open_full_debate_clicked', {
                    result_mode: result.result_mode || 'full',
                    surface: 'quick_read',
                  });
                  router.push(`/debate/${result.id}`);
                }}
                className="w-full text-sm font-bold font-mono py-3 transition-all hover:opacity-90 cursor-pointer"
                style={{
                  backgroundColor: 'var(--accent)',
                  color: 'var(--bg)',
                  borderRadius: 'var(--radius-button)',
                  boxShadow: isDark ? '0 0 20px var(--accent-glow)' : '0 2px 8px var(--accent-glow)',
                }}
              >
                {isDark ? '> View Full Debate' : 'View Full Debate'}
              </button>
            )}
            {/* Secondary row: Try Another + Share */}
            <div className="flex gap-3">
              <button
                onClick={() => {
                  setResult(null);
                  setQuestion('');
                  setError(null);
                  setEditorNotice(null);
                  setPendingPreflight(null);
                  setLastPreparedOption(null);
                  setLastTopic('');
                }}
                className="flex-1 text-sm font-bold font-mono py-3 transition-all hover:opacity-90 cursor-pointer"
                style={{
                  backgroundColor: result.id ? 'transparent' : 'var(--accent)',
                  color: result.id ? 'var(--accent)' : 'var(--bg)',
                  border: result.id ? '1px solid var(--accent)' : 'none',
                  borderRadius: 'var(--radius-button)',
                }}
              >
                Try Another
              </button>
              <button
                onClick={async () => {
                  const shareUrl = result.id
                    ? `${window.location.origin}/debate/${result.id}`
                    : window.location.href;
                  try {
                    await navigator.clipboard.writeText(shareUrl);
                  } catch {
                    const ta = document.createElement('textarea');
                    ta.value = shareUrl;
                    ta.style.position = 'fixed';
                    ta.style.opacity = '0';
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand('copy');
                    document.body.removeChild(ta);
                  }
                  setShareCopied(true);
                  setTimeout(() => setShareCopied(false), 2000);
                  trackEvent('share_clicked', {
                    result_mode: result.result_mode || 'full',
                  });
                }}
                className="flex-1 text-sm font-bold font-mono py-3 transition-all hover:opacity-80 cursor-pointer"
                style={{
                  backgroundColor: 'transparent',
                  color: 'var(--accent)',
                  border: '1px solid var(--accent)',
                  borderRadius: 'var(--radius-button)',
                  opacity: shareCopied ? 0.7 : 1,
                }}
              >
                {shareCopied ? 'Copied!' : 'Share'}
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
