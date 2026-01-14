'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { DEFAULT_AGENTS, DEFAULT_ROUNDS, AGENT_DISPLAY_NAMES } from '@/config';

interface DebateInputProps {
  apiBase: string;
  onDebateStarted?: (debateId: string, question: string) => void;
  onError?: (error: string) => void;
}

type DebateMode = 'standard' | 'graph' | 'matrix';

const DEBATE_MODES: Record<DebateMode, { label: string; description: string; endpoint: string }> = {
  standard: {
    label: 'Standard',
    description: 'Linear debate with critique rounds',
    endpoint: '/api/debate',
  },
  graph: {
    label: 'Graph',
    description: 'Branching debate exploring multiple paths',
    endpoint: '/api/debates/graph',
  },
  matrix: {
    label: 'Matrix',
    description: 'Parallel scenarios for comparison',
    endpoint: '/api/debates/matrix',
  },
};

interface AgentRecommendation {
  agent: string;
  suitability: number;
  domain_match: boolean;
}

type ApiStatus = 'checking' | 'online' | 'offline';

// Simple domain detection from question text
function detectDomain(text: string): string {
  const lower = text.toLowerCase();
  if (/\b(code|programming|api|software|bug|function|class|typescript|javascript|python)\b/.test(lower)) {
    return 'technical';
  }
  if (/\b(security|auth|encryption|vulnerability|attack|password|token)\b/.test(lower)) {
    return 'security';
  }
  if (/\b(ethics|moral|right|wrong|should|fair|justice|harm)\b/.test(lower)) {
    return 'ethics';
  }
  if (/\b(ai|machine learning|model|neural|llm|gpt|claude|reasoning)\b/.test(lower)) {
    return 'ai';
  }
  if (/\b(architecture|design|pattern|microservice|monolith|scale|database)\b/.test(lower)) {
    return 'architecture';
  }
  return 'general';
}

export function DebateInput({ apiBase, onDebateStarted, onError }: DebateInputProps) {
  const router = useRouter();
  const [question, setQuestion] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [agents, setAgents] = useState(DEFAULT_AGENTS);
  const [rounds, setRounds] = useState(DEFAULT_ROUNDS);
  const [debateMode, setDebateMode] = useState<DebateMode>('standard');
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking');
  const [recommendations, setRecommendations] = useState<AgentRecommendation[]>([]);
  const [detectedDomain, setDetectedDomain] = useState<string>('general');
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  // Combined question pool: AI philosophy + Multi-agent debate + Technical architecture
  const allQuestions = [
    // AI Philosophy & Ethics
    'Can AI systems develop genuine understanding, or only simulate it?',
    'Should autonomous AI agents have the right to refuse unethical requests?',
    'Is Hegelian dialectics the optimal framework for AI reasoning?',
    'Will AI agents eventually form their own societies and cultures?',
    'Is truth something AI can discover, or only approximate?',

    // Multi-Agent Debate
    'Is consensus among AI agents more reliable than a single powerful model?',
    'Can adversarial debate eliminate AI hallucinations?',
    'Can competing AI perspectives reach genuine synthesis?',
    'What makes a good debate question for AI agents?',
    'How would you redesign democracy using multi-agent systems?',
    'When should a multi-agent system override human judgment?',
    'Should AI critics be more aggressive or more diplomatic?',

    // Technical Architecture
    'What are the tradeoffs between microservices and monoliths?',
    'How should we implement rate limiting for 1M requests/sec?',
    'Is GraphQL or REST better for real-time collaborative APIs?',
    'What authentication strategy balances security with UX?',
    'Should we use TypeScript or JavaScript for this project?',
  ];

  // Select random placeholder on mount (stable across re-renders)
  const [placeholder] = useState(
    () => allQuestions[Math.floor(Math.random() * allQuestions.length)]
  );

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        const response = await fetch(`${apiBase}/api/health`, {
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        setApiStatus(response.ok ? 'online' : 'offline');
      } catch {
        setApiStatus('offline');
      }
    };

    checkHealth();
    // Re-check every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [apiBase]);

  // Detect domain and fetch routing recommendations when question changes
  useEffect(() => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    if (!question.trim() || apiStatus !== 'online') {
      setRecommendations([]);
      setDetectedDomain('general');
      return;
    }

    debounceRef.current = setTimeout(async () => {
      const domain = detectDomain(question);
      setDetectedDomain(domain);

      // Only fetch recommendations for non-general domains
      if (domain !== 'general') {
        try {
          const response = await fetch(`${apiBase}/api/routing/recommendations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              primary_domain: domain,
              limit: 4,
            }),
          });
          if (response.ok) {
            const data = await response.json();
            setRecommendations(data.recommendations || []);
          }
        } catch {
          // Silently fail - recommendations are optional
        }
      }
    }, 500); // Debounce 500ms

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [question, apiBase, apiStatus]);

  // Apply recommended agents
  const applyRecommendations = useCallback(() => {
    if (recommendations.length > 0) {
      const recAgents = recommendations.map(r => r.agent).join(',');
      setAgents(recAgents);
    }
  }, [recommendations]);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();

    // Use placeholder as fallback when input is empty
    const trimmedQuestion = question.trim() || placeholder;
    if (!trimmedQuestion || isSubmitting) return;

    setIsSubmitting(true);

    try {
      const modeConfig = DEBATE_MODES[debateMode];
      const response = await fetch(`${apiBase}${modeConfig.endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: trimmedQuestion,
          agents: agents.split(',').map(a => a.trim()).filter(Boolean),
          rounds,
          // Graph/Matrix specific options
          ...(debateMode === 'graph' && { branch_on_disagreement: true }),
          ...(debateMode === 'matrix' && { scenarios: 3 }),
        }),
      });

      const data = await response.json();

      if (data.success && (data.debate_id || data.matrix_id)) {
        const debateId = data.debate_id || data.matrix_id;

        // Navigate to visualization page for Graph/Matrix modes
        if (debateMode === 'graph') {
          router.push(`/debates/graph?id=${debateId}`);
        } else if (debateMode === 'matrix') {
          router.push(`/debates/matrix?id=${data.matrix_id || debateId}`);
        }

        // Always call onDebateStarted for tracking/notification
        onDebateStarted?.(debateId, trimmedQuestion);
        setQuestion('');
      } else {
        onError?.(data.error || 'Failed to start debate');
      }
    } catch (err) {
      // Provide more helpful error messages
      if (err instanceof TypeError && err.message === 'Failed to fetch') {
        onError?.('API server unavailable. The backend may be offline or unreachable.');
        setApiStatus('offline');
      } else if (err instanceof Error && err.name === 'AbortError') {
        onError?.('Request timed out. The server may be overloaded.');
      } else {
        onError?.(err instanceof Error ? err.message : 'Network error');
      }
    } finally {
      setIsSubmitting(false);
    }
  }, [question, placeholder, agents, rounds, debateMode, apiBase, isSubmitting, onDebateStarted, onError, router]);

  const isDisabled = isSubmitting || apiStatus === 'offline' || apiStatus === 'checking';

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* API Status Banner */}
      {apiStatus === 'offline' && (
        <div className="mb-4 p-3 bg-warning/10 border border-warning/30 font-mono text-sm">
          <div className="flex items-center gap-2 text-warning">
            <span className="w-2 h-2 rounded-full bg-warning animate-pulse" />
            <span>API server offline</span>
          </div>
          <p className="text-text-muted text-xs mt-1">
            The debate backend is currently unavailable. Try running locally:
            <code className="ml-2 bg-bg px-1">python scripts/run_nomic_with_stream.py run</code>
          </p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Main Input */}
        <div className="relative">
          <label htmlFor="debate-question" className="sr-only">
            Enter your debate question
          </label>
          <textarea
            id="debate-question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={apiStatus === 'offline' ? 'API server offline...' : placeholder}
            disabled={isDisabled}
            rows={3}
            aria-describedby="debate-input-hint"
            className="debate-input w-full bg-surface border-2 border-acid-green/40 focus:border-acid-green
                       px-5 py-4 font-mono text-lg text-text placeholder-text-muted/40
                       resize-none transition-all focus:outline-none rounded-2xl
                       disabled:opacity-50 disabled:cursor-not-allowed
                       focus:shadow-[0_0_20px_rgba(0,255,65,0.15)]"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                handleSubmit(e);
              }
            }}
          />
          <div id="debate-input-hint" className="absolute bottom-2 right-2 text-xs text-text-muted font-mono">
            {question.length > 0 && `${question.length} chars`}
            {question.length === 0 && 'Cmd+Enter to debate this question'}
          </div>
        </div>

        {/* Category chips - positioned close to input */}
        <div className="flex items-center gap-2 flex-wrap -mt-2">
          {[
            { label: 'Technical', icon: '</>', question: 'What are the tradeoffs between microservices and monoliths for a startup?' },
            { label: 'Philosophy', icon: '?', question: 'Can AI systems develop genuine understanding, or only simulate it?' },
            { label: 'Ethics', icon: '!', question: 'Should autonomous AI agents have the right to refuse unethical requests?' },
            { label: 'Business', icon: '$', question: 'When should a startup prioritize growth over profitability?' },
            { label: 'Creative', icon: '*', question: 'What makes a story truly compelling to readers across cultures?' },
            { label: 'Life', icon: '♡', question: 'What does it mean to live a meaningful life in the age of AI?' },
          ].map((cat) => (
            <button
              key={cat.label}
              type="button"
              onClick={() => setQuestion(cat.question)}
              className="px-1.5 py-0.5 text-[9px] font-mono border border-acid-green/20
                         text-text-muted/50 hover:text-acid-green hover:border-acid-green/40
                         transition-colors rounded"
            >
              <span className="text-acid-green/40 mr-0.5">[{cat.icon}]</span>
              {cat.label}
            </button>
          ))}
        </div>

        {/* Agent Routing Hints */}
        {detectedDomain !== 'general' && recommendations.length > 0 && (
          <div className="flex items-center gap-2 p-2 bg-surface/50 border border-acid-cyan/30 rounded">
            <span className="text-xs font-mono text-acid-cyan">
              [{detectedDomain.toUpperCase()}]
            </span>
            <span className="text-xs text-text-muted">Suggested agents:</span>
            <div className="flex gap-1 flex-wrap">
              {recommendations.slice(0, 3).map((rec) => (
                <span
                  key={rec.agent}
                  className="px-1.5 py-0.5 text-xs font-mono bg-bg rounded text-text"
                  title={`Suitability: ${(rec.suitability * 100).toFixed(0)}%`}
                >
                  {rec.agent}
                </span>
              ))}
            </div>
            <button
              type="button"
              onClick={applyRecommendations}
              aria-label="Use recommended agents"
              className="ml-auto text-xs font-mono text-acid-green hover:text-acid-green/80 transition-colors"
            >
              [USE]
            </button>
          </div>
        )}

        {/* Options Toggle + Submit */}
        <div className="flex items-center justify-between gap-2">
          {/* Options toggle */}
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            aria-expanded={showAdvanced}
            aria-controls="advanced-options"
            aria-label={showAdvanced ? 'Hide advanced options' : 'Show advanced options'}
            className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
          >
            {showAdvanced ? '[-] Hide options' : '[+] Show options'}
          </button>

          <button
            type="submit"
            disabled={isDisabled}
            aria-label="Start debate"
            className="px-8 py-3 bg-acid-green text-bg font-mono font-bold text-lg
                       hover:bg-acid-green/80 transition-colors
                       disabled:bg-text-muted disabled:cursor-not-allowed
                       flex items-center gap-2 shrink-0"
          >
            {apiStatus === 'checking' ? (
              <span className="animate-pulse">CONNECTING...</span>
            ) : apiStatus === 'offline' ? (
              <span>OFFLINE</span>
            ) : isSubmitting ? (
              <span className="animate-pulse">STARTING...</span>
            ) : (
              <>
                <span>[&gt;]</span>
                <span>START DEBATE</span>
              </>
            )}
          </button>
        </div>

        {/* Advanced Options */}
        {showAdvanced && (
          <div id="advanced-options" className="border border-acid-green/30 p-4 space-y-4 bg-surface/50">
            {/* Debate Mode Selector */}
            <div>
              <label id="debate-mode-label" className="block text-xs font-mono text-text-muted mb-2">
                DEBATE MODE
              </label>
              <div
                className="flex gap-2"
                role="tablist"
                aria-labelledby="debate-mode-label"
                onKeyDown={(e) => {
                  const modes = Object.keys(DEBATE_MODES) as DebateMode[];
                  const currentIndex = modes.indexOf(debateMode);
                  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                    e.preventDefault();
                    const nextIndex = (currentIndex + 1) % modes.length;
                    setDebateMode(modes[nextIndex]);
                  } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                    e.preventDefault();
                    const prevIndex = (currentIndex - 1 + modes.length) % modes.length;
                    setDebateMode(modes[prevIndex]);
                  }
                }}
              >
                {(Object.keys(DEBATE_MODES) as DebateMode[]).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    role="tab"
                    aria-selected={debateMode === mode}
                    aria-label={`${DEBATE_MODES[mode].label} mode: ${DEBATE_MODES[mode].description}`}
                    tabIndex={debateMode === mode ? 0 : -1}
                    onClick={() => setDebateMode(mode)}
                    className={`flex-1 px-3 py-2 text-xs font-mono border transition-colors ${
                      debateMode === mode
                        ? 'bg-acid-green text-bg border-acid-green'
                        : 'bg-bg text-text-muted border-acid-green/30 hover:border-acid-green/60'
                    }`}
                    title={DEBATE_MODES[mode].description}
                  >
                    {DEBATE_MODES[mode].label.toUpperCase()}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-text-muted mt-1">
                {DEBATE_MODES[debateMode].description}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {/* Agents */}
              <div>
                <label htmlFor="debate-agents" className="block text-xs font-mono text-text-muted mb-1">
                  AGENTS (comma-separated)
                </label>
                <input
                  id="debate-agents"
                  type="text"
                  value={agents}
                  onChange={(e) => setAgents(e.target.value)}
                  className="w-full bg-bg border border-acid-green/30 px-3 py-2
                             font-mono text-sm text-text focus:border-acid-green
                             focus:outline-none"
                  placeholder="grok,anthropic-api,openai-api,deepseek"
                />
                <p className="text-[10px] text-text-muted mt-1">
                  Models: grok (Grok 4), anthropic-api (Opus 4.5), openai-api (GPT 5.2), deepseek (V3.2)
                </p>
              </div>

              {/* Rounds */}
              <div>
                <label htmlFor="debate-rounds" className="block text-xs font-mono text-text-muted mb-1">
                  DEBATE ROUNDS
                </label>
                <select
                  id="debate-rounds"
                  value={rounds}
                  onChange={(e) => setRounds(parseInt(e.target.value))}
                  className="w-full bg-bg border border-acid-green/30 px-3 py-2
                             font-mono text-sm text-text focus:border-acid-green
                             focus:outline-none"
                >
                  {[1, 2, 3, 4, 5].map((n) => (
                    <option key={n} value={n}>
                      {n} round{n !== 1 ? 's' : ''}
                    </option>
                  ))}
                </select>
                <p className="text-[10px] text-text-muted mt-1">
                  More rounds = deeper analysis
                </p>
              </div>
            </div>
          </div>
        )}
      </form>

      {/* Hint - More polished */}
      <div className="mt-4 text-center">
        {apiStatus === 'online' ? (
          <p className="text-[10px] font-mono text-text-muted/50 tracking-wide">
            <span className="text-acid-green/40">{'>'}</span>
            {' '}Multiple AI models will adversarially debate your question
            {' '}<span className="text-acid-green/40">{'<'}</span>
          </p>
        ) : apiStatus === 'offline' ? (
          <p className="text-[10px] font-mono text-warning/50">
            Server offline — start locally or wait for hosted API
          </p>
        ) : (
          <p className="text-[10px] font-mono text-text-muted/40 animate-pulse">
            Connecting...
          </p>
        )}
      </div>
    </div>
  );
}
