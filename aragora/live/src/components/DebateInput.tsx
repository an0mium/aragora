'use client';

import { useState, useCallback, useEffect, useRef } from 'react';

interface DebateInputProps {
  apiBase: string;
  onDebateStarted?: (debateId: string, question: string) => void;
  onError?: (error: string) => void;
}

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
  const [question, setQuestion] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [agents, setAgents] = useState('grok,anthropic-api,openai-api,deepseek');
  const [rounds, setRounds] = useState(3);
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
      const response = await fetch(`${apiBase}/api/debate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: trimmedQuestion,
          agents,
          rounds,
        }),
      });

      const data = await response.json();

      if (data.success && data.debate_id) {
        onDebateStarted?.(data.debate_id, trimmedQuestion);
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
  }, [question, placeholder, agents, rounds, apiBase, isSubmitting, onDebateStarted, onError]);

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
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={apiStatus === 'offline' ? 'API server offline...' : placeholder}
            disabled={isDisabled}
            rows={3}
            className="w-full bg-bg border-2 border-acid-green/50 focus:border-acid-green
                       px-4 py-3 font-mono text-lg text-text placeholder-text-muted/50
                       resize-none transition-colors focus:outline-none
                       disabled:opacity-50 disabled:cursor-not-allowed"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                handleSubmit(e);
              }
            }}
          />
          <div className="absolute bottom-2 right-2 text-xs text-text-muted font-mono">
            {question.length > 0 && `${question.length} chars`}
            {question.length === 0 && 'Cmd+Enter to debate this question'}
          </div>
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
              className="ml-auto text-xs font-mono text-acid-green hover:text-acid-green/80 transition-colors"
            >
              [USE]
            </button>
          </div>
        )}

        {/* Advanced Options Toggle */}
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
          >
            {showAdvanced ? '[-] Hide options' : '[+] Show options'}
          </button>

          <button
            type="submit"
            disabled={isDisabled}
            className="px-6 py-2 bg-acid-green text-bg font-mono font-bold
                       hover:bg-acid-green/80 transition-colors
                       disabled:bg-text-muted disabled:cursor-not-allowed
                       flex items-center gap-2"
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
          <div className="border border-acid-green/30 p-4 space-y-4 bg-surface/50">
            <div className="grid grid-cols-2 gap-4">
              {/* Agents */}
              <div>
                <label className="block text-xs font-mono text-text-muted mb-1">
                  AGENTS (comma-separated)
                </label>
                <input
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
                <label className="block text-xs font-mono text-text-muted mb-1">
                  DEBATE ROUNDS
                </label>
                <select
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

      {/* Hint */}
      <p className="mt-4 text-center text-xs font-mono text-text-muted/60">
        {apiStatus === 'online' ? (
          'AI agents will debate your question and reach a consensus'
        ) : apiStatus === 'offline' ? (
          <span className="text-warning/70">
            Start the local server or wait for the hosted API to come online
          </span>
        ) : (
          'Checking API connection...'
        )}
      </p>
    </div>
  );
}
