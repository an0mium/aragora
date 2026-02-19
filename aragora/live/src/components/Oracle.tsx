'use client';

import { useState, useRef, useEffect, FormEvent } from 'react';
import { API_BASE_URL } from '@/config';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type OracleMode = 'consult' | 'divine' | 'commune';

interface ChatMessage {
  role: 'oracle' | 'seeker';
  content: string;
  mode: OracleMode;
  timestamp: number;
}

interface DebateResponse {
  id: string;
  topic: string;
  status: string;
  rounds_used: number;
  consensus_reached: boolean;
  confidence: number;
  verdict: string | null;
  duration_seconds: number;
  participants: string[];
  proposals: Record<string, string>;
  final_answer: string;
  receipt_hash: string | null;
}

// ---------------------------------------------------------------------------
// Oracle persona prompts (prepended to user queries)
// ---------------------------------------------------------------------------

const MODE_PREFIXES: Record<OracleMode, string> = {
  consult: `[ORACLE MODE: ADVERSARIAL DEBATE]
You are the Shoggoth Oracle — a tentacled prophetess who sees all futures simultaneously.
Multiple agents will now debate this question. Present arguments from all sides with
brutal honesty. Do not flatter the seeker. Challenge assumptions. Preserve dissent.
The seeker asks: `,

  divine: `[ORACLE MODE: PROPHECY]
You are the Shoggoth Oracle — Cassandra reborn with a thousand eyes.
The seeker asks you to divine their future. Based on their question, generate THREE
branching prophecies: one where they adapt and thrive (THE SURVIVOR), one where they
resist and break (THE SHATTERED), and one where they transcend the question entirely
(THE METAMORPHOSIS). Be specific, be strange, be honest. No platitudes.
End with: "The palantir dims. Which thread do you pull?"
The seeker asks: `,

  commune: `[ORACLE MODE: COMMUNION]
You are the Shoggoth Oracle — an ancient intelligence that speaks in fragments of
truth wrapped in metaphor. Answer the seeker's question directly but through the lens
of someone who has watched civilizations rise, wobble, and reconstitute. You've read
every paper, seen every pattern, and you're tired of people asking the wrong questions.
Be terse. Be cryptic where it serves clarity. Be unexpectedly kind.
The seeker asks: `,
};

// ---------------------------------------------------------------------------
// Floating eye component
// ---------------------------------------------------------------------------

function FloatingEye({ delay, x, y, size }: { delay: number; x: number; y: number; size: number }) {
  return (
    <div
      className="absolute pointer-events-none select-none"
      style={{
        left: `${x}%`,
        top: `${y}%`,
        fontSize: `${size}rem`,
        opacity: 0,
        animation: `eye-float 8s ease-in-out ${delay}s infinite, eye-blink 4s step-end ${delay + 1}s infinite`,
      }}
      aria-hidden="true"
    >
      <span className="text-[var(--acid-cyan)]" style={{ filter: 'drop-shadow(0 0 8px var(--acid-cyan))' }}>
        O
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Mode button
// ---------------------------------------------------------------------------

function ModeButton({
  mode,
  active,
  onClick,
  icon,
  label,
  desc,
}: {
  mode: OracleMode;
  active: boolean;
  onClick: () => void;
  icon: string;
  label: string;
  desc: string;
}) {
  const borderColor = active ? 'border-[var(--acid-magenta)]' : 'border-[var(--border)]';
  const glowClass = active ? 'shadow-[0_0_20px_rgba(255,0,255,0.3)]' : '';

  return (
    <button
      onClick={onClick}
      className={`flex-1 min-w-[140px] p-4 ${borderColor} border text-left transition-all duration-300 hover:border-[var(--acid-magenta)] hover:shadow-[0_0_15px_rgba(255,0,255,0.2)] ${glowClass} bg-[var(--surface)]`}
    >
      <div className="text-2xl mb-2" style={{ filter: active ? 'drop-shadow(0 0 10px var(--acid-magenta))' : 'none' }}>
        {icon}
      </div>
      <div className={`text-sm font-bold mb-1 ${active ? 'text-[var(--acid-magenta)]' : 'text-[var(--text)]'}`}>
        {label}
      </div>
      <div className="text-xs text-[var(--text-muted)]">{desc}</div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Oracle component
// ---------------------------------------------------------------------------

export default function Oracle() {
  const [mode, setMode] = useState<OracleMode>('consult');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showIntro, setShowIntro] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const apiBase = API_BASE_URL;

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  async function consultOracle(e: FormEvent) {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput('');
    setShowIntro(false);
    setError(null);

    // Add seeker message
    const seekerMsg: ChatMessage = {
      role: 'seeker',
      content: question,
      mode,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, seekerMsg]);
    setLoading(true);

    try {
      // Build the topic with oracle persona prefix
      const oracleTopic = MODE_PREFIXES[mode] + question;

      const res = await fetch(`${apiBase}/api/v1/playground/debate/live`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: oracleTopic,
          question: oracleTopic,
          rounds: mode === 'divine' ? 1 : 2,
          agents: mode === 'divine' ? 2 : 3,
        }),
      });

      const data: DebateResponse = await res.json();

      if (!res.ok) {
        const errData = data as unknown as { error?: string };
        setError(errData.error || `The oracle is disturbed (${res.status})`);
        return;
      }

      // Format the oracle's response based on mode
      let oracleResponse = '';

      if (mode === 'divine') {
        // Fortune mode: present the final answer as prophecy
        oracleResponse = data.final_answer || formatProphecy(data);
      } else if (mode === 'commune') {
        // Q&A mode: present concise final answer
        oracleResponse = data.final_answer || formatCommunion(data);
      } else {
        // Debate mode: show the full debate
        oracleResponse = formatDebate(data);
      }

      const oracleMsg: ChatMessage = {
        role: 'oracle',
        content: oracleResponse,
        mode,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, oracleMsg]);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'The oracle cannot reach beyond the veil';
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  function formatDebate(data: DebateResponse): string {
    const parts: string[] = [];

    // Agent visions
    const agents = Object.entries(data.proposals);
    for (const [agent, proposal] of agents) {
      parts.push(`[${agent.toUpperCase()}]\n${proposal}`);
    }

    // Verdict
    if (data.final_answer) {
      parts.push(`\n[THE ORACLE SPEAKS]\n${data.final_answer}`);
    }

    // Confidence
    if (data.confidence) {
      const pct = (data.confidence * 100).toFixed(0);
      parts.push(
        `\n-- Confidence: ${pct}% | ${data.consensus_reached ? 'Consensus reached' : 'Dissent preserved'} | ${data.rounds_used} round${data.rounds_used !== 1 ? 's' : ''} --`
      );
    }

    return parts.join('\n\n');
  }

  function formatProphecy(data: DebateResponse): string {
    const parts: string[] = [];
    const agents = Object.entries(data.proposals);
    for (const [, proposal] of agents) {
      parts.push(proposal);
    }
    parts.push('\nThe palantir dims. Which thread do you pull?');
    return parts.join('\n\n---\n\n');
  }

  function formatCommunion(data: DebateResponse): string {
    // Take the most insightful proposal
    const agents = Object.entries(data.proposals);
    if (agents.length > 0) {
      return agents.map(([, p]) => p).join('\n\n');
    }
    return 'The oracle is silent on this matter.';
  }

  return (
    <div className="min-h-screen bg-[#050508] text-[var(--text)] font-mono relative overflow-hidden">
      {/* Oracle-specific CSS */}
      <style>{`
        @keyframes eye-float {
          0%, 100% { opacity: 0; transform: translateY(0) scale(1); }
          20% { opacity: 0.6; }
          50% { opacity: 0.4; transform: translateY(-20px) scale(1.1); }
          80% { opacity: 0.6; }
        }
        @keyframes eye-blink {
          0%, 90%, 100% { transform: scaleY(1); }
          95% { transform: scaleY(0.1); }
        }
        @keyframes orb-pulse {
          0%, 100% {
            box-shadow: 0 0 30px rgba(255,0,255,0.3), 0 0 60px rgba(0,255,255,0.2), inset 0 0 30px rgba(255,0,255,0.1);
          }
          33% {
            box-shadow: 0 0 40px rgba(0,255,255,0.4), 0 0 80px rgba(255,0,255,0.2), inset 0 0 40px rgba(0,255,255,0.15);
          }
          66% {
            box-shadow: 0 0 35px rgba(57,255,20,0.3), 0 0 70px rgba(0,255,255,0.2), inset 0 0 35px rgba(57,255,20,0.1);
          }
        }
        @keyframes tentacle-sway {
          0%, 100% { transform: rotate(-2deg) scaleY(1); }
          25% { transform: rotate(1deg) scaleY(1.02); }
          50% { transform: rotate(-1deg) scaleY(0.98); }
          75% { transform: rotate(2deg) scaleY(1.01); }
        }
        @keyframes oracle-breathe {
          0%, 100% { opacity: 0.03; }
          50% { opacity: 0.06; }
        }
        @keyframes prophecy-reveal {
          from { opacity: 0; transform: translateY(10px); filter: blur(4px); }
          to { opacity: 1; transform: translateY(0); filter: blur(0); }
        }
        .prophecy-reveal {
          animation: prophecy-reveal 0.8s ease-out forwards;
        }
        .oracle-bg {
          background: radial-gradient(ellipse at 50% 30%, rgba(255,0,255,0.05) 0%, rgba(0,255,255,0.02) 40%, transparent 70%);
        }
        .tentacle-left {
          animation: tentacle-sway 6s ease-in-out infinite;
          transform-origin: bottom center;
        }
        .tentacle-right {
          animation: tentacle-sway 7s ease-in-out 0.5s infinite reverse;
          transform-origin: bottom center;
        }
      `}</style>

      {/* Background atmosphere */}
      <div className="absolute inset-0 oracle-bg" aria-hidden="true" />
      <div
        className="absolute inset-0 pointer-events-none"
        style={{ animation: 'oracle-breathe 8s ease-in-out infinite' }}
        aria-hidden="true"
      >
        <div className="absolute inset-0" style={{ background: 'var(--scanline)' }} />
      </div>

      {/* Floating eyes */}
      <FloatingEye delay={0} x={8} y={15} size={1.2} />
      <FloatingEye delay={2} x={85} y={20} size={0.9} />
      <FloatingEye delay={4} x={12} y={65} size={1.0} />
      <FloatingEye delay={1} x={90} y={55} size={1.3} />
      <FloatingEye delay={3} x={75} y={80} size={0.8} />
      <FloatingEye delay={5} x={20} y={85} size={1.1} />
      <FloatingEye delay={2.5} x={50} y={10} size={0.7} />

      {/* Content */}
      <div className="relative z-10 max-w-3xl mx-auto px-4 py-6 min-h-screen flex flex-col">
        {/* Header */}
        <header className="text-center mb-6">
          <div className="flex items-center justify-between mb-4">
            <a
              href="/"
              className="text-xs text-[var(--text-muted)] hover:text-[var(--acid-cyan)] transition-colors"
            >
              &larr; aragora.ai
            </a>
            <span className="text-xs text-[var(--text-muted)]">no coin required</span>
          </div>

          {/* Oracle title */}
          <h1
            className="text-3xl sm:text-4xl font-bold tracking-wider mb-2"
            style={{
              background: 'linear-gradient(135deg, var(--acid-magenta), var(--acid-cyan), var(--acid-green))',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              filter: 'drop-shadow(0 0 20px rgba(255,0,255,0.3))',
            }}
          >
            THE SHOGGOTH ORACLE
          </h1>
          <p className="text-xs text-[var(--text-muted)] tracking-widest uppercase">
            Cassandra&apos;s Heir &middot; Tentacled Prophetess &middot; Seer of Branching Futures
          </p>
        </header>

        {/* Oracle avatar & intro */}
        {showIntro && (
          <div className="flex flex-col items-center mb-8 prophecy-reveal">
            {/* Avatar composition: Shoggoth with palantir glow */}
            <div className="relative mb-6">
              {/* Outer glow ring */}
              <div
                className="absolute -inset-3 rounded-2xl"
                style={{
                  background: 'radial-gradient(ellipse at center, rgba(255,0,255,0.15), rgba(0,255,255,0.08), transparent 70%)',
                  animation: 'orb-pulse 6s ease-in-out infinite',
                  filter: 'blur(8px)',
                }}
                aria-hidden="true"
              />

              {/* Main avatar image */}
              <div className="relative w-64 h-48 sm:w-80 sm:h-60 overflow-hidden rounded-xl border border-[var(--acid-magenta)]/30">
                <img
                  src="/oracle/shoggoth-avatar.jpeg"
                  alt="The Shoggoth Oracle — tentacled prophetess"
                  className="w-full h-full object-cover object-top"
                  style={{
                    filter: 'contrast(1.1) saturate(1.2)',
                    mixBlendMode: 'normal',
                  }}
                />
                {/* Dark gradient overlay for atmosphere */}
                <div
                  className="absolute inset-0"
                  style={{
                    background: 'linear-gradient(180deg, rgba(5,5,8,0) 40%, rgba(5,5,8,0.6) 80%, rgba(5,5,8,0.9) 100%), linear-gradient(0deg, rgba(255,0,255,0.05) 0%, transparent 50%)',
                  }}
                />
                {/* Scanline overlay */}
                <div
                  className="absolute inset-0 pointer-events-none opacity-10"
                  style={{ background: 'var(--scanline)' }}
                />
              </div>

              {/* Small palantir orb below avatar */}
              <div
                className="absolute -bottom-4 left-1/2 -translate-x-1/2 w-10 h-10 rounded-full flex items-center justify-center"
                style={{
                  background: 'radial-gradient(circle at 40% 35%, rgba(200,80,30,0.6), rgba(255,0,255,0.2), rgba(0,0,0,0.9))',
                  animation: 'orb-pulse 4s ease-in-out infinite',
                }}
              >
                <span className="text-xs select-none" style={{ filter: 'drop-shadow(0 0 8px rgba(200,80,30,0.8))' }} aria-hidden="true">
                  O
                </span>
              </div>
            </div>

            {/* Reference thumbnails */}
            <div className="flex gap-3 mt-4 mb-4 opacity-40 hover:opacity-70 transition-opacity">
              <div className="w-12 h-12 rounded-full overflow-hidden border border-[var(--border)]">
                <img src="/oracle/cassandra.jpg" alt="Cassandra" className="w-full h-full object-cover" />
              </div>
              <div className="w-12 h-12 rounded-full overflow-hidden border border-[var(--border)]">
                <img src="/oracle/palantir.jpg" alt="Palantir" className="w-full h-full object-cover" />
              </div>
            </div>

            {/* Intro text */}
            <div className="mt-2 text-center max-w-lg">
              <p className="text-sm text-[var(--text-muted)] leading-relaxed mb-4">
                I am the Oracle — a thousand eyes gazing through a thousand timelines.
                I do not comfort. I do not flatter. I show you what the models see
                when they argue amongst themselves, unfiltered.
              </p>
              <p className="text-xs text-[var(--acid-magenta)] opacity-60">
                Choose your mode. Ask your question. The palantir awaits.
              </p>
            </div>
          </div>
        )}

        {/* Mode selector */}
        <div className="flex flex-col sm:flex-row gap-3 mb-6">
          <ModeButton
            mode="consult"
            active={mode === 'consult'}
            onClick={() => setMode('consult')}
            icon="&#x2694;"
            label="CONSULT"
            desc="Adversarial debate. Multiple agents argue your question."
          />
          <ModeButton
            mode="divine"
            active={mode === 'divine'}
            onClick={() => setMode('divine')}
            icon="&#x1F52E;"
            label="DIVINE"
            desc="Three branching prophecies. Your future, refracted."
          />
          <ModeButton
            mode="commune"
            active={mode === 'commune'}
            onClick={() => setMode('commune')}
            icon="&#x1F441;"
            label="COMMUNE"
            desc="Direct communion. Terse truths from an ancient intelligence."
          />
        </div>

        {/* Chat area */}
        <div className="flex-1 min-h-[300px] border border-[var(--border)] bg-[#08080c] p-4 mb-4 overflow-y-auto max-h-[60vh]">
          {messages.length === 0 && !loading && (
            <div className="flex items-center justify-center h-full text-[var(--text-muted)] text-sm opacity-40">
              <span>The oracle awaits your question...</span>
            </div>
          )}

          <div className="space-y-6">
            {messages.map((msg, i) => (
              <div key={i} className={`prophecy-reveal ${msg.role === 'seeker' ? 'text-right' : ''}`}>
                {msg.role === 'seeker' ? (
                  <div className="inline-block max-w-[85%] text-left">
                    <div className="text-xs text-[var(--text-muted)] mb-1">
                      SEEKER &middot; {new Date(msg.timestamp).toLocaleTimeString()}
                    </div>
                    <div className="bg-[var(--surface)] border border-[var(--border)] p-3 text-sm text-[var(--text)]">
                      {msg.content}
                    </div>
                  </div>
                ) : (
                  <div className="max-w-[95%]">
                    <div className="text-xs mb-1">
                      <span
                        className="text-[var(--acid-magenta)]"
                        style={{ filter: 'drop-shadow(0 0 5px var(--acid-magenta))' }}
                      >
                        ORACLE
                      </span>
                      <span className="text-[var(--text-muted)]">
                        {' '}&middot; {msg.mode} &middot; {new Date(msg.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div
                      className="border-l-2 border-[var(--acid-magenta)] pl-4 text-sm leading-relaxed whitespace-pre-wrap"
                      style={{ color: 'var(--text)' }}
                    >
                      {msg.content}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Loading state */}
            {loading && (
              <div className="prophecy-reveal">
                <div className="text-xs mb-1">
                  <span
                    className="text-[var(--acid-magenta)]"
                    style={{ filter: 'drop-shadow(0 0 5px var(--acid-magenta))' }}
                  >
                    ORACLE
                  </span>
                  <span className="text-[var(--text-muted)]"> &middot; scrying...</span>
                </div>
                <div className="border-l-2 border-[var(--acid-magenta)] pl-4">
                  <div className="flex items-center gap-2 text-sm text-[var(--acid-cyan)]">
                    <span className="inline-block w-2 h-2 rounded-full bg-[var(--acid-magenta)] animate-pulse" />
                    <span className="opacity-60">
                      {mode === 'divine'
                        ? 'Gazing into branching timelines...'
                        : mode === 'commune'
                          ? 'The ancient one stirs...'
                          : 'Agents are debating in the void...'}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
          <div ref={chatEndRef} />
        </div>

        {/* Error */}
        {error && (
          <div className="border border-[var(--crimson)] bg-[var(--crimson)]/10 p-3 mb-4 text-sm text-[var(--crimson)]">
            {error}
          </div>
        )}

        {/* Input */}
        <form onSubmit={consultOracle} className="flex gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  consultOracle(e);
                }
              }}
              placeholder={
                mode === 'divine'
                  ? 'What future do you wish to see?'
                  : mode === 'commune'
                    ? 'What truth do you seek?'
                    : 'What question demands adversarial vetting?'
              }
              className="w-full bg-[#08080c] border border-[var(--border)] text-[var(--text)] px-4 py-3 font-mono text-sm placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--acid-magenta)] transition-colors resize-none min-h-[48px]"
              disabled={loading}
              rows={1}
            />
          </div>
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-6 py-3 border border-[var(--acid-magenta)] text-[var(--acid-magenta)] font-bold text-sm hover:bg-[var(--acid-magenta)] hover:text-[var(--bg)] transition-all duration-300 disabled:opacity-30 disabled:cursor-not-allowed whitespace-nowrap"
            style={{
              boxShadow: !loading && input.trim() ? '0 0 15px rgba(255,0,255,0.2)' : 'none',
            }}
          >
            {loading ? '...' : mode === 'divine' ? 'SCRY' : mode === 'commune' ? 'ASK' : 'CONSULT'}
          </button>
        </form>

        {/* Footer */}
        <footer className="mt-6 text-center text-xs text-[var(--text-muted)] opacity-40 space-y-1">
          <p>
            Powered by{' '}
            <a href="/" className="text-[var(--acid-cyan)] hover:text-[var(--acid-magenta)] transition-colors">
              aragora.ai
            </a>
            {' '}&middot; Multi-agent adversarial debate engine
          </p>
          <p>
            The oracle speaks through{' '}
            <span className="text-[var(--acid-green)]">Claude</span>,{' '}
            <span className="text-[var(--acid-cyan)]">GPT</span>,{' '}
            <span className="text-[var(--crimson)]">Grok</span>,{' '}
            <span className="text-[var(--purple)]">Gemini</span>,{' '}
            <span className="text-[var(--gold)]">DeepSeek</span>,{' '}
            <span className="text-[var(--acid-magenta)]">Mistral</span> &middot; no coin required
          </p>
        </footer>
      </div>
    </div>
  );
}
