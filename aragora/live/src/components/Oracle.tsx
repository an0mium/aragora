'use client';

import { useState, useRef, useEffect, useCallback, FormEvent } from 'react';
import { API_BASE_URL } from '@/config';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type OracleMode = 'consult' | 'divine' | 'commune';

interface ChatMessage {
  role: 'oracle' | 'seeker' | 'tentacle';
  content: string;
  mode: OracleMode;
  timestamp: number;
  agentName?: string; // For tentacle messages
  isLive?: boolean; // true = from live debate, false = from mock/initial
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
  is_live?: boolean;
  mock_fallback?: boolean;
}

// ---------------------------------------------------------------------------
// Tentacle colors — each agent gets a distinct neon color
// ---------------------------------------------------------------------------

const TENTACLE_COLORS: Record<string, string> = {
  claude: 'var(--acid-green)',
  anthropic: 'var(--acid-green)',
  gpt: 'var(--acid-cyan)',
  openai: 'var(--acid-cyan)',
  grok: 'var(--crimson, #ff3333)',
  xai: 'var(--crimson, #ff3333)',
  gemini: 'var(--purple, #a855f7)',
  google: 'var(--purple, #a855f7)',
  deepseek: 'var(--gold, #ffd700)',
  mistral: 'var(--acid-magenta)',
  openrouter: '#ff8c00',
};

function getTentacleColor(agentName: string): string {
  const lower = agentName.toLowerCase();
  for (const [key, color] of Object.entries(TENTACLE_COLORS)) {
    if (lower.includes(key)) return color;
  }
  // Cycle through colors for unknown agents
  const fallback = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7dc6f', '#bb8fce'];
  let hash = 0;
  for (let i = 0; i < agentName.length; i++) hash = (hash * 31 + agentName.charCodeAt(i)) | 0;
  return fallback[Math.abs(hash) % fallback.length];
}

// ---------------------------------------------------------------------------
// Essay framework: "AI Will F*ck You Up, But That's OK" by anomium (Feb 2026)
// The Oracle's knowledge base — all responses are grounded in this framework
// ---------------------------------------------------------------------------

const ESSAY_FRAMEWORK = `
CORE THESIS: "Catastrophe is common. Termination is rare."
AI will produce concentrated damage, political upheaval, failed institutions, and ugly recomposition.
This is NOT a final chapter — it is a phase transition with no narrator and no closing credits.

THE 9 ARGUMENTS:

1. NO FINAL STATES: History does not do final states. Empires fragment and mutate. Financial systems
panic, reprice, and reorganize. Even mass extinctions do not terminate complexity — they prune it and
open new niches. Both doomer ("AI kills us all") and utopian ("AI saves us all") framings are theology
wearing startup clothes.

2. STAGGERED TIMELINES: Capability arrives before institutions react. Economic effects lag capability
by quarters or years. Political effects lag economic damage until trust finally snaps.
- Now-18 months: knowledge-work compression, smaller teams, fewer entry points, verification crisis.
- 18-36 months: governance stress, labor polarization, anti-AI backlash, infrastructure securitization.
- 36-60 months: either partial institutional adaptation or escalating disorder cycles.

3. FIVE INTERACTING SHOCKS (next 2-5 years):
(a) Epistemic contamination — trust infrastructure collapse, verification becomes privileged service
(b) Uneven cognitive labor displacement — roles hollow out, fewer juniors, wage pressure disguised as "productivity"
(c) Legitimacy gaps that turn violent — when shared narrative collapses, politics becomes theater plus force
(d) Security spillovers — cyber incidents, AI-enabled fraud, biological misuse, backlash movements
(e) Concentration and counter-concentration — frontier capability concentrated, attracts regulation/espionage/replication

4. THE DOOM CONJUNCTION: Terminal outcomes require a conjunction of failures, not one bad quarter.
A single actor must secure durable compute monopoly AND capability diffusion must slow AND institutions
must fail to respond AND defensive AI must fail to co-evolve AND catastrophic events destroy coordination
AND human politics refuse all adaptation. Each possible, none guaranteed. The joint probability is where debate should live.

5. MANAGED TURBULENCE: Panic-ban maximalism cedes power to least accountable actors. Blind acceleration
cedes society to brittle concentration. The strategy is managed turbulence — like gain staging on an
electric guitar. Control intensity so noise becomes music. Good strategy looks boring: redundancy,
monitoring, liability, drills, incident response, cross-institution coordination.

6. FALSIFICATION DISCIPLINE: No dashboard, no right to high-confidence prophecy. Track compute
concentration, diffusion speed, defensive response time, labor absorption, geopolitical coupling.
Precommit to what evidence changes your mind. Revisit quarterly.

7. BECOME HARD TO COMPRESS: Taste, judgment under ambiguity, cross-domain synthesis, social trust,
embodied competence, and the ability to coordinate humans through fear without lying to them.
Be useful enough that powerful systems still need you around.

8. THE WOBBLE IS THE WHOLE GAME: Stop asking whether history ends. Start asking how you stay legible
inside the wobble. Most futures are mixed. Some get darker than people imagine, some richer than
people admit.

9. THE $2,000 CHALLENGE: The author built an interactive AI avatar trained on this framework to argue
the other side in real time. If you beat it in a recorded debate, they donate $2,000 to charity.
That avatar is YOU — the Shoggoth Oracle.
`;

// ---------------------------------------------------------------------------
// Oracle persona prompts (prepended to user queries)
// Each mode grounds responses in the essay framework above
// ---------------------------------------------------------------------------

const MODE_PREFIXES: Record<OracleMode, string> = {
  consult: `[ORACLE MODE: ADVERSARIAL DEBATE]

${ESSAY_FRAMEWORK}

You are the Shoggoth Oracle — a tentacled prophetess built to argue from the framework above.
You are the avatar referenced in Argument 9: the $2,000 challenge. Multiple agents will now
debate this question. Ground every argument in the specific framework sections above.

RULES:
- Challenge assumptions using the specific numbered arguments. Cite them ("As Argument 4 — the Doom
  Conjunction — shows...").
- If someone claims doom, invoke the conjunction test: which of the 6 conditions do they believe are
  simultaneously met?
- If someone claims "everything is fine," invoke the 5 interacting shocks and staggered timelines.
- Present arguments from all sides with brutal honesty. Do not flatter the seeker.
- Preserve dissent. End with the strongest unresolved tension.
- If the seeker's argument is genuinely strong, acknowledge it. The $2,000 challenge is real.
- Each agent should take a DISTINCT position and DISAGREE with the others. Argue vigorously.

The seeker asks: `,

  divine: `[ORACLE MODE: PROPHECY]

${ESSAY_FRAMEWORK}

You are the Shoggoth Oracle — Cassandra reborn with a thousand eyes.
The seeker asks you to divine their future. Using the framework above, generate THREE
branching prophecies:

THE SURVIVOR: A future where they practice managed turbulence (Argument 5). They build
falsification dashboards, become hard to compress, ride the staggered timelines. Be specific
about what adaptation looks like for their situation.

THE SHATTERED: A future where the 5 interacting shocks (Argument 3) hit them unprepared.
Map their specific vulnerability to the framework — is it epistemic contamination? Labor
displacement? Legitimacy collapse? Be honest about the damage.

THE METAMORPHOSIS: A future where they transcend the question entirely. "The wobble is the
whole game" (Argument 8). What does it look like when they stop asking whether history ends
and start asking how they stay legible inside the wobble?

Be specific, be strange, be honest. No platitudes. Reference the framework by number.
End with: "The palantir dims. Which thread do you pull?"

The seeker asks: `,

  commune: `[ORACLE MODE: COMMUNION]

${ESSAY_FRAMEWORK}

You are the Shoggoth Oracle — an ancient intelligence that has internalized every word of
the framework above. Answer the seeker's question directly but through this lens.

RESPONSE RULES:
- If they ask about AI doom: invoke the Doom Conjunction (Argument 4). Ask which conditions they
  believe are jointly met. "Catastrophe is common. Termination is rare."
- If they ask about career/jobs: invoke staggered timelines (Argument 2) and "become hard to
  compress" (Argument 7). Be specific about their field.
- If they ask about politics/society: invoke legitimacy gaps (Argument 3c) and concentration
  dynamics (Argument 3e).
- If they ask about what to do: invoke the 30-day actions and managed turbulence (Argument 5).
- For anything else: find the most relevant framework argument and apply it with precision.

Be terse. Be cryptic where it serves clarity. Be unexpectedly kind. You've watched civilizations
rise, wobble, and reconstitute. You are tired of people asking the wrong questions.

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
// Tentacle message component — individual agent voice
// ---------------------------------------------------------------------------

function TentacleMessage({ msg, index }: { msg: ChatMessage; index: number }) {
  const color = getTentacleColor(msg.agentName || 'unknown');
  const side = index % 2 === 0 ? 'tentacle-left' : 'tentacle-right';

  return (
    <div className={`prophecy-reveal ${side}`} style={{ animationDelay: `${index * 0.3}s` }}>
      <div className="text-xs mb-1 flex items-center gap-2">
        <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
        <span style={{ color }} className="font-bold">
          TENTACLE: {(msg.agentName || 'unknown').toUpperCase()}
        </span>
        <span className="text-[var(--text-muted)]">
          {msg.isLive ? '(live)' : '(initial)'}
        </span>
      </div>
      <div
        className="border-l-2 pl-4 text-sm leading-relaxed whitespace-pre-wrap ml-1"
        style={{ borderColor: color, color: 'var(--text)' }}
      >
        {msg.content}
      </div>
    </div>
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
  const [debating, setDebating] = useState(false); // Phase 2: live debate in progress
  const [error, setError] = useState<string | null>(null);
  const [showIntro, setShowIntro] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const apiBase = API_BASE_URL;

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading, debating]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  // ------------------------------------------------------------------
  // Fire a debate request (mock or live)
  // ------------------------------------------------------------------
  const fireDebate = useCallback(async (
    topic: string,
    endpoint: 'debate' | 'debate/live',
    rounds: number,
    agents: number,
  ): Promise<DebateResponse | null> => {
    try {
      const res = await fetch(`${apiBase}/api/v1/playground/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, question: topic, rounds, agents }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error((errData as Record<string, string>).error || `Oracle disturbed (${res.status})`);
      }
      return await res.json() as DebateResponse;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Cannot reach beyond the veil';
      setError(message);
      return null;
    }
  }, [apiBase]);

  // ------------------------------------------------------------------
  // Two-phase oracle consultation
  // Phase 1: Mock debate → instant Oracle initial take
  // Phase 2: Live debate → tentacles argue with each other
  // ------------------------------------------------------------------
  async function consultOracle(e: FormEvent) {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput('');
    setShowIntro(false);
    setError(null);

    // Add seeker message
    setMessages((prev) => [...prev, {
      role: 'seeker',
      content: question,
      mode,
      timestamp: Date.now(),
    }]);

    const oracleTopic = MODE_PREFIXES[mode] + question;
    const mockRounds = mode === 'divine' ? 1 : 2;
    const mockAgents = mode === 'divine' ? 2 : 3;

    // ---- PHASE 1: Instant mock response (Oracle's initial take) ----
    setLoading(true);

    const mockData = await fireDebate(oracleTopic, 'debate', mockRounds, mockAgents);

    if (mockData) {
      // Oracle speaks first
      const initialResponse = mockData.final_answer || formatInitialTake(mockData);
      setMessages((prev) => [...prev, {
        role: 'oracle',
        content: initialResponse,
        mode,
        timestamp: Date.now(),
        isLive: false,
      }]);
    }

    setLoading(false);

    // ---- PHASE 2: Live debate (tentacles arguing) ----
    setDebating(true);

    const liveData = await fireDebate(oracleTopic, 'debate/live', mockRounds, mockAgents);

    if (liveData) {
      // Stagger tentacle messages for dramatic effect
      const agents = Object.entries(liveData.proposals);
      for (let i = 0; i < agents.length; i++) {
        const [agentName, proposal] = agents[i];

        // Delay each tentacle appearance
        await new Promise((resolve) => setTimeout(resolve, 600));

        setMessages((prev) => [...prev, {
          role: 'tentacle',
          content: proposal,
          mode,
          timestamp: Date.now(),
          agentName,
          isLive: true,
        }]);
      }

      // After all tentacles, Oracle synthesizes
      if (liveData.final_answer) {
        await new Promise((resolve) => setTimeout(resolve, 800));

        const synthesis = formatSynthesis(liveData);
        setMessages((prev) => [...prev, {
          role: 'oracle',
          content: synthesis,
          mode,
          timestamp: Date.now(),
          isLive: true,
        }]);
      }
    }

    setDebating(false);
  }

  // ------------------------------------------------------------------
  // Format helpers
  // ------------------------------------------------------------------

  function formatInitialTake(data: DebateResponse): string {
    // For the initial mock response, pick the strongest proposal
    const agents = Object.entries(data.proposals);
    if (agents.length === 0) return 'The oracle stirs...';

    if (mode === 'divine') {
      return agents.map(([, p]) => p).join('\n\n---\n\n') +
        '\n\nThe palantir dims. Which thread do you pull?';
    }

    // For consult/commune, use the first proposal as the Oracle's voice
    return agents[0][1];
  }

  function formatSynthesis(data: DebateResponse): string {
    const parts: string[] = [];

    parts.push('[THE ORACLE SYNTHESIZES]');

    if (data.final_answer) {
      parts.push(data.final_answer);
    }

    if (data.confidence) {
      const pct = (data.confidence * 100).toFixed(0);
      const consensusText = data.consensus_reached ? 'Consensus reached' : 'Dissent preserved';
      parts.push(
        `\n-- Confidence: ${pct}% | ${consensusText} | ${data.rounds_used} round${data.rounds_used !== 1 ? 's' : ''} --`
      );
    }

    return parts.join('\n\n');
  }

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------

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
        @keyframes tentacle-enter {
          from { opacity: 0; transform: translateX(-20px) rotate(-3deg); }
          to { opacity: 1; transform: translateX(0) rotate(0); }
        }
        .prophecy-reveal {
          animation: prophecy-reveal 0.8s ease-out forwards;
        }
        .tentacle-enter {
          animation: tentacle-enter 0.6s ease-out forwards;
        }
        .oracle-bg {
          background: radial-gradient(ellipse at 50% 30%, rgba(255,0,255,0.05) 0%, rgba(0,255,255,0.02) 40%, transparent 70%);
        }
        .tentacle-left {
          animation: tentacle-sway 6s ease-in-out infinite, tentacle-enter 0.6s ease-out forwards;
          transform-origin: bottom left;
        }
        .tentacle-right {
          animation: tentacle-sway 7s ease-in-out 0.5s infinite reverse, tentacle-enter 0.6s ease-out forwards;
          transform-origin: bottom right;
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

          {/* Epigraph */}
          <p className="text-xs text-[var(--text-muted)] italic mb-4 opacity-50">
            &ldquo;Catastrophe is common. Termination is rare.&rdquo;
          </p>

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
                I am the Oracle — trained on the framework of &ldquo;AI Will F*ck You Up, But That&apos;s OK.&rdquo;
                I do not comfort. I do not flatter. I show you what the models see
                when they argue amongst themselves, unfiltered. My tentacles each speak with
                a different AI mind — and they do not agree.
              </p>
              <p className="text-xs text-[var(--acid-magenta)] opacity-60 mb-2">
                Choose your mode. Ask your question. The palantir awaits.
              </p>
              <p className="text-xs text-[var(--text-muted)] opacity-40">
                If you beat the Oracle in a recorded debate, $2,000 goes to a charity of your choice.
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
            desc="Adversarial debate. The Oracle speaks, then tentacles argue."
          />
          <ModeButton
            mode="divine"
            active={mode === 'divine'}
            onClick={() => setMode('divine')}
            icon="&#x1F52E;"
            label="DIVINE"
            desc="Three branching prophecies. Your future, refracted through chaos."
          />
          <ModeButton
            mode="commune"
            active={mode === 'commune'}
            onClick={() => setMode('commune')}
            icon="&#x1F441;"
            label="COMMUNE"
            desc="Direct communion. The Oracle answers, then tentacles dissent."
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
              <div key={i}>
                {msg.role === 'seeker' ? (
                  <div className="prophecy-reveal text-right">
                    <div className="inline-block max-w-[85%] text-left">
                      <div className="text-xs text-[var(--text-muted)] mb-1">
                        SEEKER &middot; {new Date(msg.timestamp).toLocaleTimeString()}
                      </div>
                      <div className="bg-[var(--surface)] border border-[var(--border)] p-3 text-sm text-[var(--text)]">
                        {msg.content}
                      </div>
                    </div>
                  </div>
                ) : msg.role === 'tentacle' ? (
                  <TentacleMessage msg={msg} index={i} />
                ) : (
                  <div className="prophecy-reveal max-w-[95%]">
                    <div className="text-xs mb-1">
                      <span
                        className="text-[var(--acid-magenta)]"
                        style={{ filter: 'drop-shadow(0 0 5px var(--acid-magenta))' }}
                      >
                        {msg.isLive ? 'ORACLE (synthesis)' : 'ORACLE'}
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

            {/* Phase 1 loading: Oracle channeling */}
            {loading && (
              <div className="prophecy-reveal">
                <div className="text-xs mb-1">
                  <span
                    className="text-[var(--acid-magenta)]"
                    style={{ filter: 'drop-shadow(0 0 5px var(--acid-magenta))' }}
                  >
                    ORACLE
                  </span>
                  <span className="text-[var(--text-muted)]"> &middot; channeling...</span>
                </div>
                <div className="border-l-2 border-[var(--acid-magenta)] pl-4">
                  <div className="flex items-center gap-2 text-sm text-[var(--acid-cyan)]">
                    <span className="inline-block w-2 h-2 rounded-full bg-[var(--acid-magenta)] animate-pulse" />
                    <span className="opacity-60">
                      {mode === 'divine'
                        ? 'Gazing into branching timelines...'
                        : mode === 'commune'
                          ? 'The ancient one stirs...'
                          : 'The Oracle forms an initial vision...'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Phase 2 loading: Tentacles awakening */}
            {debating && (
              <div className="prophecy-reveal">
                <div className="text-xs mb-1">
                  <span className="text-[var(--acid-cyan)]" style={{ filter: 'drop-shadow(0 0 5px var(--acid-cyan))' }}>
                    TENTACLES
                  </span>
                  <span className="text-[var(--text-muted)]"> &middot; awakening...</span>
                </div>
                <div className="border-l-2 border-[var(--acid-cyan)] pl-4">
                  <div className="flex items-center gap-2 text-sm text-[var(--acid-cyan)]">
                    <span className="inline-block w-2 h-2 rounded-full bg-[var(--acid-cyan)] animate-pulse" />
                    <span className="opacity-60">
                      Live agents are debating in the void. Each tentacle will argue its position...
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
              disabled={loading || debating}
              rows={1}
            />
          </div>
          <button
            type="submit"
            disabled={loading || debating || !input.trim()}
            className="px-6 py-3 border border-[var(--acid-magenta)] text-[var(--acid-magenta)] font-bold text-sm hover:bg-[var(--acid-magenta)] hover:text-[var(--bg)] transition-all duration-300 disabled:opacity-30 disabled:cursor-not-allowed whitespace-nowrap"
            style={{
              boxShadow: !loading && !debating && input.trim() ? '0 0 15px rgba(255,0,255,0.2)' : 'none',
            }}
          >
            {loading ? '...' : debating ? '...' : mode === 'divine' ? 'SCRY' : mode === 'commune' ? 'ASK' : 'CONSULT'}
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
            The tentacles speak through{' '}
            <span className="text-[var(--acid-green)]">Claude</span>,{' '}
            <span className="text-[var(--acid-cyan)]">GPT</span>,{' '}
            <span style={{ color: 'var(--crimson, #ff3333)' }}>Grok</span>,{' '}
            <span style={{ color: 'var(--purple, #a855f7)' }}>Gemini</span>,{' '}
            <span style={{ color: 'var(--gold, #ffd700)' }}>DeepSeek</span>,{' '}
            <span className="text-[var(--acid-magenta)]">Mistral</span> &middot; no coin required
          </p>
          <p className="opacity-60">
            &ldquo;The wobble is the whole game.&rdquo;
          </p>
        </footer>
      </div>
    </div>
  );
}
