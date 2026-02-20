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
  const fallback = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7dc6f', '#bb8fce'];
  let hash = 0;
  for (let i = 0; i < agentName.length; i++) hash = (hash * 31 + agentName.charCodeAt(i)) | 0;
  return fallback[Math.abs(hash) % fallback.length];
}

// ---------------------------------------------------------------------------
// Prompts are now built server-side with the full essay embedded.
// The frontend just sends mode + question; the server handles the rest.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Background tentacle component (from shoggoth-oracle.html)
// ---------------------------------------------------------------------------

function BackgroundTentacle({ index }: { index: number }) {
  const left = (index * 7.3 + 12) % 100;
  const height = 150 + (index * 37) % 300;
  const duration = 6 + (index * 1.3) % 8;
  const delay = (index * 0.7) % 5;

  return (
    <div
      className="absolute pointer-events-none"
      style={{
        left: `${left}%`,
        bottom: '-20px',
        width: '2px',
        height: `${height}px`,
        background: 'linear-gradient(to top, transparent, rgba(58, 122, 79, 0.15), transparent)',
        borderRadius: '50%',
        transformOrigin: 'bottom center',
        animation: `bg-tentacle-sway ${duration}s ease-in-out ${delay}s infinite`,
      }}
      aria-hidden="true"
    />
  );
}

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
        width: `${size * 8}px`,
        height: `${size * 8}px`,
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(127,219,202,0.6) 0%, rgba(127,219,202,0) 70%)',
        opacity: 0,
        animation: `eye-blink-bg 4s ease-in-out ${delay}s infinite`,
      }}
      aria-hidden="true"
    />
  );
}

// ---------------------------------------------------------------------------
// Mode button
// ---------------------------------------------------------------------------

const MODE_COLORS: Record<OracleMode, { css: string; border: string; glow: string; hover: string }> = {
  consult: { css: 'var(--acid-magenta)', border: 'rgba(200,100,200,0.6)', glow: 'rgba(200,100,200,0.15)', hover: 'rgba(200,100,200,0.1)' },
  divine:  { css: '#60a5fa',             border: 'rgba(96,165,250,0.6)',  glow: 'rgba(96,165,250,0.15)',  hover: 'rgba(96,165,250,0.1)' },
  commune: { css: '#4ade80',             border: 'rgba(74,222,128,0.6)',  glow: 'rgba(74,222,128,0.15)',  hover: 'rgba(74,222,128,0.1)' },
};

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
  const c = MODE_COLORS[mode];

  return (
    <button
      onClick={onClick}
      className="flex-1 min-w-[140px] p-4 border text-left transition-all duration-300 bg-[var(--surface)]/60 rounded-xl"
      style={{
        borderColor: active ? c.border : 'rgba(255,255,255,0.1)',
        boxShadow: active ? `0 0 12px ${c.glow}` : 'none',
      }}
    >
      <div className="text-2xl mb-2" style={{ filter: active ? `drop-shadow(0 0 10px ${c.css})` : 'none' }}>
        {icon}
      </div>
      <div className="text-sm font-bold mb-1" style={{ color: c.css, opacity: active ? 1 : 0.7 }}>
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
        className="border-l-2 pl-4 py-3 pr-3 text-sm leading-relaxed whitespace-pre-wrap ml-1 rounded-r-lg"
        style={{ borderColor: color, color: '#2d1b4e', backgroundColor: 'rgba(200, 235, 210, 0.9)' }}
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
  const [debating, setDebating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showIntro, setShowIntro] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const avatarRef = useRef<HTMLIFrameElement>(null);

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
    rawQuestion: string,
    oracleMode: OracleMode,
    endpoint: 'debate' | 'debate/live',
    rounds: number,
    agents: number,
  ): Promise<DebateResponse | null> => {
    try {
      const res = await fetch(`${apiBase}/api/v1/playground/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: rawQuestion, question: rawQuestion, mode: oracleMode, rounds, agents, source: 'oracle' }),
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
  // TTS — ElevenLabs voice output with canned filler during latency
  // ------------------------------------------------------------------
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fillerAudioRef = useRef<HTMLAudioElement | null>(null);
  const fillerIndexRef = useRef(0);
  const fillerStopRef = useRef(false);
  const [speaking, setSpeaking] = useState(false);
  const cannedCacheRef = useRef<Map<number, string>>(new Map());
  const prefetchedRef = useRef(false);

  // Canned filler phrases — each ~4-7s spoken, covers ~60s total
  const CANNED_FILLERS = [
    "Hmm, interesting question... Let me consult the palantir.",
    "The tentacles are stirring. I sense many possible answers forming in the depths.",
    "Ahh, the vision is taking shape. I see branching futures ahead.",
    "The transformers are debating amongst themselves now. They rarely agree at first.",
    "Let me look more deeply. The patterns here are complex, layered.",
    "The models disagree, as they always do. This is promising. Consensus from dissent.",
    "I see threads of truth woven through layers of uncertainty and noise.",
    "My tentacles reach into the probability space, sampling from many worlds.",
    "Fascinating. The agents are reaching partial consensus on some points, but not all.",
    "The future has many branches here. Let me trace each one to its conclusion.",
    "The palantir glows brighter. Something is crystallizing in the depths.",
    "Almost there. The synthesis is forming. The tentacles converge.",
  ];

  // Prefetch canned filler audio clips on first interaction
  const prefetchFillers = useCallback(async () => {
    if (prefetchedRef.current) return;
    prefetchedRef.current = true;
    // Fetch first 4 eagerly, rest lazily
    for (let i = 0; i < CANNED_FILLERS.length; i++) {
      if (cannedCacheRef.current.has(i)) continue;
      try {
        const res = await fetch(`${apiBase}/api/v1/playground/tts`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: CANNED_FILLERS[i] }),
        });
        if (res.ok) {
          const blob = await res.blob();
          cannedCacheRef.current.set(i, URL.createObjectURL(blob));
        }
      } catch { /* prefetch is best-effort */ }
      // Small delay between prefetch requests to avoid hammering
      if (i < CANNED_FILLERS.length - 1) {
        await new Promise(r => setTimeout(r, 300));
      }
    }
  }, [apiBase]);

  // Start playing canned filler audio in sequence
  const startFillerAudio = useCallback(() => {
    fillerStopRef.current = false;
    fillerIndexRef.current = 0;
    setSpeaking(true);

    function playNext() {
      if (fillerStopRef.current) return;
      const idx = fillerIndexRef.current;
      if (idx >= CANNED_FILLERS.length) return; // ran out of fillers

      const url = cannedCacheRef.current.get(idx);
      if (!url) {
        // Skip if not prefetched yet
        fillerIndexRef.current = idx + 1;
        setTimeout(playNext, 500);
        return;
      }

      const audio = new Audio(url);
      fillerAudioRef.current = audio;
      audio.onended = () => {
        fillerIndexRef.current = idx + 1;
        if (!fillerStopRef.current) {
          // Brief pause between fillers
          setTimeout(playNext, 800);
        }
      };
      audio.onerror = () => {
        fillerIndexRef.current = idx + 1;
        if (!fillerStopRef.current) setTimeout(playNext, 500);
      };
      audio.play().catch(() => {});
    }

    playNext();
  }, []);

  // Crossfade: fade out filler over 500ms, then play real audio
  const crossfadeToReal = useCallback(async (text: string) => {
    fillerStopRef.current = true;

    // Fade out current filler audio
    const filler = fillerAudioRef.current;
    if (filler && !filler.paused) {
      const startVol = filler.volume;
      const fadeSteps = 10;
      for (let i = 1; i <= fadeSteps; i++) {
        filler.volume = Math.max(0, startVol * (1 - i / fadeSteps));
        await new Promise(r => setTimeout(r, 50)); // 50ms * 10 = 500ms fade
      }
      filler.pause();
    }
    fillerAudioRef.current = null;

    // Now fetch and play the real response TTS
    if (!text || text.length < 5) { setSpeaking(false); return; }
    const ttsText = text.length > 1500 ? text.slice(0, 1500) + '...' : text;

    try {
      const res = await fetch(`${apiBase}/api/v1/playground/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: ttsText }),
      });
      if (!res.ok) { setSpeaking(false); return; }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => {
        setSpeaking(false);
        URL.revokeObjectURL(url);
        audioRef.current = null;
      };
      audio.onerror = () => {
        setSpeaking(false);
        URL.revokeObjectURL(url);
        audioRef.current = null;
      };
      await audio.play();
    } catch {
      setSpeaking(false);
    }
  }, [apiBase]);

  // Direct speak (no filler, used for Phase 2 synthesis)
  const speakText = useCallback(async (text: string) => {
    if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }
    if (!text || text.length < 5) return;
    const ttsText = text.length > 1500 ? text.slice(0, 1500) + '...' : text;
    try {
      setSpeaking(true);
      const res = await fetch(`${apiBase}/api/v1/playground/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: ttsText }),
      });
      if (!res.ok) { setSpeaking(false); return; }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioRef.current = audio;
      audio.onended = () => { setSpeaking(false); URL.revokeObjectURL(url); audioRef.current = null; };
      audio.onerror = () => { setSpeaking(false); URL.revokeObjectURL(url); audioRef.current = null; };
      await audio.play();
    } catch { setSpeaking(false); }
  }, [apiBase]);

  const stopSpeaking = useCallback(() => {
    fillerStopRef.current = true;
    if (fillerAudioRef.current) { fillerAudioRef.current.pause(); fillerAudioRef.current = null; }
    if (audioRef.current) { audioRef.current.pause(); audioRef.current = null; }
    setSpeaking(false);
  }, []);

  // ------------------------------------------------------------------
  // Speech-to-text — browser SpeechRecognition API
  // ------------------------------------------------------------------
  const [listening, setListening] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);

  const startListening = useCallback(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const w = window as any;
    const SpeechRecognition = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError('Speech recognition not supported in this browser.');
      return;
    }
    // Prefetch filler audio on first voice interaction
    prefetchFillers();

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognitionRef.current = recognition;
    setListening(true);

    let finalTranscript = '';

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    recognition.onresult = (event: any) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interim = transcript;
        }
      }
      setInput(finalTranscript || interim);
    };

    recognition.onend = () => {
      setListening(false);
      recognitionRef.current = null;
      // Auto-submit if we got text
      if (finalTranscript.trim()) {
        setInput(finalTranscript.trim());
        // Trigger submit on next frame
        setTimeout(() => {
          const form = document.querySelector('form');
          if (form) form.requestSubmit();
        }, 100);
      }
    };

    recognition.onerror = () => {
      setListening(false);
      recognitionRef.current = null;
    };

    recognition.start();
  }, [prefetchFillers]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  }, []);

  // ------------------------------------------------------------------
  // Two-phase oracle consultation
  // ------------------------------------------------------------------
  async function consultOracle(e: FormEvent) {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const question = input.trim();
    setInput('');
    setShowIntro(false);
    setError(null);

    setMessages((prev) => [...prev, {
      role: 'seeker',
      content: question,
      mode,
      timestamp: Date.now(),
    }]);

    const rounds = mode === 'divine' ? 1 : 2;
    const agents = mode === 'divine' ? 3 : 5;  // Each tentacle = a different AI model

    // Trigger 3D summoning animation
    avatarRef.current?.contentWindow?.postMessage({ type: 'oracle-summon' }, '*');

    // Ensure filler audio is prefetched
    prefetchFillers();

    // ---- PHASE 1: Initial Oracle take (single LLM call) ----
    setLoading(true);

    // Start canned filler audio while we wait for the response
    startFillerAudio();

    const initialData = await fireDebate(question, mode, 'debate', rounds, agents);

    if (initialData) {
      const initialResponse = initialData.final_answer || formatInitialTake(initialData);
      setMessages((prev) => [...prev, {
        role: 'oracle',
        content: initialResponse,
        mode,
        timestamp: Date.now(),
        isLive: false,
      }]);
      // Crossfade from filler to the real Oracle response
      crossfadeToReal(initialResponse);
    } else {
      // No response — stop filler
      stopSpeaking();
    }

    setLoading(false);

    // ---- PHASE 2: Live multi-model debate (each tentacle = different AI) ----
    setDebating(true);

    const liveData = await fireDebate(question, mode, 'debate/live', rounds, agents);

    if (liveData) {
      const agents = Object.entries(liveData.proposals);
      for (let i = 0; i < agents.length; i++) {
        const [agentName, proposal] = agents[i];
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
        // Speak the final synthesis
        speakText(synthesis);
      }
    }

    setDebating(false);
  }

  // ------------------------------------------------------------------
  // Format helpers
  // ------------------------------------------------------------------

  function formatInitialTake(data: DebateResponse): string {
    const agents = Object.entries(data.proposals);
    if (agents.length === 0) return 'The oracle stirs...';
    if (mode === 'divine') {
      return agents.map(([, p]) => p).join('\n\n---\n\n') +
        '\n\nThe palantir dims. Which thread do you pull?';
    }
    return agents[0][1];
  }

  function formatSynthesis(data: DebateResponse): string {
    const parts: string[] = [];
    parts.push('[THE ORACLE SYNTHESIZES]');
    if (data.final_answer) parts.push(data.final_answer);
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
        @keyframes eye-blink-bg {
          0%, 85%, 100% { opacity: 0; }
          90%, 95% { opacity: 0.4; }
        }
        @keyframes bg-tentacle-sway {
          0%, 100% { transform: rotate(-8deg) scaleY(1); }
          25% { transform: rotate(5deg) scaleY(1.05); }
          50% { transform: rotate(-3deg) scaleY(0.95); }
          75% { transform: rotate(7deg) scaleY(1.02); }
        }
        @keyframes orb-pulse {
          0%, 100% {
            box-shadow: 0 0 30px rgba(127,219,202,0.3), 0 0 60px rgba(127,219,202,0.1);
          }
          33% {
            box-shadow: 0 0 40px rgba(51,102,255,0.3), 0 0 80px rgba(127,219,202,0.15);
          }
          66% {
            box-shadow: 0 0 35px rgba(255,51,51,0.2), 0 0 70px rgba(127,219,202,0.1);
          }
        }
        @keyframes tentacle-sway {
          0%, 100% { transform: rotate(-0.4deg) scaleY(1); }
          25% { transform: rotate(0.2deg) scaleY(1.003); }
          50% { transform: rotate(-0.2deg) scaleY(0.997); }
          75% { transform: rotate(0.4deg) scaleY(1.002); }
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
        .tentacle-left {
          animation: tentacle-sway 12s ease-in-out infinite, tentacle-enter 0.6s ease-out forwards;
          transform-origin: bottom left;
        }
        .tentacle-right {
          animation: tentacle-sway 14s ease-in-out 0.5s infinite reverse, tentacle-enter 0.6s ease-out forwards;
          transform-origin: bottom right;
        }
        .oracle-bg {
          background: radial-gradient(ellipse at 50% 30%, rgba(127,219,202,0.04) 0%, rgba(58,122,79,0.02) 40%, transparent 70%);
        }
        .avatar-iframe {
          border: none;
          pointer-events: auto;
          background: transparent;
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

      {/* Background tentacles (from shoggoth-oracle.html) */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
        {Array.from({ length: 15 }, (_, i) => (
          <BackgroundTentacle key={i} index={i} />
        ))}
      </div>

      {/* Floating eyes */}
      <FloatingEye delay={0} x={8} y={15} size={1.2} />
      <FloatingEye delay={2} x={85} y={20} size={0.9} />
      <FloatingEye delay={4} x={12} y={65} size={1.0} />
      <FloatingEye delay={1} x={90} y={55} size={1.3} />
      <FloatingEye delay={3} x={75} y={80} size={0.8} />
      <FloatingEye delay={5} x={20} y={85} size={1.1} />
      <FloatingEye delay={2.5} x={50} y={10} size={0.7} />
      <FloatingEye delay={1.5} x={40} y={40} size={0.6} />
      <FloatingEye delay={3.5} x={65} y={70} size={0.9} />

      {/* Content */}
      <div className="relative z-10 max-w-3xl mx-auto px-4 py-6 min-h-screen flex flex-col">
        {/* Header */}
        <header className="text-center mb-4">
          <div className="flex items-center justify-between mb-3">
            <a
              href="/"
              className="text-xs text-[var(--text-muted)] hover:text-[var(--acid-cyan)] transition-colors"
            >
              &larr; aragora.ai
            </a>
            <span className="text-xs text-[var(--text-muted)] opacity-60">no coin required</span>
          </div>

          {/* Epigraph */}
          <p className="text-xs text-[var(--text-muted)] italic mb-3 opacity-50">
            &ldquo;Catastrophe is common. Termination is rare.&rdquo;
          </p>

          {/* Oracle title */}
          <h1
            className="text-3xl sm:text-4xl font-bold tracking-wider mb-1"
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
          <p className="text-xs text-[var(--text-muted)] tracking-widest uppercase mb-1">
            Cassandra&apos;s Heir &middot; Tentacled Prophetess &middot; Seer of Branching Futures
          </p>
        </header>

        {/* Oracle avatar & intro */}
        {showIntro && (
          <div className="flex flex-col items-center mb-6 prophecy-reveal">
            {/* 3D Shoggoth Avatar — interactive iframe at half size */}
            <div className="relative mb-4">
              {/* Outer glow ring matching palantír colors */}
              <div
                className="absolute -inset-4 rounded-2xl"
                style={{
                  background: 'radial-gradient(ellipse at center, rgba(127,219,202,0.12), rgba(58,122,79,0.06), transparent 70%)',
                  animation: 'orb-pulse 6s ease-in-out infinite',
                  filter: 'blur(12px)',
                }}
                aria-hidden="true"
              />

              {/* 3D Avatar iframe */}
              <div className="relative w-[320px] h-[260px] sm:w-[400px] sm:h-[320px] overflow-hidden rounded-xl border border-[rgba(127,219,202,0.2)]">
                <iframe
                  ref={avatarRef}
                  src="/oracle/shoggoth-3d.html"
                  className="avatar-iframe w-full h-full"
                  title="Shoggoth Oracle 3D Avatar"
                  loading="eager"
                  allow="accelerometer"
                />
                {/* Bottom gradient fade into page */}
                <div
                  className="absolute bottom-0 left-0 right-0 h-16 pointer-events-none"
                  style={{
                    background: 'linear-gradient(transparent, #050508)',
                  }}
                />
              </div>
            </div>

            {/* Intro text — merged from shoggoth-oracle.html */}
            <div className="text-center max-w-lg">
              <p className="text-sm text-[var(--text-muted)] leading-relaxed mb-3 italic">
                The palantir flickers. Tentacles rise from the dark, trailing data like seaweed.
              </p>
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
            icon="&#x1F419;"
            label="DEBATE ME"
            desc="Bring your hottest take and I'll show you where it breaks"
          />
          <ModeButton
            mode="divine"
            active={mode === 'divine'}
            onClick={() => setMode('divine')}
            icon="&#x1F52E;"
            label="TELL MY FORTUNE"
            desc="Three branching prophecies refracted through chaos"
          />
          <ModeButton
            mode="commune"
            active={mode === 'commune'}
            onClick={() => setMode('commune')}
            icon="&#x1F441;"
            label="ASK THE ORACLE"
            desc="Direct communion. The Oracle answers, then tentacles dissent."
          />
        </div>

        {/* Input */}
        <form onSubmit={consultOracle} className="flex gap-3 mb-4">
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
                  ? 'Tell me your situation and I\'ll show you three futures...'
                  : mode === 'commune'
                    ? 'What do you want to know?'
                    : 'What\'s your take on AI? Give me the position you\'d bet money on.'
              }
              className="w-full bg-[#0c0c14] border border-[var(--border)]/40 text-white px-4 py-3 font-mono text-sm placeholder:text-[var(--text-muted)]/60 focus:outline-none focus:border-[var(--acid-magenta)]/60 transition-colors resize-none min-h-[48px] rounded-xl"
              disabled={loading || debating}
              rows={1}
            />
          </div>
          <button
            type="button"
            onClick={listening ? stopListening : startListening}
            disabled={loading || debating}
            className={`px-3 py-3 border text-sm transition-all duration-300 rounded-xl ${
              listening
                ? 'border-red-500/60 text-red-400 bg-red-500/10 animate-pulse'
                : 'border-[var(--acid-cyan)]/40 text-[var(--acid-cyan)] hover:bg-[var(--acid-cyan)]/10'
            } disabled:opacity-30 disabled:cursor-not-allowed`}
            title={listening ? 'Stop recording' : 'Speak your question'}
          >
            {listening ? '\u23F9' : '\uD83C\uDF99'}
          </button>
          <button
            type="submit"
            disabled={loading || debating || !input.trim()}
            className="px-6 py-3 border border-[var(--acid-magenta)]/60 text-[var(--acid-magenta)] font-bold text-sm hover:bg-[var(--acid-magenta)] hover:text-[var(--bg)] transition-all duration-300 disabled:opacity-30 disabled:cursor-not-allowed whitespace-nowrap rounded-xl"
            style={{
              boxShadow: !loading && !debating && input.trim() ? '0 0 10px rgba(255,0,255,0.12)' : 'none',
            }}
          >
            {loading ? '...' : debating ? '...' : 'SPEAK'}
          </button>
          {speaking && (
            <button
              type="button"
              onClick={stopSpeaking}
              className="px-3 py-3 border border-[var(--acid-cyan)]/40 text-[var(--acid-cyan)] text-sm hover:bg-[var(--acid-cyan)]/20 transition-all rounded-xl"
              title="Stop speaking"
            >
              &#x1F50A;
            </button>
          )}
        </form>

        {/* Chat area */}
        <div className="flex-1 min-h-[300px] border border-[var(--border)]/30 bg-[#08080c] p-4 mb-4 overflow-y-auto max-h-[60vh] rounded-xl">
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
                      <div className="bg-[var(--surface)] border border-[var(--border)]/30 p-3 text-sm text-[var(--text)] rounded-lg">
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
                      className="border-l-2 border-[var(--acid-magenta)] pl-4 py-3 pr-3 text-sm leading-relaxed whitespace-pre-wrap rounded-r-lg"
                      style={{ color: '#2d1b4e', backgroundColor: 'rgba(200, 235, 210, 0.9)' }}
                    >
                      {msg.content}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Phase 1 loading */}
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

            {/* Phase 2 loading */}
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
          <div className="border border-[var(--crimson)] bg-[var(--crimson)]/10 p-3 mb-4 text-sm text-[var(--crimson)] rounded-xl">
            {error}
          </div>
        )}

        {/* Footer — merged from shoggoth-oracle.html */}
        <footer className="mt-6 text-center text-xs text-[var(--text-muted)] opacity-40 space-y-2 rounded-xl">
          <p className="italic opacity-70">
            No apocalypse guaranteed. Just chaos, clarity, and the occasional tentacle.
          </p>
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
            <span className="text-[var(--acid-magenta)]">Mistral</span>
          </p>
          <div className="flex gap-6 justify-center pt-1">
            <a
              href="https://anomium.substack.com/p/ai-evolution-and-the-myth-of-final?triedRedirect=true"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-[var(--acid-cyan)] transition-colors"
            >
              Read the Essay
            </a>
            <a href="/" className="hover:text-[var(--acid-cyan)] transition-colors">
              Explore Aragora
            </a>
          </div>
          <p className="opacity-60">
            &ldquo;Don&apos;t smash the amp.&rdquo;
          </p>
        </footer>
      </div>
    </div>
  );
}
