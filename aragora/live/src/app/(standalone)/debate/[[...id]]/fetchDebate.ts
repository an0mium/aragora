/** Shape of the debate JSON returned by the backend API. */
export interface SavedDebateCritique {
  agent: string;
  target_agent: string;
  issues: string[];
  suggestions: string[];
  severity: number;
}

export interface SavedDebateVote {
  agent: string;
  choice: string;
  confidence: number;
  reasoning?: string;
}

export interface SavedDebateMessage {
  agent: string;
  role: string;
  content: string;
  round: number;
  timestamp?: string | number;
}

export interface SavedDebate {
  id: string;
  topic: string;
  status: string;
  consensus_reached: boolean;
  confidence: number;
  verdict: string;
  duration_seconds: number;
  participants: string[];
  proposals: Record<string, string>;
  critiques: SavedDebateCritique[];
  votes: SavedDebateVote[];
  final_answer: string;
  receipt_hash: string | null;
  messages: SavedDebateMessage[];
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

function parseSavedDebate(payload: unknown): SavedDebate | null {
  if (!payload || typeof payload !== 'object') {
    return null;
  }

  const debate = payload as Record<string, unknown>;

  const messages = Array.isArray(debate.messages)
    ? debate.messages.flatMap((message): SavedDebateMessage[] => {
        if (!message || typeof message !== 'object') return [];
        const item = message as Record<string, unknown>;
        const agent = item.agent ?? item.author ?? item.name ?? item.role;
        const role = item.role ?? item.position ?? 'message';
        const content = item.content;
        const round = item.round ?? 0;
        if (typeof agent !== 'string' || typeof role !== 'string' || typeof content !== 'string') {
          return [];
        }
        return [{
          agent,
          role,
          content,
          round: typeof round === 'number' ? round : Number(round) || 0,
          timestamp:
            typeof item.timestamp === 'string' || typeof item.timestamp === 'number'
              ? item.timestamp
              : undefined,
        }];
      })
    : [];

  const participants = Array.isArray(debate.participants)
    ? debate.participants.filter((agent): agent is string => typeof agent === 'string')
    : Array.isArray(debate.agents)
      ? debate.agents.filter((agent): agent is string => typeof agent === 'string')
      : Array.from(
          new Set(
            messages
              .map((message) => message.agent)
              .filter((agent) => typeof agent === 'string' && agent.length > 0),
          ),
        );

  const proposalsSource = debate.proposals;
  const proposals = typeof proposalsSource === 'object' && proposalsSource !== null
    ? Object.fromEntries(
        Object.entries(proposalsSource as Record<string, unknown>)
          .filter(([agent, text]) => typeof agent === 'string' && text != null)
          .map(([agent, text]) => [agent, String(text)]),
      )
    : Object.fromEntries(
        messages
          .filter((message) => /propos|argument|synth/i.test(message.role))
          .map((message) => [message.agent, message.content]),
      );

  const critiques = Array.isArray(debate.critiques)
    ? debate.critiques.flatMap((critique): SavedDebateCritique[] => {
        if (!critique || typeof critique !== 'object') return [];
        const item = critique as Record<string, unknown>;
        const agent = item.agent ?? item.author;
        const targetAgent = item.target_agent ?? item.target ?? '';
        const issues = Array.isArray(item.issues)
          ? item.issues.filter((value): value is string => typeof value === 'string')
          : typeof item.reasoning === 'string'
            ? [item.reasoning]
            : typeof item.content === 'string'
              ? [item.content]
              : typeof item.text === 'string'
                ? [item.text]
                : [];
        const suggestions = Array.isArray(item.suggestions)
          ? item.suggestions.filter((value): value is string => typeof value === 'string')
          : [];
        const severity = typeof item.severity === 'number' ? item.severity : 0;
        if (typeof agent !== 'string' || typeof targetAgent !== 'string') {
          return [];
        }
        return [{
          agent,
          target_agent: targetAgent,
          issues,
          suggestions,
          severity,
        }];
      })
    : [];

  const votes = Array.isArray(debate.votes)
    ? debate.votes.flatMap((vote): SavedDebateVote[] => {
        if (!vote || typeof vote !== 'object') return [];
        const item = vote as Record<string, unknown>;
        if (
          typeof item.agent !== 'string'
          || typeof item.choice !== 'string'
          || typeof item.confidence !== 'number'
        ) {
          return [];
        }
        return [{
          agent: item.agent,
          choice: item.choice,
          confidence: item.confidence,
          reasoning: typeof item.reasoning === 'string' ? item.reasoning : undefined,
        }];
      })
    : [];

  const id = typeof debate.id === 'string'
    ? debate.id
    : typeof debate.debate_id === 'string'
      ? debate.debate_id
      : null;
  const topic = typeof debate.topic === 'string'
    ? debate.topic
    : typeof debate.task === 'string'
      ? debate.task
      : typeof debate.question === 'string'
        ? debate.question
        : null;
  const status = typeof debate.status === 'string' ? debate.status : 'completed';
  const confidence = typeof debate.confidence === 'number'
    ? debate.confidence
    : typeof debate.agreement === 'number'
      ? debate.agreement
      : 0;
  const verdict = typeof debate.verdict === 'string'
    ? debate.verdict
    : typeof debate.winning_proposal === 'string'
      ? debate.winning_proposal
      : '';
  const durationSeconds = typeof debate.duration_seconds === 'number'
    ? debate.duration_seconds
    : 0;
  const consensusReached = typeof debate.consensus_reached === 'boolean'
    ? debate.consensus_reached
    : typeof debate.consensus === 'object' && debate.consensus !== null
        && typeof (debate.consensus as Record<string, unknown>).reached === 'boolean'
      ? (debate.consensus as Record<string, unknown>).reached as boolean
    : false;
  const finalAnswer = typeof debate.final_answer === 'string'
    ? debate.final_answer
    : typeof debate.conclusion === 'string'
      ? debate.conclusion
      : typeof debate.winning_proposal === 'string'
        ? debate.winning_proposal
        : '';
  const receiptHash = typeof debate.receipt_hash === 'string'
      ? debate.receipt_hash
      : typeof debate.receipt === 'object' && debate.receipt !== null
        && typeof (debate.receipt as Record<string, unknown>).signature === 'string'
        ? (debate.receipt as Record<string, unknown>).signature as string
        : null;

  if (!id || !topic) {
    return null;
  }

  return {
    id,
    topic,
    status,
    consensus_reached: consensusReached,
    confidence,
    verdict,
    duration_seconds: durationSeconds,
    participants,
    proposals,
    critiques,
    votes,
    final_answer: finalAnswer,
    receipt_hash: receiptHash,
    messages,
  };
}

async function fetchDebateFromCandidateUrls(
  debateId: string,
  init: RequestInit,
): Promise<SavedDebate | null> {
  const encodedDebateId = encodeURIComponent(debateId);
  const urls = [
    `${API_BASE}/api/v1/debates/public/${encodedDebateId}`,
    `${API_BASE}/api/v1/playground/debate/${encodedDebateId}`,
  ];

  for (const url of urls) {
    try {
      const res = await fetch(url, init);
      if (!res.ok) continue;
      const data = await res.json();
      const debate = parseSavedDebate(data?.data ?? data);
      if (debate) {
        return debate;
      }
    } catch {
      // Try the next candidate URL
    }
  }

  return null;
}

/**
 * Fetch a saved debate from the backend API (server-side).
 *
 * Tries the public viewer endpoint first (no auth required, checks shareability),
 * then falls back to the playground endpoint for backward compatibility.
 * Returns null when the debate cannot be fetched (not found, API down, etc.).
 */
export async function fetchDebate(
  debateId: string,
): Promise<SavedDebate | null> {
  return fetchDebateFromCandidateUrls(debateId, { next: { revalidate: 300 } });
}

/**
 * Fetch a saved debate from the browser runtime.
 *
 * Used by the standalone viewer as a fail-soft recovery path when the initial
 * server-side preload misses but the permalink still resolves publicly.
 */
export async function fetchDebateClient(
  debateId: string,
): Promise<SavedDebate | null> {
  return fetchDebateFromCandidateUrls(debateId, { cache: 'no-store' });
}
