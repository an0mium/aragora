export interface SavedDebateMessage {
  agent: string;
  content: string;
  role?: string;
  round?: number;
}

export interface SavedDebateCritique {
  agent: string;
  target: string;
  text: string;
}

export interface SavedDebateVote {
  agent: string;
  choice: string;
  confidence: number;
}

/** Shape of the debate JSON returned by the backend API. */
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
  task?: string;
  messages?: SavedDebateMessage[];
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

function parseSavedDebate(payload: unknown): SavedDebate | null {
  if (!payload || typeof payload !== 'object') {
    return null;
  }

  const debate = payload as Record<string, unknown>;
  const id = readString(debate.id) ?? readString(debate.debate_id);
  const task = readString(debate.task);
  const topic = readString(debate.topic) ?? task;

  if (!id || !topic) {
    return null;
  }

  const messages = parseMessages(debate.messages ?? debate.transcript);
  const participants = parseParticipants(debate, messages);
  const finalAnswer =
    readString(debate.final_answer)
    ?? readString(debate.conclusion)
    ?? readString(debate.winning_proposal)
    ?? readString(debate.verdict)
    ?? '';
  const verdict =
    readString(debate.verdict)
    ?? readString(debate.winning_proposal)
    ?? finalAnswer
    ?? '';

  return {
    id,
    topic,
    status: readString(debate.status) ?? 'completed',
    consensus_reached: readBoolean(debate.consensus_reached) ?? readConsensusReached(debate.consensus),
    confidence: readNumber(debate.confidence) ?? readNumber(debate.agreement) ?? readConsensusConfidence(debate.consensus),
    verdict,
    duration_seconds: readNumber(debate.duration_seconds) ?? 0,
    participants,
    proposals: parseProposals(debate.proposals),
    critiques: parseCritiques(debate.critiques),
    votes: parseVotes(debate.votes),
    final_answer: finalAnswer,
    receipt_hash: readNullableString(debate.receipt_hash) ?? readNullableString(debate.signature),
    task: task ?? undefined,
    messages: messages.length > 0 ? messages : undefined,
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

function readString(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function readNullableString(value: unknown): string | null {
  if (value === null) {
    return null;
  }
  return readString(value);
}

function readBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function readNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function readConsensusReached(value: unknown): boolean {
  if (!value || typeof value !== 'object') {
    return false;
  }
  return readBoolean((value as Record<string, unknown>).reached) ?? false;
}

function readConsensusConfidence(value: unknown): number {
  if (!value || typeof value !== 'object') {
    return 0;
  }
  const consensus = value as Record<string, unknown>;
  return readNumber(consensus.confidence) ?? readNumber(consensus.agreement) ?? 0;
}

function parseParticipants(
  debate: Record<string, unknown>,
  messages: SavedDebateMessage[],
): string[] {
  const participants = parseStringArray(debate.participants);
  if (participants.length > 0) {
    return participants;
  }

  const agents = parseStringArray(debate.agents);
  if (agents.length > 0) {
    return agents;
  }

  const seen = new Set<string>();
  for (const message of messages) {
    if (message.agent) {
      seen.add(message.agent);
    }
  }
  return Array.from(seen);
}

function parseStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === 'string');
}

function parseProposals(value: unknown): Record<string, string> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {};
  }

  const proposals: Record<string, string> = {};
  for (const [agent, content] of Object.entries(value as Record<string, unknown>)) {
    if (typeof content === 'string') {
      proposals[agent] = content;
    }
  }
  return proposals;
}

function parseCritiques(value: unknown): SavedDebateCritique[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((entry) => {
    if (!entry || typeof entry !== 'object') {
      return [];
    }
    const critique = entry as Record<string, unknown>;
    const agent = readString(critique.agent) ?? readString(critique.author);
    const target = readString(critique.target) ?? '';
    const text =
      readString(critique.text)
      ?? readString(critique.reasoning)
      ?? readString(critique.content);

    if (!agent || !text) {
      return [];
    }

    return [{ agent, target, text }];
  });
}

function parseVotes(value: unknown): SavedDebateVote[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((entry) => {
    if (!entry || typeof entry !== 'object') {
      return [];
    }
    const vote = entry as Record<string, unknown>;
    const agent = readString(vote.agent);
    const choice = readString(vote.choice) ?? readString(vote.vote);
    if (!agent || !choice) {
      return [];
    }

    return [
      {
        agent,
        choice,
        confidence: readNumber(vote.confidence) ?? 0,
      },
    ];
  });
}

function parseMessages(value: unknown): SavedDebateMessage[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.flatMap((entry) => {
    if (!entry || typeof entry !== 'object') {
      return [];
    }

    const message = entry as Record<string, unknown>;
    const agent =
      readString(message.agent)
      ?? readString(message.author)
      ?? readString(message.name)
      ?? readString(message.role);
    const content =
      readString(message.content)
      ?? readString(message.text)
      ?? readString(message.message);

    if (!agent || !content) {
      return [];
    }

    return [
      {
        agent,
        content,
        role: readString(message.role) ?? undefined,
        round: readNumber(message.round) ?? undefined,
      },
    ];
  });
}
