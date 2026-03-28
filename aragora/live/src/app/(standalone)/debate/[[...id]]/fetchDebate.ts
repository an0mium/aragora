/** Shape of the debate JSON returned by the backend API. */
export interface SavedDebateMessage {
  agent: string;
  role: string;
  content: string;
  round: number;
  target?: string;
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
  critiques: Array<{ agent: string; target: string; text: string }>;
  votes: Array<{ agent: string; choice: string; confidence: number }>;
  final_answer: string;
  receipt_hash: string | null;
  messages?: SavedDebateMessage[];
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function asString(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function normalizeStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === 'string');
}

function normalizeProposalMap(value: unknown): Record<string, string> {
  const obj = asObject(value);
  if (!obj) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(obj).filter((entry): entry is [string, string] => typeof entry[1] === 'string'),
  );
}

function normalizeMessages(value: unknown): SavedDebateMessage[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => {
      const obj = asObject(item);
      if (!obj) {
        return null;
      }

      const agent =
        asString(obj.agent)
        || asString(obj.author)
        || asString(obj.position)
        || asString(obj.role)
        || 'unknown';
      const role = asString(obj.role) || asString(obj.position) || 'message';
      const content = asString(obj.content);
      const round = asNumber(obj.round) ?? 0;
      const target = asString(obj.target) || undefined;

      if (!content) {
        return null;
      }

      return {
        agent,
        role,
        content,
        round,
        target,
      };
    })
    .filter((item): item is SavedDebateMessage => item !== null);
}

function deriveProposals(messages: SavedDebateMessage[]): Record<string, string> {
  const grouped = new Map<string, string[]>();

  messages.forEach((message) => {
    const role = message.role.toLowerCase();
    if (role.includes('critique') || role.includes('critic')) {
      return;
    }

    const existing = grouped.get(message.agent) || [];
    existing.push(message.content);
    grouped.set(message.agent, existing);
  });

  return Object.fromEntries(
    Array.from(grouped.entries()).map(([agent, parts]) => [agent, parts.join('\n\n')]),
  );
}

function normalizeCritiques(value: unknown, messages: SavedDebateMessage[]): Array<{ agent: string; target: string; text: string }> {
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        const obj = asObject(item);
        if (!obj) {
          return null;
        }

        const agent = asString(obj.agent) || asString(obj.author);
        const text =
          asString(obj.text)
          || asString(obj.reasoning)
          || asString(obj.content);
        if (!agent || !text) {
          return null;
        }

        return {
          agent,
          target: asString(obj.target) || '',
          text,
        };
      })
      .filter((item): item is { agent: string; target: string; text: string } => item !== null);
  }

  return messages
    .filter((message) => {
      const role = message.role.toLowerCase();
      return role.includes('critique') || role.includes('critic');
    })
    .map((message) => ({
      agent: message.agent,
      target: message.target || '',
      text: message.content,
    }));
}

function normalizeVotes(value: unknown): Array<{ agent: string; choice: string; confidence: number }> {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => {
      const obj = asObject(item);
      if (!obj) {
        return null;
      }

      const agent = asString(obj.agent);
      const choice = asString(obj.choice);
      if (!agent || !choice) {
        return null;
      }

      return {
        agent,
        choice,
        confidence: asNumber(obj.confidence) ?? 0,
      };
    })
    .filter((item): item is { agent: string; choice: string; confidence: number } => item !== null);
}

function parseSavedDebate(payload: unknown): SavedDebate | null {
  const debate = asObject(payload);
  if (!debate) {
    return null;
  }

  const result = asObject(debate.result);
  const messages = normalizeMessages(debate.messages);
  const directParticipants = normalizeStringArray(debate.participants);
  const storedParticipants = normalizeStringArray(result?.participants);
  const agentParticipants = normalizeStringArray(debate.agents);
  const participants = directParticipants.length > 0
    ? directParticipants
    : storedParticipants.length > 0
      ? storedParticipants
      : agentParticipants;

  const proposals = (() => {
    const direct = normalizeProposalMap(debate.proposals);
    return Object.keys(direct).length > 0 ? direct : deriveProposals(messages);
  })();

  const normalized: SavedDebate = {
    id: asString(debate.id) || asString(debate.debate_id) || '',
    topic:
      asString(debate.topic)
      || asString(debate.question)
      || asString(debate.task)
      || '',
    status: asString(debate.status) || 'completed',
    consensus_reached:
      asBoolean(debate.consensus_reached)
      ?? asBoolean(result?.consensus_reached)
      ?? false,
    confidence:
      asNumber(debate.confidence)
      ?? asNumber(result?.confidence)
      ?? 0,
    verdict:
      asString(debate.verdict)
      || asString(result?.verdict)
      || asString(result?.final_answer)
      || asString(debate.final_answer)
      || '',
    duration_seconds:
      asNumber(debate.duration_seconds)
      ?? asNumber(result?.duration_seconds)
      ?? 0,
    participants,
    proposals,
    critiques: normalizeCritiques(debate.critiques, messages),
    votes: normalizeVotes(debate.votes),
    final_answer:
      asString(debate.final_answer)
      || asString(result?.final_answer)
      || '',
    receipt_hash: (() => {
      const receiptHash = debate.receipt_hash;
      if (receiptHash === null) {
        return null;
      }
      return asString(receiptHash);
    })(),
    messages,
  };

  if (
    !normalized.id
    || !normalized.topic
    || !normalized.status
    || !normalized.verdict
    || !normalized.final_answer
  ) {
    return null;
  }

  return normalized;
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
