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
  critiques: Array<{ agent: string; target: string; text: string }>;
  votes: Array<{ agent: string; choice: string; confidence: number }>;
  final_answer: string;
  receipt_hash: string;
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

function parseSavedDebate(payload: unknown): SavedDebate | null {
  if (!payload || typeof payload !== 'object') {
    return null;
  }

  const debate = payload as Partial<SavedDebate>;
  if (
    typeof debate.id !== 'string'
    || typeof debate.topic !== 'string'
    || typeof debate.status !== 'string'
    || typeof debate.consensus_reached !== 'boolean'
    || typeof debate.confidence !== 'number'
    || typeof debate.verdict !== 'string'
    || typeof debate.duration_seconds !== 'number'
    || !Array.isArray(debate.participants)
    || typeof debate.proposals !== 'object'
    || debate.proposals === null
    || !Array.isArray(debate.critiques)
    || !Array.isArray(debate.votes)
    || typeof debate.final_answer !== 'string'
    || typeof debate.receipt_hash !== 'string'
  ) {
    return null;
  }

  return debate as SavedDebate;
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
