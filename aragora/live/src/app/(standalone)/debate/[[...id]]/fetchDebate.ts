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

async function fetchDebateFromCandidateUrls(
  debateId: string,
  init: RequestInit,
): Promise<SavedDebate | null> {
  const urls = [
    `${API_BASE}/api/v1/debates/public/${debateId}`,
    `${API_BASE}/api/v1/playground/debate/${debateId}`,
  ];

  for (const url of urls) {
    try {
      const res = await fetch(url, init);
      if (!res.ok) continue;
      const data = await res.json();
      return (data?.data ?? data) as SavedDebate;
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
