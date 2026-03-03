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

/**
 * Fetch a saved debate from the backend API (server-side).
 * Returns null when the debate cannot be fetched (not found, API down, etc.).
 */
export async function fetchDebate(
  debateId: string,
): Promise<SavedDebate | null> {
  try {
    const res = await fetch(
      `${API_BASE}/api/v1/playground/debate/${debateId}`,
      { next: { revalidate: 300 } },
    );
    if (!res.ok) return null;
    const data = await res.json();
    // API may wrap in { data: ... } or return the debate directly
    return (data?.data ?? data) as SavedDebate;
  } catch {
    return null;
  }
}
