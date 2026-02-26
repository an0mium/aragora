export interface DecisionPackage {
  id: string;
  question: string;
  verdict: string;
  confidence: number;
  consensus_reached: boolean;
  explanation: string;
  final_answer: string;
  agents: string[];
  rounds: number;
  arguments: Array<{
    agent: string;
    round: number;
    position: string;
    content: string;
  }>;
  cost_breakdown: Array<{
    agent: string;
    tokens: number;
    cost: number;
  }>;
  total_cost: number;
  receipt: {
    hash: string;
    timestamp: string;
    signers: string[];
  } | null;
  next_steps: string[];
  created_at: string;
  duration_seconds: number;
}

function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === 'string');
}

function normalizeArguments(value: unknown): DecisionPackage['arguments'] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      const obj = asObject(item);
      if (!obj) return null;
      return {
        agent: asString(obj.agent, 'unknown'),
        round: asNumber(obj.round, 0),
        position: asString(obj.position, ''),
        content: asString(obj.content, ''),
      };
    })
    .filter((item): item is DecisionPackage['arguments'][number] => item !== null);
}

function normalizeCostBreakdown(value: unknown): DecisionPackage['cost_breakdown'] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      const obj = asObject(item);
      if (!obj) return null;
      return {
        agent: asString(obj.agent, 'unknown'),
        tokens: asNumber(obj.tokens, 0),
        cost: asNumber(obj.cost, 0),
      };
    })
    .filter((item): item is DecisionPackage['cost_breakdown'][number] => item !== null);
}

function normalizeReceipt(value: unknown): DecisionPackage['receipt'] {
  const obj = asObject(value);
  if (!obj) return null;

  const hash = asString(obj.hash);
  const timestamp = asString(obj.timestamp);
  if (!hash || !timestamp) return null;

  return {
    hash,
    timestamp,
    signers: asStringArray(obj.signers),
  };
}

export function normalizeDecisionPackage(raw: unknown, fallbackId: string): DecisionPackage {
  const obj = asObject(raw) ?? {};

  return {
    id: asString(obj.id, fallbackId),
    question: asString(obj.question),
    verdict: asString(obj.verdict),
    confidence: asNumber(obj.confidence, 0),
    consensus_reached: Boolean(obj.consensus_reached),
    explanation: asString(obj.explanation),
    final_answer: asString(obj.final_answer),
    agents: asStringArray(obj.agents),
    rounds: asNumber(obj.rounds, 0),
    arguments: normalizeArguments(obj.arguments),
    cost_breakdown: normalizeCostBreakdown(obj.cost_breakdown),
    total_cost: asNumber(obj.total_cost, 0),
    receipt: normalizeReceipt(obj.receipt),
    next_steps: asStringArray(obj.next_steps),
    created_at: asString(obj.created_at, new Date().toISOString()),
    duration_seconds: asNumber(obj.duration_seconds, 0),
  };
}
