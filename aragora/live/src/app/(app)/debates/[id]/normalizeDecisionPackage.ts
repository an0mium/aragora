export interface DecisionPackageReceiptCostAgentModel {
  model: string;
  call_count: number;
}

export interface DecisionPackageReceiptCostAgent {
  agent: string;
  total_cost_usd: number;
  total_tokens_in: number;
  total_tokens_out: number;
  call_count: number;
  models_used: DecisionPackageReceiptCostAgentModel[];
}

export interface DecisionPackageReceiptModelUsage {
  key: string;
  label: string;
  provider: string;
  model: string;
  total_cost_usd: number;
  total_tokens_in: number;
  total_tokens_out: number;
  call_count: number;
}

export interface DecisionPackageReceiptCostSummary {
  total_cost_usd: number | null;
  total_tokens_in: number;
  total_tokens_out: number;
  total_calls: number;
  per_agent: DecisionPackageReceiptCostAgent[];
  model_usage: DecisionPackageReceiptModelUsage[];
}

export interface DecisionPackageReceipt {
  hash: string;
  timestamp: string;
  signers: string[];
  cost_summary: DecisionPackageReceiptCostSummary | null;
}

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
  receipt: DecisionPackageReceipt | null;
  next_steps: Array<{
    action: string;
    priority: 'high' | 'medium' | 'low';
  }>;
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
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === 'string');
}

function hasObjectEntries(value: Record<string, unknown> | null): value is Record<string, unknown> {
  return value !== null && Object.keys(value).length > 0;
}

function tokenMapFromCostSummary(value: unknown): Record<string, unknown> | null {
  const obj = asObject(value);
  const perAgent = asObject(obj?.per_agent);
  if (!perAgent) return null;

  const tokenEntries: Array<[string, unknown]> = [];
  for (const [agent, summary] of Object.entries(perAgent)) {
    const summaryObj = asObject(summary);
    if (!summaryObj) continue;

    const totalTokens = summaryObj.total_tokens ?? (
      asNumber(summaryObj.total_tokens_in, 0) + asNumber(summaryObj.total_tokens_out, 0)
    );

    tokenEntries.push([agent, totalTokens]);
  }

  const tokenMap = Object.fromEntries(tokenEntries) as Record<string, unknown>;

  return Object.keys(tokenMap).length > 0 ? tokenMap : null;
}

function formatModelLabel(provider: string, model: string, fallback = ''): string {
  if (provider && model) return `${provider}/${model}`;
  if (model) return model;
  if (provider) return provider;
  return fallback;
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

function normalizeCostBreakdown(
  value: unknown,
  tokenMap: Record<string, unknown> | null = null
): DecisionPackage['cost_breakdown'] {
  if (Array.isArray(value)) {
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

  const obj = asObject(value);
  const perAgentCost = asObject(obj?.per_agent_cost);
  if (perAgentCost) {
    return Object.entries(perAgentCost).map(([agent, cost]) => ({
      agent,
      tokens: asNumber(tokenMap?.[agent], 0),
      cost: asNumber(cost, 0),
    }));
  }

  const perAgent = asObject(obj?.per_agent);
  if (!perAgent) return [];

  return Object.entries(perAgent).map(([agent, summary]) => {
    const summaryObj = asObject(summary);
    const tokens = asNumber(
      summaryObj?.total_tokens,
      asNumber(summaryObj?.total_tokens_in, 0) + asNumber(summaryObj?.total_tokens_out, 0)
    );
    return {
      agent,
      tokens: tokens || asNumber(tokenMap?.[agent], 0),
      cost: asNumber(summaryObj?.total_cost_usd, asNumber(summary, 0)),
    };
  });
}

function normalizeReceiptCostSummary(value: unknown): DecisionPackageReceiptCostSummary | null {
  const obj = asObject(value);
  if (!obj) return null;

  const perAgentRecord = asObject(obj.per_agent);
  const perAgent = perAgentRecord
    ? Object.entries(perAgentRecord)
        .map(([agentName, rawSummary]) => {
          const summary = asObject(rawSummary);
          if (!summary) return null;

          const rawModelsUsed = asObject(summary.models_used);
          const modelsUsed = rawModelsUsed
            ? Object.entries(rawModelsUsed)
                .map(([model, callCount]) => ({
                  model,
                  call_count: asNumber(callCount, 0),
                }))
                .filter((entry) => entry.call_count > 0)
            : [];

          return {
            agent: asString(summary.agent_name, agentName),
            total_cost_usd: asNumber(summary.total_cost_usd, 0),
            total_tokens_in: asNumber(summary.total_tokens_in, 0),
            total_tokens_out: asNumber(summary.total_tokens_out, 0),
            call_count: asNumber(summary.call_count, 0),
            models_used: modelsUsed,
          };
        })
        .filter((entry): entry is DecisionPackageReceiptCostAgent => entry !== null)
    : [];

  const modelUsageRecord = asObject(obj.model_usage);
  const modelUsage = modelUsageRecord
    ? Object.entries(modelUsageRecord)
        .map(([key, rawSummary]) => {
          const summary = asObject(rawSummary);
          if (!summary) return null;

          const provider = asString(summary.provider);
          const model = asString(summary.model, key);
          return {
            key,
            label: formatModelLabel(provider, model, key),
            provider,
            model,
            total_cost_usd: asNumber(summary.total_cost_usd, 0),
            total_tokens_in: asNumber(summary.total_tokens_in, 0),
            total_tokens_out: asNumber(summary.total_tokens_out, 0),
            call_count: asNumber(summary.call_count, 0),
          };
        })
        .filter((entry): entry is DecisionPackageReceiptModelUsage => entry !== null)
    : [];

  const totalCost =
    obj.total_cost_usd === undefined ? null : asNumber(obj.total_cost_usd, 0);

  if (
    totalCost === null &&
    asNumber(obj.total_tokens_in, 0) === 0 &&
    asNumber(obj.total_tokens_out, 0) === 0 &&
    asNumber(obj.total_calls, 0) === 0 &&
    perAgent.length === 0 &&
    modelUsage.length === 0
  ) {
    return null;
  }

  return {
    total_cost_usd: totalCost,
    total_tokens_in: asNumber(obj.total_tokens_in, 0),
    total_tokens_out: asNumber(obj.total_tokens_out, 0),
    total_calls: asNumber(obj.total_calls, 0),
    per_agent: perAgent,
    model_usage: modelUsage,
  };
}

function normalizeReceipt(
  value: unknown,
  fallbackTimestamp = ''
): DecisionPackage['receipt'] {
  const obj = asObject(value);
  if (!obj) return null;

  const hash = asString(obj.hash, asString(obj.checksum));
  const timestamp = asString(obj.timestamp, asString(obj.created_at, fallbackTimestamp));
  if (!hash || !timestamp) return null;

  return {
    hash,
    timestamp,
    signers: asStringArray(obj.signers),
    cost_summary: normalizeReceiptCostSummary(obj.cost_summary),
  };
}

function normalizeNextSteps(value: unknown): DecisionPackage['next_steps'] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (typeof item === 'string') {
        return { action: item, priority: 'medium' as const };
      }
      const obj = asObject(item);
      if (!obj) return null;
      const action = asString(obj.action);
      if (!action) return null;
      const rawPriority = asString(obj.priority, 'medium').toLowerCase();
      const priority = (
        rawPriority === 'high' || rawPriority === 'low' ? rawPriority : 'medium'
      ) as 'high' | 'medium' | 'low';
      return { action, priority };
    })
    .filter((item): item is DecisionPackage['next_steps'][number] => item !== null);
}

export function normalizeDecisionPackage(raw: unknown, fallbackId: string): DecisionPackage {
  const obj = asObject(raw) ?? {};
  const agents = asStringArray(obj.agents);
  const participants = asStringArray(obj.participants);
  const argumentsList = normalizeArguments(obj.arguments ?? obj.messages);
  const rawReceipt = asObject(obj.receipt);
  const receiptCostSummary = asObject(rawReceipt?.cost_summary);
  const explicitCost = asObject(obj.cost);
  const cost = hasObjectEntries(explicitCost) ? explicitCost : receiptCostSummary;
  const tokenMap = asObject(obj.per_agent_tokens) ?? tokenMapFromCostSummary(receiptCostSummary);
  const createdAt = asString(obj.created_at, asString(obj.assembled_at, new Date().toISOString()));

  return {
    id: asString(obj.id, asString(obj.debate_id, fallbackId)),
    question: asString(obj.question, asString(obj.task)),
    verdict: asString(obj.verdict, asString(rawReceipt?.verdict)),
    confidence: asNumber(obj.confidence, asNumber(rawReceipt?.confidence, 0)),
    consensus_reached: Boolean(obj.consensus_reached),
    explanation: asString(obj.explanation, asString(obj.explanation_summary)),
    final_answer: asString(obj.final_answer),
    agents: agents.length > 0 ? agents : participants,
    rounds: asNumber(
      obj.rounds,
      argumentsList.reduce((maxRound, arg) => Math.max(maxRound, arg.round), 0)
    ),
    arguments: argumentsList,
    cost_breakdown: normalizeCostBreakdown(obj.cost_breakdown ?? cost, tokenMap),
    total_cost: asNumber(obj.total_cost, asNumber(cost?.total_cost_usd, 0)),
    receipt: normalizeReceipt(rawReceipt, createdAt),
    next_steps: normalizeNextSteps(obj.next_steps),
    created_at: createdAt,
    duration_seconds: asNumber(obj.duration_seconds, 0),
  };
}
