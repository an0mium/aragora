'use client';

import { FormEvent, useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { useBackend } from '@/components/BackendSelector';
import {
  normalizeDecisionPackage,
  type DecisionPackage,
} from '@/app/(app)/debates/[id]/normalizeDecisionPackage';

interface DebateResultComparePanelProps {
  initialLeftDebateId?: string;
  initialRightDebateId?: string;
}

function formatSigned(value: number, suffix = ''): string {
  const rounded = Math.round(value);
  return `${rounded > 0 ? '+' : ''}${rounded}${suffix}`;
}

function formatCurrencyDelta(value: number): string {
  const prefix = value > 0 ? '+' : '';
  return `${prefix}$${value.toFixed(4)}`;
}

function getAgentConfigEntries(pkg: DecisionPackage): Array<{ agent: string; label: string }> {
  const orderedAgents = Array.from(new Set([...pkg.agents, ...Object.keys(pkg.agent_models)]));

  return orderedAgents.map((agent) => {
    const meta = pkg.agent_models[agent];
    return {
      agent,
      label:
        meta?.llm_label ||
        meta?.model ||
        meta?.provider_display ||
        meta?.provider ||
        'Agent only',
    };
  });
}

function getModelSignature(pkg: DecisionPackage, agent: string): string {
  const meta = pkg.agent_models[agent];
  if (!meta) return '';
  return [meta.provider, meta.provider_display, meta.model, meta.llm_label].join('|');
}

function verdictTone(pkg: DecisionPackage): string {
  if (pkg.consensus_reached) {
    return 'bg-[var(--acid-green)]/10 text-[var(--acid-green)] border-[var(--acid-green)]/40';
  }
  return 'bg-[var(--warning)]/10 text-[var(--warning)] border-[var(--warning)]/40';
}

function differenceSummary(
  left: DecisionPackage,
  right: DecisionPackage
): {
  addedAgents: string[];
  removedAgents: string[];
  reconfiguredAgents: string[];
  verdictShift: string;
  consensusShift: string;
} {
  const leftAgents = new Set(left.agents);
  const rightAgents = new Set(right.agents);

  const addedAgents = right.agents.filter((agent) => !leftAgents.has(agent));
  const removedAgents = left.agents.filter((agent) => !rightAgents.has(agent));
  const reconfiguredAgents = left.agents.filter(
    (agent) => rightAgents.has(agent) && getModelSignature(left, agent) !== getModelSignature(right, agent)
  );

  let verdictShift = 'ALIGNED';
  if (left.verdict !== right.verdict) {
    verdictShift = `${left.verdict || 'UNKNOWN'} -> ${right.verdict || 'UNKNOWN'}`;
  } else if (left.final_answer !== right.final_answer) {
    verdictShift = 'FINAL ANSWER SHIFT';
  }

  const consensusShift =
    left.consensus_reached === right.consensus_reached
      ? left.consensus_reached
        ? 'CONSENSUS HELD'
        : 'NO CONSENSUS'
      : left.consensus_reached
        ? 'CONSENSUS LOST'
        : 'CONSENSUS GAINED';

  return {
    addedAgents,
    removedAgents,
    reconfiguredAgents,
    verdictShift,
    consensusShift,
  };
}

async function fetchDecisionPackage(apiBase: string, debateId: string): Promise<DecisionPackage> {
  const response = await fetch(`${apiBase}/api/v1/debates/${encodeURIComponent(debateId)}/package`);

  if (response.status === 404) {
    throw new Error(`Debate ${debateId} was not found.`);
  }
  if (!response.ok) {
    throw new Error(`Failed to load ${debateId} (HTTP ${response.status}).`);
  }

  const data = await response.json();
  return normalizeDecisionPackage(data, debateId);
}

function DebateResultCard({
  label,
  pkg,
  accentClass,
}: {
  label: string;
  pkg: DecisionPackage;
  accentClass: string;
}) {
  const configEntries = getAgentConfigEntries(pkg);

  return (
    <article className={`bg-[var(--surface)] border p-5 ${accentClass}`}>
      <div className="flex items-start justify-between gap-3 mb-4">
        <div>
          <div className="text-xs font-mono text-[var(--text-muted)] mb-1">{label}</div>
          <Link
            href={`/debates/${pkg.id}`}
            className="text-sm font-mono text-[var(--acid-cyan)] hover:text-[var(--acid-green)] transition-colors"
          >
            {pkg.id}
          </Link>
        </div>
        <span className={`px-2 py-1 text-xs font-mono border ${verdictTone(pkg)}`}>
          {pkg.verdict || (pkg.consensus_reached ? 'CONSENSUS' : 'DIVERGED')}
        </span>
      </div>

      <h4 className="text-sm font-mono text-[var(--text)] mb-2 whitespace-pre-wrap">
        {pkg.question || 'Untitled debate'}
      </h4>

      {pkg.final_answer && (
        <div className="mb-4 p-3 bg-[var(--bg)] border border-[var(--border)]">
          <div className="text-[10px] font-mono text-[var(--text-muted)] mb-1">OUTCOME</div>
          <p className="text-sm font-mono text-[var(--text)] whitespace-pre-wrap">
            {pkg.final_answer}
          </p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="border border-[var(--border)] bg-[var(--bg)] p-3">
          <div className="text-[10px] font-mono text-[var(--text-muted)] mb-1">CONFIDENCE</div>
          <div className="text-base font-mono text-[var(--acid-green)]">
            {Math.round(pkg.confidence * 100)}%
          </div>
        </div>
        <div className="border border-[var(--border)] bg-[var(--bg)] p-3">
          <div className="text-[10px] font-mono text-[var(--text-muted)] mb-1">ROUNDS</div>
          <div className="text-base font-mono text-[var(--acid-cyan)]">{pkg.rounds || '--'}</div>
        </div>
        <div className="border border-[var(--border)] bg-[var(--bg)] p-3">
          <div className="text-[10px] font-mono text-[var(--text-muted)] mb-1">COST</div>
          <div className="text-base font-mono text-[var(--text)]">${pkg.total_cost.toFixed(4)}</div>
        </div>
        <div className="border border-[var(--border)] bg-[var(--bg)] p-3">
          <div className="text-[10px] font-mono text-[var(--text-muted)] mb-1">DURATION</div>
          <div className="text-base font-mono text-[var(--text)]">
            {pkg.duration_seconds ? `${Math.round(pkg.duration_seconds)}s` : '--'}
          </div>
        </div>
      </div>

      <div className="mb-4">
        <div className="text-xs font-mono text-[var(--acid-green)] mb-2">{'>'} AGENT CONFIGURATION</div>
        <div className="space-y-2">
          {configEntries.length > 0 ? (
            configEntries.map((entry) => (
              <div
                key={`${pkg.id}-${entry.agent}`}
                className="flex items-start justify-between gap-3 border border-[var(--border)] bg-[var(--bg)] p-2"
              >
                <span className="text-xs font-mono text-[var(--acid-cyan)]">{entry.agent}</span>
                <span className="text-xs font-mono text-[var(--text-muted)] text-right">
                  {entry.label}
                </span>
              </div>
            ))
          ) : (
            <div className="text-xs font-mono text-[var(--text-muted)] border border-[var(--border)] bg-[var(--bg)] p-2">
              No agent metadata available.
            </div>
          )}
        </div>
      </div>

      {pkg.next_steps.length > 0 && (
        <div>
          <div className="text-xs font-mono text-[var(--acid-green)] mb-2">{'>'} NEXT STEPS</div>
          <div className="space-y-2">
            {pkg.next_steps.slice(0, 3).map((step, index) => (
              <div key={`${pkg.id}-step-${index}`} className="border border-[var(--border)] bg-[var(--bg)] p-2">
                <div className="text-[10px] font-mono text-[var(--text-muted)] mb-1">
                  {step.priority.toUpperCase()}
                </div>
                <div className="text-xs font-mono text-[var(--text)]">{step.action}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </article>
  );
}

export function DebateResultComparePanel({
  initialLeftDebateId = '',
  initialRightDebateId = '',
}: DebateResultComparePanelProps) {
  const { config: backendConfig } = useBackend();
  const [leftDebateId, setLeftDebateId] = useState(initialLeftDebateId);
  const [rightDebateId, setRightDebateId] = useState(initialRightDebateId);
  const [leftPackage, setLeftPackage] = useState<DecisionPackage | null>(null);
  const [rightPackage, setRightPackage] = useState<DecisionPackage | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLeftDebateId(initialLeftDebateId);
    setRightDebateId(initialRightDebateId);
  }, [initialLeftDebateId, initialRightDebateId]);

  const loadComparison = useCallback(
    async (leftCandidate: string, rightCandidate: string) => {
      const leftId = leftCandidate.trim();
      const rightId = rightCandidate.trim();

      if (!leftId || !rightId) {
        setError('Enter two debate IDs to compare.');
        setLeftPackage(null);
        setRightPackage(null);
        return;
      }

      if (leftId === rightId) {
        setError('Choose two different debate IDs.');
        setLeftPackage(null);
        setRightPackage(null);
        return;
      }

      try {
        setLoading(true);
        setError(null);
        const [left, right] = await Promise.all([
          fetchDecisionPackage(backendConfig.api, leftId),
          fetchDecisionPackage(backendConfig.api, rightId),
        ]);
        setLeftPackage(left);
        setRightPackage(right);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to compare debate results.');
        setLeftPackage(null);
        setRightPackage(null);
      } finally {
        setLoading(false);
      }
    },
    [backendConfig.api]
  );

  useEffect(() => {
    const leftId = initialLeftDebateId.trim();
    const rightId = initialRightDebateId.trim();

    if (!leftId || !rightId || leftId === rightId) {
      return;
    }

    void loadComparison(leftId, rightId);
  }, [initialLeftDebateId, initialRightDebateId, loadComparison]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    await loadComparison(leftDebateId, rightDebateId);
  };

  const handleSwap = () => {
    setLeftDebateId(rightDebateId);
    setRightDebateId(leftDebateId);
    setLeftPackage(rightPackage);
    setRightPackage(leftPackage);
    setError(null);
  };

  const comparedPackages =
    leftPackage && rightPackage ? { left: leftPackage, right: rightPackage } : null;
  const summary = comparedPackages
    ? differenceSummary(comparedPackages.left, comparedPackages.right)
    : null;

  return (
    <section className="bg-[var(--surface)] border border-[var(--border)] p-5">
      <div className="mb-4">
        <h2 className="text-lg font-mono text-[var(--acid-green)] mb-2">
          {'>'} DEBATE RESULT COMPARISON
        </h2>
        <p className="text-sm font-mono text-[var(--text-muted)]">
          Compare two completed decision packages side by side to see how agent lineup and model
          choices shifted the outcome.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="grid grid-cols-1 xl:grid-cols-[1fr_auto_1fr_auto] gap-3 mb-4">
        <label className="block">
          <span className="block text-xs font-mono text-[var(--text-muted)] mb-1">Left debate</span>
          <input
            value={leftDebateId}
            onChange={(event) => setLeftDebateId(event.target.value)}
            placeholder="debate-123"
            aria-label="Left debate ID"
            className="w-full bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-sm font-mono text-[var(--text)]"
          />
        </label>

        <button
          type="button"
          onClick={handleSwap}
          className="px-3 py-2 text-xs font-mono border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)] hover:border-[var(--acid-cyan)]/40 transition-colors self-end"
        >
          SWAP
        </button>

        <label className="block">
          <span className="block text-xs font-mono text-[var(--text-muted)] mb-1">Right debate</span>
          <input
            value={rightDebateId}
            onChange={(event) => setRightDebateId(event.target.value)}
            placeholder="debate-456"
            aria-label="Right debate ID"
            className="w-full bg-[var(--bg)] border border-[var(--border)] px-3 py-2 text-sm font-mono text-[var(--text)]"
          />
        </label>

        <button
          type="submit"
          disabled={loading}
          className="px-4 py-2 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 disabled:opacity-50 transition-colors self-end"
        >
          {loading ? 'COMPARING...' : 'COMPARE'}
        </button>
      </form>

      {error && (
        <div className="mb-4 border border-[var(--warning)]/40 bg-[var(--warning)]/10 px-3 py-2 text-sm font-mono text-[var(--warning)]">
          {error}
        </div>
      )}

      {!comparedPackages && !loading && !error && (
        <div className="border border-dashed border-[var(--border)] px-4 py-6 text-sm font-mono text-[var(--text-muted)]">
          Enter two debate IDs, or open this page from a debate detail view to prefill the left side.
        </div>
      )}

      {comparedPackages && summary && (
        <div className="space-y-4">
          {comparedPackages.left.question !== comparedPackages.right.question && (
            <div className="border border-[var(--warning)]/40 bg-[var(--warning)]/10 px-3 py-2 text-sm font-mono text-[var(--warning)]">
              Questions differ between these debates. Compare only if the prompt drift is intentional.
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
            <div className="border border-[var(--border)] bg-[var(--bg)] p-3">
              <div className="text-[10px] font-mono text-[var(--text-muted)] mb-1">OUTCOME</div>
              <div className="text-sm font-mono text-[var(--acid-green)]">{summary.verdictShift}</div>
              <div className="text-[10px] font-mono text-[var(--text-muted)] mt-1">
                {summary.consensusShift}
              </div>
            </div>
            <div className="border border-[var(--border)] bg-[var(--bg)] p-3">
              <div className="text-[10px] font-mono text-[var(--text-muted)] mb-1">CONFIDENCE Δ</div>
              <div className="text-sm font-mono text-[var(--acid-cyan)]">
                {formatSigned(
                  (comparedPackages.right.confidence - comparedPackages.left.confidence) * 100,
                  ' pts'
                )}
              </div>
              <div className="text-[10px] font-mono text-[var(--text-muted)] mt-1">
                {Math.round(comparedPackages.left.confidence * 100)}% ->{' '}
                {Math.round(comparedPackages.right.confidence * 100)}%
              </div>
            </div>
            <div className="border border-[var(--border)] bg-[var(--bg)] p-3">
              <div className="text-[10px] font-mono text-[var(--text-muted)] mb-1">CONFIG CHANGE</div>
              <div className="text-sm font-mono text-[var(--acid-green)]">
                {summary.addedAgents.length} added / {summary.removedAgents.length} removed
              </div>
              <div className="text-[10px] font-mono text-[var(--text-muted)] mt-1">
                {summary.reconfiguredAgents.length} shared agents reconfigured
              </div>
            </div>
            <div className="border border-[var(--border)] bg-[var(--bg)] p-3">
              <div className="text-[10px] font-mono text-[var(--text-muted)] mb-1">COST Δ</div>
              <div className="text-sm font-mono text-[var(--text)]">
                {formatCurrencyDelta(
                  comparedPackages.right.total_cost - comparedPackages.left.total_cost
                )}
              </div>
              <div className="text-[10px] font-mono text-[var(--text-muted)] mt-1">
                {formatSigned(
                  comparedPackages.right.duration_seconds - comparedPackages.left.duration_seconds,
                  's'
                )}{' '}
                duration
              </div>
            </div>
          </div>

          {(summary.addedAgents.length > 0 || summary.removedAgents.length > 0 || summary.reconfiguredAgents.length > 0) && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="border border-[var(--border)] bg-[var(--bg)] p-3">
                <div className="text-[10px] font-mono text-[var(--text-muted)] mb-2">ADDED ON RIGHT</div>
                <div className="flex flex-wrap gap-2">
                  {summary.addedAgents.length > 0 ? (
                    summary.addedAgents.map((agent) => (
                      <span
                        key={`added-${agent}`}
                        className="px-2 py-1 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30"
                      >
                        {agent}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs font-mono text-[var(--text-muted)]">No additions</span>
                  )}
                </div>
              </div>
              <div className="border border-[var(--border)] bg-[var(--bg)] p-3">
                <div className="text-[10px] font-mono text-[var(--text-muted)] mb-2">REMOVED ON RIGHT</div>
                <div className="flex flex-wrap gap-2">
                  {summary.removedAgents.length > 0 ? (
                    summary.removedAgents.map((agent) => (
                      <span
                        key={`removed-${agent}`}
                        className="px-2 py-1 text-xs font-mono bg-[var(--warning)]/10 text-[var(--warning)] border border-[var(--warning)]/30"
                      >
                        {agent}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs font-mono text-[var(--text-muted)]">No removals</span>
                  )}
                </div>
              </div>
              <div className="border border-[var(--border)] bg-[var(--bg)] p-3">
                <div className="text-[10px] font-mono text-[var(--text-muted)] mb-2">RECONFIGURED</div>
                <div className="flex flex-wrap gap-2">
                  {summary.reconfiguredAgents.length > 0 ? (
                    summary.reconfiguredAgents.map((agent) => (
                      <span
                        key={`reconfigured-${agent}`}
                        className="px-2 py-1 text-xs font-mono bg-[var(--acid-cyan)]/10 text-[var(--acid-cyan)] border border-[var(--acid-cyan)]/30"
                      >
                        {agent}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs font-mono text-[var(--text-muted)]">No model changes</span>
                  )}
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <DebateResultCard
              label="LEFT RESULT"
              pkg={comparedPackages.left}
              accentClass="border-[var(--acid-cyan)]/30"
            />
            <DebateResultCard
              label="RIGHT RESULT"
              pkg={comparedPackages.right}
              accentClass="border-[var(--acid-green)]/30"
            />
          </div>
        </div>
      )}
    </section>
  );
}
