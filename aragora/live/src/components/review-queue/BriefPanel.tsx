'use client';

import type { ReviewQueueBrief } from '@/hooks/useReviewQueue';

export interface BriefPanelProps {
  brief: ReviewQueueBrief | null;
  loading?: boolean;
  error?: string | null;
}

/**
 * Renders the role-structured synthesis sections from a PDB brief.
 *
 * For v0, the source briefs are manually written (schema from
 * ``docs/plans/2026-04-19-pr-intelligence-brief-addendum.md §5``). Fields
 * that are missing simply render "—" rather than breaking the layout.
 */
export function BriefPanel({ brief, loading, error }: BriefPanelProps) {
  const panelStyle = {
    borderColor: 'var(--border)',
    backgroundColor: 'var(--surface-elevated)',
  };

  if (loading) {
    return (
      <div
        data-testid="brief-panel-loading"
        className="rounded-lg border px-4 py-3 text-xs"
        style={{
          ...panelStyle,
          color: 'var(--text-muted)',
        }}
      >
        Loading brief…
      </div>
    );
  }

  if (error) {
    return (
      <div
        data-testid="brief-panel-error"
        role="alert"
        className="rounded-lg border px-4 py-3 text-xs"
        style={{
          borderColor: 'var(--crimson)',
          backgroundColor: 'rgba(255, 0, 64, 0.08)',
          color: 'var(--crimson)',
        }}
      >
        {error}
      </div>
    );
  }

  if (!brief) {
    return (
      <div
        data-testid="brief-panel-empty"
        className="rounded-lg border px-4 py-3 text-xs italic"
        style={{
          ...panelStyle,
          color: 'var(--text-muted)',
        }}
      >
        Brief generation is not enabled yet. The PDB pipeline — heterogeneous
        debate, synthesis, and signed brief output — is on the roadmap (see
        <code className="font-theme-data not-italic"> #6306 </code>).
        Approve decisions currently rely on CI status + your own reading of the diff.
      </div>
    );
  }

  const sections: Array<{ key: keyof ReviewQueueBrief; label: string }> = [
    { key: 'logic', label: 'Logic' },
    { key: 'security', label: 'Security' },
    { key: 'maintainability', label: 'Maintainability' },
    { key: 'skeptic', label: 'Skeptic' },
  ];

  return (
    <div
      data-testid="brief-panel"
      className="rounded-lg border px-4 py-4 text-sm"
      style={{
        ...panelStyle,
        color: 'var(--text)',
      }}
    >
      <div
        className="flex flex-wrap items-center gap-3 border-b pb-3"
        style={{ borderColor: 'var(--border)' }}
      >
        <span
          className="font-theme-data uppercase tracking-wider"
          style={{ fontSize: '10px', color: 'var(--text-muted)' }}
        >
          Verdict
        </span>
        <span className="font-theme-data text-sm" data-testid="brief-verdict">
          {brief.verdict || '—'}
        </span>
        {brief.confidence !== null && brief.confidence !== undefined && (
          <span
            className="text-xs"
            style={{ color: 'var(--text-muted)' }}
            data-testid="brief-confidence"
          >
            confidence {brief.confidence}/5
          </span>
        )}
        <span
          className="ml-auto font-theme-data text-xs"
          style={{ color: 'var(--text-muted)' }}
        >
          head {brief.head_sha?.slice(0, 12) || '—'}
        </span>
      </div>
      <div className="mt-3 space-y-4">
        {sections.map(({ key, label }) => {
          const value = brief[key];
          const text = typeof value === 'string' && value.trim() ? value : '—';
          return (
            <div key={key} data-testid={`brief-section-${key}`}>
              <div
                className="mb-1 font-theme-data uppercase tracking-wider"
                style={{ fontSize: '10px', color: 'var(--text-muted)' }}
              >
                {label}
              </div>
              <div className="whitespace-pre-wrap leading-relaxed">{text}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
