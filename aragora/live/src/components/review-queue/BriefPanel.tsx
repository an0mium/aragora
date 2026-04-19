import clsx from 'clsx';

import type { ReviewQueueDetail } from './types';
import { verdictLabel, verdictTone } from './utils';

interface BriefPanelProps {
  detail: ReviewQueueDetail | null;
  loading: boolean;
}

export function BriefPanel({ detail, loading }: BriefPanelProps) {
  if (loading) {
    return (
      <div className="rounded-xl border border-[var(--accent)]/15 bg-bg/45 px-4 py-5 text-sm font-theme-data text-text-muted">
        Loading brief and packet details…
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="rounded-xl border border-dashed border-border bg-bg/35 px-4 py-5 text-sm font-theme-data text-text-muted">
        No brief loaded yet.
      </div>
    );
  }

  const { brief, packet, checks, files, diff_url } = detail;
  const verdict = brief?.verdict ?? packet.machine_recommendation;

  return (
    <div className="space-y-4 rounded-xl border border-[var(--accent)]/15 bg-[linear-gradient(180deg,rgba(12,18,24,0.92),rgba(9,13,18,0.92))] p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className={clsx('rounded-full border px-2.5 py-1 text-[11px] font-theme-data uppercase tracking-[0.16em]', verdictTone(verdict))}>
              {verdictLabel(verdict)}
            </span>
            {brief?.confidence ? (
              <span className="text-[11px] font-theme-data uppercase tracking-[0.16em] text-text-muted">
                Confidence {brief.confidence}/5
              </span>
            ) : null}
          </div>
          <h3 className="text-lg font-theme-data text-text">{packet.title}</h3>
          <p className="text-sm font-theme-data text-text-muted">
            {brief?.recommended_action || packet.machine_recommendation_reason}
          </p>
        </div>
        <a
          href={diff_url}
          target="_blank"
          rel="noreferrer"
          className="rounded-full border border-[var(--accent)]/25 px-3 py-1 text-xs font-theme-data text-[var(--accent)] hover:bg-[var(--accent)]/10"
        >
          Transcript / diff
        </a>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <SectionCard title="Logic" body={brief?.logic || packet.machine_recommendation_reason} />
        <SectionCard title="Security" body={brief?.security || 'No security-specific note in the available brief.'} />
        <SectionCard
          title="Maintainability"
          body={brief?.maintainability || 'No maintainability note in the available brief.'}
        />
        <SectionCard title="Skeptic" body={brief?.skeptic || 'No skeptic pass captured in the available brief.'} />
      </div>

      {packet.risk_flags.length > 0 ? (
        <div className="rounded-xl border border-[var(--acid-yellow)]/20 bg-[var(--acid-yellow)]/8 p-4">
          <p className="text-[11px] font-theme-data uppercase tracking-[0.18em] text-[var(--acid-yellow)]">
            Risk Flags
          </p>
          <ul className="mt-3 space-y-2 text-sm font-theme-data text-text-muted">
            {packet.risk_flags.map((flag) => (
              <li key={flag}>• {flag}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
        <div className="rounded-xl border border-[var(--accent)]/12 bg-bg/45 p-4">
          <p className="text-[11px] font-theme-data uppercase tracking-[0.18em] text-text-muted">
            Checks
          </p>
          <div className="mt-3 space-y-2">
            {checks.slice(0, 8).map((check) => (
              <div
                key={`${check.name}-${check.details_url || check.status}`}
                className="flex items-center justify-between gap-3 rounded-lg border border-[var(--accent)]/10 px-3 py-2"
              >
                <span className="text-sm font-theme-data text-text">{check.name}</span>
                <span className="text-xs font-theme-data text-text-muted">
                  {check.conclusion || check.status}
                </span>
              </div>
            ))}
            {checks.length === 0 ? (
              <div className="text-sm font-theme-data text-text-muted">No check detail available.</div>
            ) : null}
          </div>
        </div>

        <div className="rounded-xl border border-[var(--accent)]/12 bg-bg/45 p-4">
          <p className="text-[11px] font-theme-data uppercase tracking-[0.18em] text-text-muted">
            Changed Files
          </p>
          <div className="mt-3 space-y-2">
            {files.slice(0, 10).map((file) => (
              <div key={file.path} className="rounded-lg border border-[var(--accent)]/10 px-3 py-2">
                <div className="truncate text-sm font-theme-data text-text">{file.path}</div>
                <div className="mt-1 text-xs font-theme-data text-text-muted">
                  +{file.additions} / -{file.deletions}
                </div>
              </div>
            ))}
            {files.length > 10 ? (
              <div className="text-xs font-theme-data text-text-muted">+{files.length - 10} more files</div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

function SectionCard({ title, body }: { title: string; body: string | null | undefined }) {
  return (
    <section className="rounded-xl border border-[var(--accent)]/12 bg-bg/45 p-4">
      <p className="text-[11px] font-theme-data uppercase tracking-[0.18em] text-text-muted">{title}</p>
      <p className="mt-3 text-sm font-theme-data leading-6 text-text">
        {body || 'No signal captured.'}
      </p>
    </section>
  );
}
