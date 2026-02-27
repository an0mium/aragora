'use client';

interface SpecEditorProps {
  spec: Record<string, unknown>;
  confidence: number | null;
  onApprove: () => void;
  onSkipValidation: () => void;
}

export function SpecEditor({ spec, confidence, onApprove, onSkipValidation }: SpecEditorProps) {
  const goal = (spec.refined_goal as string) || (spec.raw_goal as string) || '';
  const criteria = (spec.acceptance_criteria as string[]) || [];
  const constraints = (spec.constraints as string[]) || [];
  const tracks = (spec.track_hints as string[]) || [];
  const complexity = (spec.estimated_complexity as string) || 'medium';

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-sm text-[var(--acid-green)] font-bold">Generated Spec</h2>
        {confidence !== null && (
          <span className="font-mono text-xs text-[var(--text-muted)]">
            Confidence: {(confidence * 100).toFixed(0)}%
          </span>
        )}
      </div>

      <div className="border border-[var(--border)] p-4 space-y-4">
        <div>
          <label className="font-mono text-xs text-[var(--text-muted)] block mb-1">Goal</label>
          <p className="font-mono text-sm text-[var(--text)]">{goal}</p>
        </div>

        {criteria.length > 0 && (
          <div>
            <label className="font-mono text-xs text-[var(--text-muted)] block mb-1">Acceptance Criteria</label>
            <ul className="space-y-1">
              {criteria.map((c, i) => (
                <li key={i} className="font-mono text-xs text-[var(--text)] flex items-start gap-2">
                  <span className="text-[var(--acid-green)]">&#10003;</span> {c}
                </li>
              ))}
            </ul>
          </div>
        )}

        {constraints.length > 0 && (
          <div>
            <label className="font-mono text-xs text-[var(--text-muted)] block mb-1">Constraints</label>
            <ul className="space-y-1">
              {constraints.map((c, i) => (
                <li key={i} className="font-mono text-xs text-[var(--text-muted)]">- {c}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="flex gap-4 text-xs font-mono text-[var(--text-muted)]">
          <span>Complexity: <strong className="text-[var(--text)]">{complexity}</strong></span>
          {tracks.length > 0 && (
            <span>Tracks: {tracks.map(t => (
              <span key={t} className="inline-block px-1.5 py-0.5 border border-[var(--border)] ml-1">{t}</span>
            ))}</span>
          )}
        </div>
      </div>

      <div className="flex gap-3">
        <button
          onClick={onApprove}
          className="px-6 py-2 bg-[var(--acid-green)] text-[var(--bg)] font-mono font-bold text-sm hover:opacity-90"
        >
          Approve &amp; Validate
        </button>
        <button
          onClick={onSkipValidation}
          className="px-6 py-2 border border-[var(--border)] text-[var(--text-muted)] font-mono text-sm hover:border-[var(--acid-green)] hover:text-[var(--acid-green)]"
        >
          Skip Validation
        </button>
      </div>
    </div>
  );
}
