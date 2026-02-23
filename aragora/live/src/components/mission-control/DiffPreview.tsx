'use client';

import { memo } from 'react';

export interface DiffPreviewProps {
  diff: string;
  maxLines?: number;
}

export const DiffPreview = memo(function DiffPreview({ diff, maxLines = 20 }: DiffPreviewProps) {
  const lines = diff.split('\n').slice(0, maxLines);
  const truncated = diff.split('\n').length > maxLines;

  return (
    <div
      className="rounded border border-[var(--border)] bg-[var(--bg)] overflow-hidden"
      data-testid="diff-preview"
    >
      <div className="px-2 py-1 border-b border-[var(--border)] text-xs font-mono text-[var(--text-muted)]">
        Diff Preview
      </div>
      <pre className="p-2 text-xs font-mono overflow-x-auto max-h-48 overflow-y-auto">
        {lines.map((line, i) => {
          let className = 'text-[var(--text-muted)]';
          if (line.startsWith('+') && !line.startsWith('+++')) {
            className = 'text-emerald-400 bg-emerald-500/10';
          } else if (line.startsWith('-') && !line.startsWith('---')) {
            className = 'text-red-400 bg-red-500/10';
          } else if (line.startsWith('@@')) {
            className = 'text-blue-400';
          }
          return (
            <div key={i} className={`${className} px-1`}>
              {line}
            </div>
          );
        })}
        {truncated && (
          <div className="text-[var(--text-muted)] px-1 italic">
            ... {diff.split('\n').length - maxLines} more lines
          </div>
        )}
      </pre>
    </div>
  );
});

export default DiffPreview;
