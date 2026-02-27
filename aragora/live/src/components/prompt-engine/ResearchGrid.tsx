'use client';

import type { ResearchProgress } from '@/hooks/usePromptEngine';

const SOURCE_LABELS: Record<string, string> = {
  km: 'Knowledge Mound',
  codebase: 'Codebase',
  obsidian: 'Obsidian',
  web: 'Web Search',
};

interface ResearchGridProps {
  progress: ResearchProgress[];
}

export function ResearchGrid({ progress }: ResearchGridProps) {
  return (
    <div className="max-w-2xl mx-auto space-y-4">
      <h2 className="font-mono text-sm text-[var(--acid-green)] font-bold">
        Research Sources
      </h2>
      <div className="grid grid-cols-2 gap-4">
        {progress.map(p => (
          <div
            key={p.source}
            className={`border p-4 font-mono text-xs ${
              p.status === 'complete'
                ? 'border-[var(--acid-green)]/30'
                : p.status === 'failed'
                ? 'border-[var(--crimson)]/30'
                : 'border-[var(--border)]'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <span className={`w-2 h-2 rounded-full ${
                p.status === 'complete' ? 'bg-[var(--acid-green)]' :
                p.status === 'failed' ? 'bg-[var(--crimson)]' :
                'bg-[var(--acid-yellow)] animate-pulse'
              }`} />
              <span className="text-[var(--text)] font-bold">
                {SOURCE_LABELS[p.source] || p.source}
              </span>
            </div>
            <div className="text-[var(--text-muted)]">
              {p.status === 'pending' ? 'Searching...' :
               p.status === 'complete' ? `${p.results} results found` :
               'Source unavailable'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
