'use client';

import type { EngineStage } from '@/hooks/usePromptEngine';

interface ProvenanceBarProps {
  hash: string | null;
  stage: EngineStage | null;
}

export function ProvenanceBar({ hash, stage }: ProvenanceBarProps) {
  if (!hash) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 h-8 bg-[var(--surface)] border-t border-[var(--border)] flex items-center px-4 gap-4 font-mono text-xs text-[var(--text-muted)] z-10">
      <span>Provenance:</span>
      <span className="text-[var(--acid-cyan)]">{hash}</span>
      {stage && (
        <>
          <span className="text-[var(--border)]">|</span>
          <span>Stage: <strong className="text-[var(--text)]">{stage}</strong></span>
        </>
      )}
    </div>
  );
}
