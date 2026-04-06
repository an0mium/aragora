'use client';

import { API_BASE_URL } from '@/config';
import { AudioDownloadSection } from './AudioDownloadSection';

interface InlineDownloadPanelProps {
  debateId: string;
}

export function InlineDownloadPanel({ debateId }: InlineDownloadPanelProps) {
  return (
    <div className="mt-6 pt-6 border-t border-[var(--accent)]/30">
      <div className="text-xs font-theme-data text-[var(--accent)] uppercase tracking-wider mb-4">
        {'>'} DOWNLOAD DEBATE
      </div>

      {/* Transcript Format Downloads */}
      <div className="mb-4">
        <div className="text-xs font-theme-data text-text-muted mb-2">Transcript</div>
        <div className="flex flex-wrap gap-2">
          <a
            href={`${API_BASE_URL}/api/debates/${debateId}/export/txt`}
            download
            className="px-3 py-1.5 text-xs font-theme-data bg-bg border border-[var(--accent)]/40 text-[var(--accent)] hover:bg-[var(--accent)]/10 transition-colors"
          >
            [TXT]
          </a>
          <a
            href={`${API_BASE_URL}/api/debates/${debateId}/export/md`}
            download
            className="px-3 py-1.5 text-xs font-theme-data bg-bg border border-[var(--accent)]/40 text-[var(--accent)] hover:bg-[var(--accent)]/10 transition-colors"
          >
            [MARKDOWN]
          </a>
          <a
            href={`${API_BASE_URL}/api/debates/${debateId}/export/json`}
            download
            className="px-3 py-1.5 text-xs font-theme-data bg-bg border border-border text-text-muted hover:border-[var(--accent)]/40 transition-colors"
          >
            [JSON]
          </a>
          <a
            href={`${API_BASE_URL}/api/debates/${debateId}/export/html`}
            download
            className="px-3 py-1.5 text-xs font-theme-data bg-bg border border-border text-text-muted hover:border-[var(--accent)]/40 transition-colors"
          >
            [HTML]
          </a>
          <a
            href={`${API_BASE_URL}/api/debates/${debateId}/export/csv?table=messages`}
            download
            className="px-3 py-1.5 text-xs font-theme-data bg-bg border border-border text-text-muted hover:border-[var(--accent)]/40 transition-colors"
          >
            [CSV]
          </a>
        </div>
      </div>

      {/* Audio Generation */}
      <div>
        <div className="text-xs font-theme-data text-text-muted mb-2">Audio</div>
        <AudioDownloadSection debateId={debateId} />
      </div>
    </div>
  );
}
