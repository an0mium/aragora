'use client';

import { useState } from 'react';
import Link from 'next/link';
import { PublicNav } from '@/components/PublicNav';
import { PublicFooter } from '@/components/PublicFooter';

type DocView = 'swagger' | 'redoc';

export default function DocsPage() {
  const [view, setView] = useState<DocView>('swagger');
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || '';

  const urls: Record<DocView, string> = {
    swagger: `${apiUrl}/api/v2/docs`,
    redoc: `${apiUrl}/api/v2/redoc`,
  };

  return (
    <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] flex flex-col">
      <PublicNav>
        <div className="flex items-center gap-1">
          {(['swagger', 'redoc'] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-3 py-1.5 text-xs font-mono font-bold rounded-full transition-colors ${
                view === v
                  ? 'bg-[var(--acid-green)] text-[var(--bg)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--acid-green)]'
              }`}
            >
              {v === 'swagger' ? 'SWAGGER' : 'REDOC'}
            </button>
          ))}
        </div>
      </PublicNav>

      <div className="flex-1 relative">
        {!apiUrl && (
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--bg)]">
            <div className="text-center space-y-4 max-w-md px-4">
              <p className="font-mono text-sm text-[var(--text-muted)]">
                Set <code className="text-[var(--acid-green)]">NEXT_PUBLIC_API_URL</code> to load
                live API documentation.
              </p>
              <p className="font-mono text-xs text-[var(--text-muted)]">
                Example: <code>NEXT_PUBLIC_API_URL=http://localhost:8080</code>
              </p>
            </div>
          </div>
        )}
        {apiUrl && (
          <iframe
            key={view}
            src={urls[view]}
            className="w-full h-full border-0"
            style={{ minHeight: 'calc(100vh - 49px)' }}
            title={`API Documentation - ${view}`}
          />
        )}
      </div>
    </main>
  );
}
