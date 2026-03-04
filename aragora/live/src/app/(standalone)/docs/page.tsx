'use client';

import { useState } from 'react';
import Link from 'next/link';

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
      <nav className="border-b border-[var(--border)] bg-[var(--surface)]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
          <Link
            href="/"
            className="font-mono text-[var(--acid-green)] font-bold text-sm tracking-wider"
          >
            ARAGORA
          </Link>
          <div className="flex items-center gap-1">
            {(['swagger', 'redoc'] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-3 py-1.5 text-xs font-mono font-bold transition-colors ${
                  view === v
                    ? 'bg-[var(--acid-green)] text-[var(--bg)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--acid-green)]'
                }`}
              >
                {v === 'swagger' ? 'SWAGGER' : 'REDOC'}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/playground"
              className="text-xs font-mono text-[var(--text-muted)] hover:text-[var(--acid-green)] transition-colors"
            >
              PLAYGROUND
            </Link>
            <Link
              href="/signup"
              className="text-xs font-mono px-3 py-1.5 bg-[var(--acid-green)] text-[var(--bg)] hover:bg-[var(--acid-green)]/80 transition-colors font-bold"
            >
              SIGN UP FREE
            </Link>
          </div>
        </div>
      </nav>

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
