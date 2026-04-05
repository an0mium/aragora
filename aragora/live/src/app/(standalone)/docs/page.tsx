'use client';

import { useState } from 'react';
import Link from 'next/link';
import { getRuntimeBackendConfig } from '@/lib/runtimeBackend';

type DocView = 'swagger' | 'redoc';

export default function DocsPage() {
  const [view, setView] = useState<DocView>('swagger');
  const apiUrl = getRuntimeBackendConfig().config.api.replace(/\/$/, '');

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
            className="font-theme-data text-[var(--acid-green)] font-bold text-sm tracking-wider"
          >
            ARAGORA
          </Link>
          <div className="flex items-center gap-1">
            {(['swagger', 'redoc'] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`px-3 py-1.5 text-xs font-theme-data font-bold transition-colors ${
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
              className="text-xs font-theme-data text-[var(--text-muted)] hover:text-[var(--acid-green)] transition-colors"
            >
              PLAYGROUND
            </Link>
            <Link
              href="/signup"
              className="text-xs font-theme-data px-3 py-1.5 bg-[var(--acid-green)] text-[var(--bg)] hover:bg-[var(--acid-green)]/80 transition-colors font-bold"
            >
              SIGN UP FREE
            </Link>
          </div>
        </div>
      </nav>

      <div className="flex-1 relative">
        <iframe
          key={view}
          src={urls[view]}
          className="w-full h-full border-0"
          style={{ minHeight: 'calc(100vh - 49px)' }}
          title={`API Documentation - ${view}`}
        />
      </div>
    </main>
  );
}
