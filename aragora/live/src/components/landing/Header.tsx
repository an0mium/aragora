'use client';

import { useState } from 'react';
import { AsciiBannerCompact } from '../AsciiBanner';
import { ThemeToggle } from '../ThemeToggle';
import { BackendSelector } from '../BackendSelector';

export function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          <AsciiBannerCompact connected={true} />

          {/* Desktop Navigation */}
          <div className="hidden sm:flex items-center gap-4">
            <a
              href="/debates"
              className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
            >
              [DEBATES]
            </a>
            <a
              href="/agents"
              className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
            >
              [AGENTS]
            </a>
            <a
              href="/insights"
              className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
            >
              [INSIGHTS]
            </a>
            <a
              href="/evidence"
              className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
            >
              [EVIDENCE]
            </a>
            <a
              href="/memory"
              className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
            >
              [MEMORY]
            </a>
            <a
              href="/tournaments"
              className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
            >
              [TOURNAMENTS]
            </a>
            <a
              href="/developer"
              className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
            >
              [DEV]
            </a>
            <a
              href="https://live.aragora.ai"
              className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
            >
              [LIVE]
            </a>
            <BackendSelector compact />
            <ThemeToggle />
          </div>

          {/* Mobile Menu Button */}
          <div className="flex sm:hidden items-center gap-2">
            <ThemeToggle />
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 text-acid-green hover:text-acid-cyan transition-colors"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <span className="font-mono text-lg">✕</span>
              ) : (
                <span className="font-mono text-lg">☰</span>
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="sm:hidden mt-4 pb-2 border-t border-acid-green/20 pt-4 space-y-3">
            <a
              href="/debates"
              className="block text-sm font-mono text-text-muted hover:text-acid-green transition-colors py-2"
            >
              [DEBATES]
            </a>
            <a
              href="/agents"
              className="block text-sm font-mono text-text-muted hover:text-acid-green transition-colors py-2"
            >
              [AGENTS]
            </a>
            <a
              href="/insights"
              className="block text-sm font-mono text-text-muted hover:text-acid-green transition-colors py-2"
            >
              [INSIGHTS]
            </a>
            <a
              href="/evidence"
              className="block text-sm font-mono text-text-muted hover:text-acid-green transition-colors py-2"
            >
              [EVIDENCE]
            </a>
            <a
              href="/memory"
              className="block text-sm font-mono text-text-muted hover:text-acid-green transition-colors py-2"
            >
              [MEMORY]
            </a>
            <a
              href="/tournaments"
              className="block text-sm font-mono text-text-muted hover:text-acid-green transition-colors py-2"
            >
              [TOURNAMENTS]
            </a>
            <a
              href="/developer"
              className="block text-sm font-mono text-text-muted hover:text-acid-green transition-colors py-2"
            >
              [DEVELOPER]
            </a>
            <a
              href="https://live.aragora.ai"
              className="block text-sm font-mono text-acid-cyan hover:text-acid-green transition-colors py-2"
            >
              [LIVE DASHBOARD]
            </a>
            <div className="pt-2">
              <BackendSelector compact />
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
