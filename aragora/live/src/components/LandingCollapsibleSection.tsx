'use client';

import { useState, useEffect, ReactNode } from 'react';

interface LandingCollapsibleSectionProps {
  id: string;
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
}

/**
 * A collapsible section for the landing page.
 * Persists open/closed state in localStorage.
 */
export function LandingCollapsibleSection({
  id,
  title,
  defaultOpen = false,
  children,
}: LandingCollapsibleSectionProps) {
  const storageKey = `aragora-landing-${id}`;

  // Initialize from localStorage if available, otherwise use defaultOpen
  const [isOpen, setIsOpen] = useState<boolean>(() => {
    if (typeof window === 'undefined') return defaultOpen;
    const stored = localStorage.getItem(storageKey);
    return stored !== null ? stored === 'true' : defaultOpen;
  });

  // Persist state changes to localStorage
  useEffect(() => {
    localStorage.setItem(storageKey, String(isOpen));
  }, [isOpen, storageKey]);

  return (
    <section className="py-12 border-t border-acid-green/20">
      <div className="container mx-auto px-4">
        {/* Clickable Header */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full text-center mb-8 group cursor-pointer"
          aria-expanded={isOpen}
          aria-controls={`${id}-content`}
        >
          <p className="text-acid-green/50 font-mono text-xs mb-2">{'═'.repeat(30)}</p>
          <h2 className="text-acid-green font-mono text-lg flex items-center justify-center gap-3 group-hover:text-acid-cyan transition-colors">
            <span
              className={`text-acid-green text-sm transition-transform duration-200 ${
                isOpen ? 'rotate-90' : ''
              }`}
            >
              {'>'}
            </span>
            {title}
            <span className="text-text-muted text-xs font-mono ml-2">
              {isOpen ? '[−]' : '[+]'}
            </span>
          </h2>
          <p className="text-acid-green/50 font-mono text-xs mt-2">{'═'.repeat(30)}</p>
        </button>

        {/* Collapsible Content */}
        <div
          id={`${id}-content`}
          className={`transition-all duration-300 ease-in-out overflow-hidden ${
            isOpen ? 'max-h-[5000px] opacity-100' : 'max-h-0 opacity-0'
          }`}
        >
          {children}
        </div>
      </div>
    </section>
  );
}
