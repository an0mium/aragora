'use client';

import { useState, useEffect, ReactNode } from 'react';

interface CollapsibleSectionProps {
  id: string;
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
  badge?: number;
}

/**
 * A collapsible section that persists its open/closed state in localStorage.
 * Used to organize the sidebar panels into logical groups.
 */
export function CollapsibleSection({
  id,
  title,
  defaultOpen = false,
  children,
  badge,
}: CollapsibleSectionProps) {
  const storageKey = `aragora-section-${id}`;

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
    <div className="border border-acid-green/20 rounded-lg overflow-hidden mb-3 bg-surface/30">
      <button
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-controls={`section-${id}-content`}
        aria-label={`${isOpen ? 'Collapse' : 'Expand'} ${title} section`}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-acid-green/5 transition-colors group"
      >
        <div className="flex items-center gap-2">
          <span
            className={`text-acid-green text-xs transition-transform duration-200 ${
              isOpen ? 'rotate-90' : ''
            }`}
          >
            ▶
          </span>
          <span className="font-mono text-sm text-text group-hover:text-acid-green transition-colors">
            {title}
          </span>
          {badge !== undefined && badge > 0 && (
            <span className="px-1.5 py-0.5 text-[10px] font-mono bg-acid-green/20 text-acid-green rounded">
              {badge}
            </span>
          )}
        </div>
        <span className="text-text-muted text-xs font-mono">
          {isOpen ? '[−]' : '[+]'}
        </span>
      </button>

      <div
        id={`section-${id}-content`}
        className={`transition-all duration-200 ease-in-out overflow-hidden ${
          isOpen ? 'max-h-[5000px] opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <div className="px-2 pb-2 space-y-3">
          {children}
        </div>
      </div>
    </div>
  );
}
