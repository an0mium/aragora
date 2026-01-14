'use client';

import { useState, useEffect, ReactNode } from 'react';

type SectionPriority = 'core' | 'secondary' | 'advanced';

interface CollapsibleSectionProps {
  id: string;
  title: string;
  defaultOpen?: boolean;
  children: ReactNode;
  badge?: number;
  /** Section priority for Focus Mode filtering */
  priority?: SectionPriority;
  /** Hide entire section in Focus Mode (collapsed sections still visible) */
  hideInFocusMode?: boolean;
  /** Override open state from parent (for dashboard mode control) */
  forceOpen?: boolean;
  /** Description shown on hover */
  description?: string;
}

/**
 * A collapsible section that persists its open/closed state in localStorage.
 * Used to organize the sidebar panels into logical groups.
 *
 * Supports Focus Mode where only core sections are visible.
 */
export function CollapsibleSection({
  id,
  title,
  defaultOpen = false,
  children,
  badge,
  priority = 'secondary',
  hideInFocusMode = false,
  forceOpen,
  description,
}: CollapsibleSectionProps) {
  const storageKey = `aragora-section-${id}`;

  // Initialize from localStorage if available, otherwise use defaultOpen
  const [isOpen, setIsOpen] = useState<boolean>(() => {
    if (typeof window === 'undefined') return defaultOpen;
    const stored = localStorage.getItem(storageKey);
    return stored !== null ? stored === 'true' : defaultOpen;
  });

  // Handle forceOpen override from parent
  useEffect(() => {
    if (forceOpen !== undefined) {
      setIsOpen(forceOpen);
    }
  }, [forceOpen]);

  // Persist state changes to localStorage
  useEffect(() => {
    if (forceOpen === undefined) {
      localStorage.setItem(storageKey, String(isOpen));
    }
  }, [isOpen, storageKey, forceOpen]);

  const isForced = forceOpen !== undefined;
  const isExpanded = isForced ? forceOpen : isOpen;

  if (hideInFocusMode && forceOpen === false) {
    return null;
  }

  // Priority indicator styling
  const priorityStyles = {
    core: 'border-l-2 border-l-acid-green',
    secondary: '',
    advanced: 'border-l-2 border-l-acid-cyan/50',
  };

  return (
    <div className={`border border-acid-green/20 rounded-lg overflow-hidden mb-3 bg-surface/30 ${priorityStyles[priority]}`}>
      <button
        onClick={() => {
          if (!isForced) {
            setIsOpen(!isOpen);
          }
        }}
        aria-expanded={isExpanded}
        aria-controls={`section-${id}-content`}
        aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${title} section`}
        title={description}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-acid-green/5 transition-colors group"
      >
        <div className="flex items-center gap-2">
          <span
            className={`text-acid-green text-xs transition-transform duration-200 ${
              isExpanded ? 'rotate-90' : ''
            }`}
          >
            ▶
          </span>
          <span className="font-mono text-sm text-text group-hover:text-acid-green transition-colors">
            {title}
          </span>
          {priority === 'core' && (
            <span className="px-1 py-0.5 text-[9px] font-mono bg-acid-green/20 text-acid-green rounded">
              CORE
            </span>
          )}
          {badge !== undefined && badge > 0 && (
            <span className="px-1.5 py-0.5 text-[10px] font-mono bg-acid-cyan/20 text-acid-cyan rounded">
              {badge}
            </span>
          )}
        </div>
        <span className="text-text-muted text-xs font-mono">
          {isExpanded ? '[−]' : '[+]'}
        </span>
      </button>

      <div
        id={`section-${id}-content`}
        className={`transition-all duration-200 ease-in-out overflow-hidden ${
          isExpanded ? 'max-h-[5000px] opacity-100' : 'max-h-0 opacity-0'
        }`}
      >
        <div className="px-2 pb-2 space-y-3">
          {children}
        </div>
      </div>
    </div>
  );
}
