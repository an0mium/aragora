'use client';

export interface ExpandToggleProps {
  expanded: boolean;
  onToggle: () => void;
  className?: string;
}

export function ExpandToggle({ expanded, onToggle, className = '' }: ExpandToggleProps) {
  return (
    <button
      onClick={onToggle}
      className={`text-xs font-theme-data text-text-muted hover:text-text transition-colors ${className}`}
      aria-expanded={expanded}
      aria-label={expanded ? 'Collapse' : 'Expand'}
    >
      [{expanded ? '-' : '+'}]
    </button>
  );
}
