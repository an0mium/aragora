'use client';

export interface RefreshButtonProps {
  onClick: () => void;
  loading?: boolean;
  className?: string;
}

export function RefreshButton({ onClick, loading, className = '' }: RefreshButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      aria-label={loading ? 'Refreshing data' : 'Refresh data'}
      className={`text-xs font-theme-data text-text-muted hover:text-[var(--accent)] disabled:opacity-50 transition-colors ${className}`}
    >
      [{loading ? '...' : 'REFRESH'}]
    </button>
  );
}
