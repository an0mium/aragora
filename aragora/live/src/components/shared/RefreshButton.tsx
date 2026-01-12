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
      className={`text-xs font-mono text-text-muted hover:text-acid-green disabled:opacity-50 transition-colors ${className}`}
    >
      [{loading ? '...' : 'REFRESH'}]
    </button>
  );
}
