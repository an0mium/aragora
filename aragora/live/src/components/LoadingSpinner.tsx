'use client';

interface LoadingSpinnerProps {
  /** Loading message to display */
  message?: string;
  /** Use compact inline style */
  compact?: boolean;
  /** Custom class name */
  className?: string;
}

/**
 * Terminal-styled loading indicator
 *
 * Matches the Aragora CRT/terminal aesthetic with animated blocks.
 */
export function LoadingSpinner({
  message = 'Loading...',
  compact = false,
  className = '',
}: LoadingSpinnerProps) {
  if (compact) {
    return (
      <div className={`text-accent text-xs font-mono animate-pulse ${className}`}>
        {'>'} {message}
      </div>
    );
  }

  return (
    <div className={`flex items-center justify-center p-8 ${className}`}>
      <div className="text-accent font-mono text-center">
        <div className="text-lg mb-2 flex items-center justify-center gap-2">
          <span className="animate-pulse">{'>'}</span>
          <span>{message}</span>
        </div>
        <div className="flex gap-1 justify-center">
          <span
            className="animate-pulse"
            style={{ animationDelay: '0ms' }}
          >
            █
          </span>
          <span
            className="animate-pulse"
            style={{ animationDelay: '150ms' }}
          >
            █
          </span>
          <span
            className="animate-pulse"
            style={{ animationDelay: '300ms' }}
          >
            █
          </span>
        </div>
      </div>
    </div>
  );
}
