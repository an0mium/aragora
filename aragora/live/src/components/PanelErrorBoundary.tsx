'use client';

import { Component, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  panelName: string;
  onRetry?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * Granular error boundary for individual panels.
 * Prevents one panel crash from taking down the entire dashboard.
 */
export class PanelErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error(`[${this.props.panelName}] Error:`, error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null });
    this.props.onRetry?.();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="bg-surface border border-red-500/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-red-400 text-lg">⚠</span>
            <h3 className="text-sm font-medium text-red-400">
              {this.props.panelName} Error
            </h3>
          </div>
          <p className="text-xs text-text-muted mb-3">
            This panel encountered an error and couldn&apos;t render.
          </p>
          {this.state.error && (
            <details className="mb-3">
              <summary className="text-xs text-text-muted cursor-pointer hover:text-text">
                Show details
              </summary>
              <pre className="mt-2 text-xs text-red-300/70 bg-red-900/10 p-2 rounded overflow-x-auto">
                {this.state.error.message}
              </pre>
            </details>
          )}
          <button
            onClick={this.handleRetry}
            className="px-3 py-1 text-xs bg-surface border border-border rounded hover:bg-surface-hover text-text transition-colors"
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * HOC to wrap any component with PanelErrorBoundary.
 */
export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  panelName: string
) {
  return function WithErrorBoundary(props: P) {
    return (
      <PanelErrorBoundary panelName={panelName}>
        <WrappedComponent {...props} />
      </PanelErrorBoundary>
    );
  };
}
