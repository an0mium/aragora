'use client';

import { useState } from 'react';
import { useOptionalConnection, type OverallConnectionStatus } from '@/context/ConnectionContext';

/**
 * Global connection status indicator that appears in the layout.
 *
 * Shows a minimal indicator when all services are connected,
 * and expands to show details when there are issues.
 *
 * Uses ConnectionContext to aggregate status from multiple WebSocket services.
 */

const STATUS_CONFIG: Record<OverallConnectionStatus, {
  color: string;
  bgColor: string;
  borderColor: string;
  label: string;
  icon: string;
}> = {
  connected: {
    color: 'text-acid-green',
    bgColor: 'bg-acid-green/10',
    borderColor: 'border-acid-green/30',
    label: 'ALL SYSTEMS ONLINE',
    icon: '●',
  },
  partial: {
    color: 'text-acid-yellow',
    bgColor: 'bg-acid-yellow/10',
    borderColor: 'border-acid-yellow/30',
    label: 'PARTIAL CONNECTION',
    icon: '◐',
  },
  connecting: {
    color: 'text-acid-cyan',
    bgColor: 'bg-acid-cyan/10',
    borderColor: 'border-acid-cyan/30',
    label: 'CONNECTING...',
    icon: '◌',
  },
  disconnected: {
    color: 'text-text-muted',
    bgColor: 'bg-surface',
    borderColor: 'border-border',
    label: 'DISCONNECTED',
    icon: '○',
  },
  error: {
    color: 'text-crimson',
    bgColor: 'bg-crimson/10',
    borderColor: 'border-crimson/30',
    label: 'CONNECTION ERROR',
    icon: '✕',
  },
};

export function GlobalConnectionStatus() {
  const connection = useOptionalConnection();
  const [isExpanded, setIsExpanded] = useState(false);

  // Don't render if outside provider or no services registered
  if (!connection || connection.totalServices === 0) {
    return null;
  }

  const { overallStatus, services, connectedCount, totalServices, isReconnecting } = connection;
  const config = STATUS_CONFIG[overallStatus];

  // Only show expanded view for non-connected states by default
  const shouldShowDetails = overallStatus !== 'connected' || isExpanded;

  return (
    <div className="fixed bottom-4 right-4 z-50 font-mono text-xs">
      {/* Minimal indicator for connected state */}
      {overallStatus === 'connected' && !isExpanded && (
        <button
          onClick={() => setIsExpanded(true)}
          className={`flex items-center gap-2 px-3 py-1.5 ${config.bgColor} border ${config.borderColor} rounded-full ${config.color} hover:opacity-80 transition-opacity`}
          title="Click for connection details"
        >
          <span className="animate-pulse">{config.icon}</span>
          <span>{connectedCount}/{totalServices}</span>
        </button>
      )}

      {/* Expanded view */}
      {shouldShowDetails && (
        <div className={`${config.bgColor} border ${config.borderColor} rounded-lg shadow-lg overflow-hidden min-w-[240px]`}>
          {/* Header */}
          <div
            className={`flex items-center justify-between px-3 py-2 border-b ${config.borderColor} cursor-pointer`}
            onClick={() => overallStatus === 'connected' && setIsExpanded(false)}
          >
            <div className={`flex items-center gap-2 ${config.color}`}>
              <span className={overallStatus === 'connecting' || isReconnecting ? 'animate-spin' : 'animate-pulse'}>
                {config.icon}
              </span>
              <span>{config.label}</span>
            </div>
            <span className="text-text-muted">
              {connectedCount}/{totalServices}
            </span>
          </div>

          {/* Service list */}
          <div className="px-3 py-2 space-y-1.5 max-h-[200px] overflow-y-auto">
            {Array.from(services.values()).map((service) => {
              const isConnected = service.status === 'connected' || service.status === 'streaming';
              const hasError = service.status === 'error';
              const isConnecting = service.status === 'connecting';

              return (
                <div
                  key={service.name}
                  className="flex items-center justify-between gap-2"
                >
                  <div className="flex items-center gap-2">
                    <span className={`w-1.5 h-1.5 rounded-full ${
                      isConnected ? 'bg-acid-green' :
                      hasError ? 'bg-crimson' :
                      isConnecting ? 'bg-acid-cyan animate-pulse' :
                      'bg-text-muted'
                    }`} />
                    <span className="text-text">{service.displayName}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {service.reconnectAttempt > 0 && (
                      <span className="text-acid-yellow text-[10px]">
                        retry #{service.reconnectAttempt}
                      </span>
                    )}
                    {hasError && service.error && (
                      <span className="text-crimson text-[10px] truncate max-w-[100px]" title={service.error}>
                        {service.error}
                      </span>
                    )}
                    {isConnected && (
                      <span className="text-acid-green text-[10px]">OK</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Reconnecting indicator */}
          {isReconnecting && (
            <div className="px-3 py-1.5 border-t border-border text-acid-yellow text-center">
              Attempting to reconnect...
            </div>
          )}

          {/* Close button for expanded connected state */}
          {overallStatus === 'connected' && isExpanded && (
            <button
              onClick={() => setIsExpanded(false)}
              className="w-full px-3 py-1.5 border-t border-border text-text-muted hover:text-text text-center"
            >
              Collapse
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default GlobalConnectionStatus;
