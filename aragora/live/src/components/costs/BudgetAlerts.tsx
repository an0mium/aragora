'use client';

import { useState } from 'react';

interface Alert {
  id: string;
  type: string;
  message: string;
  severity: string;
  timestamp: string;
}

interface BudgetAlertsProps {
  alerts: Alert[];
}

const SEVERITY_CONFIG: Record<string, { color: string; bgColor: string; icon: string }> = {
  critical: { color: 'text-red-400', bgColor: 'bg-red-500/10 border-red-500/30', icon: '🚨' },
  warning: { color: 'text-yellow-400', bgColor: 'bg-yellow-500/10 border-yellow-500/30', icon: '⚠️' },
  info: { color: 'text-blue-400', bgColor: 'bg-blue-500/10 border-blue-500/30', icon: 'ℹ️' },
};

export function BudgetAlerts({ alerts }: BudgetAlertsProps) {
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState(true);

  const visibleAlerts = alerts.filter(a => !dismissedAlerts.has(a.id));

  if (visibleAlerts.length === 0) return null;

  const dismissAlert = (id: string) => {
    setDismissedAlerts(prev => new Set([...prev, id]));
  };

  const dismissAll = () => {
    setDismissedAlerts(new Set(alerts.map(a => a.id)));
  };

  const formatTimeAgo = (timestamp: string): string => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] rounded overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 flex items-center justify-between hover:bg-[var(--bg)] transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className="text-xl">🔔</span>
          <div className="text-left">
            <h3 className="text-sm font-mono text-[var(--acid-green)]">
              BUDGET ALERTS
            </h3>
            <p className="text-xs text-[var(--text-muted)]">
              {visibleAlerts.length} active alert{visibleAlerts.length !== 1 ? 's' : ''}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              dismissAll();
            }}
            className="px-2 py-1 text-xs font-mono text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
          >
            Dismiss All
          </button>
          <span className="text-[var(--text-muted)]">
            {expanded ? '[-]' : '[+]'}
          </span>
        </div>
      </button>

      {/* Alerts List */}
      {expanded && (
        <div className="divide-y divide-[var(--border)]">
          {visibleAlerts.map((alert) => {
            const config = SEVERITY_CONFIG[alert.severity] || SEVERITY_CONFIG.info;

            return (
              <div
                key={alert.id}
                className={`p-4 ${config.bgColor} border-l-4`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start gap-3">
                    <span className="text-lg">{config.icon}</span>
                    <div>
                      <p className={`text-sm ${config.color}`}>{alert.message}</p>
                      <div className="flex items-center gap-3 mt-1">
                        <span className="text-xs text-[var(--text-muted)]">
                          {formatTimeAgo(alert.timestamp)}
                        </span>
                        <span className="text-xs text-[var(--text-muted)]">
                          • {alert.type.replace(/_/g, ' ')}
                        </span>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => dismissAlert(alert.id)}
                    className="text-[var(--text-muted)] hover:text-[var(--text)] p-1"
                  >
                    ✕
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default BudgetAlerts;
