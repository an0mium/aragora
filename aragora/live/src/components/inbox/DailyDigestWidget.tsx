'use client';

import { useState, useEffect } from 'react';

interface DigestStats {
  emailsReceived: number;
  emailsProcessed: number;
  criticalHandled: number;
  timeSaved: string;
  topSenders: Array<{ name: string; count: number }>;
  categoryBreakdown: Array<{ category: string; count: number; percentage: number }>;
}

interface DailyDigestWidgetProps {
  compact?: boolean;
}

export function DailyDigestWidget({ compact = false }: DailyDigestWidgetProps) {
  const [expanded, setExpanded] = useState(false);
  const [stats, setStats] = useState<DigestStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDigest = async () => {
      try {
        const response = await fetch('/api/email/daily-digest');
        if (response.ok) {
          const data = await response.json();
          setStats(data);
        } else {
          // Use mock data if API not available
          setStats({
            emailsReceived: 47,
            emailsProcessed: 42,
            criticalHandled: 3,
            timeSaved: '1.5 hrs',
            topSenders: [
              { name: 'team@company.com', count: 12 },
              { name: 'client@example.com', count: 8 },
              { name: 'notifications@github.com', count: 6 },
            ],
            categoryBreakdown: [
              { category: 'Work', count: 28, percentage: 60 },
              { category: 'Updates', count: 12, percentage: 25 },
              { category: 'Personal', count: 5, percentage: 11 },
              { category: 'Spam', count: 2, percentage: 4 },
            ],
          });
        }
      } catch {
        // Use mock data on error
        setStats({
          emailsReceived: 47,
          emailsProcessed: 42,
          criticalHandled: 3,
          timeSaved: '1.5 hrs',
          topSenders: [
            { name: 'team@company.com', count: 12 },
            { name: 'client@example.com', count: 8 },
            { name: 'notifications@github.com', count: 6 },
          ],
          categoryBreakdown: [
            { category: 'Work', count: 28, percentage: 60 },
            { category: 'Updates', count: 12, percentage: 25 },
            { category: 'Personal', count: 5, percentage: 11 },
            { category: 'Spam', count: 2, percentage: 4 },
          ],
        });
      } finally {
        setLoading(false);
      }
    };

    fetchDigest();
  }, []);

  if (compact) {
    return (
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 px-3 py-1.5 text-xs font-mono bg-[var(--surface)] border border-[var(--border)] rounded hover:border-[var(--acid-green)]/30 transition-colors"
      >
        <span>📊</span>
        <span className="text-[var(--text-muted)]">Daily Digest</span>
        {stats && (
          <span className="text-[var(--acid-green)]">
            {stats.timeSaved} saved
          </span>
        )}
      </button>
    );
  }

  if (loading) {
    return (
      <div className="bg-[var(--surface)] border border-[var(--border)] p-4 rounded animate-pulse">
        <div className="h-4 bg-[var(--bg)] rounded w-1/3 mb-4" />
        <div className="grid grid-cols-3 gap-4">
          <div className="h-16 bg-[var(--bg)] rounded" />
          <div className="h-16 bg-[var(--bg)] rounded" />
          <div className="h-16 bg-[var(--bg)] rounded" />
        </div>
      </div>
    );
  }

  if (!stats) return null;

  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] rounded overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-[var(--border)] flex items-center justify-between">
        <div>
          <h3 className="text-sm font-mono text-[var(--acid-green)]">
            {'>'} TODAY'S DIGEST
          </h3>
          <p className="text-xs text-[var(--text-muted)]">
            {new Date().toLocaleDateString('en-US', {
              weekday: 'long',
              month: 'short',
              day: 'numeric',
            })}
          </p>
        </div>
        <div className="text-right">
          <div className="text-lg font-mono font-bold text-[var(--acid-green)]">
            {stats.timeSaved}
          </div>
          <div className="text-xs text-[var(--text-muted)]">Time Saved</div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-4 p-4 border-b border-[var(--border)]">
        <DigestStat
          label="Received"
          value={stats.emailsReceived}
          icon="📥"
        />
        <DigestStat
          label="Processed"
          value={stats.emailsProcessed}
          icon="✅"
          color="green"
        />
        <DigestStat
          label="Critical"
          value={stats.criticalHandled}
          icon="🔴"
          color="red"
        />
      </div>

      {/* Category Breakdown */}
      <div className="p-4 border-b border-[var(--border)]">
        <h4 className="text-xs font-mono text-[var(--text-muted)] mb-3">
          Category Breakdown
        </h4>
        <div className="space-y-2">
          {stats.categoryBreakdown.map((cat) => (
            <CategoryBar
              key={cat.category}
              category={cat.category}
              count={cat.count}
              percentage={cat.percentage}
            />
          ))}
        </div>
      </div>

      {/* Top Senders */}
      <div className="p-4">
        <h4 className="text-xs font-mono text-[var(--text-muted)] mb-3">
          Top Senders Today
        </h4>
        <div className="space-y-2">
          {stats.topSenders.map((sender, i) => (
            <div
              key={sender.name}
              className="flex items-center justify-between text-xs font-mono"
            >
              <span className="flex items-center gap-2">
                <span className="text-[var(--text-muted)]">{i + 1}.</span>
                <span className="text-[var(--text)] truncate max-w-[150px]">
                  {sender.name}
                </span>
              </span>
              <span className="text-[var(--acid-cyan)]">{sender.count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DigestStat({
  label,
  value,
  icon,
  color = 'default',
}: {
  label: string;
  value: number;
  icon: string;
  color?: 'default' | 'green' | 'red';
}) {
  const colorClasses = {
    default: 'text-[var(--text)]',
    green: 'text-green-400',
    red: 'text-red-400',
  };

  return (
    <div className="text-center">
      <div className="text-lg mb-1">{icon}</div>
      <div className={`text-xl font-mono font-bold ${colorClasses[color]}`}>
        {value}
      </div>
      <div className="text-xs text-[var(--text-muted)]">{label}</div>
    </div>
  );
}

function CategoryBar({
  category,
  count,
  percentage,
}: {
  category: string;
  count: number;
  percentage: number;
}) {
  const colors: Record<string, string> = {
    Work: 'bg-blue-500',
    Updates: 'bg-yellow-500',
    Personal: 'bg-green-500',
    Spam: 'bg-red-500',
  };

  return (
    <div>
      <div className="flex items-center justify-between text-xs mb-1">
        <span className="text-[var(--text-muted)]">{category}</span>
        <span className="text-[var(--text)]">{count}</span>
      </div>
      <div className="h-1.5 bg-[var(--bg)] rounded-full overflow-hidden">
        <div
          className={`h-full ${colors[category] || 'bg-[var(--acid-cyan)]'} transition-all`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default DailyDigestWidget;
