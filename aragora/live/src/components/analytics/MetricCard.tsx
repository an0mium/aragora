'use client';

import React from 'react';

export interface MetricCardProps {
  /** Title of the metric */
  title: string;
  /** Main value to display */
  value: string | number;
  /** Optional subtitle or description */
  subtitle?: string;
  /** Optional change indicator (e.g., +15.3%, -5.2%) */
  change?: number;
  /** Change period label (e.g., "vs last week") */
  changePeriod?: string;
  /** Color theme */
  color?: 'green' | 'cyan' | 'yellow' | 'magenta' | 'red' | 'purple';
  /** Optional icon or prefix character */
  icon?: string;
  /** Show loading state */
  loading?: boolean;
  /** Additional CSS classes */
  className?: string;
}

const colorClasses = {
  green: {
    border: 'border-acid-green/30',
    bg: 'bg-acid-green/5',
    text: 'text-acid-green',
    glow: 'shadow-[0_0_10px_rgba(57,255,20,0.1)]',
  },
  cyan: {
    border: 'border-acid-cyan/30',
    bg: 'bg-acid-cyan/5',
    text: 'text-acid-cyan',
    glow: 'shadow-[0_0_10px_rgba(0,255,255,0.1)]',
  },
  yellow: {
    border: 'border-acid-yellow/30',
    bg: 'bg-acid-yellow/5',
    text: 'text-acid-yellow',
    glow: 'shadow-[0_0_10px_rgba(255,255,0,0.1)]',
  },
  magenta: {
    border: 'border-acid-magenta/30',
    bg: 'bg-acid-magenta/5',
    text: 'text-acid-magenta',
    glow: 'shadow-[0_0_10px_rgba(255,0,255,0.1)]',
  },
  red: {
    border: 'border-crimson/30',
    bg: 'bg-crimson/5',
    text: 'text-crimson',
    glow: 'shadow-[0_0_10px_rgba(220,20,60,0.1)]',
  },
  purple: {
    border: 'border-purple-500/30',
    bg: 'bg-purple-500/5',
    text: 'text-purple-400',
    glow: 'shadow-[0_0_10px_rgba(168,85,247,0.1)]',
  },
};

export function MetricCard({
  title,
  value,
  subtitle,
  change,
  changePeriod,
  color = 'green',
  icon,
  loading = false,
  className = '',
}: MetricCardProps) {
  const colors = colorClasses[color];

  const formatChange = (val: number): string => {
    const sign = val >= 0 ? '+' : '';
    return `${sign}${val.toFixed(1)}%`;
  };

  const changeColor = change !== undefined
    ? change >= 0 ? 'text-acid-green' : 'text-crimson'
    : '';

  if (loading) {
    return (
      <div className={`border ${colors.border} ${colors.bg} p-4 rounded ${colors.glow} ${className}`}>
        <div className="animate-pulse space-y-3">
          <div className="h-3 bg-surface rounded w-1/2" />
          <div className="h-8 bg-surface rounded w-3/4" />
          <div className="h-2 bg-surface rounded w-1/3" />
        </div>
      </div>
    );
  }

  return (
    <div
      className={`border ${colors.border} ${colors.bg} p-4 rounded ${colors.glow} transition-all hover:border-opacity-50 ${className}`}
    >
      {/* Title */}
      <div className="flex items-center gap-2 mb-2">
        {icon && <span className={`${colors.text} font-mono`}>{icon}</span>}
        <span className="text-text-muted font-mono text-xs uppercase tracking-wider">
          {title}
        </span>
      </div>

      {/* Value */}
      <div className={`text-2xl font-mono ${colors.text} mb-1`}>
        {value}
      </div>

      {/* Subtitle and Change */}
      <div className="flex items-center justify-between">
        {subtitle && (
          <span className="text-text-muted text-xs font-mono">
            {subtitle}
          </span>
        )}
        {change !== undefined && (
          <div className="flex items-center gap-1">
            <span className={`text-xs font-mono ${changeColor}`}>
              {formatChange(change)}
            </span>
            {changePeriod && (
              <span className="text-text-muted text-[10px]">
                {changePeriod}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default MetricCard;
