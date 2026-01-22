'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ProgressiveMode, useProgressiveMode } from '@/context/ProgressiveModeContext';

interface FeatureItem {
  label: string;
  href: string;
  description?: string;
  icon?: string;
  minMode?: ProgressiveMode;
}

interface FeatureCardProps {
  title: string;
  description?: string;
  icon?: string;
  features: FeatureItem[];
  minMode?: ProgressiveMode;
  className?: string;
  defaultExpanded?: boolean;
}

/**
 * Feature card with progressive disclosure
 *
 * Shows features based on current mode, with expandable "show more"
 * for features above current mode level.
 */
export function FeatureCard({
  title,
  description,
  icon,
  features,
  minMode = 'simple',
  className = '',
  defaultExpanded = false,
}: FeatureCardProps) {
  const { isFeatureVisible } = useProgressiveMode();
  const [expanded, setExpanded] = useState(defaultExpanded);

  // Don't render card if user's mode is below minMode
  if (!isFeatureVisible(minMode)) {
    return null;
  }

  // Separate features into visible (current mode) and expandable (higher modes)
  const visibleFeatures = features.filter(
    f => !f.minMode || isFeatureVisible(f.minMode)
  );
  const expandableFeatures = features.filter(
    f => f.minMode && !isFeatureVisible(f.minMode)
  );

  const hasExpandable = expandableFeatures.length > 0;

  return (
    <div
      className={`
        relative bg-surface border border-acid-green/30
        hover:border-acid-green/50 transition-colors
        ${className}
      `}
    >
      {/* Header */}
      <div className="border-b border-acid-green/20 px-4 py-3">
        <div className="flex items-center gap-2">
          {icon && (
            <span className="text-acid-green font-mono text-lg">{icon}</span>
          )}
          <h3 className="text-text font-bold font-mono">{title}</h3>
        </div>
        {description && (
          <p className="text-text-muted text-sm mt-1">{description}</p>
        )}
      </div>

      {/* Feature list */}
      <div className="p-2">
        {visibleFeatures.map((feature, idx) => (
          <FeatureLink key={idx} feature={feature} />
        ))}

        {/* Expandable section */}
        {hasExpandable && (
          <>
            {expanded && (
              <div className="mt-2 pt-2 border-t border-acid-green/10">
                {expandableFeatures.map((feature, idx) => (
                  <FeatureLink key={idx} feature={feature} locked />
                ))}
              </div>
            )}

            <button
              onClick={() => setExpanded(!expanded)}
              className="
                w-full mt-2 px-3 py-1.5
                text-xs font-mono text-acid-green/70
                hover:text-acid-green hover:bg-acid-green/5
                transition-colors text-left
                flex items-center gap-2
              "
            >
              <span>{expanded ? '[-]' : '[+]'}</span>
              <span>
                {expanded
                  ? 'Show less'
                  : `Show ${expandableFeatures.length} more features`}
              </span>
            </button>
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Individual feature link within a card
 */
function FeatureLink({
  feature,
  locked = false,
}: {
  feature: FeatureItem;
  locked?: boolean;
}) {
  const content = (
    <div
      className={`
        flex items-center gap-3 px-3 py-2 rounded
        ${locked
          ? 'opacity-50 cursor-not-allowed'
          : 'hover:bg-acid-green/10 cursor-pointer'
        }
        transition-colors
      `}
    >
      {feature.icon && (
        <span className="w-5 text-center text-acid-green/70 font-mono">
          {feature.icon}
        </span>
      )}
      <div className="flex-1 min-w-0">
        <span className={`text-sm font-mono ${locked ? 'text-text-muted' : 'text-text'}`}>
          {feature.label}
        </span>
        {feature.description && (
          <p className="text-xs text-text-muted truncate">
            {feature.description}
          </p>
        )}
      </div>
      {locked && (
        <span className="text-xs text-acid-cyan/50 font-mono">
          [{feature.minMode}]
        </span>
      )}
    </div>
  );

  if (locked) {
    return content;
  }

  return (
    <Link href={feature.href} className="block">
      {content}
    </Link>
  );
}

/**
 * Mode complexity indicator (dots)
 */
export function ModeIndicator({ mode }: { mode: ProgressiveMode }) {
  const modeIndex = ['simple', 'standard', 'advanced', 'expert'].indexOf(mode);
  const dots = modeIndex + 1;

  return (
    <span className="inline-flex gap-0.5">
      {Array.from({ length: 4 }).map((_, i) => (
        <span
          key={i}
          className={`
            w-1.5 h-1.5 rounded-full
            ${i < dots ? 'bg-acid-green' : 'bg-acid-green/20'}
          `}
        />
      ))}
    </span>
  );
}

/**
 * Mode selector component
 */
export function ModeSelector({ compact = false }: { compact?: boolean }) {
  const { mode, setMode } = useProgressiveMode();

  const modes: { value: ProgressiveMode; label: string; short: string }[] = [
    { value: 'simple', label: 'Simple', short: 'S' },
    { value: 'standard', label: 'Standard', short: 'ST' },
    { value: 'advanced', label: 'Advanced', short: 'A' },
    { value: 'expert', label: 'Expert', short: 'E' },
  ];

  if (compact) {
    return (
      <div className="flex border border-acid-green/30 rounded overflow-hidden">
        {modes.map(m => (
          <button
            key={m.value}
            onClick={() => setMode(m.value)}
            className={`
              px-2 py-1 text-xs font-mono
              ${mode === m.value
                ? 'bg-acid-green text-bg'
                : 'text-acid-green/70 hover:bg-acid-green/10'
              }
              transition-colors
            `}
            title={m.label}
          >
            {m.short}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="flex flex-wrap gap-2">
      {modes.map(m => (
        <button
          key={m.value}
          onClick={() => setMode(m.value)}
          className={`
            px-3 py-1.5 text-sm font-mono border
            ${mode === m.value
              ? 'bg-acid-green text-bg border-acid-green'
              : 'border-acid-green/30 text-acid-green/70 hover:border-acid-green/50 hover:text-acid-green'
            }
            transition-colors
          `}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
