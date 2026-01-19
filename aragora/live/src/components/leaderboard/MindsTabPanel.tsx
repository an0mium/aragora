'use client';

import { memo } from 'react';
import Link from 'next/link';
import { IntrospectionListSkeleton } from '../Skeleton';
import type { AgentIntrospection } from './types';

interface MindsTabPanelProps {
  introspections: AgentIntrospection[];
  loading: boolean;
}

function MindsTabPanelComponent({ introspections, loading }: MindsTabPanelProps) {
  return (
    <div
      id="minds-panel"
      role="tabpanel"
      aria-labelledby="minds-tab"
      className="space-y-3 max-h-80 overflow-y-auto"
    >
      {loading && <IntrospectionListSkeleton count={2} />}

      {!loading && introspections.length === 0 && (
        <div className="text-center text-text-muted py-4">
          No introspection data yet. Agents build self-models through debate participation.
        </div>
      )}

      {introspections.map((intro) => (
        <div
          key={intro.agent}
          className="p-3 bg-bg border border-border rounded-lg hover:border-purple-500/30 transition-colors"
        >
          <div className="flex items-center justify-between mb-2">
            <Link
              href={`/agent/${encodeURIComponent(intro.agent)}/`}
              className="font-medium text-text hover:text-accent transition-colors"
            >
              {intro.agent}
            </Link>
            <span
              className={`text-xs px-2 py-0.5 rounded ${
                intro.confidence_calibration >= 0.8
                  ? 'bg-green-500/20 text-green-400'
                  : intro.confidence_calibration >= 0.5
                    ? 'bg-yellow-500/20 text-yellow-400'
                    : 'bg-red-500/20 text-red-400'
              }`}
            >
              {Math.round(intro.confidence_calibration * 100)}% calibrated
            </span>
          </div>

          {/* Performance Assessment */}
          {intro.recent_performance_assessment && (
            <p className="text-sm text-text-muted mb-2 italic">
              &quot;{intro.recent_performance_assessment}&quot;
            </p>
          )}

          {/* Strengths / Weaknesses / Biases */}
          <div className="grid grid-cols-3 gap-2 text-xs mb-2">
            {intro.self_model?.strengths?.length > 0 && (
              <div>
                <div className="text-green-400 mb-1">Strengths</div>
                {intro.self_model.strengths.slice(0, 2).map((s, i) => (
                  <div key={i} className="text-text-muted truncate">
                    {s}
                  </div>
                ))}
              </div>
            )}
            {intro.self_model?.weaknesses?.length > 0 && (
              <div>
                <div className="text-yellow-400 mb-1">Weaknesses</div>
                {intro.self_model.weaknesses.slice(0, 2).map((w, i) => (
                  <div key={i} className="text-text-muted truncate">
                    {w}
                  </div>
                ))}
              </div>
            )}
            {intro.self_model?.biases?.length > 0 && (
              <div>
                <div className="text-red-400 mb-1">Known Biases</div>
                {intro.self_model.biases.slice(0, 2).map((b, i) => (
                  <div key={i} className="text-text-muted truncate">
                    {b}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Improvement Focus */}
          {intro.improvement_focus?.length > 0 && (
            <div className="text-xs">
              <span className="text-purple-400">Focus:</span>{' '}
              <span className="text-text-muted">
                {intro.improvement_focus.slice(0, 2).join(', ')}
              </span>
            </div>
          )}

          <div className="text-xs text-text-muted mt-2">
            Updated:{' '}
            {intro.last_updated ? new Date(intro.last_updated).toLocaleDateString() : 'N/A'}
          </div>
        </div>
      ))}
    </div>
  );
}

export const MindsTabPanel = memo(MindsTabPanelComponent);
