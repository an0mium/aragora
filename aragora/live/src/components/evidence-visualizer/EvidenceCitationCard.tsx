'use client';

/**
 * Evidence citation card component for displaying individual evidence items
 */

import React from 'react';
import { ConfidenceBar } from './ConfidenceBar';
import { SourceTypeBadge } from './SourceTypeBadge';
import type { EvidenceCitation } from './types';

interface EvidenceCitationCardProps {
  citation: EvidenceCitation;
}

export function EvidenceCitationCard({ citation }: EvidenceCitationCardProps) {
  return (
    <div className="p-4 bg-surface rounded border border-acid-green/20">
      {/* Header row with round, agent, and source type */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-acid-cyan">
            Round {citation.round}
          </span>
          <span className="font-mono text-xs text-text-muted">
            {citation.agent}
          </span>
          <SourceTypeBadge sourceType={citation.source_type} />
        </div>
        {citation.reliability_score !== undefined && (
          <span className={`font-mono text-xs px-2 py-0.5 rounded ${
            citation.reliability_score >= 0.7 ? 'bg-acid-green/20 text-acid-green' :
            citation.reliability_score >= 0.4 ? 'bg-acid-yellow/20 text-acid-yellow' :
            'bg-acid-red/20 text-acid-red'
          }`}>
            {(citation.reliability_score * 100).toFixed(0)}% reliable
          </span>
        )}
      </div>

      {/* Claim content */}
      <p className="font-mono text-sm text-text mb-3">{citation.claim}</p>

      {/* Source with optional link */}
      <div className="flex items-center gap-2 mb-3">
        <span className="font-mono text-xs text-text-muted">Source:</span>
        {citation.url ? (
          <a
            href={citation.url}
            target="_blank"
            rel="noopener noreferrer"
            className="font-mono text-xs text-acid-green hover:underline truncate max-w-md"
          >
            {citation.source}
          </a>
        ) : (
          <span className="font-mono text-xs text-acid-green truncate max-w-md">
            {citation.source}
          </span>
        )}
      </div>

      {/* Confidence metrics */}
      {(citation.confidence !== undefined || citation.freshness !== undefined || citation.authority !== undefined) && (
        <div className="space-y-1.5 pt-3 border-t border-acid-green/10">
          {citation.confidence !== undefined && (
            <ConfidenceBar value={citation.confidence} label="Conf." color="acid-green" />
          )}
          {citation.freshness !== undefined && (
            <ConfidenceBar value={citation.freshness} label="Fresh" color="acid-cyan" />
          )}
          {citation.authority !== undefined && (
            <ConfidenceBar value={citation.authority} label="Auth." color="acid-yellow" />
          )}
        </div>
      )}
    </div>
  );
}

export default EvidenceCitationCard;
