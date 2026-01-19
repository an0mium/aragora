'use client';

import { getAgentColors } from '@/utils/agentColors';
import type { DebateNode } from './types';
import { getBranchColor } from './utils';

export interface NodeDetailPanelProps {
  node: DebateNode;
  onClose: () => void;
}

export function NodeDetailPanel({ node, onClose }: NodeDetailPanelProps) {
  const colors = getAgentColors(node.agent_id);

  return (
    <div className="absolute top-4 right-4 w-96 bg-surface border border-acid-green/30 shadow-lg z-10">
      <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 ${colors.bg} ${colors.text} text-xs font-mono`}>
            {node.agent_id}
          </span>
          <span className="text-xs font-mono text-text-muted uppercase">
            {node.node_type.replace('_', ' ')}
          </span>
        </div>
        <button
          onClick={onClose}
          className="text-text-muted hover:text-acid-green text-xs font-mono"
          aria-label="Close node details"
        >
          [X]
        </button>
      </div>

      <div className="p-4 space-y-3 max-h-[60vh] overflow-y-auto">
        {/* Content */}
        <div>
          <div className="text-xs font-mono text-text-muted mb-1">CONTENT</div>
          <div className="text-sm font-mono text-text whitespace-pre-wrap">
            {node.content.length > 500 ? node.content.slice(0, 500) + '...' : node.content}
          </div>
        </div>

        {/* Claims */}
        {node.claims.length > 0 && (
          <div>
            <div className="text-xs font-mono text-acid-cyan mb-1">CLAIMS ({node.claims.length})</div>
            <ul className="space-y-1">
              {node.claims.slice(0, 5).map((claim, i) => (
                <li key={i} className="text-xs font-mono text-text-muted pl-2 border-l border-acid-cyan/30">
                  {claim.slice(0, 100)}{claim.length > 100 ? '...' : ''}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Metadata */}
        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          <div>
            <span className="text-text-muted">Branch: </span>
            <span className={getBranchColor(node.branch_id || 'main')}>
              {node.branch_id || 'main'}
            </span>
          </div>
          <div>
            <span className="text-text-muted">Confidence: </span>
            <span className="text-acid-green">{(node.confidence * 100).toFixed(0)}%</span>
          </div>
          <div>
            <span className="text-text-muted">Parents: </span>
            <span className="text-text">{node.parent_ids.length}</span>
          </div>
          <div>
            <span className="text-text-muted">Children: </span>
            <span className="text-text">{node.child_ids.length}</span>
          </div>
        </div>

        {/* Hash */}
        <div className="text-[10px] font-mono text-text-muted/50 pt-2 border-t border-border">
          Hash: {node.hash}
        </div>
      </div>
    </div>
  );
}

export default NodeDetailPanel;
