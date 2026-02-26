'use client';

import { useState, type ReactNode } from 'react';

interface DAGToolbarProps {
  onBrainDump: (text: string) => void;
  onAutoFlow: (ideas: string[]) => void;
  onUndo: () => void;
  onRedo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  loading: boolean;
  stageFilter: string | null;
  onStageFilterChange: (stage: string | null) => void;
  children?: ReactNode;
}

const STAGES = ['ideas', 'goals', 'actions', 'orchestration'] as const;

export function DAGToolbar({
  onBrainDump,
  onAutoFlow,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  loading,
  stageFilter,
  onStageFilterChange,
  children,
}: DAGToolbarProps) {
  const [showBrainDump, setShowBrainDump] = useState(false);
  const [brainDumpText, setBrainDumpText] = useState('');

  const handleBrainDumpSubmit = () => {
    if (!brainDumpText.trim()) return;
    const ideas = brainDumpText
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);
    onBrainDump(brainDumpText);
    onAutoFlow(ideas);
    setBrainDumpText('');
    setShowBrainDump(false);
  };

  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b border-border bg-surface">
      {/* Brain Dump */}
      <button
        onClick={() => setShowBrainDump(!showBrainDump)}
        disabled={loading}
        className="px-3 py-1.5 bg-indigo-600 text-white text-sm font-mono rounded hover:bg-indigo-500 transition-colors disabled:opacity-50"
      >
        Brain Dump
      </button>

      {/* Auto-Flow */}
      <button
        onClick={() => showBrainDump && handleBrainDumpSubmit()}
        disabled={loading || !showBrainDump || !brainDumpText.trim()}
        className="px-3 py-1.5 bg-emerald-600 text-white text-sm font-mono rounded hover:bg-emerald-500 transition-colors disabled:opacity-50"
      >
        {loading ? 'Running...' : 'Auto-Flow'}
      </button>

      {/* Divider */}
      <div className="w-px h-6 bg-border" />

      {/* Stage filters */}
      <button
        onClick={() => onStageFilterChange(null)}
        className={`px-2 py-1 text-xs font-mono rounded transition-colors ${
          stageFilter === null ? 'bg-indigo-600 text-white' : 'text-text-muted hover:text-text'
        }`}
      >
        All
      </button>
      {STAGES.map((s) => (
        <button
          key={s}
          onClick={() => onStageFilterChange(s === stageFilter ? null : s)}
          className={`px-2 py-1 text-xs font-mono rounded capitalize transition-colors ${
            stageFilter === s ? 'bg-indigo-600 text-white' : 'text-text-muted hover:text-text'
          }`}
        >
          {s}
        </button>
      ))}

      {/* Divider */}
      <div className="w-px h-6 bg-border" />

      {/* Undo/Redo */}
      <button
        onClick={onUndo}
        disabled={!canUndo}
        className="px-2 py-1 text-sm font-mono text-text-muted hover:text-text disabled:opacity-30 transition-colors"
        title="Undo"
      >
        {'\u21B6'}
      </button>
      <button
        onClick={onRedo}
        disabled={!canRedo}
        className="px-2 py-1 text-sm font-mono text-text-muted hover:text-text disabled:opacity-30 transition-colors"
        title="Redo"
      >
        {'\u21B7'}
      </button>

      {/* Additional toolbar items (e.g. execution toggle) */}
      {children}

      {/* Brain Dump Input */}
      {showBrainDump && (
        <div className="absolute top-14 left-4 z-30 w-96 bg-surface border border-border rounded-lg shadow-elevated p-3">
          <textarea
            value={brainDumpText}
            onChange={(e) => setBrainDumpText(e.target.value)}
            placeholder="Paste your ideas here, one per line..."
            className="w-full h-32 bg-bg text-text text-sm font-mono p-2 rounded border border-border resize-none focus:outline-none focus:border-indigo-500"
          />
          <div className="flex justify-end gap-2 mt-2">
            <button
              onClick={() => setShowBrainDump(false)}
              className="px-3 py-1 text-xs font-mono text-text-muted hover:text-text"
            >
              Cancel
            </button>
            <button
              onClick={handleBrainDumpSubmit}
              disabled={!brainDumpText.trim() || loading}
              className="px-3 py-1 text-xs font-mono bg-indigo-600 text-white rounded hover:bg-indigo-500 disabled:opacity-50"
            >
              Process
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
