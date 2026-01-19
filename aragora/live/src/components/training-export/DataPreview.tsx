'use client';

import { memo } from 'react';
import type { ExportType } from './types';

interface DataPreviewProps {
  records: unknown[];
  exportType: ExportType;
}

function DataPreviewComponent({ records, exportType }: DataPreviewProps) {
  if (!records.length) return null;

  const sample = records.slice(0, 3);

  return (
    <div className="bg-slate-800 rounded-lg p-4 mb-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-white">Data Preview</h4>
        <span className="text-xs text-slate-400">
          Showing {sample.length} of {records.length} records
        </span>
      </div>
      <div className="space-y-2 max-h-60 overflow-y-auto">
        {sample.map((record, i) => (
          <div
            key={i}
            className="bg-slate-900 rounded p-2 text-xs font-mono text-slate-300 overflow-x-auto"
          >
            <pre className="whitespace-pre-wrap break-words">
              {JSON.stringify(record, null, 2).slice(0, 500)}
              {JSON.stringify(record, null, 2).length > 500 && '...'}
            </pre>
          </div>
        ))}
      </div>
      <div className="mt-3 flex items-center gap-2 text-xs">
        <span className="text-slate-400">Format:</span>
        <span className="px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded">
          {exportType.toUpperCase()}
        </span>
      </div>
    </div>
  );
}

export const DataPreview = memo(DataPreviewComponent);
