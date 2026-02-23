'use client';

import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { ToolCatalog } from '@/components/tools/ToolCatalog';
import { TOOL_CATEGORIES, ALL_TOOLS } from '@/lib/mcp-tools-registry';

export default function ToolsPage() {
  return (
    <div className="relative min-h-screen">
      <Scanlines />
      <CRTVignette />

      <div className="relative z-10">
        {/* Header */}
        <div className="border-b border-[var(--acid-green)]/20 bg-[var(--surface)]/30">
          <div className="container mx-auto px-4 py-10">
            <h1 className="text-2xl md:text-3xl font-mono text-[var(--acid-green)] mb-3">
              {'>'} MCP TOOL PLAYGROUND
            </h1>
            <p className="text-sm font-mono text-[var(--text-muted)] max-w-2xl">
              Explore and discover {ALL_TOOLS.length}+ AI-powered tools across debate orchestration,
              knowledge management, verification, and autonomous self-improvement.
            </p>

            {/* Summary stats */}
            <div className="flex flex-wrap gap-4 mt-6">
              <div className="border border-[var(--acid-green)]/30 bg-[var(--acid-green)]/5 rounded px-4 py-2">
                <div className="text-lg font-mono text-[var(--acid-green)] font-bold">
                  {ALL_TOOLS.length}
                </div>
                <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
                  Tools
                </div>
              </div>
              <div className="border border-[var(--border)] bg-[var(--surface)]/50 rounded px-4 py-2">
                <div className="text-lg font-mono text-[var(--text)] font-bold">
                  {TOOL_CATEGORIES.length}
                </div>
                <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
                  Categories
                </div>
              </div>
              <div className="border border-[var(--border)] bg-[var(--surface)]/50 rounded px-4 py-2">
                <div className="text-xs font-mono text-[var(--text)] font-bold leading-6">
                  API + SDK
                </div>
                <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
                  Access
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Catalog */}
        <div className="container mx-auto px-4 py-8">
          <ToolCatalog tools={ALL_TOOLS} categories={TOOL_CATEGORIES} />
        </div>
      </div>
    </div>
  );
}
