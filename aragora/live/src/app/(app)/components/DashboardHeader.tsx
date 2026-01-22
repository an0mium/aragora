'use client';

import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { StatusPill } from '@/components/StatusBar';
import { BackendSelector } from '@/components/BackendSelector';
import { ThemeToggle } from '@/components/ThemeToggle';
import { DashboardModeToggle } from '@/components/DashboardModeToggle';
import { DeepAuditToggle } from '@/components/deep-audit';
import { CompareButton } from '@/components/CompareView';
import { ModeSelector } from '@/components/ui/FeatureCard';
import type { DashboardMode } from '@/hooks/useDashboardPreferences';

type ViewMode = 'tabs' | 'stream' | 'deep-audit';

interface ActiveLoop {
  loop_id: string;
  name: string;
  cycle: number;
  started_at: number;
}

interface DashboardHeaderProps {
  connected: boolean;
  showHeaderAscii: boolean;
  isMobile: boolean;
  showSidebar: boolean;
  onToggleSidebar: () => void;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  dashboardMode: DashboardMode;
  onDashboardModeChange: (mode: DashboardMode) => void;
  currentPhase: string;
  onShowCompare: () => void;
  activeLoops: ActiveLoop[];
  selectedLoopId: string | null;
  onSelectLoop: (loopId: string) => void;
}

function formatRelativeTime(timestamp: number): string {
  const seconds = Math.floor((Date.now() / 1000) - timestamp);
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

export function DashboardHeader({
  connected,
  showHeaderAscii,
  isMobile,
  showSidebar,
  onToggleSidebar,
  viewMode,
  onViewModeChange,
  dashboardMode,
  onDashboardModeChange,
  currentPhase,
  onShowCompare,
  activeLoops,
  selectedLoopId,
  onSelectLoop,
}: DashboardHeaderProps) {
  return (
    <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-screen-2xl mx-auto px-3 sm:px-4 lg:px-6 py-2 sm:py-3">
        <div className="flex items-center justify-between gap-2">
          {/* ASCII Logo */}
          <AsciiBannerCompact connected={connected} showAsciiArt={showHeaderAscii} />

          {/* Controls */}
          <div className="flex items-center gap-1 sm:gap-2 lg:gap-3">
            {/* Mobile Sidebar Toggle */}
            {isMobile && (
              <button
                onClick={onToggleSidebar}
                className="px-2 py-1 border border-acid-green/30 text-xs font-mono text-acid-green hover:bg-acid-green/10 transition-colors"
                aria-label={showSidebar ? 'Hide sidebar panels' : 'Show sidebar panels'}
                aria-expanded={showSidebar}
                aria-controls="sidebar-panels"
              >
                {showSidebar ? '[HIDE PANELS]' : '[PANELS]'}
              </button>
            )}

            {/* View Mode Toggle - Hidden on mobile */}
            <div className="hidden sm:flex items-center gap-0.5 bg-bg border border-acid-green/30 p-0.5 font-mono text-xs" role="tablist" aria-label="View mode selection">
              <button
                onClick={() => onViewModeChange('tabs')}
                role="tab"
                aria-selected={viewMode === 'tabs'}
                aria-label="Switch to tabbed view"
                className={`px-2 py-1 transition-colors ${
                  viewMode === 'tabs'
                    ? 'bg-acid-green text-bg'
                    : 'text-text-muted hover:text-acid-green'
                }`}
              >
                [TABS]
              </button>
              <button
                onClick={() => onViewModeChange('stream')}
                role="tab"
                aria-selected={viewMode === 'stream'}
                aria-label="Switch to stream view"
                className={`px-2 py-1 transition-colors ${
                  viewMode === 'stream'
                    ? 'bg-acid-green text-bg'
                    : 'text-text-muted hover:text-acid-green'
                }`}
              >
                [STREAM]
              </button>
            </div>

            {/* Dashboard Mode Toggle - Focus vs Explorer */}
            <div className="hidden sm:block">
              <DashboardModeToggle
                mode={dashboardMode}
                onModeChange={onDashboardModeChange}
                compact
              />
            </div>

            {/* Progressive Mode Selector */}
            <div className="hidden lg:block">
              <ModeSelector compact />
            </div>

            {/* Deep Audit Toggle - Hidden on small screens */}
            <div className="hidden md:block">
              <DeepAuditToggle
                isActive={viewMode === 'deep-audit'}
                onToggle={() => onViewModeChange(viewMode === 'deep-audit' ? 'tabs' : 'deep-audit')}
              />
            </div>

            {/* Compare Button - Hidden on mobile */}
            <div className="hidden lg:block">
              <CompareButton onClick={onShowCompare} />
            </div>

            {/* Status Pill - Always visible */}
            <StatusPill connected={connected} phase={currentPhase} />

            {/* Backend Selector - Hidden on mobile */}
            <div className="hidden sm:block">
              <BackendSelector compact />
            </div>

            {/* Theme Toggle - Always visible */}
            <ThemeToggle />

            {/* Loop Selector - Only show if multiple loops */}
            {activeLoops.length > 1 && (
              <div className="hidden md:flex items-center gap-2">
                <label htmlFor="loop-selector" className="text-text-muted text-xs font-mono">{activeLoops.length} LOOPS</label>
                <select
                  id="loop-selector"
                  value={selectedLoopId || ''}
                  onChange={(e) => onSelectLoop(e.target.value)}
                  aria-label="Select active loop"
                  className="bg-bg border border-acid-green/30 px-2 py-1 text-xs font-mono text-acid-green"
                >
                  {activeLoops.map((loop) => (
                    <option key={loop.loop_id} value={loop.loop_id}>
                      {loop.name} (C{loop.cycle}, {formatRelativeTime(loop.started_at)})
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Single loop indicator */}
            {activeLoops.length === 1 && (
              <span className="hidden sm:inline text-acid-cyan text-xs font-mono">
                {activeLoops[0].name}
              </span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
