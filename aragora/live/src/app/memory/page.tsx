'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

const MemoryExplorerPanel = dynamic(
  () => import('@/components/MemoryExplorerPanel').then(m => ({ default: m.MemoryExplorerPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-96 bg-surface rounded" />
      </div>
    ),
  }
);

const MemoryAnalyticsPanel = dynamic(
  () => import('@/components/MemoryAnalyticsPanel').then(m => ({ default: m.MemoryAnalyticsPanel })),
  {
    ssr: false,
    loading: () => (
      <div className="card p-4 animate-pulse">
        <div className="h-48 bg-surface rounded" />
      </div>
    ),
  }
);

interface MemoryPressure {
  overall_pressure: number;
  tier_pressure: {
    fast: number;
    medium: number;
    slow: number;
    glacial: number;
  };
  alerts: string[];
  recommendation?: string;
}

function PressureGauge({ value, label, color }: { value: number; label: string; color: string }) {
  const percentage = Math.min(100, Math.max(0, value * 100));
  const barColor = percentage > 80 ? 'bg-warning' : percentage > 60 ? 'bg-acid-yellow' : color;

  return (
    <div className="flex-1">
      <div className="flex justify-between text-xs font-mono mb-1">
        <span className="text-text-muted">{label}</span>
        <span className={percentage > 80 ? 'text-warning' : 'text-text'}>{percentage.toFixed(0)}%</span>
      </div>
      <div className="h-2 bg-bg rounded overflow-hidden">
        <div className={`h-full transition-all ${barColor}`} style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
}

export default function MemoryPage() {
  const { config: backendConfig } = useBackend();
  const [pressure, setPressure] = useState<MemoryPressure | null>(null);
  const [activeTab, setActiveTab] = useState<'explorer' | 'analytics'>('explorer');

  // Fetch memory pressure data
  useEffect(() => {
    const fetchPressure = async () => {
      try {
        const res = await fetch(`${backendConfig.api}/api/memory/pressure`);
        if (res.ok) {
          const data = await res.json();
          setPressure(data.pressure || data);
        }
      } catch {
        // Pressure endpoint may not exist
      }
    };

    fetchPressure();
    const interval = setInterval(fetchPressure, 30000);
    return () => clearInterval(interval);
  }, [backendConfig.api]);

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-3">
              <Link
                href="/"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <Link
                href="/memory-analytics"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [ANALYTICS]
              </Link>
              <Link
                href="/insights"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [INSIGHTS]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} CONTINUUM MEMORY
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Explore the multi-tier memory system: fast, medium, slow, and glacial storage.
            </p>
          </div>

          {/* Memory Pressure Section */}
          {pressure && (
            <div className="mb-6 p-4 border border-acid-cyan/30 bg-acid-cyan/5 rounded">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-mono text-acid-cyan">Memory Pressure</h3>
                <span className={`text-xs font-mono px-2 py-0.5 rounded ${
                  pressure.overall_pressure > 0.8 ? 'bg-warning/20 text-warning' :
                  pressure.overall_pressure > 0.6 ? 'bg-acid-yellow/20 text-acid-yellow' :
                  'bg-acid-green/20 text-acid-green'
                }`}>
                  {pressure.overall_pressure > 0.8 ? 'HIGH' :
                   pressure.overall_pressure > 0.6 ? 'MODERATE' : 'NORMAL'}
                </span>
              </div>
              <div className="flex gap-4">
                <PressureGauge value={pressure.tier_pressure.fast} label="Fast" color="bg-acid-green" />
                <PressureGauge value={pressure.tier_pressure.medium} label="Medium" color="bg-acid-cyan" />
                <PressureGauge value={pressure.tier_pressure.slow} label="Slow" color="bg-gold" />
                <PressureGauge value={pressure.tier_pressure.glacial} label="Glacial" color="bg-acid-purple" />
              </div>
              {pressure.alerts && pressure.alerts.length > 0 && (
                <div className="mt-3 pt-3 border-t border-acid-cyan/20">
                  {pressure.alerts.map((alert, i) => (
                    <div key={i} className="text-xs font-mono text-warning flex items-center gap-2">
                      <span>!</span> {alert}
                    </div>
                  ))}
                </div>
              )}
              {pressure.recommendation && (
                <div className="mt-2 text-xs font-mono text-text-muted">
                  Recommendation: {pressure.recommendation}
                </div>
              )}
            </div>
          )}

          {/* Tab Navigation */}
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => setActiveTab('explorer')}
              className={`px-4 py-2 font-mono text-sm border transition-colors ${
                activeTab === 'explorer'
                  ? 'border-acid-green bg-acid-green/10 text-acid-green'
                  : 'border-acid-green/30 text-text-muted hover:text-text'
              }`}
            >
              [EXPLORER]
            </button>
            <button
              onClick={() => setActiveTab('analytics')}
              className={`px-4 py-2 font-mono text-sm border transition-colors ${
                activeTab === 'analytics'
                  ? 'border-acid-green bg-acid-green/10 text-acid-green'
                  : 'border-acid-green/30 text-text-muted hover:text-text'
              }`}
            >
              [ANALYTICS]
            </button>
          </div>

          {activeTab === 'explorer' ? (
            <PanelErrorBoundary panelName="Memory Explorer">
              <MemoryExplorerPanel backendConfig={{ apiUrl: backendConfig.api, wsUrl: backendConfig.ws }} />
            </PanelErrorBoundary>
          ) : (
            <PanelErrorBoundary panelName="Memory Analytics">
              <MemoryAnalyticsPanel apiBase={backendConfig.api} />
            </PanelErrorBoundary>
          )}
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // MEMORY EXPLORER
          </p>
        </footer>
      </main>
    </>
  );
}
