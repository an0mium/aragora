'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { ErrorWithRetry } from '@/components/RetryButton';
import { fetchWithRetry } from '@/utils/retry';

interface Plugin {
  id: string;
  name: string;
  description: string;
  version: string;
  enabled: boolean;
  category: string;
  author?: string;
  config?: Record<string, unknown>;
}

function PluginCard({ plugin, onToggle }: { plugin: Plugin; onToggle: (id: string, enabled: boolean) => void }) {
  const [toggling, setToggling] = useState(false);

  const handleToggle = async () => {
    setToggling(true);
    await onToggle(plugin.id, !plugin.enabled);
    setToggling(false);
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-mono text-acid-green font-medium">{plugin.name}</h3>
          <div className="text-xs text-text-muted mt-1">
            v{plugin.version} {plugin.author && `by ${plugin.author}`}
          </div>
        </div>
        <button
          onClick={handleToggle}
          disabled={toggling}
          className={`px-3 py-1 rounded text-xs font-mono transition-colors ${
            plugin.enabled
              ? 'bg-acid-green/20 text-acid-green border border-acid-green/30 hover:bg-acid-green/30'
              : 'bg-crimson/20 text-crimson border border-crimson/30 hover:bg-crimson/30'
          } disabled:opacity-50`}
        >
          {toggling ? '...' : plugin.enabled ? 'ENABLED' : 'DISABLED'}
        </button>
      </div>

      <p className="text-sm text-text-muted mb-3">{plugin.description}</p>

      <div className="flex items-center gap-2">
        <span className="px-2 py-0.5 bg-bg rounded text-xs text-text-muted font-mono">
          {plugin.category}
        </span>
      </div>
    </div>
  );
}

function PluginManager({ apiBase }: { apiBase: string }) {
  const [plugins, setPlugins] = useState<Plugin[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'enabled' | 'disabled'>('all');

  const fetchPlugins = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetchWithRetry(`${apiBase}/api/plugins`, undefined, { maxRetries: 2 });
      if (response.ok) {
        const data = await response.json();
        setPlugins(data.plugins || []);
      } else {
        throw new Error(`Failed to fetch plugins: ${response.statusText}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch plugins');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPlugins();
  }, [apiBase]);

  const handleToggle = async (pluginId: string, enabled: boolean) => {
    try {
      const response = await fetchWithRetry(
        `${apiBase}/api/plugins/${pluginId}/toggle`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled }),
        },
        { maxRetries: 2 }
      );

      if (response.ok) {
        setPlugins((prev) =>
          prev.map((p) => (p.id === pluginId ? { ...p, enabled } : p))
        );
      } else {
        throw new Error('Failed to toggle plugin');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle plugin');
    }
  };

  const filteredPlugins = plugins.filter((p) => {
    if (filter === 'enabled') return p.enabled;
    if (filter === 'disabled') return !p.enabled;
    return true;
  });

  const categories = Array.from(new Set(plugins.map((p) => p.category)));

  if (loading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="bg-surface border border-border rounded-lg p-4 animate-pulse">
            <div className="h-6 bg-bg rounded w-1/3 mb-2" />
            <div className="h-4 bg-bg rounded w-2/3" />
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return <ErrorWithRetry error={error} onRetry={fetchPlugins} />;
  }

  return (
    <div>
      {/* Filter bar */}
      <div className="flex items-center gap-4 mb-6">
        <div className="flex items-center gap-2">
          {(['all', 'enabled', 'disabled'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 rounded text-xs font-mono transition-colors ${
                filter === f
                  ? 'bg-acid-green text-bg'
                  : 'bg-surface text-text-muted hover:text-text border border-border'
              }`}
            >
              {f.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="text-xs text-text-muted font-mono">
          {filteredPlugins.length} plugin{filteredPlugins.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-surface border border-border rounded-lg p-4">
          <div className="text-2xl font-mono text-acid-green">{plugins.length}</div>
          <div className="text-xs text-text-muted">Total Plugins</div>
        </div>
        <div className="bg-surface border border-border rounded-lg p-4">
          <div className="text-2xl font-mono text-acid-green">
            {plugins.filter((p) => p.enabled).length}
          </div>
          <div className="text-xs text-text-muted">Enabled</div>
        </div>
        <div className="bg-surface border border-border rounded-lg p-4">
          <div className="text-2xl font-mono text-acid-green">{categories.length}</div>
          <div className="text-xs text-text-muted">Categories</div>
        </div>
      </div>

      {/* Plugin list */}
      {filteredPlugins.length === 0 ? (
        <div className="text-center py-12 text-text-muted font-mono">
          <div className="text-4xl mb-4">{'>'}</div>
          <p>No plugins found</p>
          <p className="text-xs mt-2">
            {filter !== 'all' && 'Try changing the filter or '}
            Check that the backend is running
          </p>
        </div>
      ) : (
        <div className="grid gap-4">
          {categories.map((category) => {
            const categoryPlugins = filteredPlugins.filter((p) => p.category === category);
            if (categoryPlugins.length === 0) return null;

            return (
              <div key={category}>
                <h3 className="text-sm font-mono text-acid-cyan mb-3 uppercase tracking-wider">
                  {category}
                </h3>
                <div className="grid gap-4 md:grid-cols-2">
                  {categoryPlugins.map((plugin) => (
                    <PluginCard key={plugin.id} plugin={plugin} onToggle={handleToggle} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function PluginsPage() {
  const { config: backendConfig } = useBackend();

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
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">Plugin Manager</h1>
            <p className="text-text-muted font-mono text-sm">
              Configure and manage agent plugins. Enable or disable capabilities as needed.
            </p>
          </div>

          <PanelErrorBoundary panelName="Plugin Manager">
            <PluginManager apiBase={backendConfig.api} />
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // PLUGIN MANAGER
          </p>
        </footer>
      </main>
    </>
  );
}
