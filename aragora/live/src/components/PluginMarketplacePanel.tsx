'use client';

import { useState, useEffect, useCallback } from 'react';
import { ErrorWithRetry } from './RetryButton';
import { fetchWithRetry } from '@/utils/retry';

interface PluginManifest {
  name: string;
  version: string;
  description: string;
  author: string;
  capabilities: string[];
  requirements: string[];
  entry_point: string;
  timeout_seconds: number;
  max_memory_mb: number;
  python_packages: string[];
  system_tools: string[];
  license: string;
  homepage: string;
  tags: string[];
  created_at: string;
  requirements_satisfied?: boolean;
  missing_requirements?: string[];
}

interface BackendConfig {
  apiUrl: string;
  wsUrl: string;
}

interface PluginMarketplacePanelProps {
  backendConfig?: BackendConfig;
}

const DEFAULT_API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

const CAPABILITY_COLORS: Record<string, { text: string; bg: string }> = {
  code_analysis: { text: 'text-acid-cyan', bg: 'bg-acid-cyan/20' },
  lint: { text: 'text-acid-yellow', bg: 'bg-acid-yellow/20' },
  security_scan: { text: 'text-acid-red', bg: 'bg-acid-red/20' },
  type_check: { text: 'text-acid-green', bg: 'bg-acid-green/20' },
  test_runner: { text: 'text-acid-cyan', bg: 'bg-acid-cyan/20' },
  benchmark: { text: 'text-acid-yellow', bg: 'bg-acid-yellow/20' },
  formatter: { text: 'text-acid-green', bg: 'bg-acid-green/20' },
  evidence_fetch: { text: 'text-acid-cyan', bg: 'bg-acid-cyan/20' },
  documentation: { text: 'text-text', bg: 'bg-surface' },
  formal_verify: { text: 'text-acid-red', bg: 'bg-acid-red/20' },
  property_check: { text: 'text-acid-yellow', bg: 'bg-acid-yellow/20' },
  custom: { text: 'text-text-muted', bg: 'bg-surface' },
};

const REQUIREMENT_INFO: Record<string, { icon: string; description: string }> = {
  read_files: { icon: '📖', description: 'Can read local files' },
  write_files: { icon: '📝', description: 'Can write local files' },
  run_commands: { icon: '⚡', description: 'Can execute shell commands' },
  network: { icon: '🌐', description: 'Makes network requests' },
  high_memory: { icon: '🧠', description: 'Requires >1GB RAM' },
  long_running: { icon: '⏱️', description: 'May run >60 seconds' },
  python_packages: { icon: '📦', description: 'External Python packages' },
  system_tools: { icon: '🔧', description: 'External system tools' },
};

export function PluginMarketplacePanel({ backendConfig }: PluginMarketplacePanelProps) {
  const apiBase = backendConfig?.apiUrl || DEFAULT_API_BASE;

  const [plugins, setPlugins] = useState<PluginManifest[]>([]);
  const [selectedPlugin, setSelectedPlugin] = useState<PluginManifest | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [usingDemoData, setUsingDemoData] = useState(false);
  const [filterCapability, setFilterCapability] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');

  const fetchPlugins = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetchWithRetry(`${apiBase}/api/plugins`, undefined, { maxRetries: 2 });

      if (response.ok) {
        const data = await response.json();
        setPlugins(data.plugins || []);
        setError(null);
        setUsingDemoData(false);
      } else {
        // Demo data when API unavailable
        setUsingDemoData(true);
        setPlugins([
          {
            name: 'security-scan',
            version: '1.0.0',
            description: 'Scan code for security vulnerabilities using static analysis',
            author: 'aragora',
            capabilities: ['security_scan', 'code_analysis'],
            requirements: ['read_files'],
            entry_point: 'security_scan:run',
            timeout_seconds: 120,
            max_memory_mb: 512,
            python_packages: ['bandit'],
            system_tools: [],
            license: 'MIT',
            homepage: 'https://github.com/aragora/plugins',
            tags: ['security', 'analysis'],
            created_at: new Date().toISOString(),
          },
          {
            name: 'test-runner',
            version: '1.2.0',
            description: 'Execute test suites and report results',
            author: 'aragora',
            capabilities: ['test_runner'],
            requirements: ['read_files', 'run_commands'],
            entry_point: 'test_runner:run',
            timeout_seconds: 300,
            max_memory_mb: 1024,
            python_packages: ['pytest'],
            system_tools: [],
            license: 'MIT',
            homepage: '',
            tags: ['testing'],
            created_at: new Date().toISOString(),
          },
          {
            name: 'code-formatter',
            version: '2.0.0',
            description: 'Format code according to style guidelines',
            author: 'aragora',
            capabilities: ['formatter'],
            requirements: ['read_files', 'write_files'],
            entry_point: 'code_formatter:run',
            timeout_seconds: 60,
            max_memory_mb: 256,
            python_packages: ['black', 'isort'],
            system_tools: [],
            license: 'MIT',
            homepage: '',
            tags: ['formatting', 'style'],
            created_at: new Date().toISOString(),
          },
        ]);
        setError(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch plugins');
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  const fetchPluginDetails = useCallback(async (pluginName: string) => {
    try {
      const response = await fetchWithRetry(
        `${apiBase}/api/plugins/${pluginName}`,
        undefined,
        { maxRetries: 2 }
      );

      if (response.ok) {
        const data = await response.json();
        setSelectedPlugin(data);
      }
    } catch (err) {
      console.error('Failed to fetch plugin details:', err);
    }
  }, [apiBase]);

  useEffect(() => {
    fetchPlugins();
  }, [fetchPlugins]);

  // Filter plugins
  const filteredPlugins = plugins.filter((plugin) => {
    const matchesCapability = !filterCapability || plugin.capabilities.includes(filterCapability);
    const matchesSearch = !searchQuery ||
      plugin.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      plugin.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      plugin.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesCapability && matchesSearch;
  });

  // Get unique capabilities for filter
  const allCapabilities = Array.from(new Set(plugins.flatMap(p => p.capabilities)));

  if (loading && plugins.length === 0) {
    return (
      <div className="card p-6">
        <div className="flex items-center gap-3">
          <div className="animate-spin w-5 h-5 border-2 border-acid-green border-t-transparent rounded-full" />
          <span className="font-mono text-text-muted">Loading plugins...</span>
        </div>
      </div>
    );
  }

  if (error && plugins.length === 0) {
    return (
      <ErrorWithRetry
        error={error || "Failed to load plugins"}
        onRetry={fetchPlugins}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Demo Mode Indicator */}
      {usingDemoData && (
        <div className="bg-warning/10 border border-warning/30 rounded px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-warning">⚠</span>
            <span className="font-mono text-sm text-warning">
              Demo Mode - Showing example plugins (API unavailable)
            </span>
          </div>
          <button
            onClick={fetchPlugins}
            className="font-mono text-xs text-warning hover:text-warning/80 transition-colors"
          >
            [RETRY]
          </button>
        </div>
      )}

      {/* Search and Filter */}
      <div className="card p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block font-mono text-xs text-text-muted mb-2">
              Search Plugins
            </label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by name, description, or tags..."
              className="w-full bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
            />
          </div>
          <div>
            <label className="block font-mono text-xs text-text-muted mb-2">
              Filter by Capability
            </label>
            <select
              value={filterCapability}
              onChange={(e) => setFilterCapability(e.target.value)}
              className="w-full bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
            >
              <option value="">All Capabilities</option>
              {allCapabilities.map((cap) => (
                <option key={cap} value={cap}>
                  {cap.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase())}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="card p-4">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-mono text-acid-green">{plugins.length}</div>
            <div className="text-xs font-mono text-text-muted">Total Plugins</div>
          </div>
          <div>
            <div className="text-2xl font-mono text-acid-cyan">{allCapabilities.length}</div>
            <div className="text-xs font-mono text-text-muted">Capabilities</div>
          </div>
          <div>
            <div className="text-2xl font-mono text-acid-yellow">{filteredPlugins.length}</div>
            <div className="text-xs font-mono text-text-muted">Showing</div>
          </div>
        </div>
      </div>

      {/* Plugin Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredPlugins.map((plugin) => (
          <button
            key={plugin.name}
            onClick={() => {
              setSelectedPlugin(plugin);
              fetchPluginDetails(plugin.name);
            }}
            className={`card p-4 text-left transition-all hover:border-acid-green/60 ${
              selectedPlugin?.name === plugin.name ? 'border-acid-green bg-acid-green/5' : ''
            }`}
          >
            <div className="flex items-start justify-between mb-2">
              <h3 className="font-mono text-acid-green font-bold">{plugin.name}</h3>
              <span className="text-xs font-mono text-text-muted">v{plugin.version}</span>
            </div>

            <p className="font-mono text-xs text-text-muted mb-3 line-clamp-2">
              {plugin.description}
            </p>

            {/* Capabilities */}
            <div className="flex flex-wrap gap-1 mb-3">
              {plugin.capabilities.slice(0, 3).map((cap) => {
                const style = CAPABILITY_COLORS[cap] || CAPABILITY_COLORS.custom;
                return (
                  <span
                    key={cap}
                    className={`text-xs font-mono px-2 py-0.5 rounded ${style.bg} ${style.text}`}
                  >
                    {cap.replace(/_/g, ' ')}
                  </span>
                );
              })}
              {plugin.capabilities.length > 3 && (
                <span className="text-xs font-mono text-text-muted">
                  +{plugin.capabilities.length - 3}
                </span>
              )}
            </div>

            {/* Tags */}
            {plugin.tags.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {plugin.tags.slice(0, 3).map((tag) => (
                  <span key={tag} className="text-xs font-mono text-text-muted">
                    #{tag}
                  </span>
                ))}
              </div>
            )}

            {/* Author */}
            <div className="mt-2 text-xs font-mono text-text-muted">
              by {plugin.author}
            </div>
          </button>
        ))}
      </div>

      {filteredPlugins.length === 0 && (
        <div className="card p-8 text-center">
          <p className="text-text-muted font-mono">
            No plugins match your search criteria.
          </p>
        </div>
      )}

      {/* Selected Plugin Details */}
      {selectedPlugin && (
        <div className="card p-6 border-2 border-acid-green/40">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h2 className="text-xl font-mono text-acid-green font-bold">
                {selectedPlugin.name}
              </h2>
              <p className="text-sm font-mono text-text-muted">
                v{selectedPlugin.version} by {selectedPlugin.author}
              </p>
            </div>
            <button
              onClick={() => setSelectedPlugin(null)}
              className="text-text-muted hover:text-text transition-colors"
              aria-label="Close plugin details"
            >
              [X]
            </button>
          </div>

          <p className="font-mono text-sm text-text mb-6">
            {selectedPlugin.description}
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Capabilities */}
            <div>
              <h4 className="font-mono text-xs text-acid-cyan mb-2">CAPABILITIES</h4>
              <div className="space-y-1">
                {selectedPlugin.capabilities.map((cap) => {
                  const style = CAPABILITY_COLORS[cap] || CAPABILITY_COLORS.custom;
                  return (
                    <div
                      key={cap}
                      className={`px-3 py-1.5 rounded ${style.bg} ${style.text} font-mono text-sm`}
                    >
                      {cap.replace(/_/g, ' ')}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Requirements */}
            <div>
              <h4 className="font-mono text-xs text-acid-yellow mb-2">REQUIREMENTS</h4>
              <div className="space-y-1">
                {selectedPlugin.requirements.map((req) => {
                  const info = REQUIREMENT_INFO[req] || { icon: '?', description: req };
                  return (
                    <div
                      key={req}
                      className="px-3 py-1.5 rounded bg-surface font-mono text-sm flex items-center gap-2"
                    >
                      <span>{info.icon}</span>
                      <span className="text-text-muted">{info.description}</span>
                    </div>
                  );
                })}
              </div>

              {selectedPlugin.requirements_satisfied !== undefined && (
                <div className={`mt-2 text-xs font-mono ${
                  selectedPlugin.requirements_satisfied ? 'text-acid-green' : 'text-acid-red'
                }`}>
                  {selectedPlugin.requirements_satisfied
                    ? 'All requirements satisfied'
                    : `Missing: ${selectedPlugin.missing_requirements?.join(', ')}`}
                </div>
              )}
            </div>
          </div>

          {/* Technical Details */}
          <div className="mt-6 p-4 bg-surface rounded">
            <h4 className="font-mono text-xs text-acid-cyan mb-3">TECHNICAL DETAILS</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono text-xs">
              <div>
                <span className="text-text-muted">Entry Point:</span>
                <div className="text-text">{selectedPlugin.entry_point}</div>
              </div>
              <div>
                <span className="text-text-muted">Timeout:</span>
                <div className="text-text">{selectedPlugin.timeout_seconds}s</div>
              </div>
              <div>
                <span className="text-text-muted">Max Memory:</span>
                <div className="text-text">{selectedPlugin.max_memory_mb}MB</div>
              </div>
              <div>
                <span className="text-text-muted">License:</span>
                <div className="text-text">{selectedPlugin.license}</div>
              </div>
            </div>

            {selectedPlugin.python_packages.length > 0 && (
              <div className="mt-3">
                <span className="text-text-muted text-xs">Python Packages: </span>
                <span className="text-text text-xs">
                  {selectedPlugin.python_packages.join(', ')}
                </span>
              </div>
            )}

            {selectedPlugin.system_tools.length > 0 && (
              <div className="mt-1">
                <span className="text-text-muted text-xs">System Tools: </span>
                <span className="text-text text-xs">
                  {selectedPlugin.system_tools.join(', ')}
                </span>
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="mt-6 flex gap-4">
            {selectedPlugin.homepage && (
              <a
                href={selectedPlugin.homepage}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-surface border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/10 transition-colors"
              >
                View Documentation
              </a>
            )}
            <button
              className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors"
              onClick={() => {
                alert('Plugin execution requires authentication. Run via CLI: aragora plugins run ' + selectedPlugin.name);
              }}
            >
              Run Plugin
            </button>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-4">
        <button
          onClick={fetchPlugins}
          disabled={loading}
          className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50"
        >
          {loading ? 'Refreshing...' : 'Refresh Plugins'}
        </button>
      </div>
    </div>
  );
}
