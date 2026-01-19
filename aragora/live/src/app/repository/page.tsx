'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { ErrorWithRetry } from '@/components/ErrorWithRetry';

interface IndexProgress {
  repository_id: string;
  status: 'pending' | 'indexing' | 'completed' | 'failed' | 'unknown';
  files_discovered: number;
  files_processed: number;
  nodes_created: number;
  current_file?: string;
  started_at?: string;
  error?: string;
}

interface Entity {
  id: string;
  content?: string;
  metadata: {
    kind?: string;
    name?: string;
    file_path?: string;
    line_start?: number;
    [key: string]: unknown;
  };
}

interface GraphStats {
  total_nodes: number;
  total_edges: number;
  node_types: Record<string, number>;
  edge_types: Record<string, number>;
}

const statusColors: Record<string, string> = {
  pending: 'text-acid-yellow',
  indexing: 'text-acid-cyan animate-pulse',
  completed: 'text-acid-green',
  failed: 'text-crimson',
  unknown: 'text-text-muted',
};

export default function RepositoryPage() {
  const { config: backendConfig } = useBackend();
  const [activeTab, setActiveTab] = useState<'index' | 'browse' | 'graph'>('index');
  const [error, setError] = useState<string | null>(null);

  // Index state
  const [repoPath, setRepoPath] = useState('');
  const [workspaceId, setWorkspaceId] = useState('default');
  const [includePatterns, setIncludePatterns] = useState('*,**/*');
  const [excludePatterns, setExcludePatterns] = useState('node_modules/**,*.pyc,__pycache__/**,.git/**');
  const [maxFileSize, setMaxFileSize] = useState(1000000);
  const [maxFiles, setMaxFiles] = useState(10000);
  const [extractSymbols, setExtractSymbols] = useState(true);
  const [extractDeps, setExtractDeps] = useState(true);
  const [indexing, setIndexing] = useState(false);
  const [progress, setProgress] = useState<IndexProgress | null>(null);

  // Browse state
  const [browseRepoId, setBrowseRepoId] = useState('');
  const [entityKind, setEntityKind] = useState('');
  const [filePath, setFilePath] = useState('');
  const [entities, setEntities] = useState<Entity[]>([]);
  const [browsing, setBrowsing] = useState(false);
  const [totalEntities, setTotalEntities] = useState(0);

  // Graph state
  const [graphRepoId, setGraphRepoId] = useState('');
  const [entityId, setEntityId] = useState('');
  const [graphDepth, setGraphDepth] = useState(2);
  const [graphDirection, setGraphDirection] = useState<'both' | 'dependencies' | 'dependents'>('both');
  const [graphStats, setGraphStats] = useState<GraphStats | null>(null);
  const [graphEntities, setGraphEntities] = useState<Entity[]>([]);
  const [loadingGraph, setLoadingGraph] = useState(false);

  const handleStartIndex = async () => {
    if (!repoPath.trim()) return;
    setIndexing(true);
    setProgress(null);
    setError(null);

    try {
      const res = await fetch(`${backendConfig.api}/api/repository/index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repo_path: repoPath,
          workspace_id: workspaceId,
          crawl_config: {
            include_patterns: includePatterns.split(',').map(p => p.trim()).filter(Boolean),
            exclude_patterns: excludePatterns.split(',').map(p => p.trim()).filter(Boolean),
            max_file_size_bytes: maxFileSize,
            max_files: maxFiles,
            extract_symbols: extractSymbols,
            extract_dependencies: extractDeps,
          },
        }),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.result) {
          setProgress({
            repository_id: repoPath,
            status: data.success ? 'completed' : 'failed',
            files_discovered: data.result.files_discovered || 0,
            files_processed: data.result.files_processed || 0,
            nodes_created: data.result.nodes_created || 0,
            error: data.result.errors?.join(', '),
          });
        }
      } else {
        const errData = await res.json().catch(() => ({}));
        setError(errData.error || 'Indexing failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Indexing failed');
    } finally {
      setIndexing(false);
    }
  };

  const handleBrowse = async () => {
    if (!browseRepoId.trim()) return;
    setBrowsing(true);
    setEntities([]);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (entityKind) params.set('kind', entityKind);
      if (filePath) params.set('file_path', filePath);
      params.set('limit', '50');

      const res = await fetch(
        `${backendConfig.api}/api/repository/${encodeURIComponent(browseRepoId)}/entities?${params}`
      );

      if (res.ok) {
        const data = await res.json();
        setEntities(data.entities || []);
        setTotalEntities(data.total || 0);
      } else {
        const errData = await res.json().catch(() => ({}));
        setError(errData.error || 'Failed to browse entities');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to browse entities');
    } finally {
      setBrowsing(false);
    }
  };

  const handleLoadGraph = async () => {
    if (!graphRepoId.trim()) return;
    setLoadingGraph(true);
    setGraphStats(null);
    setGraphEntities([]);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (entityId) params.set('entity_id', entityId);
      params.set('depth', graphDepth.toString());
      params.set('direction', graphDirection);

      const res = await fetch(
        `${backendConfig.api}/api/repository/${encodeURIComponent(graphRepoId)}/graph?${params}`
      );

      if (res.ok) {
        const data = await res.json();
        if (data.statistics) {
          setGraphStats(data.statistics);
        }
        if (data.entities) {
          setGraphEntities(data.entities);
        }
      } else {
        const errData = await res.json().catch(() => ({}));
        setError(errData.error || 'Failed to load graph');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load graph');
    } finally {
      setLoadingGraph(false);
    }
  };

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
              <BackendSelector />
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-8">
          {/* Title */}
          <div className="mb-8">
            <h1 className="text-2xl font-mono font-bold text-acid-green mb-2">
              [REPOSITORY_INDEXER]
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Index and explore codebase structure for knowledge retrieval
            </p>
          </div>

          {error && (
            <ErrorWithRetry error={error} onRetry={() => setError(null)} className="mb-6" />
          )}

          {/* Tabs */}
          <div className="flex gap-2 mb-6 border-b border-border pb-2">
            {(['index', 'browse', 'graph'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 font-mono text-sm transition-colors ${
                  activeTab === tab
                    ? 'text-acid-green border-b-2 border-acid-green'
                    : 'text-text-muted hover:text-text'
                }`}
              >
                {tab.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Index Tab */}
          {activeTab === 'index' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="card p-4">
                <h3 className="text-lg font-mono text-acid-green mb-4">[INDEX REPOSITORY]</h3>

                <div className="space-y-4">
                  <div>
                    <label className="block text-xs font-mono text-text-muted mb-1">Repository Path *</label>
                    <input
                      type="text"
                      value={repoPath}
                      onChange={e => setRepoPath(e.target.value)}
                      placeholder="/path/to/repository"
                      className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-mono text-text-muted mb-1">Workspace ID</label>
                    <input
                      type="text"
                      value={workspaceId}
                      onChange={e => setWorkspaceId(e.target.value)}
                      placeholder="default"
                      className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-mono text-text-muted mb-1">Include Patterns</label>
                      <input
                        type="text"
                        value={includePatterns}
                        onChange={e => setIncludePatterns(e.target.value)}
                        placeholder="*,**/*"
                        className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-mono text-text-muted mb-1">Exclude Patterns</label>
                      <input
                        type="text"
                        value={excludePatterns}
                        onChange={e => setExcludePatterns(e.target.value)}
                        placeholder="node_modules/**,.git/**"
                        className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-mono text-text-muted mb-1">Max File Size (bytes)</label>
                      <input
                        type="number"
                        value={maxFileSize}
                        onChange={e => setMaxFileSize(parseInt(e.target.value) || 1000000)}
                        className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-mono text-text-muted mb-1">Max Files</label>
                      <input
                        type="number"
                        value={maxFiles}
                        onChange={e => setMaxFiles(parseInt(e.target.value) || 10000)}
                        className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                      />
                    </div>
                  </div>

                  <div className="flex gap-4">
                    <label className="flex items-center gap-2 text-sm font-mono">
                      <input
                        type="checkbox"
                        checked={extractSymbols}
                        onChange={e => setExtractSymbols(e.target.checked)}
                        className="rounded border-border"
                      />
                      <span className="text-text-muted">Extract Symbols</span>
                    </label>
                    <label className="flex items-center gap-2 text-sm font-mono">
                      <input
                        type="checkbox"
                        checked={extractDeps}
                        onChange={e => setExtractDeps(e.target.checked)}
                        className="rounded border-border"
                      />
                      <span className="text-text-muted">Extract Dependencies</span>
                    </label>
                  </div>

                  <button
                    onClick={handleStartIndex}
                    disabled={indexing || !repoPath.trim()}
                    className="w-full py-3 font-mono text-sm bg-acid-green/20 border border-acid-green text-acid-green hover:bg-acid-green/30 transition-colors disabled:opacity-50"
                  >
                    {indexing ? '[INDEXING...]' : '[START INDEX]'}
                  </button>
                </div>
              </div>

              <div className="card p-4">
                <h3 className="text-lg font-mono text-acid-green mb-4">[PROGRESS]</h3>

                {progress ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-sm text-text-muted">Status</span>
                      <span className={`font-mono text-sm ${statusColors[progress.status]}`}>
                        {progress.status.toUpperCase()}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                      <div className="p-3 bg-bg rounded border border-border">
                        <div className="text-xs font-mono text-text-muted mb-1">Files Discovered</div>
                        <div className="text-xl font-mono text-acid-green">{progress.files_discovered}</div>
                      </div>
                      <div className="p-3 bg-bg rounded border border-border">
                        <div className="text-xs font-mono text-text-muted mb-1">Files Processed</div>
                        <div className="text-xl font-mono text-acid-cyan">{progress.files_processed}</div>
                      </div>
                      <div className="p-3 bg-bg rounded border border-border">
                        <div className="text-xs font-mono text-text-muted mb-1">Nodes Created</div>
                        <div className="text-xl font-mono text-accent">{progress.nodes_created}</div>
                      </div>
                    </div>

                    {progress.current_file && (
                      <div>
                        <div className="text-xs font-mono text-text-muted mb-1">Current File</div>
                        <div className="text-sm font-mono text-text truncate">{progress.current_file}</div>
                      </div>
                    )}

                    {progress.error && (
                      <div className="p-3 bg-crimson/10 border border-crimson/30 rounded">
                        <div className="text-xs font-mono text-crimson">{progress.error}</div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-12 text-text-muted font-mono text-sm">
                    Enter a repository path and click Start Index
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Browse Tab */}
          {activeTab === 'browse' && (
            <div className="space-y-6">
              <div className="card p-4">
                <h3 className="text-lg font-mono text-acid-green mb-4">[BROWSE ENTITIES]</h3>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                  <div>
                    <label className="block text-xs font-mono text-text-muted mb-1">Repository ID *</label>
                    <input
                      type="text"
                      value={browseRepoId}
                      onChange={e => setBrowseRepoId(e.target.value)}
                      placeholder="repository-name"
                      className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-mono text-text-muted mb-1">Kind (optional)</label>
                    <select
                      value={entityKind}
                      onChange={e => setEntityKind(e.target.value)}
                      className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                    >
                      <option value="">All</option>
                      <option value="file">File</option>
                      <option value="class">Class</option>
                      <option value="function">Function</option>
                      <option value="method">Method</option>
                      <option value="variable">Variable</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-mono text-text-muted mb-1">File Path (optional)</label>
                    <input
                      type="text"
                      value={filePath}
                      onChange={e => setFilePath(e.target.value)}
                      placeholder="src/..."
                      className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                    />
                  </div>
                  <div className="flex items-end">
                    <button
                      onClick={handleBrowse}
                      disabled={browsing || !browseRepoId.trim()}
                      className="w-full py-2 font-mono text-sm bg-acid-green/20 border border-acid-green text-acid-green hover:bg-acid-green/30 transition-colors disabled:opacity-50"
                    >
                      {browsing ? '[LOADING...]' : '[BROWSE]'}
                    </button>
                  </div>
                </div>
              </div>

              {entities.length > 0 && (
                <div className="card p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-mono text-acid-green">[ENTITIES]</h3>
                    <span className="text-xs font-mono text-text-muted">
                      Showing {entities.length} of {totalEntities}
                    </span>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full font-mono text-sm">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="text-left py-2 px-2 text-text-muted">ID</th>
                          <th className="text-left py-2 px-2 text-text-muted">Kind</th>
                          <th className="text-left py-2 px-2 text-text-muted">Name</th>
                          <th className="text-left py-2 px-2 text-text-muted">File</th>
                          <th className="text-left py-2 px-2 text-text-muted">Line</th>
                        </tr>
                      </thead>
                      <tbody>
                        {entities.map(entity => (
                          <tr key={entity.id} className="border-b border-border/50 hover:bg-surface/50">
                            <td className="py-2 px-2 text-acid-cyan truncate max-w-[120px]" title={entity.id}>
                              {entity.id.slice(0, 12)}...
                            </td>
                            <td className="py-2 px-2 text-acid-green">
                              {entity.metadata.kind || '-'}
                            </td>
                            <td className="py-2 px-2">{entity.metadata.name || '-'}</td>
                            <td className="py-2 px-2 text-text-muted truncate max-w-[200px]">
                              {entity.metadata.file_path || '-'}
                            </td>
                            <td className="py-2 px-2 text-text-muted">
                              {entity.metadata.line_start || '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {entities.length === 0 && !browsing && browseRepoId && (
                <div className="card p-8 text-center">
                  <div className="text-text-muted font-mono">
                    No entities found. The repository may not be indexed.
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Graph Tab */}
          {activeTab === 'graph' && (
            <div className="space-y-6">
              <div className="card p-4">
                <h3 className="text-lg font-mono text-acid-green mb-4">[RELATIONSHIP GRAPH]</h3>

                <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-4">
                  <div>
                    <label className="block text-xs font-mono text-text-muted mb-1">Repository ID *</label>
                    <input
                      type="text"
                      value={graphRepoId}
                      onChange={e => setGraphRepoId(e.target.value)}
                      placeholder="repository-name"
                      className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-mono text-text-muted mb-1">Entity ID (optional)</label>
                    <input
                      type="text"
                      value={entityId}
                      onChange={e => setEntityId(e.target.value)}
                      placeholder="For specific entity"
                      className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-mono text-text-muted mb-1">Depth</label>
                    <input
                      type="number"
                      value={graphDepth}
                      onChange={e => setGraphDepth(parseInt(e.target.value) || 2)}
                      min={1}
                      max={5}
                      className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-mono text-text-muted mb-1">Direction</label>
                    <select
                      value={graphDirection}
                      onChange={e => setGraphDirection(e.target.value as typeof graphDirection)}
                      className="w-full p-2 bg-bg border border-border rounded font-mono text-sm focus:border-acid-green focus:outline-none"
                    >
                      <option value="both">Both</option>
                      <option value="dependencies">Dependencies</option>
                      <option value="dependents">Dependents</option>
                    </select>
                  </div>
                  <div className="flex items-end">
                    <button
                      onClick={handleLoadGraph}
                      disabled={loadingGraph || !graphRepoId.trim()}
                      className="w-full py-2 font-mono text-sm bg-acid-green/20 border border-acid-green text-acid-green hover:bg-acid-green/30 transition-colors disabled:opacity-50"
                    >
                      {loadingGraph ? '[LOADING...]' : '[LOAD GRAPH]'}
                    </button>
                  </div>
                </div>
              </div>

              {graphStats && (
                <div className="card p-4">
                  <h3 className="text-lg font-mono text-acid-green mb-4">[GRAPH STATISTICS]</h3>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="p-3 bg-bg rounded border border-border">
                      <div className="text-xs font-mono text-text-muted mb-1">Total Nodes</div>
                      <div className="text-xl font-mono text-acid-green">{graphStats.total_nodes}</div>
                    </div>
                    <div className="p-3 bg-bg rounded border border-border">
                      <div className="text-xs font-mono text-text-muted mb-1">Total Edges</div>
                      <div className="text-xl font-mono text-acid-cyan">{graphStats.total_edges}</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="text-xs font-mono text-text-muted mb-2">Node Types</div>
                      <div className="space-y-1">
                        {Object.entries(graphStats.node_types || {}).map(([type, count]) => (
                          <div key={type} className="flex items-center justify-between text-sm font-mono">
                            <span className="text-text-muted">{type}</span>
                            <span className="text-acid-green">{count}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-mono text-text-muted mb-2">Edge Types</div>
                      <div className="space-y-1">
                        {Object.entries(graphStats.edge_types || {}).map(([type, count]) => (
                          <div key={type} className="flex items-center justify-between text-sm font-mono">
                            <span className="text-text-muted">{type}</span>
                            <span className="text-acid-cyan">{count}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {graphEntities.length > 0 && (
                <div className="card p-4">
                  <h3 className="text-lg font-mono text-acid-green mb-4">[RELATED ENTITIES]</h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {graphEntities.map((entity: any) => (
                      <div key={entity.id} className="p-3 bg-bg rounded border border-border">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-mono text-sm text-acid-green">{entity.name}</span>
                          <span className="text-xs font-mono text-text-muted px-2 py-0.5 bg-surface rounded">
                            {entity.kind}
                          </span>
                        </div>
                        <div className="text-xs font-mono text-text-muted truncate">
                          {entity.file_path}
                          {entity.line_start && `:${entity.line_start}`}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </>
  );
}
