'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { AgentWorkflowVisualization } from '@/components/AgentWorkflowVisualization';
import { useControlPlaneWebSocket } from '@/hooks/useControlPlaneWebSocket';

// Control Plane Components
import {
  AgentCatalog,
  WorkflowBuilder,
  KnowledgeExplorer,
  ExecutionMonitor,
  PolicyDashboard,
  WorkspaceManager,
  ConnectorDashboard,
} from '@/components/control-plane';

// Verticals Components
import {
  VerticalSelector,
  KnowledgeExplorer as VerticalKnowledgeExplorer,
  ExecutionMonitor as VerticalExecutionMonitor,
} from '@/components/verticals';

interface Agent {
  id: string;
  name: string;
  model: string;
  status: 'idle' | 'working' | 'error' | 'rate_limited';
  current_task?: string;
  requests_today: number;
  tokens_used: number;
  last_active?: string;
}

interface ProcessingJob {
  id: string;
  type: 'document_processing' | 'audit' | 'debate' | 'batch_upload';
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  started_at?: string;
  document_count?: number;
  agents_assigned: string[];
}

interface SystemMetrics {
  active_jobs: number;
  queued_jobs: number;
  agents_available: number;
  agents_busy: number;
  documents_processed_today: number;
  audits_completed_today: number;
  tokens_used_today: number;
}

type TabId = 'overview' | 'agents' | 'workflows' | 'knowledge' | 'connectors' | 'executions' | 'queue' | 'verticals' | 'policy' | 'workspace' | 'settings';

export default function ControlPlanePage() {
  const { config: backendConfig } = useBackend();
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [agents, setAgents] = useState<Agent[]>([]);
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [usingMockData, setUsingMockData] = useState(false);
  const [useWebSocket, setUseWebSocket] = useState(true);

  // WebSocket connection for real-time updates
  const {
    isConnected: wsConnected,
    agents: wsAgents,
    jobs: wsJobs,
    metrics: wsMetrics,
    recentFindings,
    reconnect: wsReconnect,
  } = useControlPlaneWebSocket({
    enabled: useWebSocket,
    autoReconnect: true,
    onFindingDetected: (_finding) => {
      // Could show a toast notification here
    },
  });

  // Use WebSocket data when connected, otherwise use REST data
  const displayAgents = wsConnected && wsAgents.length > 0 ? wsAgents.map(a => ({
    id: a.id,
    name: a.name,
    model: a.model,
    status: a.status,
    current_task: a.current_task,
    requests_today: a.requests_today,
    tokens_used: a.tokens_used,
    last_active: a.last_active,
  })) : agents;

  const displayJobs = wsConnected && wsJobs.length > 0 ? wsJobs.map(j => ({
    id: j.id,
    type: j.type,
    name: j.name,
    status: j.status,
    progress: j.progress,
    started_at: j.started_at,
    document_count: j.document_count,
    agents_assigned: j.agents_assigned,
  })) : jobs;

  const displayMetrics = wsConnected && wsMetrics ? {
    active_jobs: wsMetrics.active_jobs,
    queued_jobs: wsMetrics.queued_jobs,
    agents_available: wsMetrics.agents_available,
    agents_busy: wsMetrics.agents_busy,
    documents_processed_today: wsMetrics.documents_processed_today,
    audits_completed_today: wsMetrics.audits_completed_today,
    tokens_used_today: wsMetrics.tokens_used_today,
  } : metrics;

  // Fetch agents
  const fetchAgents = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.api}/api/agents`);
      if (!response.ok) throw new Error('Failed to fetch agents');
      const data = await response.json();
      setAgents(data.agents || []);
      return true; // Success
    } catch {
      // Use mock data if endpoint not available
      setAgents([
        { id: 'claude', name: 'Claude', model: 'claude-3.5-sonnet', status: 'idle', requests_today: 45, tokens_used: 125000 },
        { id: 'gemini', name: 'Gemini', model: 'gemini-3-pro', status: 'working', current_task: 'Document audit scan', requests_today: 32, tokens_used: 890000 },
        { id: 'gpt4', name: 'GPT-4', model: 'gpt-4-turbo', status: 'idle', requests_today: 28, tokens_used: 78000 },
        { id: 'codex', name: 'Codex', model: 'claude-3.5-sonnet', status: 'idle', requests_today: 15, tokens_used: 45000 },
      ]);
      return false; // Used mock
    }
  }, [backendConfig.api]);

  // Fetch jobs
  const fetchJobs = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.api}/api/control-plane/queue`);
      if (!response.ok) throw new Error('Failed to fetch jobs');
      const data = await response.json();
      setJobs(data.jobs || []);
      return true; // Success
    } catch {
      // Use mock data if endpoint not available
      setJobs([
        { id: 'job1', type: 'audit', name: 'Security Audit - Q1 Contracts', status: 'running', progress: 0.45, started_at: new Date().toISOString(), document_count: 12, agents_assigned: ['gemini', 'claude'] },
        { id: 'job2', type: 'document_processing', name: 'Batch Import - Legal Docs', status: 'queued', progress: 0, document_count: 48, agents_assigned: [] },
        { id: 'job3', type: 'audit', name: 'Compliance Check - HR Policies', status: 'completed', progress: 1, document_count: 5, agents_assigned: ['gemini'] },
      ]);
      return false; // Used mock
    }
  }, [backendConfig.api]);

  // Fetch metrics
  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.api}/api/control-plane/metrics`);
      if (!response.ok) throw new Error('Failed to fetch metrics');
      const data = await response.json();
      setMetrics(data);
      return true; // Success
    } catch {
      // Use mock data
      setMetrics({
        active_jobs: 1,
        queued_jobs: 2,
        agents_available: 3,
        agents_busy: 1,
        documents_processed_today: 67,
        audits_completed_today: 4,
        tokens_used_today: 1138000,
      });
      return false; // Used mock
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api]);

  // Track mock data usage
  const fetchAllData = useCallback(async () => {
    const [agentsOk, jobsOk, metricsOk] = await Promise.all([
      fetchAgents(),
      fetchJobs(),
      fetchMetrics(),
    ]);
    setUsingMockData(!agentsOk || !jobsOk || !metricsOk);
  }, [fetchAgents, fetchJobs, fetchMetrics]);

  useEffect(() => {
    fetchAllData();
  }, [fetchAllData]);

  // Auto-refresh (fallback to polling when WebSocket is not connected)
  useEffect(() => {
    // Skip polling if WebSocket is connected and providing data
    if (wsConnected && (wsAgents.length > 0 || wsJobs.length > 0)) return;
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchAllData();
    }, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchAllData, wsConnected, wsAgents.length, wsJobs.length]);

  const pauseJob = async (jobId: string) => {
    try {
      await fetch(`${backendConfig.api}/api/control-plane/jobs/${jobId}/pause`, { method: 'POST' });
      fetchJobs();
    } catch {
      // Handle error
    }
  };

  const resumeJob = async (jobId: string) => {
    try {
      await fetch(`${backendConfig.api}/api/control-plane/jobs/${jobId}/resume`, { method: 'POST' });
      fetchJobs();
    } catch {
      // Handle error
    }
  };

  const cancelJob = async (jobId: string) => {
    try {
      await fetch(`${backendConfig.api}/api/control-plane/jobs/${jobId}/cancel`, { method: 'POST' });
      fetchJobs();
    } catch {
      // Handle error
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'idle':
      case 'completed':
        return 'text-success';
      case 'working':
      case 'running':
        return 'text-acid-cyan';
      case 'queued':
        return 'text-acid-yellow';
      case 'error':
      case 'failed':
      case 'rate_limited':
        return 'text-crimson';
      case 'paused':
        return 'text-text-muted';
      default:
        return 'text-text-muted';
    }
  };

  const formatTokens = (tokens: number) => {
    if (tokens < 1000) return tokens.toString();
    if (tokens < 1000000) return `${(tokens / 1000).toFixed(1)}K`;
    return `${(tokens / 1000000).toFixed(2)}M`;
  };

  const tabs = [
    { id: 'overview' as TabId, label: 'OVERVIEW' },
    { id: 'agents' as TabId, label: 'AGENTS', count: displayAgents.length },
    { id: 'workflows' as TabId, label: 'WORKFLOWS' },
    { id: 'knowledge' as TabId, label: 'KNOWLEDGE' },
    { id: 'connectors' as TabId, label: 'CONNECTORS' },
    { id: 'executions' as TabId, label: 'EXECUTIONS' },
    { id: 'queue' as TabId, label: 'QUEUE', count: displayJobs.filter(j => j.status === 'running' || j.status === 'queued').length },
    { id: 'verticals' as TabId, label: 'VERTICALS' },
    { id: 'policy' as TabId, label: 'POLICY' },
    { id: 'workspace' as TabId, label: 'WORKSPACE' },
    { id: 'settings' as TabId, label: 'SETTINGS' },
  ];

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
              {usingMockData && (
                <div className="flex items-center gap-1.5 px-2 py-1 bg-yellow-900/20 border border-yellow-600/30 rounded text-xs">
                  <span className="w-1.5 h-1.5 rounded-full bg-yellow-400" />
                  <span className="font-mono text-yellow-400">DEMO MODE</span>
                </div>
              )}
              <div className="flex items-center gap-2">
                {wsConnected ? (
                  <span className="w-2 h-2 rounded-full bg-acid-green animate-pulse" title="WebSocket connected" />
                ) : (
                  <span className={`w-2 h-2 rounded-full ${autoRefresh ? 'bg-success animate-pulse' : 'bg-text-muted'}`} />
                )}
                <button
                  onClick={() => {
                    if (wsConnected) {
                      setUseWebSocket(!useWebSocket);
                    } else {
                      setAutoRefresh(!autoRefresh);
                    }
                  }}
                  className="text-xs font-mono text-text-muted hover:text-text transition-colors"
                >
                  {wsConnected ? 'WS LIVE' : autoRefresh ? 'POLLING' : 'PAUSED'}
                </button>
              </div>
              <Link
                href="/admin"
                className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
              >
                [ADMIN]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Sub Navigation */}
        <div className="border-b border-acid-green/20 bg-surface/40">
          <div className="container mx-auto px-4">
            <div className="flex gap-4 overflow-x-auto">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-4 py-2 font-mono text-sm transition-colors flex items-center gap-2 ${
                    activeTab === tab.id
                      ? 'text-acid-green border-b-2 border-acid-green'
                      : 'text-text-muted hover:text-text'
                  }`}
                >
                  {tab.label}
                  {tab.count !== undefined && (
                    <span className="px-1.5 py-0.5 bg-surface rounded text-xs">
                      {tab.count}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <PanelErrorBoundary panelName="ControlPlane">
            {/* Page Header */}
            <div className="mb-6">
              <h1 className="text-2xl font-mono text-acid-green mb-2">
                Control Plane
              </h1>
              <p className="text-text-muted font-mono text-sm">
                Monitor and orchestrate multi-agent document processing and auditing.
              </p>
            </div>

            {loading ? (
              <div className="card p-8 text-center">
                <div className="animate-pulse font-mono text-text-muted">Loading control plane...</div>
              </div>
            ) : (
              <>
                {/* Overview Tab */}
                {activeTab === 'overview' && displayMetrics && (
                  <div className="space-y-6">
                    {/* Metrics Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="card p-4">
                        <div className="text-xs font-mono text-text-muted mb-1">ACTIVE JOBS</div>
                        <div className="text-2xl font-mono text-acid-cyan">{displayMetrics.active_jobs}</div>
                      </div>
                      <div className="card p-4">
                        <div className="text-xs font-mono text-text-muted mb-1">QUEUED</div>
                        <div className="text-2xl font-mono text-acid-yellow">{displayMetrics.queued_jobs}</div>
                      </div>
                      <div className="card p-4">
                        <div className="text-xs font-mono text-text-muted mb-1">AGENTS AVAILABLE</div>
                        <div className="text-2xl font-mono text-success">{displayMetrics.agents_available}/{displayMetrics.agents_available + displayMetrics.agents_busy}</div>
                      </div>
                      <div className="card p-4">
                        <div className="text-xs font-mono text-text-muted mb-1">TOKENS TODAY</div>
                        <div className="text-2xl font-mono">{formatTokens(displayMetrics.tokens_used_today)}</div>
                      </div>
                    </div>

                    {/* Agent Workflow Visualization */}
                    <div className="card p-4">
                      <AgentWorkflowVisualization
                        agents={displayAgents.map(a => ({
                          ...a,
                          requests_today: a.requests_today,
                          tokens_used: a.tokens_used,
                        }))}
                        jobs={displayJobs.map(j => ({
                          ...j,
                          agents_assigned: j.agents_assigned,
                        }))}
                        width={850}
                        height={350}
                        onAgentClick={(_agent) => {
                          // Could navigate to agent detail or show modal
                        }}
                      />
                    </div>

                    {/* Active Jobs */}
                    <div className="card">
                      <div className="p-4 border-b border-border">
                        <h2 className="font-mono text-sm text-acid-green">Active Jobs</h2>
                      </div>
                      <div className="p-4 space-y-3">
                        {displayJobs.filter(j => j.status === 'running').length === 0 ? (
                          <div className="text-center text-text-muted font-mono text-sm py-4">
                            No active jobs
                          </div>
                        ) : (
                          displayJobs.filter(j => j.status === 'running').map(job => (
                            <div key={job.id} className="bg-surface p-3 rounded border border-border">
                              <div className="flex items-center justify-between mb-2">
                                <span className="font-mono text-sm">{job.name}</span>
                                <span className={`text-xs font-mono uppercase ${getStatusColor(job.status)}`}>
                                  {job.status}
                                </span>
                              </div>
                              <div className="h-1.5 bg-bg rounded overflow-hidden mb-2">
                                <div
                                  className="h-full bg-acid-cyan transition-all"
                                  style={{ width: `${job.progress * 100}%` }}
                                />
                              </div>
                              <div className="flex items-center justify-between text-xs font-mono text-text-muted">
                                <span>{Math.round(job.progress * 100)}% - {job.document_count} documents</span>
                                <span>Agents: {job.agents_assigned.join(', ')}</span>
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    </div>

                    {/* Agent Status */}
                    <div className="card">
                      <div className="p-4 border-b border-border">
                        <h2 className="font-mono text-sm text-acid-green">Agent Status</h2>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4">
                        {displayAgents.map(agent => (
                          <div key={agent.id} className="bg-surface p-3 rounded border border-border">
                            <div className="flex items-center gap-2 mb-2">
                              <span className={`w-2 h-2 rounded-full ${
                                agent.status === 'working' ? 'bg-acid-cyan animate-pulse' :
                                agent.status === 'idle' ? 'bg-success' :
                                'bg-crimson'
                              }`} />
                              <span className="font-mono text-sm">{agent.name}</span>
                            </div>
                            <div className="text-xs font-mono text-text-muted">
                              {agent.current_task || agent.model}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Agents Tab - Enhanced with AgentCatalog */}
                {activeTab === 'agents' && (
                  <AgentCatalog
                    onSelectAgent={(_agent) => {
                      // Agent selection handler
                    }}
                    onConfigureAgent={(_agent) => {
                      // Agent configuration handler
                    }}
                  />
                )}

                {/* Workflows Tab */}
                {activeTab === 'workflows' && (
                  <div className="h-[calc(100vh-280px)]">
                    <WorkflowBuilder
                      onSave={() => {
                        // Workflow saved
                      }}
                      onExecute={(_executionId) => {
                        setActiveTab('executions');
                      }}
                    />
                  </div>
                )}

                {/* Knowledge Tab */}
                {activeTab === 'knowledge' && (
                  <KnowledgeExplorer
                    onSelectNode={(_node) => {
                      // Knowledge node selection handler
                    }}
                  />
                )}

                {/* Connectors Tab */}
                {activeTab === 'connectors' && (
                  <ConnectorDashboard
                    onSelectConnector={(_connector) => {
                      // Connector selection handler
                    }}
                  />
                )}

                {/* Executions Tab */}
                {activeTab === 'executions' && (
                  <ExecutionMonitor
                    onSelectExecution={(_execution) => {
                      // Execution selection handler
                    }}
                  />
                )}

                {/* Queue Tab */}
                {activeTab === 'queue' && (
                  <div className="space-y-4">
                    {/* Real-time indicator */}
                    {wsConnected && (
                      <div className="flex items-center gap-2 text-xs font-mono text-acid-green mb-2">
                        <span className="w-2 h-2 rounded-full bg-acid-green animate-pulse" />
                        Real-time updates via WebSocket
                        {recentFindings.length > 0 && (
                          <span className="ml-2 px-2 py-0.5 bg-acid-green/20 rounded">
                            {recentFindings.length} recent findings
                          </span>
                        )}
                      </div>
                    )}
                    {displayJobs.length === 0 ? (
                      <div className="card p-8 text-center">
                        <div className="font-mono text-text-muted">No jobs in queue</div>
                      </div>
                    ) : (
                      displayJobs.map(job => (
                        <div key={job.id} className="card p-4">
                          <div className="flex items-start justify-between mb-3">
                            <div>
                              <div className="font-mono font-medium">{job.name}</div>
                              <div className="text-xs text-text-muted font-mono mt-1">
                                {job.type.replace('_', ' ').toUpperCase()} | {job.document_count} documents
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className={`text-xs font-mono uppercase ${getStatusColor(job.status)}`}>
                                {job.status}
                              </span>
                            </div>
                          </div>

                          {(job.status === 'running' || job.status === 'paused') && (
                            <div className="mb-3">
                              <div className="h-1.5 bg-surface rounded overflow-hidden">
                                <div
                                  className={`h-full transition-all ${job.status === 'paused' ? 'bg-acid-yellow' : 'bg-acid-cyan'}`}
                                  style={{ width: `${job.progress * 100}%` }}
                                />
                              </div>
                              <div className="text-xs text-text-muted font-mono mt-1 text-right">
                                {Math.round(job.progress * 100)}%
                              </div>
                            </div>
                          )}

                          <div className="flex items-center justify-between">
                            <div className="text-xs font-mono text-text-muted">
                              {job.agents_assigned.length > 0 && (
                                <span>Agents: {job.agents_assigned.join(', ')}</span>
                              )}
                              {job.started_at && (
                                <span className="ml-4">Started: {new Date(job.started_at).toLocaleString()}</span>
                              )}
                            </div>

                            {job.status !== 'completed' && job.status !== 'failed' && (
                              <div className="flex gap-2">
                                {job.status === 'running' && (
                                  <button
                                    onClick={() => pauseJob(job.id)}
                                    className="px-2 py-1 text-xs font-mono border border-border rounded hover:border-acid-yellow transition-colors"
                                  >
                                    Pause
                                  </button>
                                )}
                                {job.status === 'paused' && (
                                  <button
                                    onClick={() => resumeJob(job.id)}
                                    className="px-2 py-1 text-xs font-mono border border-border rounded hover:border-acid-green transition-colors"
                                  >
                                    Resume
                                  </button>
                                )}
                                <button
                                  onClick={() => cancelJob(job.id)}
                                  className="px-2 py-1 text-xs font-mono border border-crimson/30 text-crimson rounded hover:bg-crimson/10 transition-colors"
                                >
                                  Cancel
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}

                {/* Verticals Tab */}
                {activeTab === 'verticals' && (
                  <div className="space-y-6">
                    <VerticalSelector
                      onSelect={(_vertical) => {
                        // Vertical selection handler
                      }}
                    />
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      <div className="h-[400px]">
                        <VerticalKnowledgeExplorer />
                      </div>
                      <div className="h-[400px]">
                        <VerticalExecutionMonitor />
                      </div>
                    </div>
                  </div>
                )}

                {/* Policy Tab */}
                {activeTab === 'policy' && (
                  <PolicyDashboard />
                )}

                {/* Workspace Tab */}
                {activeTab === 'workspace' && (
                  <WorkspaceManager
                    onWorkspaceSelect={(_workspace) => {
                      // Workspace selection handler
                    }}
                    onWorkspaceUpdate={(_workspace) => {
                      // Workspace update handler
                    }}
                  />
                )}

                {/* Settings Tab */}
                {activeTab === 'settings' && (
                  <div className="max-w-2xl space-y-6">
                    <div className="card p-4">
                      <h3 className="font-mono text-sm text-acid-green mb-4">Processing Settings</h3>
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-mono text-sm">Max Concurrent Documents</div>
                            <div className="text-xs text-text-muted">Limit parallel document processing</div>
                          </div>
                          <select className="bg-surface border border-border rounded px-3 py-1.5 text-sm font-mono">
                            <option>5</option>
                            <option>10</option>
                            <option>20</option>
                            <option>50</option>
                          </select>
                        </div>
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-mono text-sm">Max Concurrent Chunks</div>
                            <div className="text-xs text-text-muted">Chunks processed in parallel per job</div>
                          </div>
                          <select className="bg-surface border border-border rounded px-3 py-1.5 text-sm font-mono">
                            <option>10</option>
                            <option>20</option>
                            <option>50</option>
                          </select>
                        </div>
                      </div>
                    </div>

                    <div className="card p-4">
                      <h3 className="font-mono text-sm text-acid-green mb-4">Audit Settings</h3>
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-mono text-sm">Primary Scan Model</div>
                            <div className="text-xs text-text-muted">Model for initial document scanning</div>
                          </div>
                          <select className="bg-surface border border-border rounded px-3 py-1.5 text-sm font-mono">
                            <option>gemini-3-pro</option>
                            <option>claude-3.5-sonnet</option>
                            <option>gpt-4-turbo</option>
                          </select>
                        </div>
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-mono text-sm">Verification Model</div>
                            <div className="text-xs text-text-muted">Model for finding verification</div>
                          </div>
                          <select className="bg-surface border border-border rounded px-3 py-1.5 text-sm font-mono">
                            <option>claude-3.5-sonnet</option>
                            <option>gpt-4-turbo</option>
                          </select>
                        </div>
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-mono text-sm">Require Multi-Agent Confirmation</div>
                            <div className="text-xs text-text-muted">Findings must be verified by multiple agents</div>
                          </div>
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input type="checkbox" defaultChecked className="sr-only peer" />
                            <div className="w-11 h-6 bg-surface peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-text-muted after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-acid-green/30 peer-checked:after:bg-acid-green"></div>
                          </label>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // CONTROL PLANE
          </p>
        </footer>
      </main>
    </>
  );
}
