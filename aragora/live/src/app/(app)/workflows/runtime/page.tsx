'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import Link from 'next/link';
import { API_BASE_URL } from '@/config';
import { ExecutionDAGView, StepDetailPanel } from '@/components/workflow-runtime';

type ViewMode = 'list' | 'dag';

interface WorkflowStep {
  id: string;
  name: string;
  type: 'agent' | 'task' | 'decision' | 'human_checkpoint' | 'parallel' | 'memory';
  status: 'pending' | 'running' | 'completed' | 'failed' | 'waiting_approval';
  startedAt?: string;
  completedAt?: string;
  error?: string;
  output?: Record<string, unknown>;
  approvalRequired?: boolean;
  approvalMessage?: string;
}

interface WorkflowExecution {
  id: string;
  workflowId: string;
  workflowName: string;
  status: 'running' | 'completed' | 'failed' | 'paused' | 'waiting_approval';
  progress: number;
  currentStep: string;
  steps: WorkflowStep[];
  startedAt: string;
  completedAt?: string;
  error?: string;
  metadata?: Record<string, unknown>;
}

const STATUS_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  running: { bg: 'bg-blue-900/30', text: 'text-blue-400', border: 'border-blue-500' },
  completed: { bg: 'bg-green-900/30', text: 'text-green-400', border: 'border-green-500' },
  failed: { bg: 'bg-red-900/30', text: 'text-red-400', border: 'border-red-500' },
  paused: { bg: 'bg-yellow-900/30', text: 'text-yellow-400', border: 'border-yellow-500' },
  waiting_approval: { bg: 'bg-purple-900/30', text: 'text-purple-400', border: 'border-purple-500' },
  pending: { bg: 'bg-gray-900/30', text: 'text-gray-400', border: 'border-gray-500' },
};

const STEP_ICONS: Record<string, string> = {
  agent: '🤖',
  task: '📋',
  decision: '🔀',
  human_checkpoint: '👤',
  parallel: '⚡',
  memory: '💾',
};

function formatDuration(startedAt: string, completedAt?: string): string {
  const start = new Date(startedAt).getTime();
  const end = completedAt ? new Date(completedAt).getTime() : Date.now();
  const seconds = Math.floor((end - start) / 1000);

  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function formatTime(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export default function WorkflowRuntimePage() {
  const [executions, setExecutions] = useState<WorkflowExecution[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedExecution, setSelectedExecution] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [usingMockData, setUsingMockData] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [selectedStep, setSelectedStep] = useState<WorkflowStep | null>(null);

  const fetchExecutions = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/workflow-executions`);
      if (!response.ok) {
        // Demo mode: use mock data if API not available
        setExecutions(getMockExecutions());
        setUsingMockData(true);
        setError(null);
        return;
      }
      const data = await response.json();
      setExecutions(data.executions || []);
      setUsingMockData(false);
      setError(null);
    } catch {
      // Demo mode: use mock data on error
      setExecutions(getMockExecutions());
      setUsingMockData(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchExecutions();
    // Poll for updates
    const interval = setInterval(fetchExecutions, 5000);
    return () => clearInterval(interval);
  }, [fetchExecutions]);

  const filteredExecutions = useMemo(() => {
    if (statusFilter === 'all') return executions;
    return executions.filter((e) => e.status === statusFilter);
  }, [executions, statusFilter]);

  const stats = useMemo(() => ({
    total: executions.length,
    running: executions.filter((e) => e.status === 'running').length,
    completed: executions.filter((e) => e.status === 'completed').length,
    failed: executions.filter((e) => e.status === 'failed').length,
    waitingApproval: executions.filter((e) => e.status === 'waiting_approval').length,
  }), [executions]);

  const selectedExecutionData = useMemo(() => {
    return executions.find((e) => e.id === selectedExecution);
  }, [executions, selectedExecution]);

  const handleApprove = async (executionId: string, stepId: string) => {
    try {
      await fetch(`${API_BASE_URL}/api/workflow-executions/${executionId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stepId, approved: true }),
      });
      fetchExecutions();
    } catch (err) {
      console.error('Approval failed:', err);
    }
  };

  const handleReject = async (executionId: string, stepId: string) => {
    try {
      await fetch(`${API_BASE_URL}/api/workflow-executions/${executionId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stepId, approved: false }),
      });
      fetchExecutions();
    } catch (err) {
      console.error('Rejection failed:', err);
    }
  };

  const handleRetry = async (executionId: string) => {
    try {
      await fetch(`${API_BASE_URL}/api/workflow-executions/${executionId}/retry`, {
        method: 'POST',
      });
      fetchExecutions();
    } catch (err) {
      console.error('Retry failed:', err);
    }
  };

  return (
    <main className="min-h-screen bg-bg p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-3xl font-mono font-bold text-text">
                Workflow Runtime
              </h1>
              {usingMockData && (
                <div className="flex items-center gap-1.5 px-2 py-1 bg-yellow-900/20 border border-yellow-600/30 rounded text-xs">
                  <span className="w-1.5 h-1.5 rounded-full bg-yellow-400" />
                  <span className="font-mono text-yellow-400">DEMO MODE</span>
                </div>
              )}
            </div>
            <p className="text-text-muted">Monitor active workflow executions</p>
          </div>
          <Link
            href="/workflows"
            className="px-4 py-2 bg-surface border border-border text-text font-mono hover:border-acid-green transition-colors rounded"
          >
            ← Back to Workflows
          </Link>
        </div>

        {/* Stats Bar */}
        <div className="grid grid-cols-5 gap-4 mb-6">
          <div className="p-4 bg-surface border border-border rounded-lg text-center">
            <div className="text-2xl font-mono text-acid-green">{stats.total}</div>
            <div className="text-xs text-text-muted">Total</div>
          </div>
          <div className="p-4 bg-surface border border-border rounded-lg text-center">
            <div className="text-2xl font-mono text-blue-400">{stats.running}</div>
            <div className="text-xs text-text-muted">Running</div>
          </div>
          <div className="p-4 bg-surface border border-border rounded-lg text-center">
            <div className="text-2xl font-mono text-green-400">{stats.completed}</div>
            <div className="text-xs text-text-muted">Completed</div>
          </div>
          <div className="p-4 bg-surface border border-border rounded-lg text-center">
            <div className="text-2xl font-mono text-red-400">{stats.failed}</div>
            <div className="text-xs text-text-muted">Failed</div>
          </div>
          <div className="p-4 bg-surface border border-border rounded-lg text-center">
            <div className="text-2xl font-mono text-purple-400">{stats.waitingApproval}</div>
            <div className="text-xs text-text-muted">Awaiting Approval</div>
          </div>
        </div>

        {/* Filters and View Toggle */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex gap-2">
            {['all', 'running', 'waiting_approval', 'completed', 'failed'].map((status) => (
              <button
                key={status}
                onClick={() => setStatusFilter(status)}
                className={`px-3 py-1.5 text-xs font-mono rounded transition-colors ${
                  statusFilter === status
                    ? 'bg-acid-green text-bg'
                    : 'bg-surface text-text-muted hover:text-text border border-border'
                }`}
              >
                {status === 'all' ? 'All' : status.replace('_', ' ').toUpperCase()}
              </button>
            ))}
          </div>

          {/* View Mode Toggle */}
          <div className="flex gap-1 border border-border rounded overflow-hidden">
            <button
              onClick={() => setViewMode('list')}
              className={`px-3 py-1.5 text-xs font-mono flex items-center gap-1.5 transition-colors ${
                viewMode === 'list'
                  ? 'bg-acid-green text-bg'
                  : 'bg-surface text-text-muted hover:text-text'
              }`}
              title="List View"
            >
              <span>≡</span> List
            </button>
            <button
              onClick={() => setViewMode('dag')}
              className={`px-3 py-1.5 text-xs font-mono flex items-center gap-1.5 transition-colors ${
                viewMode === 'dag'
                  ? 'bg-acid-green text-bg'
                  : 'bg-surface text-text-muted hover:text-text'
              }`}
              title="DAG View"
            >
              <span>◇</span> DAG
            </button>
          </div>
        </div>

        {/* Executions Sidebar - always visible */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-4">
            <h2 className="text-sm font-mono text-acid-green uppercase">Executions</h2>

            {loading ? (
              <div className="text-center py-8 text-text-muted font-mono">Loading...</div>
            ) : error ? (
              <div className="text-center py-8 text-red-400">{error}</div>
            ) : filteredExecutions.length === 0 ? (
              <div className="text-center py-8 text-text-muted">No executions found</div>
            ) : (
              filteredExecutions.map((execution) => {
                const colors = STATUS_COLORS[execution.status];
                return (
                  <div
                    key={execution.id}
                    onClick={() => setSelectedExecution(execution.id)}
                    className={`p-4 bg-surface border rounded-lg cursor-pointer transition-all ${
                      selectedExecution === execution.id
                        ? 'border-acid-green'
                        : 'border-border hover:border-text-muted'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <h3 className="font-mono font-bold text-text">{execution.workflowName}</h3>
                        <span className="text-xs text-text-muted font-mono">{execution.id}</span>
                      </div>
                      <span className={`px-2 py-0.5 text-xs font-mono uppercase rounded ${colors.bg} ${colors.text}`}>
                        {execution.status.replace('_', ' ')}
                      </span>
                    </div>

                    {/* Progress bar */}
                    <div className="w-full h-2 bg-bg rounded-full overflow-hidden mb-2">
                      <div
                        className={`h-full transition-all ${colors.bg}`}
                        style={{ width: `${execution.progress}%` }}
                      />
                    </div>

                    <div className="flex items-center justify-between text-xs text-text-muted">
                      <span>Step: {execution.currentStep}</span>
                      <span>{execution.progress}% complete</span>
                    </div>

                    <div className="flex items-center justify-between text-xs text-text-muted mt-2">
                      <span>Started: {formatTime(execution.startedAt)}</span>
                      <span>Duration: {formatDuration(execution.startedAt, execution.completedAt)}</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Execution Details - List or DAG view */}
          <div className="lg:col-span-2">
            <h2 className="text-sm font-mono text-acid-green uppercase mb-4">
              Execution Details {viewMode === 'dag' && '(DAG View)'}
            </h2>

            {selectedExecutionData ? (
              viewMode === 'dag' ? (
                /* DAG View */
                <div className="bg-surface border border-border rounded-lg overflow-hidden" style={{ height: '600px' }}>
                  <ExecutionDAGView
                    execution={selectedExecutionData}
                    onStepSelect={setSelectedStep}
                    selectedStepId={selectedStep?.id}
                  />
                </div>
              ) : (
                /* List View - Original Timeline */
                <div className="bg-surface border border-border rounded-lg p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-mono font-bold text-text text-lg">
                      {selectedExecutionData.workflowName}
                    </h3>
                    <div className="flex gap-2">
                      {selectedExecutionData.status === 'failed' && (
                        <button
                          onClick={() => handleRetry(selectedExecutionData.id)}
                          className="px-3 py-1.5 text-xs font-mono bg-yellow-900/30 text-yellow-400 border border-yellow-800/30 rounded hover:bg-yellow-900/50"
                        >
                          Retry
                        </button>
                      )}
                    </div>
                  </div>

                  {selectedExecutionData.error && (
                    <div className="p-3 bg-red-900/20 border border-red-800/30 rounded mb-4">
                      <span className="text-xs font-mono text-red-400 uppercase">Error</span>
                      <p className="text-sm text-red-300 mt-1">{selectedExecutionData.error}</p>
                    </div>
                  )}

                  {/* Steps Timeline */}
                  <div className="space-y-3">
                    {selectedExecutionData.steps.map((step, index) => {
                      const stepColors = STATUS_COLORS[step.status];
                      const isWaitingApproval = step.status === 'waiting_approval';

                      return (
                        <div
                          key={step.id}
                          onClick={() => setSelectedStep(step)}
                          className={`p-3 bg-bg border rounded-lg cursor-pointer ${stepColors.border} ${
                            step.status === 'running' ? 'animate-pulse' : ''
                          } ${selectedStep?.id === step.id ? 'ring-2 ring-acid-green' : ''}`}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className="text-lg">{STEP_ICONS[step.type] || '📦'}</span>
                              <span className="font-mono font-bold text-text">{step.name}</span>
                              <span className="text-xs text-text-muted">({step.type})</span>
                            </div>
                            <span className={`px-2 py-0.5 text-xs font-mono uppercase rounded ${stepColors.bg} ${stepColors.text}`}>
                              {step.status.replace('_', ' ')}
                            </span>
                          </div>

                          {step.startedAt && (
                            <div className="text-xs text-text-muted">
                              {step.completedAt
                                ? `Completed in ${formatDuration(step.startedAt, step.completedAt)}`
                                : `Running for ${formatDuration(step.startedAt)}`}
                            </div>
                          )}

                          {step.error && (
                            <div className="mt-2 p-2 bg-red-900/20 rounded text-xs text-red-400">
                              {step.error}
                            </div>
                          )}

                          {/* Human Checkpoint Approval */}
                          {isWaitingApproval && (
                            <div className="mt-3 p-3 bg-purple-900/20 border border-purple-800/30 rounded">
                              <p className="text-sm text-purple-300 mb-3">
                                {step.approvalMessage || 'This step requires human approval to continue.'}
                              </p>
                              <div className="flex gap-2">
                                <button
                                  onClick={(e) => { e.stopPropagation(); handleApprove(selectedExecutionData.id, step.id); }}
                                  className="flex-1 px-3 py-2 text-xs font-mono bg-green-900/30 text-green-400 border border-green-800/30 rounded hover:bg-green-900/50"
                                >
                                  Approve
                                </button>
                                <button
                                  onClick={(e) => { e.stopPropagation(); handleReject(selectedExecutionData.id, step.id); }}
                                  className="flex-1 px-3 py-2 text-xs font-mono bg-red-900/30 text-red-400 border border-red-800/30 rounded hover:bg-red-900/50"
                                >
                                  Reject
                                </button>
                              </div>
                            </div>
                          )}

                          {/* Step connector line */}
                          {index < selectedExecutionData.steps.length - 1 && (
                            <div className="flex justify-center mt-2">
                              <div className="w-0.5 h-4 bg-border" />
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )
            ) : (
              <div className="bg-surface border border-border rounded-lg p-8 text-center text-text-muted">
                Select an execution to view details
              </div>
            )}
          </div>
        </div>

        {/* Step Detail Panel */}
        {selectedStep && (
          <>
            <div
              className="fixed inset-0 bg-bg/60 z-40"
              onClick={() => setSelectedStep(null)}
            />
            <StepDetailPanel
              step={selectedStep}
              onClose={() => setSelectedStep(null)}
              onApprove={(stepId) => {
                if (selectedExecutionData) {
                  handleApprove(selectedExecutionData.id, stepId);
                  setSelectedStep(null);
                }
              }}
              onReject={(stepId) => {
                if (selectedExecutionData) {
                  handleReject(selectedExecutionData.id, stepId);
                  setSelectedStep(null);
                }
              }}
            />
          </>
        )}
      </div>
    </main>
  );
}

// Mock data for development
function getMockExecutions(): WorkflowExecution[] {
  return [
    {
      id: 'exec_001',
      workflowId: 'wf_contract_review',
      workflowName: 'Contract Review Pipeline',
      status: 'running',
      progress: 60,
      currentStep: 'Legal Analysis',
      startedAt: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
      steps: [
        { id: 's1', name: 'Document Ingestion', type: 'task', status: 'completed', startedAt: new Date(Date.now() - 1000 * 60 * 15).toISOString(), completedAt: new Date(Date.now() - 1000 * 60 * 12).toISOString() },
        { id: 's2', name: 'Clause Extraction', type: 'agent', status: 'completed', startedAt: new Date(Date.now() - 1000 * 60 * 12).toISOString(), completedAt: new Date(Date.now() - 1000 * 60 * 8).toISOString() },
        { id: 's3', name: 'Legal Analysis', type: 'agent', status: 'running', startedAt: new Date(Date.now() - 1000 * 60 * 8).toISOString() },
        { id: 's4', name: 'Risk Assessment', type: 'parallel', status: 'pending' },
        { id: 's5', name: 'Partner Review', type: 'human_checkpoint', status: 'pending', approvalRequired: true, approvalMessage: 'Senior partner must approve high-risk clauses' },
      ],
    },
    {
      id: 'exec_002',
      workflowId: 'wf_code_audit',
      workflowName: 'Security Code Audit',
      status: 'waiting_approval',
      progress: 80,
      currentStep: 'Security Review',
      startedAt: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
      steps: [
        { id: 's1', name: 'Code Analysis', type: 'agent', status: 'completed', startedAt: new Date(Date.now() - 1000 * 60 * 30).toISOString(), completedAt: new Date(Date.now() - 1000 * 60 * 20).toISOString() },
        { id: 's2', name: 'Vulnerability Scan', type: 'task', status: 'completed', startedAt: new Date(Date.now() - 1000 * 60 * 20).toISOString(), completedAt: new Date(Date.now() - 1000 * 60 * 10).toISOString() },
        { id: 's3', name: 'Security Review', type: 'human_checkpoint', status: 'waiting_approval', approvalRequired: true, approvalMessage: '3 critical vulnerabilities found. Approve to proceed with remediation recommendations.' },
        { id: 's4', name: 'Generate Report', type: 'task', status: 'pending' },
      ],
    },
    {
      id: 'exec_003',
      workflowId: 'wf_hipaa_check',
      workflowName: 'HIPAA Compliance Check',
      status: 'failed',
      progress: 40,
      currentStep: 'PHI Detection',
      startedAt: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
      error: 'Connection to document store timed out after 30 seconds',
      steps: [
        { id: 's1', name: 'Document Scan', type: 'task', status: 'completed', startedAt: new Date(Date.now() - 1000 * 60 * 45).toISOString(), completedAt: new Date(Date.now() - 1000 * 60 * 40).toISOString() },
        { id: 's2', name: 'PHI Detection', type: 'agent', status: 'failed', startedAt: new Date(Date.now() - 1000 * 60 * 40).toISOString(), error: 'Connection to document store timed out' },
        { id: 's3', name: 'Compliance Report', type: 'task', status: 'pending' },
      ],
    },
    {
      id: 'exec_004',
      workflowId: 'wf_financial_audit',
      workflowName: 'Q4 Financial Audit',
      status: 'completed',
      progress: 100,
      currentStep: 'Complete',
      startedAt: new Date(Date.now() - 1000 * 60 * 120).toISOString(),
      completedAt: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
      steps: [
        { id: 's1', name: 'Data Import', type: 'task', status: 'completed', startedAt: new Date(Date.now() - 1000 * 60 * 120).toISOString(), completedAt: new Date(Date.now() - 1000 * 60 * 110).toISOString() },
        { id: 's2', name: 'Transaction Analysis', type: 'agent', status: 'completed', startedAt: new Date(Date.now() - 1000 * 60 * 110).toISOString(), completedAt: new Date(Date.now() - 1000 * 60 * 90).toISOString() },
        { id: 's3', name: 'Anomaly Detection', type: 'parallel', status: 'completed', startedAt: new Date(Date.now() - 1000 * 60 * 90).toISOString(), completedAt: new Date(Date.now() - 1000 * 60 * 70).toISOString() },
        { id: 's4', name: 'Final Review', type: 'human_checkpoint', status: 'completed', startedAt: new Date(Date.now() - 1000 * 60 * 70).toISOString(), completedAt: new Date(Date.now() - 1000 * 60 * 65).toISOString() },
        { id: 's5', name: 'Generate Report', type: 'task', status: 'completed', startedAt: new Date(Date.now() - 1000 * 60 * 65).toISOString(), completedAt: new Date(Date.now() - 1000 * 60 * 60).toISOString() },
      ],
    },
  ];
}
