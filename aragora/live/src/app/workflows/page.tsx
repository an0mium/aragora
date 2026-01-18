'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { API_BASE_URL } from '@/config';

interface WorkflowSummary {
  id: string;
  name: string;
  description: string;
  category: string;
  version: string;
  tags: string[];
  stepCount: number;
  createdAt: string;
  updatedAt: string;
}

const categoryIcons: Record<string, string> = {
  legal: '⚖️',
  healthcare: '🏥',
  code: '💻',
  accounting: '📊',
  custom: '🔧',
};

const categoryColors: Record<string, string> = {
  legal: 'border-purple-500 bg-purple-500/10',
  healthcare: 'border-green-500 bg-green-500/10',
  code: 'border-blue-500 bg-blue-500/10',
  accounting: 'border-yellow-500 bg-yellow-500/10',
  custom: 'border-gray-500 bg-gray-500/10',
};

export default function WorkflowsPage() {
  const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
  const [templates, setTemplates] = useState<WorkflowSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'my-workflows' | 'templates'>('templates');

  const fetchTemplates = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/workflow-templates`);
      if (!response.ok) throw new Error('Failed to fetch templates');

      const data = await response.json();
      const templateList: WorkflowSummary[] = (data.templates || []).map(
        (t: Record<string, unknown>) => ({
          id: t.id,
          name: t.name,
          description: t.description,
          category: t.category,
          version: t.version,
          tags: t.tags || [],
          stepCount: (t.steps as unknown[])?.length || 0,
          createdAt: '',
          updatedAt: '',
        })
      );
      setTemplates(templateList);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTemplates();
  }, [fetchTemplates]);

  const displayedWorkflows = activeTab === 'templates' ? templates : workflows;

  return (
    <main className="min-h-screen bg-bg p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-mono font-bold text-text mb-2">
              Workflows
            </h1>
            <p className="text-text-muted">
              Design and manage multi-agent workflows for your organization
            </p>
          </div>

          <div className="flex gap-3">
            <Link
              href="/workflows/runtime"
              className="px-4 py-3 bg-surface border border-border text-text font-mono hover:border-acid-green transition-colors rounded flex items-center gap-2"
            >
              <span>📊</span>
              <span>Runtime</span>
            </Link>
            <Link
              href="/workflows/builder"
              className="px-6 py-3 bg-acid-green text-bg font-mono font-bold hover:bg-acid-green/80 transition-colors rounded flex items-center gap-2"
            >
              <span>+</span>
              <span>New Workflow</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="max-w-7xl mx-auto mb-6">
        <div className="flex gap-1 bg-surface border border-border rounded-lg p-1 w-fit">
          <button
            onClick={() => setActiveTab('templates')}
            className={`px-4 py-2 font-mono text-sm rounded transition-colors ${
              activeTab === 'templates'
                ? 'bg-acid-green text-bg font-bold'
                : 'text-text-muted hover:text-text'
            }`}
          >
            Templates ({templates.length})
          </button>
          <button
            onClick={() => setActiveTab('my-workflows')}
            className={`px-4 py-2 font-mono text-sm rounded transition-colors ${
              activeTab === 'my-workflows'
                ? 'bg-acid-green text-bg font-bold'
                : 'text-text-muted hover:text-text'
            }`}
          >
            My Workflows ({workflows.length})
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto">
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="animate-pulse text-text-muted font-mono">
              Loading workflows...
            </div>
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 mb-6">
            {error}
          </div>
        )}

        {!loading && !error && displayedWorkflows.length === 0 && (
          <div className="text-center py-12 bg-surface border border-border rounded-lg">
            <div className="text-4xl mb-4">📁</div>
            <h3 className="text-lg font-mono font-bold text-text mb-2">
              {activeTab === 'templates' ? 'No templates available' : 'No workflows yet'}
            </h3>
            <p className="text-text-muted mb-4">
              {activeTab === 'templates'
                ? 'Templates will appear here once configured'
                : 'Create your first workflow using the builder'}
            </p>
            <Link
              href="/workflows/builder"
              className="inline-flex items-center gap-2 px-4 py-2 bg-acid-green text-bg font-mono font-bold hover:bg-acid-green/80 transition-colors rounded"
            >
              Create Workflow
            </Link>
          </div>
        )}

        {/* Workflow Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {displayedWorkflows.map((workflow) => (
            <Link
              key={workflow.id}
              href={`/workflows/builder?template=${workflow.id}`}
              className={`
                p-5 rounded-lg border-2 transition-all
                hover:scale-[1.02] hover:shadow-lg
                ${categoryColors[workflow.category] || categoryColors.custom}
              `}
            >
              <div className="flex items-center gap-3 mb-3">
                <span className="text-2xl">
                  {categoryIcons[workflow.category] || '📁'}
                </span>
                <div>
                  <h3 className="font-mono font-bold text-text">
                    {workflow.name}
                  </h3>
                  <span className="text-xs text-text-muted font-mono capitalize">
                    {workflow.category}
                  </span>
                </div>
              </div>

              <p className="text-sm text-text-muted mb-4 line-clamp-2">
                {workflow.description}
              </p>

              <div className="flex flex-wrap gap-1 mb-3">
                {workflow.tags.slice(0, 3).map((tag) => (
                  <span
                    key={tag}
                    className="px-2 py-0.5 text-xs bg-bg/50 rounded font-mono text-text-muted"
                  >
                    {tag}
                  </span>
                ))}
                {workflow.tags.length > 3 && (
                  <span className="px-2 py-0.5 text-xs bg-bg/50 rounded font-mono text-text-muted">
                    +{workflow.tags.length - 3}
                  </span>
                )}
              </div>

              <div className="flex items-center justify-between text-xs font-mono text-text-muted">
                <span>{workflow.stepCount} steps</span>
                <span>v{workflow.version}</span>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Quick Start Guide */}
      {activeTab === 'templates' && templates.length > 0 && (
        <div className="max-w-7xl mx-auto mt-12">
          <div className="bg-surface border border-border rounded-lg p-6">
            <h3 className="text-lg font-mono font-bold text-text mb-4">
              Quick Start
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
              <div>
                <div className="text-2xl mb-2">1️⃣</div>
                <h4 className="font-mono font-bold text-text mb-1">
                  Choose a Template
                </h4>
                <p className="text-text-muted">
                  Select an industry template above to start with a pre-built workflow structure
                </p>
              </div>
              <div>
                <div className="text-2xl mb-2">2️⃣</div>
                <h4 className="font-mono font-bold text-text mb-1">
                  Customize Steps
                </h4>
                <p className="text-text-muted">
                  Modify agents, add review checkpoints, and configure each step for your needs
                </p>
              </div>
              <div>
                <div className="text-2xl mb-2">3️⃣</div>
                <h4 className="font-mono font-bold text-text mb-1">
                  Execute & Monitor
                </h4>
                <p className="text-text-muted">
                  Run workflows and track progress with real-time updates and audit trails
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
