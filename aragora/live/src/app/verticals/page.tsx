'use client';
import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { ErrorWithRetry } from '@/components/ErrorWithRetry';

interface Vertical {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  agents: string[];
  tools: string[];
  compliance_frameworks: string[];
  enabled: boolean;
}

interface Tool {
  name: string;
  description: string;
  category: string;
}

interface ComplianceFramework {
  name: string;
  description: string;
  requirements: string[];
}

interface Suggestion {
  vertical_id: string;
  confidence: number;
  reason: string;
}

type TabType = 'browse' | 'suggest' | 'detail';

export default function VerticalsPage() {
  const { config } = useBackend();
  const backendUrl = config.api;
  const [activeTab, setActiveTab] = useState<TabType>('browse');
  const [verticals, setVerticals] = useState<Vertical[]>([]);
  const [selectedVertical, setSelectedVertical] = useState<Vertical | null>(null);
  const [verticalTools, setVerticalTools] = useState<Tool[]>([]);
  const [verticalCompliance, setVerticalCompliance] = useState<ComplianceFramework[]>([]);
  const [suggestTask, setSuggestTask] = useState('');
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [suggesting, setSuggesting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchVerticals = useCallback(async () => {
    try {
      const response = await fetch(`${backendUrl}/api/verticals`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setVerticals(data.verticals || []);
    } catch (err) {
      console.error('Failed to fetch verticals:', err);
      throw err;
    }
  }, [backendUrl]);

  const fetchVerticalDetail = useCallback(async (verticalId: string) => {
    try {
      const [toolsRes, complianceRes] = await Promise.all([
        fetch(`${backendUrl}/api/verticals/${verticalId}/tools`),
        fetch(`${backendUrl}/api/verticals/${verticalId}/compliance`),
      ]);

      if (toolsRes.ok) {
        const toolsData = await toolsRes.json();
        setVerticalTools(toolsData.tools || []);
      }

      if (complianceRes.ok) {
        const complianceData = await complianceRes.json();
        setVerticalCompliance(complianceData.frameworks || []);
      }
    } catch (err) {
      console.error('Failed to fetch vertical detail:', err);
    }
  }, [backendUrl]);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await fetchVerticals();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [fetchVerticals]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleVerticalSelect = async (vertical: Vertical) => {
    setSelectedVertical(vertical);
    setActiveTab('detail');
    await fetchVerticalDetail(vertical.id);
  };

  const handleSuggest = async () => {
    if (!suggestTask.trim()) return;

    setSuggesting(true);
    setError(null);

    try {
      const response = await fetch(
        `${backendUrl}/api/verticals/suggest?task=${encodeURIComponent(suggestTask)}`
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setSuggestions(data.suggestions || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Suggestion failed');
    } finally {
      setSuggesting(false);
    }
  };

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      finance: 'text-green-400 bg-green-500/10 border-green-500/30',
      legal: 'text-blue-400 bg-blue-500/10 border-blue-500/30',
      healthcare: 'text-red-400 bg-red-500/10 border-red-500/30',
      technology: 'text-purple-400 bg-purple-500/10 border-purple-500/30',
      education: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
      default: 'text-text-muted bg-surface border-border',
    };
    return colors[category.toLowerCase()] || colors.default;
  };

  const renderBrowseTab = () => (
    <div className="space-y-6">
      <h2 className="text-xl font-mono font-bold text-acid-green mb-4">Domain Specialists</h2>

      {verticals.length === 0 ? (
        <p className="text-text-muted">No verticals configured</p>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {verticals.map((vertical) => (
            <button
              key={vertical.id}
              onClick={() => handleVerticalSelect(vertical)}
              className="p-4 bg-surface border border-border rounded-lg hover:border-acid-green/50 transition-all text-left group"
            >
              <div className="flex items-center gap-3 mb-2">
                <span className="text-2xl">{vertical.icon || '🔧'}</span>
                <div>
                  <span className="font-mono font-bold text-text group-hover:text-acid-green transition-colors">
                    {vertical.name}
                  </span>
                  <span
                    className={`ml-2 px-2 py-0.5 text-xs font-mono rounded border ${getCategoryColor(vertical.category)}`}
                  >
                    {vertical.category}
                  </span>
                </div>
              </div>
              <p className="text-sm text-text-muted mb-3 line-clamp-2">{vertical.description}</p>
              <div className="flex items-center justify-between text-xs text-text-muted">
                <span>{vertical.agents.length} agents</span>
                <span>{vertical.tools.length} tools</span>
              </div>
              {!vertical.enabled && (
                <div className="mt-2 text-xs text-yellow-400">Disabled</div>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );

  const renderSuggestTab = () => (
    <div className="space-y-6">
      <h2 className="text-xl font-mono font-bold text-acid-green mb-4">Find Best Vertical</h2>

      <div className="p-4 bg-surface border border-border rounded-lg">
        <label className="block text-xs font-mono text-text-muted uppercase mb-2">
          Describe your task
        </label>
        <textarea
          value={suggestTask}
          onChange={(e) => setSuggestTask(e.target.value)}
          placeholder="e.g., Review a contract for compliance with GDPR requirements..."
          rows={4}
          className="w-full px-3 py-2 bg-bg border border-border rounded font-mono text-sm focus:outline-none focus:border-acid-green/50 resize-none"
        />
        <button
          onClick={handleSuggest}
          disabled={!suggestTask.trim() || suggesting}
          className={`mt-3 w-full px-4 py-2 rounded font-mono font-bold transition-all ${
            !suggestTask.trim() || suggesting
              ? 'bg-border text-text-muted cursor-not-allowed'
              : 'bg-acid-green/20 border-2 border-acid-green text-acid-green hover:bg-acid-green/30'
          }`}
        >
          {suggesting ? 'Analyzing...' : 'Suggest Vertical'}
        </button>
      </div>

      {suggestions.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-mono font-bold text-text-muted uppercase">Recommendations</h3>
          {suggestions.map((suggestion, idx) => {
            const vertical = verticals.find(v => v.id === suggestion.vertical_id);
            return (
              <div
                key={idx}
                className="p-4 bg-surface border border-border rounded-lg hover:border-acid-green/50 cursor-pointer transition-all"
                onClick={() => vertical && handleVerticalSelect(vertical)}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xl">{vertical?.icon || '🔧'}</span>
                    <span className="font-mono font-bold text-text">
                      {vertical?.name || suggestion.vertical_id}
                    </span>
                  </div>
                  <span className="text-sm font-mono text-acid-green">
                    {(suggestion.confidence * 100).toFixed(0)}% match
                  </span>
                </div>
                <p className="text-sm text-text-muted">{suggestion.reason}</p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  const renderDetailTab = () => {
    if (!selectedVertical) {
      return <p className="text-text-muted">No vertical selected</p>;
    }

    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">{selectedVertical.icon || '🔧'}</span>
            <div>
              <h2 className="text-xl font-mono font-bold text-acid-green">{selectedVertical.name}</h2>
              <span className={`px-2 py-0.5 text-xs font-mono rounded border ${getCategoryColor(selectedVertical.category)}`}>
                {selectedVertical.category}
              </span>
            </div>
          </div>
          <button
            onClick={() => {
              setActiveTab('browse');
              setSelectedVertical(null);
            }}
            className="px-3 py-1 text-sm font-mono border border-border rounded hover:border-acid-green/50 transition-colors"
          >
            Back to list
          </button>
        </div>

        <p className="text-text-muted">{selectedVertical.description}</p>

        {/* Specialist Agents */}
        <div className="p-4 bg-surface border border-border rounded-lg">
          <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">Specialist Agents</h3>
          <div className="flex flex-wrap gap-2">
            {selectedVertical.agents.map((agent, i) => (
              <Link
                key={i}
                href={`/agents/${agent}`}
                className="px-3 py-1 text-sm font-mono bg-acid-green/10 border border-acid-green/30 text-acid-green rounded hover:bg-acid-green/20 transition-colors"
              >
                {agent}
              </Link>
            ))}
          </div>
        </div>

        {/* Tools */}
        {verticalTools.length > 0 && (
          <div className="p-4 bg-surface border border-border rounded-lg">
            <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">Available Tools</h3>
            <div className="space-y-2">
              {verticalTools.map((tool, i) => (
                <div key={i} className="p-2 bg-bg rounded">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm text-text">{tool.name}</span>
                    {tool.category && (
                      <span className="px-1 py-0.5 text-xs font-mono bg-blue-500/10 text-blue-400 rounded">
                        {tool.category}
                      </span>
                    )}
                  </div>
                  {tool.description && (
                    <p className="text-xs text-text-muted mt-1">{tool.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Compliance Frameworks */}
        {verticalCompliance.length > 0 && (
          <div className="p-4 bg-surface border border-border rounded-lg">
            <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">Compliance Frameworks</h3>
            <div className="space-y-3">
              {verticalCompliance.map((framework, i) => (
                <div key={i} className="p-3 bg-bg rounded">
                  <div className="font-mono text-sm text-text mb-1">{framework.name}</div>
                  <p className="text-xs text-text-muted mb-2">{framework.description}</p>
                  {framework.requirements && framework.requirements.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {framework.requirements.slice(0, 5).map((req, j) => (
                        <span key={j} className="px-1 py-0.5 text-xs font-mono bg-yellow-500/10 text-yellow-400 rounded">
                          {req}
                        </span>
                      ))}
                      {framework.requirements.length > 5 && (
                        <span className="text-xs text-text-muted">
                          +{framework.requirements.length - 5} more
                        </span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Start Debate */}
        <div className="p-4 bg-surface border border-acid-green/30 rounded-lg">
          <h3 className="text-sm font-mono font-bold text-acid-green uppercase mb-3">Start Specialist Debate</h3>
          <p className="text-sm text-text-muted mb-4">
            Launch a debate using agents specialized for {selectedVertical.name.toLowerCase()} tasks.
          </p>
          <Link
            href={`/debate?vertical=${selectedVertical.id}`}
            className="inline-block px-4 py-2 bg-acid-green/20 border border-acid-green text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors"
          >
            Start {selectedVertical.name} Debate
          </Link>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-bg text-text relative overflow-hidden">
      <Scanlines />
      <CRTVignette />

      <div className="max-w-6xl mx-auto px-4 py-8 relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Link href="/" className="hover:opacity-80 transition-opacity">
            <AsciiBannerCompact />
          </Link>
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <BackendSelector />
          </div>
        </div>

        {/* Title */}
        <div className="mb-8">
          <h1 className="text-3xl font-mono font-bold text-acid-green mb-2">Domain Verticals</h1>
          <p className="text-text-muted font-mono text-sm">
            Specialized agents and tools for domain-specific debates
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6">
            <ErrorWithRetry
              error={error}
              onRetry={loadData}
            />
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-2 mb-6 border-b border-border pb-2">
          <button
            onClick={() => { setActiveTab('browse'); setSelectedVertical(null); }}
            className={`px-4 py-2 font-mono text-sm rounded-t transition-colors ${
              activeTab === 'browse'
                ? 'bg-acid-green/10 text-acid-green border-b-2 border-acid-green'
                : 'text-text-muted hover:text-text'
            }`}
          >
            Browse
          </button>
          <button
            onClick={() => setActiveTab('suggest')}
            className={`px-4 py-2 font-mono text-sm rounded-t transition-colors ${
              activeTab === 'suggest'
                ? 'bg-acid-green/10 text-acid-green border-b-2 border-acid-green'
                : 'text-text-muted hover:text-text'
            }`}
          >
            Find Best Match
          </button>
          {selectedVertical && (
            <button
              onClick={() => setActiveTab('detail')}
              className={`px-4 py-2 font-mono text-sm rounded-t transition-colors ${
                activeTab === 'detail'
                  ? 'bg-acid-green/10 text-acid-green border-b-2 border-acid-green'
                  : 'text-text-muted hover:text-text'
              }`}
            >
              {selectedVertical.name}
            </button>
          )}
        </div>

        {/* Content */}
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-acid-green font-mono animate-pulse">Loading...</div>
          </div>
        ) : (
          <div>
            {activeTab === 'browse' && renderBrowseTab()}
            {activeTab === 'suggest' && renderSuggestTab()}
            {activeTab === 'detail' && renderDetailTab()}
          </div>
        )}
      </div>
    </div>
  );
}
