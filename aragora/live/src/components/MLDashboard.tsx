'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { apiFetch } from '@/config';

interface MLStats {
  router_available: boolean;
  scorer_available: boolean;
  predictor_available: boolean;
  embeddings_available: boolean;
  exporter_available: boolean;
  total_routings?: number;
  total_scorings?: number;
  total_predictions?: number;
  avg_routing_confidence?: number;
  avg_quality_score?: number;
}

interface RoutingResult {
  selected_agents: string[];
  task_type: string;
  confidence: number;
  reasoning: string[];
  agent_scores: Record<string, number>;
  diversity_score: number;
}

interface ScoringResult {
  quality_score: number;
  dimensions: Record<string, number>;
  issues: string[];
  suggestions: string[];
}

interface ConsensusPrediction {
  will_converge: boolean;
  confidence: number;
  estimated_rounds: number;
  risk_factors: string[];
  recommended_actions: string[];
}

interface MLDashboardProps {
  apiBase: string;
}

export function MLDashboard({ apiBase }: MLDashboardProps) {
  const [stats, setStats] = useState<MLStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Routing state
  const [routingTask, setRoutingTask] = useState('');
  const [availableAgents, setAvailableAgents] = useState('anthropic-api,openai-api,grok,deepseek,mistral');
  const [teamSize, setTeamSize] = useState(3);
  const [routingResult, setRoutingResult] = useState<RoutingResult | null>(null);
  const [routingLoading, setRoutingLoading] = useState(false);

  // Scoring state
  const [scoreText, setScoreText] = useState('');
  const [scoreTask, setScoreTask] = useState('');
  const [scoringResult, setScoringResult] = useState<ScoringResult | null>(null);
  const [scoringLoading, setScoringLoading] = useState(false);

  // Consensus prediction state
  const [predictionTask, setPredictionTask] = useState('');
  const [predictionAgents, setPredictionAgents] = useState('anthropic-api,openai-api,grok');
  const [predictionResult, setPredictionResult] = useState<ConsensusPrediction | null>(null);
  const [predictionLoading, setPredictionLoading] = useState(false);

  // Active tab
  const [activeTab, setActiveTab] = useState<'route' | 'score' | 'predict' | 'stats'>('stats');

  const fetchStats = useCallback(async () => {
    setLoading(true);
    setError(null);
    const { data, error: fetchError } = await apiFetch<MLStats>('/api/ml/stats');
    if (fetchError) {
      setError(fetchError);
    } else if (data) {
      setStats(data);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  const handleRouting = async () => {
    if (!routingTask.trim()) return;
    setRoutingLoading(true);
    setRoutingResult(null);

    const agents = availableAgents.split(',').map(a => a.trim()).filter(Boolean);
    const { data, error: routeError } = await apiFetch<RoutingResult>('/api/ml/route', {
      method: 'POST',
      body: JSON.stringify({
        task: routingTask,
        available_agents: agents,
        team_size: teamSize,
      }),
    });

    if (routeError) {
      setError(routeError);
    } else if (data) {
      setRoutingResult(data);
    }
    setRoutingLoading(false);
  };

  const handleScoring = async () => {
    if (!scoreText.trim()) return;
    setScoringLoading(true);
    setScoringResult(null);

    const { data, error: scoreError } = await apiFetch<ScoringResult>('/api/ml/score', {
      method: 'POST',
      body: JSON.stringify({
        response: scoreText,
        task: scoreTask || undefined,
      }),
    });

    if (scoreError) {
      setError(scoreError);
    } else if (data) {
      setScoringResult(data);
    }
    setScoringLoading(false);
  };

  const handlePrediction = async () => {
    if (!predictionTask.trim()) return;
    setPredictionLoading(true);
    setPredictionResult(null);

    const agents = predictionAgents.split(',').map(a => a.trim()).filter(Boolean);
    const { data, error: predError } = await apiFetch<ConsensusPrediction>('/api/ml/consensus', {
      method: 'POST',
      body: JSON.stringify({
        task: predictionTask,
        agents: agents,
      }),
    });

    if (predError) {
      setError(predError);
    } else if (data) {
      setPredictionResult(data);
    }
    setPredictionLoading(false);
  };

  if (loading) {
    return (
      <div className="card p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-surface rounded w-1/4" />
          <div className="h-32 bg-surface rounded" />
          <div className="h-32 bg-surface rounded" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Error display */}
      {error && (
        <div className="p-4 border border-red-500/30 bg-red-500/10 rounded text-red-400 text-sm font-mono">
          {error}
          <button
            onClick={() => setError(null)}
            className="ml-4 text-red-500 hover:text-red-400"
          >
            [DISMISS]
          </button>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 border-b border-acid-green/30 pb-2">
        {(['stats', 'route', 'score', 'predict'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-xs font-mono rounded-t transition-colors ${
              activeTab === tab
                ? 'bg-acid-green/20 text-acid-green border border-acid-green/50 border-b-0'
                : 'text-text-muted hover:text-acid-green hover:bg-acid-green/5'
            }`}
          >
            [{tab.toUpperCase()}]
          </button>
        ))}
      </div>

      {/* Stats Tab */}
      {activeTab === 'stats' && stats && (
        <div className="card p-6 space-y-6">
          <h3 className="text-lg font-mono text-acid-green">ML Module Status</h3>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <StatusIndicator
              label="Agent Router"
              available={stats.router_available}
            />
            <StatusIndicator
              label="Quality Scorer"
              available={stats.scorer_available}
            />
            <StatusIndicator
              label="Consensus Predictor"
              available={stats.predictor_available}
            />
            <StatusIndicator
              label="Embeddings"
              available={stats.embeddings_available}
            />
            <StatusIndicator
              label="Training Exporter"
              available={stats.exporter_available}
            />
          </div>

          {stats.total_routings !== undefined && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
              <MetricCard
                label="Total Routings"
                value={stats.total_routings}
              />
              <MetricCard
                label="Avg Routing Confidence"
                value={`${((stats.avg_routing_confidence || 0) * 100).toFixed(1)}%`}
              />
              <MetricCard
                label="Total Scorings"
                value={stats.total_scorings || 0}
              />
              <MetricCard
                label="Avg Quality Score"
                value={`${((stats.avg_quality_score || 0) * 100).toFixed(1)}%`}
              />
            </div>
          )}

          <button
            onClick={fetchStats}
            className="px-4 py-2 text-xs font-mono bg-acid-green/20 text-acid-green border border-acid-green/50 rounded hover:bg-acid-green/30 transition-colors"
          >
            [REFRESH STATS]
          </button>
        </div>
      )}

      {/* Routing Tab */}
      {activeTab === 'route' && (
        <div className="card p-6 space-y-4">
          <h3 className="text-lg font-mono text-acid-green">Agent Routing</h3>
          <p className="text-xs text-text-muted font-mono">
            Get ML-based recommendations for which agents to include in a debate team.
          </p>

          <div className="space-y-3">
            <div>
              <label className="block text-xs font-mono text-text-muted mb-1">Task Description</label>
              <textarea
                value={routingTask}
                onChange={e => setRoutingTask(e.target.value)}
                placeholder="Describe the task for the debate..."
                className="w-full h-24 p-3 bg-surface border border-acid-green/30 rounded font-mono text-sm focus:outline-none focus:border-acid-green"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-mono text-text-muted mb-1">Available Agents (comma-separated)</label>
                <input
                  type="text"
                  value={availableAgents}
                  onChange={e => setAvailableAgents(e.target.value)}
                  className="w-full p-2 bg-surface border border-acid-green/30 rounded font-mono text-sm focus:outline-none focus:border-acid-green"
                />
              </div>
              <div>
                <label className="block text-xs font-mono text-text-muted mb-1">Team Size</label>
                <input
                  type="number"
                  value={teamSize}
                  onChange={e => setTeamSize(parseInt(e.target.value) || 3)}
                  min={1}
                  max={10}
                  className="w-full p-2 bg-surface border border-acid-green/30 rounded font-mono text-sm focus:outline-none focus:border-acid-green"
                />
              </div>
            </div>

            <button
              onClick={handleRouting}
              disabled={routingLoading || !routingTask.trim()}
              className="px-4 py-2 text-xs font-mono bg-acid-green/20 text-acid-green border border-acid-green/50 rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {routingLoading ? '[ROUTING...]' : '[GET RECOMMENDATION]'}
            </button>
          </div>

          {routingResult && (
            <div className="mt-6 p-4 border border-acid-cyan/30 bg-acid-cyan/5 rounded space-y-3">
              <h4 className="text-sm font-mono text-acid-cyan">Routing Result</h4>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-xs text-text-muted">Selected Agents:</span>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {routingResult.selected_agents.map(agent => (
                      <span key={agent} className="px-2 py-1 bg-acid-green/20 text-acid-green text-xs font-mono rounded">
                        {agent}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <span className="text-xs text-text-muted">Task Type:</span>
                  <p className="text-acid-green font-mono">{routingResult.task_type}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-xs text-text-muted">Confidence:</span>
                  <ProgressBar value={routingResult.confidence} />
                </div>
                <div>
                  <span className="text-xs text-text-muted">Diversity Score:</span>
                  <ProgressBar value={routingResult.diversity_score} />
                </div>
              </div>

              {routingResult.reasoning.length > 0 && (
                <div>
                  <span className="text-xs text-text-muted">Reasoning:</span>
                  <ul className="mt-1 text-xs font-mono text-text-muted">
                    {routingResult.reasoning.map((r, i) => (
                      <li key={i}>- {r}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div>
                <span className="text-xs text-text-muted">Agent Scores:</span>
                <div className="mt-1 space-y-1">
                  {Object.entries(routingResult.agent_scores)
                    .sort(([, a], [, b]) => b - a)
                    .map(([agent, score]) => (
                      <div key={agent} className="flex items-center gap-2">
                        <span className="text-xs font-mono w-32">{agent}</span>
                        <div className="flex-1">
                          <ProgressBar value={score} />
                        </div>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Scoring Tab */}
      {activeTab === 'score' && (
        <div className="card p-6 space-y-4">
          <h3 className="text-lg font-mono text-acid-green">Quality Scoring</h3>
          <p className="text-xs text-text-muted font-mono">
            Analyze response quality across multiple dimensions.
          </p>

          <div className="space-y-3">
            <div>
              <label className="block text-xs font-mono text-text-muted mb-1">Response to Score</label>
              <textarea
                value={scoreText}
                onChange={e => setScoreText(e.target.value)}
                placeholder="Paste response text to analyze..."
                className="w-full h-32 p-3 bg-surface border border-acid-green/30 rounded font-mono text-sm focus:outline-none focus:border-acid-green"
              />
            </div>

            <div>
              <label className="block text-xs font-mono text-text-muted mb-1">Original Task (optional)</label>
              <input
                type="text"
                value={scoreTask}
                onChange={e => setScoreTask(e.target.value)}
                placeholder="What task was this response for?"
                className="w-full p-2 bg-surface border border-acid-green/30 rounded font-mono text-sm focus:outline-none focus:border-acid-green"
              />
            </div>

            <button
              onClick={handleScoring}
              disabled={scoringLoading || !scoreText.trim()}
              className="px-4 py-2 text-xs font-mono bg-acid-green/20 text-acid-green border border-acid-green/50 rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {scoringLoading ? '[SCORING...]' : '[ANALYZE QUALITY]'}
            </button>
          </div>

          {scoringResult && (
            <div className="mt-6 p-4 border border-acid-cyan/30 bg-acid-cyan/5 rounded space-y-3">
              <h4 className="text-sm font-mono text-acid-cyan">Quality Analysis</h4>

              <div>
                <span className="text-xs text-text-muted">Overall Quality:</span>
                <div className="flex items-center gap-3 mt-1">
                  <ProgressBar value={scoringResult.quality_score} />
                  <span className="text-acid-green font-mono">
                    {(scoringResult.quality_score * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              {Object.keys(scoringResult.dimensions).length > 0 && (
                <div>
                  <span className="text-xs text-text-muted">Dimensions:</span>
                  <div className="mt-1 space-y-1">
                    {Object.entries(scoringResult.dimensions).map(([dim, score]) => (
                      <div key={dim} className="flex items-center gap-2">
                        <span className="text-xs font-mono w-32 capitalize">{dim}</span>
                        <div className="flex-1">
                          <ProgressBar value={score} />
                        </div>
                        <span className="text-xs font-mono w-12 text-right">
                          {(score * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {scoringResult.issues.length > 0 && (
                <div>
                  <span className="text-xs text-text-muted">Issues Detected:</span>
                  <ul className="mt-1 text-xs font-mono text-yellow-500">
                    {scoringResult.issues.map((issue, i) => (
                      <li key={i}>- {issue}</li>
                    ))}
                  </ul>
                </div>
              )}

              {scoringResult.suggestions.length > 0 && (
                <div>
                  <span className="text-xs text-text-muted">Suggestions:</span>
                  <ul className="mt-1 text-xs font-mono text-acid-cyan">
                    {scoringResult.suggestions.map((s, i) => (
                      <li key={i}>- {s}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Prediction Tab */}
      {activeTab === 'predict' && (
        <div className="card p-6 space-y-4">
          <h3 className="text-lg font-mono text-acid-green">Consensus Prediction</h3>
          <p className="text-xs text-text-muted font-mono">
            Predict likelihood of consensus and estimated rounds to convergence.
          </p>

          <div className="space-y-3">
            <div>
              <label className="block text-xs font-mono text-text-muted mb-1">Task Description</label>
              <textarea
                value={predictionTask}
                onChange={e => setPredictionTask(e.target.value)}
                placeholder="Describe the debate topic..."
                className="w-full h-24 p-3 bg-surface border border-acid-green/30 rounded font-mono text-sm focus:outline-none focus:border-acid-green"
              />
            </div>

            <div>
              <label className="block text-xs font-mono text-text-muted mb-1">Participating Agents (comma-separated)</label>
              <input
                type="text"
                value={predictionAgents}
                onChange={e => setPredictionAgents(e.target.value)}
                className="w-full p-2 bg-surface border border-acid-green/30 rounded font-mono text-sm focus:outline-none focus:border-acid-green"
              />
            </div>

            <button
              onClick={handlePrediction}
              disabled={predictionLoading || !predictionTask.trim()}
              className="px-4 py-2 text-xs font-mono bg-acid-green/20 text-acid-green border border-acid-green/50 rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {predictionLoading ? '[PREDICTING...]' : '[PREDICT CONSENSUS]'}
            </button>
          </div>

          {predictionResult && (
            <div className="mt-6 p-4 border border-acid-cyan/30 bg-acid-cyan/5 rounded space-y-3">
              <h4 className="text-sm font-mono text-acid-cyan">Prediction Result</h4>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <span className="text-xs text-text-muted">Will Converge:</span>
                  <p className={`font-mono text-lg ${predictionResult.will_converge ? 'text-acid-green' : 'text-yellow-500'}`}>
                    {predictionResult.will_converge ? 'YES' : 'UNCERTAIN'}
                  </p>
                </div>
                <div>
                  <span className="text-xs text-text-muted">Confidence:</span>
                  <p className="font-mono text-lg text-acid-green">
                    {(predictionResult.confidence * 100).toFixed(0)}%
                  </p>
                </div>
                <div>
                  <span className="text-xs text-text-muted">Est. Rounds:</span>
                  <p className="font-mono text-lg text-acid-cyan">
                    {predictionResult.estimated_rounds}
                  </p>
                </div>
              </div>

              {predictionResult.risk_factors.length > 0 && (
                <div>
                  <span className="text-xs text-text-muted">Risk Factors:</span>
                  <ul className="mt-1 text-xs font-mono text-yellow-500">
                    {predictionResult.risk_factors.map((r, i) => (
                      <li key={i}>- {r}</li>
                    ))}
                  </ul>
                </div>
              )}

              {predictionResult.recommended_actions.length > 0 && (
                <div>
                  <span className="text-xs text-text-muted">Recommended Actions:</span>
                  <ul className="mt-1 text-xs font-mono text-acid-cyan">
                    {predictionResult.recommended_actions.map((a, i) => (
                      <li key={i}>- {a}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StatusIndicator({ label, available }: { label: string; available: boolean }) {
  return (
    <div className={`p-3 rounded border ${available ? 'border-acid-green/30 bg-acid-green/5' : 'border-red-500/30 bg-red-500/5'}`}>
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${available ? 'bg-acid-green animate-pulse' : 'bg-red-500'}`} />
        <span className="text-xs font-mono">{label}</span>
      </div>
      <p className={`text-xs font-mono mt-1 ${available ? 'text-acid-green' : 'text-red-400'}`}>
        {available ? 'ONLINE' : 'UNAVAILABLE'}
      </p>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="p-3 rounded border border-acid-green/20 bg-surface">
      <p className="text-xs font-mono text-text-muted">{label}</p>
      <p className="text-lg font-mono text-acid-green">{value}</p>
    </div>
  );
}

function ProgressBar({ value }: { value: number }) {
  const pct = Math.min(100, Math.max(0, value * 100));
  return (
    <div className="h-2 bg-surface rounded overflow-hidden flex-1">
      <div
        className="h-full bg-acid-green transition-all duration-300"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
