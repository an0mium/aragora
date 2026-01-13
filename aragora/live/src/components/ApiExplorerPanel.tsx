'use client';

import { useState, useMemo, useCallback } from 'react';
import { CollapsibleSection } from '@/components/CollapsibleSection';
import { API_BASE_URL } from '@/config';

// API endpoint definition
interface Endpoint {
  path: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  summary: string;
  description?: string;
  auth?: boolean;
  rateLimit?: string;
  requestBody?: {
    required?: boolean;
    example?: Record<string, unknown>;
  };
  queryParams?: Array<{
    name: string;
    type: string;
    required?: boolean;
    description?: string;
  }>;
  pathParams?: Array<{
    name: string;
    type: string;
    description?: string;
  }>;
}

interface EndpointCategory {
  name: string;
  description: string;
  endpoints: Endpoint[];
}

// Define all API endpoints
const API_ENDPOINTS: EndpointCategory[] = [
  {
    name: 'Debates',
    description: 'Core debate management and history',
    endpoints: [
      { path: '/api/debates', method: 'GET', summary: 'List all debates', auth: true, queryParams: [{ name: 'limit', type: 'integer', description: 'Max results (1-100)' }] },
      { path: '/api/debates/slug/{slug}', method: 'GET', summary: 'Get debate by slug', pathParams: [{ name: 'slug', type: 'string' }] },
      { path: '/api/debates/{id}/export/{format}', method: 'GET', summary: 'Export debate', auth: true, pathParams: [{ name: 'id', type: 'string' }, { name: 'format', type: 'string', description: 'json, csv, or html' }] },
      { path: '/api/debates/{id}/impasse', method: 'GET', summary: 'Detect debate impasse', pathParams: [{ name: 'id', type: 'string' }] },
      { path: '/api/debates/{id}/convergence', method: 'GET', summary: 'Get convergence status', pathParams: [{ name: 'id', type: 'string' }] },
      { path: '/api/debates/{id}/citations', method: 'GET', summary: 'Get evidence citations', auth: true, pathParams: [{ name: 'id', type: 'string' }] },
      { path: '/api/debates/{id}/fork', method: 'POST', summary: 'Fork debate at branch point', auth: true, description: 'Create a counterfactual fork of the debate', requestBody: { required: true, example: { branch_point: 2, modified_context: 'Alternative scenario' } } },
      { path: '/api/debates/{id}/verify-outcome', method: 'POST', summary: 'Verify debate outcome', auth: true, description: 'Record verification of winning position correctness', requestBody: { required: true, example: { correct: true, source: 'manual' } } },
      { path: '/api/debates/{id}/followup-suggestions', method: 'GET', summary: 'Get follow-up suggestions', auth: true, description: 'Get suggestions based on disagreement cruxes' },
      { path: '/api/debates/{id}/followup', method: 'POST', summary: 'Create follow-up debate', auth: true, requestBody: { example: { crux_id: 'crux-123', agents: ['claude', 'gpt4'] } } },
    ],
  },
  {
    name: 'Batch Debates',
    description: 'Submit and manage batch debate operations',
    endpoints: [
      { path: '/api/debates/batch', method: 'POST', summary: 'Submit batch of debates', auth: true, rateLimit: '10/min', description: 'Submit multiple debates for async processing', requestBody: { required: true, example: { items: [{ question: 'Should we use microservices?', agents: 'claude,gpt4', rounds: 3 }], webhook_url: 'https://example.com/callback' } } },
      { path: '/api/debates/batch', method: 'GET', summary: 'List batch requests', auth: true, queryParams: [{ name: 'limit', type: 'integer' }, { name: 'status', type: 'string', description: 'pending, processing, completed, failed' }] },
      { path: '/api/debates/batch/{batch_id}/status', method: 'GET', summary: 'Get batch status', auth: true, pathParams: [{ name: 'batch_id', type: 'string' }] },
      { path: '/api/debates/batch/queue', method: 'GET', summary: 'Get queue status', auth: true, description: 'Overall queue health and statistics' },
    ],
  },
  {
    name: 'Graph Debates',
    description: 'Run debates with automatic branching and merging',
    endpoints: [
      { path: '/api/debates/graph', method: 'POST', summary: 'Run graph-structured debate', auth: true, rateLimit: '5/min', description: 'Automatic branching on disagreements', requestBody: { required: true, example: { task: 'Evaluate trade-offs of eventual consistency', agents: ['claude', 'gpt4', 'gemini'], max_rounds: 5, branch_policy: { min_disagreement: 0.7, max_branches: 3, merge_strategy: 'synthesis' } } } },
      { path: '/api/debates/graph/{id}', method: 'GET', summary: 'Get graph debate', pathParams: [{ name: 'id', type: 'string' }] },
      { path: '/api/debates/graph/{id}/branches', method: 'GET', summary: 'Get debate branches', pathParams: [{ name: 'id', type: 'string' }] },
      { path: '/api/debates/graph/{id}/nodes', method: 'GET', summary: 'Get debate nodes', pathParams: [{ name: 'id', type: 'string' }] },
    ],
  },
  {
    name: 'Matrix Debates',
    description: 'Run parallel scenario debates for sensitivity analysis',
    endpoints: [
      { path: '/api/debates/matrix', method: 'POST', summary: 'Run matrix debate', auth: true, rateLimit: '5/min', description: 'Explore how conclusions change under different assumptions', requestBody: { required: true, example: { task: 'Should we expand to European markets?', scenarios: [{ name: 'Optimistic', parameters: { growth_rate: 0.15 } }, { name: 'Pessimistic', parameters: { growth_rate: 0.02 } }], agents: ['claude', 'gpt4'] } } },
      { path: '/api/debates/matrix/{id}', method: 'GET', summary: 'Get matrix debate', pathParams: [{ name: 'id', type: 'string' }] },
      { path: '/api/debates/matrix/{id}/scenarios', method: 'GET', summary: 'Get scenario results', pathParams: [{ name: 'id', type: 'string' }] },
      { path: '/api/debates/matrix/{id}/conclusions', method: 'GET', summary: 'Get universal/conditional conclusions', pathParams: [{ name: 'id', type: 'string' }] },
    ],
  },
  {
    name: 'Agents',
    description: 'Agent profiles, rankings, and performance',
    endpoints: [
      { path: '/api/leaderboard', method: 'GET', summary: 'Get agent rankings', queryParams: [{ name: 'limit', type: 'integer' }, { name: 'domain', type: 'string' }] },
      { path: '/api/matches/recent', method: 'GET', summary: 'Get recent matches', queryParams: [{ name: 'limit', type: 'integer' }] },
      { path: '/api/agent/{name}/history', method: 'GET', summary: 'Get agent match history', pathParams: [{ name: 'name', type: 'string' }] },
      { path: '/api/agent/{name}/consistency', method: 'GET', summary: 'Get consistency score', description: 'Flip detection metrics', pathParams: [{ name: 'name', type: 'string' }] },
      { path: '/api/agent/{name}/network', method: 'GET', summary: 'Get relationship network', description: 'Rivals and allies', pathParams: [{ name: 'name', type: 'string' }] },
      { path: '/api/agent/compare', method: 'GET', summary: 'Compare multiple agents', queryParams: [{ name: 'agents', type: 'array', required: true, description: 'Agent names to compare' }] },
      { path: '/api/flips/recent', method: 'GET', summary: 'Get recent position flips', queryParams: [{ name: 'limit', type: 'integer' }] },
    ],
  },
  {
    name: 'System',
    description: 'Health checks and system status',
    endpoints: [
      { path: '/api/health', method: 'GET', summary: 'Health check' },
      { path: '/api/health/detailed', method: 'GET', summary: 'Detailed health status', auth: true },
      { path: '/api/health/ws', method: 'GET', summary: 'WebSocket health' },
      { path: '/api/status', method: 'GET', summary: 'System status' },
      { path: '/api/agents/available', method: 'GET', summary: 'List available agents' },
    ],
  },
  {
    name: 'Memory',
    description: 'Continuum memory management',
    endpoints: [
      { path: '/api/memory/query', method: 'GET', summary: 'Query memory', auth: true, queryParams: [{ name: 'query', type: 'string', required: true }, { name: 'tier', type: 'string' }, { name: 'limit', type: 'integer' }] },
      { path: '/api/memory/store', method: 'POST', summary: 'Store memory', auth: true, requestBody: { required: true, example: { key: 'debate-123', content: 'Key insight from debate', tier: 'medium', ttl: 3600 } } },
      { path: '/api/memory/stats', method: 'GET', summary: 'Memory statistics', auth: true },
    ],
  },
  {
    name: 'Analytics',
    description: 'System analytics and metrics',
    endpoints: [
      { path: '/api/analytics/disagreement', method: 'GET', summary: 'Disagreement statistics' },
      { path: '/api/analytics/rankings', method: 'GET', summary: 'Ranking statistics' },
      { path: '/api/analytics/relationships', method: 'GET', summary: 'Relationship summary' },
      { path: '/api/analytics/moments', method: 'GET', summary: 'Moments summary' },
      { path: '/api/dashboard', method: 'GET', summary: 'Dashboard metrics' },
    ],
  },
];

const METHOD_COLORS: Record<string, string> = {
  GET: 'text-acid-cyan bg-acid-cyan/10 border-acid-cyan/30',
  POST: 'text-acid-green bg-acid-green/10 border-acid-green/30',
  PUT: 'text-acid-yellow bg-acid-yellow/10 border-acid-yellow/30',
  DELETE: 'text-red-400 bg-red-400/10 border-red-400/30',
  PATCH: 'text-acid-purple bg-acid-purple/10 border-acid-purple/30',
};

interface TryItFormProps {
  endpoint: Endpoint;
  baseUrl: string;
}

function TryItForm({ endpoint, baseUrl }: TryItFormProps) {
  const [pathValues, setPathValues] = useState<Record<string, string>>({});
  const [queryValues, setQueryValues] = useState<Record<string, string>>({});
  const [bodyValue, setBodyValue] = useState(
    endpoint.requestBody?.example ? JSON.stringify(endpoint.requestBody.example, null, 2) : ''
  );
  const [response, setResponse] = useState<{ status: number; data: unknown } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const buildUrl = useCallback(() => {
    let url = `${baseUrl}${endpoint.path}`;

    // Replace path params
    for (const [key, value] of Object.entries(pathValues)) {
      url = url.replace(`{${key}}`, encodeURIComponent(value));
    }

    // Add query params
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(queryValues)) {
      if (value) params.append(key, value);
    }
    const queryString = params.toString();
    if (queryString) url += `?${queryString}`;

    return url;
  }, [baseUrl, endpoint.path, pathValues, queryValues]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const url = buildUrl();
      const options: RequestInit = {
        method: endpoint.method,
        headers: {
          'Content-Type': 'application/json',
        },
      };

      if (endpoint.method !== 'GET' && bodyValue) {
        options.body = bodyValue;
      }

      const res = await fetch(url);
      const data = await res.json().catch(() => res.text());

      setResponse({ status: res.status, data });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Path Parameters */}
      {endpoint.pathParams && endpoint.pathParams.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-mono text-text-muted uppercase">Path Parameters</h4>
          {endpoint.pathParams.map(param => (
            <div key={param.name} className="flex items-center gap-2">
              <label className="text-sm font-mono w-32 text-acid-cyan">{param.name}:</label>
              <input
                type="text"
                value={pathValues[param.name] || ''}
                onChange={e => setPathValues(p => ({ ...p, [param.name]: e.target.value }))}
                placeholder={param.description || param.type}
                className="flex-1 bg-black/30 border border-acid-green/30 rounded px-2 py-1 text-sm font-mono text-text focus:border-acid-green focus:outline-none"
              />
            </div>
          ))}
        </div>
      )}

      {/* Query Parameters */}
      {endpoint.queryParams && endpoint.queryParams.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-mono text-text-muted uppercase">Query Parameters</h4>
          {endpoint.queryParams.map(param => (
            <div key={param.name} className="flex items-center gap-2">
              <label className="text-sm font-mono w-32 text-acid-cyan">
                {param.name}{param.required && <span className="text-red-400">*</span>}:
              </label>
              <input
                type="text"
                value={queryValues[param.name] || ''}
                onChange={e => setQueryValues(p => ({ ...p, [param.name]: e.target.value }))}
                placeholder={param.description || param.type}
                className="flex-1 bg-black/30 border border-acid-green/30 rounded px-2 py-1 text-sm font-mono text-text focus:border-acid-green focus:outline-none"
              />
            </div>
          ))}
        </div>
      )}

      {/* Request Body */}
      {endpoint.method !== 'GET' && (
        <div className="space-y-2">
          <h4 className="text-xs font-mono text-text-muted uppercase">
            Request Body {endpoint.requestBody?.required && <span className="text-red-400">*</span>}
          </h4>
          <textarea
            value={bodyValue}
            onChange={e => setBodyValue(e.target.value)}
            rows={6}
            className="w-full bg-black/30 border border-acid-green/30 rounded px-3 py-2 text-sm font-mono text-text focus:border-acid-green focus:outline-none resize-y"
            placeholder="JSON request body"
          />
        </div>
      )}

      {/* Built URL Preview */}
      <div className="p-2 bg-black/30 rounded border border-acid-green/20">
        <span className="text-xs font-mono text-text-muted">URL: </span>
        <code className="text-xs font-mono text-acid-cyan break-all">{buildUrl()}</code>
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        disabled={loading}
        className="px-4 py-2 bg-acid-green/20 border border-acid-green text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50"
      >
        {loading ? 'Sending...' : `Send ${endpoint.method}`}
      </button>

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded">
          <span className="text-sm font-mono text-red-400">{error}</span>
        </div>
      )}

      {/* Response */}
      {response && (
        <div className="space-y-2">
          <h4 className="text-xs font-mono text-text-muted uppercase">
            Response{' '}
            <span className={response.status < 400 ? 'text-acid-green' : 'text-red-400'}>
              {response.status}
            </span>
          </h4>
          <pre className="p-3 bg-black/30 rounded border border-acid-green/20 overflow-x-auto text-xs font-mono text-text max-h-64 overflow-y-auto">
            {typeof response.data === 'string'
              ? response.data
              : JSON.stringify(response.data, null, 2)}
          </pre>
        </div>
      )}
    </form>
  );
}

export function ApiExplorerPanel() {
  const [selectedEndpoint, setSelectedEndpoint] = useState<Endpoint | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [baseUrl, setBaseUrl] = useState(
    typeof window !== 'undefined'
      ? window.location.origin
      : API_BASE_URL
  );

  const filteredCategories = useMemo(() => {
    if (!searchQuery) return API_ENDPOINTS;

    const query = searchQuery.toLowerCase();
    return API_ENDPOINTS.map(category => ({
      ...category,
      endpoints: category.endpoints.filter(
        ep =>
          ep.path.toLowerCase().includes(query) ||
          ep.summary.toLowerCase().includes(query) ||
          ep.description?.toLowerCase().includes(query)
      ),
    })).filter(cat => cat.endpoints.length > 0);
  }, [searchQuery]);

  const totalEndpoints = API_ENDPOINTS.reduce((sum, cat) => sum + cat.endpoints.length, 0);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Left Panel - Endpoint List */}
      <div className="lg:col-span-1 space-y-4">
        {/* Search */}
        <div className="card p-4">
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search endpoints..."
            className="w-full bg-black/30 border border-acid-green/30 rounded px-3 py-2 text-sm font-mono text-text focus:border-acid-green focus:outline-none"
          />
          <p className="text-xs font-mono text-text-muted mt-2">
            {totalEndpoints} endpoints available
          </p>
        </div>

        {/* Endpoint Categories */}
        <div className="card p-4 space-y-4 max-h-[600px] overflow-y-auto">
          {filteredCategories.map(category => (
            <CollapsibleSection
              key={category.name}
              id={`api-category-${category.name.toLowerCase().replace(/\s+/g, '-')}`}
              title={category.name}
              defaultOpen={true}
            >
              <p className="text-xs font-mono text-text-muted mb-2">
                {category.description}
              </p>
              <div className="space-y-1">
                {category.endpoints.map(endpoint => (
                  <button
                    key={`${endpoint.method}-${endpoint.path}`}
                    onClick={() => setSelectedEndpoint(endpoint)}
                    className={`w-full text-left p-2 rounded border transition-colors ${
                      selectedEndpoint?.path === endpoint.path && selectedEndpoint?.method === endpoint.method
                        ? 'border-acid-green bg-acid-green/10'
                        : 'border-transparent hover:border-acid-green/30 hover:bg-surface'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span className={`text-xs font-mono px-1.5 py-0.5 rounded border ${METHOD_COLORS[endpoint.method]}`}>
                        {endpoint.method}
                      </span>
                      <span className="text-xs font-mono text-text truncate flex-1">
                        {endpoint.path}
                      </span>
                    </div>
                    <p className="text-xs text-text-muted mt-1 truncate">
                      {endpoint.summary}
                    </p>
                  </button>
                ))}
              </div>
            </CollapsibleSection>
          ))}
        </div>
      </div>

      {/* Right Panel - Endpoint Details */}
      <div className="lg:col-span-2">
        {selectedEndpoint ? (
          <div className="card p-6 space-y-6">
            {/* Header */}
            <div>
              <div className="flex items-center gap-3 mb-2">
                <span className={`text-sm font-mono px-2 py-1 rounded border ${METHOD_COLORS[selectedEndpoint.method]}`}>
                  {selectedEndpoint.method}
                </span>
                <code className="text-lg font-mono text-acid-cyan">{selectedEndpoint.path}</code>
              </div>
              <h2 className="text-xl font-mono text-acid-green">{selectedEndpoint.summary}</h2>
              {selectedEndpoint.description && (
                <p className="text-sm text-text-muted mt-2">{selectedEndpoint.description}</p>
              )}
              <div className="flex gap-4 mt-3 text-xs font-mono">
                {selectedEndpoint.auth && (
                  <span className="text-acid-yellow">Requires Auth</span>
                )}
                {selectedEndpoint.rateLimit && (
                  <span className="text-acid-purple">Rate Limit: {selectedEndpoint.rateLimit}</span>
                )}
              </div>
            </div>

            {/* Base URL */}
            <div className="space-y-2">
              <label className="text-xs font-mono text-text-muted uppercase">Base URL</label>
              <input
                type="text"
                value={baseUrl}
                onChange={e => setBaseUrl(e.target.value)}
                className="w-full bg-black/30 border border-acid-green/30 rounded px-3 py-2 text-sm font-mono text-text focus:border-acid-green focus:outline-none"
              />
            </div>

            {/* Try It */}
            <CollapsibleSection id="api-try-it" title="Try It" defaultOpen={true}>
              <TryItForm endpoint={selectedEndpoint} baseUrl={baseUrl} />
            </CollapsibleSection>
          </div>
        ) : (
          <div className="card p-6 flex items-center justify-center h-96">
            <div className="text-center">
              <p className="text-text-muted font-mono mb-2">Select an endpoint to explore</p>
              <p className="text-xs text-text-muted font-mono">
                Browse the list on the left or search for a specific endpoint
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
