'use client';

import { useState, useMemo, useCallback, useEffect } from 'react';
import { CollapsibleSection } from '@/components/CollapsibleSection';
import { API_BASE_URL } from '@/config';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
  icon: string;
  description: string;
  endpoints: Endpoint[];
}

// ---------------------------------------------------------------------------
// Endpoint Catalog (~50 most-used endpoints across 12 categories)
// ---------------------------------------------------------------------------

const API_CATALOG: EndpointCategory[] = [
  {
    name: 'Debates',
    icon: '#',
    description: 'Core debate management, forking, and history',
    endpoints: [
      { path: '/api/debates', method: 'GET', summary: 'List all debates', auth: true, queryParams: [{ name: 'limit', type: 'integer', description: 'Max results (1-100)' }, { name: 'offset', type: 'integer', description: 'Pagination offset' }] },
      { path: '/api/debates', method: 'POST', summary: 'Create new debate', auth: true, rateLimit: '10/min', requestBody: { required: true, example: { topic: 'Should AI be regulated?', agents: ['claude', 'gpt4'], rounds: 3 } } },
      { path: '/api/debates/{id}', method: 'GET', summary: 'Get debate by ID', pathParams: [{ name: 'id', type: 'string', description: 'Debate UUID' }] },
      { path: '/api/debates/slug/{slug}', method: 'GET', summary: 'Get debate by slug', pathParams: [{ name: 'slug', type: 'string' }] },
      { path: '/api/debates/{id}/export/{format}', method: 'GET', summary: 'Export debate', auth: true, pathParams: [{ name: 'id', type: 'string' }, { name: 'format', type: 'string', description: 'json, csv, or html' }] },
      { path: '/api/debates/{id}/fork', method: 'POST', summary: 'Fork debate at branch point', auth: true, requestBody: { required: true, example: { branch_point: 2, modified_context: 'Alternative scenario' } } },
      { path: '/api/debates/{id}/convergence', method: 'GET', summary: 'Get convergence metrics', pathParams: [{ name: 'id', type: 'string' }] },
      { path: '/api/debates/{id}/citations', method: 'GET', summary: 'Get evidence citations', auth: true, pathParams: [{ name: 'id', type: 'string' }] },
    ],
  },
  {
    name: 'Batch Debates',
    icon: '=',
    description: 'Submit and manage batch debate operations',
    endpoints: [
      { path: '/api/debates/batch', method: 'POST', summary: 'Submit batch of debates', auth: true, rateLimit: '10/min', requestBody: { required: true, example: { items: [{ question: 'Should we use microservices?', agents: 'claude,gpt4', rounds: 3 }], webhook_url: 'https://example.com/callback' } } },
      { path: '/api/debates/batch', method: 'GET', summary: 'List batch requests', auth: true, queryParams: [{ name: 'limit', type: 'integer' }, { name: 'status', type: 'string', description: 'pending, processing, completed, failed' }] },
      { path: '/api/debates/batch/{batch_id}/status', method: 'GET', summary: 'Get batch status', auth: true, pathParams: [{ name: 'batch_id', type: 'string' }] },
    ],
  },
  {
    name: 'Agents',
    icon: '&',
    description: 'Agent profiles, rankings, and performance',
    endpoints: [
      { path: '/api/agents/available', method: 'GET', summary: 'List available agents' },
      { path: '/api/leaderboard', method: 'GET', summary: 'Get agent rankings', queryParams: [{ name: 'limit', type: 'integer' }, { name: 'domain', type: 'string' }] },
      { path: '/api/agent/{name}/history', method: 'GET', summary: 'Get agent match history', pathParams: [{ name: 'name', type: 'string', description: 'Agent name (e.g. claude)' }] },
      { path: '/api/agent/{name}/consistency', method: 'GET', summary: 'Get consistency score', pathParams: [{ name: 'name', type: 'string' }] },
      { path: '/api/agent/{name}/network', method: 'GET', summary: 'Get relationship network', pathParams: [{ name: 'name', type: 'string' }] },
      { path: '/api/agent/compare', method: 'GET', summary: 'Compare multiple agents', queryParams: [{ name: 'agents', type: 'string', required: true, description: 'Comma-separated agent names' }] },
    ],
  },
  {
    name: 'Auth',
    icon: '>',
    description: 'Authentication, API keys, and user management',
    endpoints: [
      { path: '/api/auth/me', method: 'GET', summary: 'Get current user info', auth: true },
      { path: '/api/auth/login', method: 'POST', summary: 'Login with credentials', requestBody: { required: true, example: { email: 'user@example.com', password: 'password' } } },
      { path: '/api/auth/register', method: 'POST', summary: 'Register new user', requestBody: { required: true, example: { email: 'user@example.com', password: 'password', name: 'User' } } },
      { path: '/api/auth/api-key', method: 'POST', summary: 'Generate API key', auth: true },
      { path: '/api/auth/api-key', method: 'DELETE', summary: 'Revoke API key', auth: true },
    ],
  },
  {
    name: 'Memory',
    icon: '=',
    description: 'Continuum memory management across tiers',
    endpoints: [
      { path: '/api/memory/query', method: 'GET', summary: 'Query memory', auth: true, queryParams: [{ name: 'query', type: 'string', required: true }, { name: 'tier', type: 'string', description: 'fast, medium, slow, glacial' }, { name: 'limit', type: 'integer' }] },
      { path: '/api/memory/store', method: 'POST', summary: 'Store memory entry', auth: true, requestBody: { required: true, example: { key: 'debate-123', content: 'Key insight from debate', tier: 'medium', ttl: 3600 } } },
      { path: '/api/memory/stats', method: 'GET', summary: 'Memory statistics', auth: true },
    ],
  },
  {
    name: 'Knowledge',
    icon: '?',
    description: 'Knowledge Mound search, adapters, and federation',
    endpoints: [
      { path: '/api/v1/knowledge/search', method: 'GET', summary: 'Semantic knowledge search', auth: true, queryParams: [{ name: 'q', type: 'string', required: true, description: 'Search query' }, { name: 'limit', type: 'integer' }, { name: 'adapter', type: 'string' }] },
      { path: '/api/v1/knowledge/adapters', method: 'GET', summary: 'List KM adapters', auth: true },
      { path: '/api/v1/knowledge/stats', method: 'GET', summary: 'Knowledge store statistics', auth: true },
      { path: '/api/v1/knowledge/entries', method: 'POST', summary: 'Store knowledge entry', auth: true, requestBody: { required: true, example: { content: 'Insight text', source: 'debate', metadata: { debate_id: 'abc-123' } } } },
    ],
  },
  {
    name: 'Billing',
    icon: '$',
    description: 'Usage tracking, costs, and budget management',
    endpoints: [
      { path: '/api/billing/usage', method: 'GET', summary: 'Get usage statistics', auth: true },
      { path: '/api/v1/billing/costs', method: 'GET', summary: 'Get cost breakdown', auth: true, queryParams: [{ name: 'time_range', type: 'string', description: '7d, 30d, 90d' }] },
      { path: '/api/v1/billing/budget', method: 'GET', summary: 'Get budget status', auth: true },
      { path: '/api/v1/billing/budget', method: 'PUT', summary: 'Update budget limits', auth: true, requestBody: { required: true, example: { monthly_limit_usd: 100, alert_threshold: 0.8 } } },
    ],
  },
  {
    name: 'Analytics',
    icon: '~',
    description: 'Debate metrics, trends, and performance analytics',
    endpoints: [
      { path: '/api/v1/analytics/debates/overview', method: 'GET', summary: 'Debate overview metrics', queryParams: [{ name: 'time_range', type: 'string', description: '7d, 30d, 90d' }] },
      { path: '/api/v1/analytics/debates/trends', method: 'GET', summary: 'Debate activity trends', queryParams: [{ name: 'time_range', type: 'string' }, { name: 'granularity', type: 'string', description: 'daily, weekly, monthly' }] },
      { path: '/api/v1/analytics/agents/leaderboard', method: 'GET', summary: 'Agent leaderboard', queryParams: [{ name: 'limit', type: 'integer' }] },
      { path: '/api/v1/analytics/usage/tokens', method: 'GET', summary: 'Token usage analytics', auth: true, queryParams: [{ name: 'time_range', type: 'string' }] },
      { path: '/api/v1/analytics/usage/costs', method: 'GET', summary: 'Cost analytics', auth: true, queryParams: [{ name: 'time_range', type: 'string' }] },
      { path: '/api/dashboard', method: 'GET', summary: 'Dashboard metrics' },
    ],
  },
  {
    name: 'Compliance',
    icon: '\u2713',
    description: 'SOC 2 controls, GDPR, and audit trails',
    endpoints: [
      { path: '/api/v1/compliance/status', method: 'GET', summary: 'Compliance status overview', auth: true },
      { path: '/api/v1/compliance/controls', method: 'GET', summary: 'List compliance controls', auth: true, queryParams: [{ name: 'framework', type: 'string', description: 'soc2, gdpr, hipaa' }] },
      { path: '/api/v1/compliance/audit-log', method: 'GET', summary: 'Get audit log entries', auth: true, queryParams: [{ name: 'limit', type: 'integer' }, { name: 'action', type: 'string' }] },
    ],
  },
  {
    name: 'Control Plane',
    icon: '\u25CE',
    description: 'Agent registry, scheduling, and policy governance',
    endpoints: [
      { path: '/api/v1/control-plane/agents', method: 'GET', summary: 'List registered agents', auth: true },
      { path: '/api/v1/control-plane/health', method: 'GET', summary: 'Control plane health', auth: true },
      { path: '/api/v1/control-plane/policies', method: 'GET', summary: 'List active policies', auth: true },
      { path: '/api/v1/control-plane/policies', method: 'POST', summary: 'Create policy', auth: true, requestBody: { required: true, example: { name: 'rate-limit-policy', type: 'rate_limit', config: { max_requests: 100, window_seconds: 60 } } } },
    ],
  },
  {
    name: 'Workflows',
    icon: '>',
    description: 'DAG-based workflow automation',
    endpoints: [
      { path: '/api/v1/workflows', method: 'GET', summary: 'List workflows', auth: true },
      { path: '/api/v1/workflows', method: 'POST', summary: 'Create workflow', auth: true, requestBody: { required: true, example: { name: 'review-pipeline', nodes: [{ type: 'debate', config: { topic: 'Review code change' } }] } } },
      { path: '/api/v1/workflows/{id}', method: 'GET', summary: 'Get workflow details', auth: true, pathParams: [{ name: 'id', type: 'string' }] },
      { path: '/api/v1/workflows/{id}/execute', method: 'POST', summary: 'Execute workflow', auth: true, pathParams: [{ name: 'id', type: 'string' }] },
    ],
  },
  {
    name: 'System',
    icon: '!',
    description: 'Health checks and system status',
    endpoints: [
      { path: '/api/health', method: 'GET', summary: 'Health check' },
      { path: '/api/health/detailed', method: 'GET', summary: 'Detailed health status', auth: true },
      { path: '/api/health/ws', method: 'GET', summary: 'WebSocket health' },
      { path: '/api/status', method: 'GET', summary: 'System status' },
    ],
  },
];

const METHOD_COLORS: Record<string, string> = {
  GET: 'text-acid-cyan bg-acid-cyan/10 border-acid-cyan/30',
  POST: 'text-acid-green bg-acid-green/10 border-acid-green/30',
  PUT: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30',
  DELETE: 'text-red-400 bg-red-400/10 border-red-400/30',
  PATCH: 'text-purple-400 bg-purple-400/10 border-purple-400/30',
};

const STATUS_COLORS: Record<string, string> = {
  '2': 'text-acid-green',
  '3': 'text-acid-cyan',
  '4': 'text-yellow-400',
  '5': 'text-red-400',
};

function getStatusColor(status: number): string {
  const key = String(status).charAt(0);
  return STATUS_COLORS[key] || 'text-text-muted';
}

// ---------------------------------------------------------------------------
// TryItForm -- request builder with headers, auth, response display
// ---------------------------------------------------------------------------

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
  const [customHeaders, setCustomHeaders] = useState<Array<{ key: string; value: string }>>([]);
  const [response, setResponse] = useState<{
    status: number;
    statusText: string;
    headers: Record<string, string>;
    data: unknown;
    elapsed: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showHeaders, setShowHeaders] = useState(false);
  const [showResponseHeaders, setShowResponseHeaders] = useState(false);

  // Auto-detect auth token from localStorage
  const authToken = useMemo(() => {
    if (typeof window === 'undefined') return null;
    const stored = localStorage.getItem('aragora_tokens');
    if (!stored) return null;
    try {
      return (JSON.parse(stored) as { access_token?: string }).access_token || null;
    } catch {
      return null;
    }
  }, []);

  // Reset form when endpoint changes
  useEffect(() => {
    setPathValues({});
    setQueryValues({});
    setBodyValue(
      endpoint.requestBody?.example ? JSON.stringify(endpoint.requestBody.example, null, 2) : ''
    );
    setResponse(null);
    setError(null);
  }, [endpoint]);

  const buildUrl = useCallback(() => {
    let url = `${baseUrl}${endpoint.path}`;

    for (const [key, value] of Object.entries(pathValues)) {
      url = url.replace(`{${key}}`, encodeURIComponent(value));
    }

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

    const start = performance.now();

    try {
      const url = buildUrl();
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      // Auto-inject auth token
      if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
      }

      // Add custom headers
      for (const h of customHeaders) {
        if (h.key.trim()) {
          headers[h.key.trim()] = h.value;
        }
      }

      const options: RequestInit = {
        method: endpoint.method,
        headers,
      };

      if (endpoint.method !== 'GET' && bodyValue) {
        options.body = bodyValue;
      }

      const res = await fetch(url, options);
      const elapsed = Math.round(performance.now() - start);
      const data = await res.json().catch(() => res.text());

      // Collect response headers
      const resHeaders: Record<string, string> = {};
      res.headers.forEach((value, key) => {
        resHeaders[key] = value;
      });

      setResponse({ status: res.status, statusText: res.statusText, headers: resHeaders, data, elapsed });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  const addHeader = () => {
    setCustomHeaders([...customHeaders, { key: '', value: '' }]);
  };

  const removeHeader = (idx: number) => {
    setCustomHeaders(customHeaders.filter((_, i) => i !== idx));
  };

  const updateHeader = (idx: number, field: 'key' | 'value', val: string) => {
    const next = [...customHeaders];
    next[idx] = { ...next[idx], [field]: val };
    setCustomHeaders(next);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Path Parameters */}
      {endpoint.pathParams && endpoint.pathParams.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-mono text-text-muted uppercase tracking-wider">Path Parameters</h4>
          {endpoint.pathParams.map(param => (
            <div key={param.name} className="flex items-center gap-2">
              <label className="text-sm font-mono w-28 text-acid-cyan shrink-0">{param.name}:</label>
              <input
                type="text"
                value={pathValues[param.name] || ''}
                onChange={e => setPathValues(p => ({ ...p, [param.name]: e.target.value }))}
                placeholder={param.description || param.type}
                className="flex-1 bg-black/30 border border-acid-green/30 px-2 py-1.5 text-sm font-mono text-text focus:border-acid-green focus:outline-none"
              />
            </div>
          ))}
        </div>
      )}

      {/* Query Parameters */}
      {endpoint.queryParams && endpoint.queryParams.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-xs font-mono text-text-muted uppercase tracking-wider">Query Parameters</h4>
          {endpoint.queryParams.map(param => (
            <div key={param.name} className="flex items-center gap-2">
              <label className="text-sm font-mono w-28 text-acid-cyan shrink-0">
                {param.name}{param.required && <span className="text-red-400">*</span>}:
              </label>
              <input
                type="text"
                value={queryValues[param.name] || ''}
                onChange={e => setQueryValues(p => ({ ...p, [param.name]: e.target.value }))}
                placeholder={param.description || param.type}
                className="flex-1 bg-black/30 border border-acid-green/30 px-2 py-1.5 text-sm font-mono text-text focus:border-acid-green focus:outline-none"
              />
            </div>
          ))}
        </div>
      )}

      {/* Request Body */}
      {endpoint.method !== 'GET' && (
        <div className="space-y-2">
          <h4 className="text-xs font-mono text-text-muted uppercase tracking-wider">
            Request Body {endpoint.requestBody?.required && <span className="text-red-400">*</span>}
          </h4>
          <textarea
            value={bodyValue}
            onChange={e => setBodyValue(e.target.value)}
            rows={8}
            className="w-full bg-black/30 border border-acid-green/30 px-3 py-2 text-sm font-mono text-acid-green focus:border-acid-green focus:outline-none resize-y"
            placeholder="JSON request body"
            spellCheck={false}
          />
        </div>
      )}

      {/* Headers Section */}
      <div className="space-y-2">
        <button
          type="button"
          onClick={() => setShowHeaders(!showHeaders)}
          className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
        >
          [{showHeaders ? '-' : '+'}] HEADERS
          {authToken && <span className="text-acid-green/60 ml-2">(auth token detected)</span>}
        </button>

        {showHeaders && (
          <div className="space-y-2 border border-acid-green/20 p-3 bg-black/20">
            {/* Auto-injected auth */}
            {authToken && (
              <div className="flex items-center gap-2 opacity-60">
                <span className="text-xs font-mono w-28 text-acid-green shrink-0">Authorization:</span>
                <span className="text-xs font-mono text-text truncate">Bearer {authToken.slice(0, 20)}...</span>
                <span className="text-xs font-mono text-acid-green/50 ml-auto">(auto)</span>
              </div>
            )}
            <div className="flex items-center gap-2 opacity-60">
              <span className="text-xs font-mono w-28 text-acid-green shrink-0">Content-Type:</span>
              <span className="text-xs font-mono text-text">application/json</span>
              <span className="text-xs font-mono text-acid-green/50 ml-auto">(auto)</span>
            </div>

            {/* Custom headers */}
            {customHeaders.map((h, idx) => (
              <div key={idx} className="flex items-center gap-2">
                <input
                  type="text"
                  value={h.key}
                  onChange={e => updateHeader(idx, 'key', e.target.value)}
                  placeholder="Header name"
                  className="w-28 bg-black/30 border border-acid-green/30 px-2 py-1 text-xs font-mono text-text focus:border-acid-green focus:outline-none shrink-0"
                />
                <input
                  type="text"
                  value={h.value}
                  onChange={e => updateHeader(idx, 'value', e.target.value)}
                  placeholder="Value"
                  className="flex-1 bg-black/30 border border-acid-green/30 px-2 py-1 text-xs font-mono text-text focus:border-acid-green focus:outline-none"
                />
                <button
                  type="button"
                  onClick={() => removeHeader(idx)}
                  className="text-red-400 hover:text-red-300 text-xs font-mono px-1"
                >
                  x
                </button>
              </div>
            ))}

            <button
              type="button"
              onClick={addHeader}
              className="text-xs font-mono text-acid-green/70 hover:text-acid-green transition-colors"
            >
              + Add Header
            </button>
          </div>
        )}
      </div>

      {/* URL Preview */}
      <div className="p-2 bg-black/30 border border-acid-green/20">
        <span className="text-xs font-mono text-text-muted">URL: </span>
        <code className="text-xs font-mono text-acid-cyan break-all">{buildUrl()}</code>
      </div>

      {/* Send Button */}
      <button
        type="submit"
        disabled={loading}
        className="px-6 py-2 bg-acid-green/20 border border-acid-green text-acid-green font-mono text-sm hover:bg-acid-green/30 transition-colors disabled:opacity-50"
      >
        {loading ? 'SENDING...' : `SEND ${endpoint.method}`}
      </button>

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30">
          <span className="text-sm font-mono text-red-400">{error}</span>
        </div>
      )}

      {/* Response */}
      {response && (
        <div className="space-y-3 border-t border-acid-green/20 pt-4">
          {/* Status line */}
          <div className="flex items-center gap-4 font-mono text-sm">
            <span className="text-text-muted">Status:</span>
            <span className={`font-bold ${getStatusColor(response.status)}`}>
              {response.status} {response.statusText}
            </span>
            <span className="text-text-muted text-xs ml-auto">{response.elapsed}ms</span>
          </div>

          {/* Response Headers toggle */}
          <button
            type="button"
            onClick={() => setShowResponseHeaders(!showResponseHeaders)}
            className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
          >
            [{showResponseHeaders ? '-' : '+'}] RESPONSE HEADERS ({Object.keys(response.headers).length})
          </button>

          {showResponseHeaders && (
            <div className="bg-black/30 border border-acid-green/20 p-3 max-h-40 overflow-y-auto">
              {Object.entries(response.headers).map(([key, value]) => (
                <div key={key} className="text-xs font-mono">
                  <span className="text-acid-cyan">{key}</span>
                  <span className="text-text-muted">: </span>
                  <span className="text-text">{value}</span>
                </div>
              ))}
            </div>
          )}

          {/* Response Body */}
          <div>
            <h4 className="text-xs font-mono text-text-muted uppercase tracking-wider mb-2">Response Body</h4>
            <pre className="p-3 bg-black/30 border border-acid-green/20 overflow-x-auto text-xs font-mono text-acid-green max-h-80 overflow-y-auto whitespace-pre-wrap">
              {typeof response.data === 'string'
                ? response.data
                : JSON.stringify(response.data, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </form>
  );
}

// ---------------------------------------------------------------------------
// ApiExplorerPanel -- main exported component
// ---------------------------------------------------------------------------

export function ApiExplorerPanel() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [selectedEndpoint, setSelectedEndpoint] = useState<Endpoint | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [baseUrl, setBaseUrl] = useState(
    typeof window !== 'undefined' ? window.location.origin : API_BASE_URL
  );

  const filteredCategories = useMemo(() => {
    if (!searchQuery) return API_CATALOG;

    const query = searchQuery.toLowerCase();
    return API_CATALOG.map(category => ({
      ...category,
      endpoints: category.endpoints.filter(
        ep =>
          ep.path.toLowerCase().includes(query) ||
          ep.summary.toLowerCase().includes(query) ||
          ep.method.toLowerCase().includes(query) ||
          ep.description?.toLowerCase().includes(query)
      ),
    })).filter(cat => cat.endpoints.length > 0);
  }, [searchQuery]);

  const totalEndpoints = API_CATALOG.reduce((sum, cat) => sum + cat.endpoints.length, 0);
  const filteredCount = filteredCategories.reduce((sum, cat) => sum + cat.endpoints.length, 0);

  const selectEndpoint = (endpoint: Endpoint, categoryName: string) => {
    setSelectedEndpoint(endpoint);
    setSelectedCategory(categoryName);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* ── Left Sidebar: Category Tree ────────────────────────────────── */}
      <div className="lg:col-span-4 space-y-4">
        {/* Search */}
        <div className="border border-acid-green/30 bg-surface/30 p-3">
          <div className="relative">
            <span className="absolute left-2 top-1/2 -translate-y-1/2 text-acid-green/50 font-mono text-sm">
              /
            </span>
            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search endpoints..."
              className="w-full bg-black/30 border border-acid-green/30 pl-6 pr-3 py-2 text-sm font-mono text-text focus:border-acid-green focus:outline-none"
            />
          </div>
          <p className="text-xs font-mono text-text-muted mt-2">
            {searchQuery ? `${filteredCount} of ${totalEndpoints}` : `${totalEndpoints}`} endpoints
          </p>
        </div>

        {/* Category Tree */}
        <div className="border border-acid-green/30 bg-surface/30 p-3 space-y-1 max-h-[calc(100vh-16rem)] overflow-y-auto">
          {filteredCategories.map(category => (
            <CollapsibleSection
              key={category.name}
              id={`api-cat-${category.name.toLowerCase().replace(/\s+/g, '-')}`}
              title={`${category.icon} ${category.name}`}
              defaultOpen={selectedCategory === category.name || !!searchQuery}
            >
              <p className="text-xs font-mono text-text-muted mb-2 px-1">
                {category.description}
              </p>
              <div className="space-y-0.5">
                {category.endpoints.map(endpoint => {
                  const isSelected =
                    selectedEndpoint?.path === endpoint.path &&
                    selectedEndpoint?.method === endpoint.method;
                  return (
                    <button
                      key={`${endpoint.method}-${endpoint.path}`}
                      onClick={() => selectEndpoint(endpoint, category.name)}
                      className={`w-full text-left px-2 py-1.5 transition-colors ${
                        isSelected
                          ? 'bg-acid-green/15 border-l-2 border-acid-green'
                          : 'hover:bg-acid-green/5 border-l-2 border-transparent'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className={`text-[10px] font-mono font-bold px-1 py-0.5 border ${METHOD_COLORS[endpoint.method]} shrink-0 w-12 text-center`}
                        >
                          {endpoint.method}
                        </span>
                        <span className="text-xs font-mono text-text truncate">
                          {endpoint.path}
                        </span>
                      </div>
                      <p className="text-[11px] text-text-muted mt-0.5 truncate pl-14">
                        {endpoint.summary}
                      </p>
                    </button>
                  );
                })}
              </div>
            </CollapsibleSection>
          ))}

          {filteredCategories.length === 0 && (
            <div className="text-center py-8">
              <p className="text-text-muted font-mono text-sm">No endpoints match your search</p>
            </div>
          )}
        </div>
      </div>

      {/* ── Main Panel: Request Builder + Response ─────────────────────── */}
      <div className="lg:col-span-8">
        {selectedEndpoint ? (
          <div className="border border-acid-green/30 bg-surface/30 p-6 space-y-6">
            {/* Endpoint Header */}
            <div>
              <div className="flex items-center gap-3 mb-2 flex-wrap">
                <span
                  className={`text-sm font-mono font-bold px-2 py-1 border ${METHOD_COLORS[selectedEndpoint.method]}`}
                >
                  {selectedEndpoint.method}
                </span>
                <code className="text-base font-mono text-acid-cyan break-all">
                  {selectedEndpoint.path}
                </code>
              </div>
              <h2 className="text-lg font-mono text-acid-green">{selectedEndpoint.summary}</h2>
              {selectedEndpoint.description && (
                <p className="text-sm text-text-muted mt-1 font-mono">{selectedEndpoint.description}</p>
              )}
              <div className="flex gap-4 mt-2 text-xs font-mono flex-wrap">
                {selectedEndpoint.auth && (
                  <span className="text-yellow-400 border border-yellow-400/30 px-1.5 py-0.5">AUTH REQUIRED</span>
                )}
                {selectedEndpoint.rateLimit && (
                  <span className="text-purple-400 border border-purple-400/30 px-1.5 py-0.5">RATE LIMIT: {selectedEndpoint.rateLimit}</span>
                )}
              </div>
            </div>

            {/* Base URL */}
            <div className="space-y-1">
              <label className="text-xs font-mono text-text-muted uppercase tracking-wider">Base URL</label>
              <input
                type="text"
                value={baseUrl}
                onChange={e => setBaseUrl(e.target.value)}
                className="w-full bg-black/30 border border-acid-green/30 px-3 py-2 text-sm font-mono text-text focus:border-acid-green focus:outline-none"
              />
            </div>

            {/* Request Builder */}
            <CollapsibleSection id="api-try-it" title="REQUEST BUILDER" defaultOpen={true}>
              <TryItForm endpoint={selectedEndpoint} baseUrl={baseUrl} />
            </CollapsibleSection>
          </div>
        ) : (
          <div className="border border-acid-green/30 bg-surface/30 flex items-center justify-center h-96">
            <div className="text-center space-y-3">
              <div className="text-4xl font-mono text-acid-green/30">{'{}'}</div>
              <p className="text-text-muted font-mono text-sm">
                Select an endpoint to explore
              </p>
              <p className="text-xs text-text-muted/70 font-mono">
                Browse categories on the left or search by path
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
