'use client';

import { useState, useEffect, useCallback } from 'react';

interface ServiceStatus {
  name: string;
  status: 'operational' | 'degraded' | 'outage' | 'unknown';
  latency?: number;
  lastChecked: string;
  details?: string;
}

interface HealthResponse {
  status: string;
  version?: string;
  uptime?: number;
  checks?: Record<string, { status: string; latency_ms?: number; error?: string }>;
  timestamp?: string;
}

const STATUS_COLORS = {
  operational: 'bg-green-500',
  degraded: 'bg-yellow-500',
  outage: 'bg-red-500',
  unknown: 'bg-gray-500',
};

const STATUS_LABELS = {
  operational: 'Operational',
  degraded: 'Degraded',
  outage: 'Outage',
  unknown: 'Unknown',
};

function ServiceCard({ service }: { service: ServiceStatus }) {
  return (
    <div className="flex items-center justify-between p-4 bg-[#1a1a1a] rounded-lg border border-[#333]">
      <div className="flex items-center gap-3">
        <div className={`w-3 h-3 rounded-full ${STATUS_COLORS[service.status]}`} />
        <div>
          <div className="font-mono text-sm text-white">{service.name}</div>
          {service.details && (
            <div className="text-xs text-gray-500">{service.details}</div>
          )}
        </div>
      </div>
      <div className="text-right">
        <div className={`text-sm font-mono ${
          service.status === 'operational' ? 'text-green-400' :
          service.status === 'degraded' ? 'text-yellow-400' :
          service.status === 'outage' ? 'text-red-400' : 'text-gray-400'
        }`}>
          {STATUS_LABELS[service.status]}
        </div>
        {service.latency !== undefined && (
          <div className="text-xs text-gray-500">{service.latency}ms</div>
        )}
      </div>
    </div>
  );
}

export default function StatusPage() {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [overallStatus, setOverallStatus] = useState<'operational' | 'degraded' | 'outage' | 'unknown'>('unknown');
  const [lastUpdated, setLastUpdated] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';

  const checkHealth = useCallback(async () => {
    try {
      const startTime = Date.now();
      const response = await fetch(`${API_URL}/api/health/detailed`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });
      const latency = Date.now() - startTime;

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data: HealthResponse = await response.json();
      const now = new Date().toISOString();

      // Map health checks to services
      const serviceList: ServiceStatus[] = [
        {
          name: 'API Server',
          status: data.status === 'healthy' ? 'operational' :
                  data.status === 'degraded' ? 'degraded' : 'outage',
          latency,
          lastChecked: now,
          details: data.version ? `v${data.version}` : undefined,
        },
      ];

      // Add individual service checks if available
      if (data.checks) {
        Object.entries(data.checks).forEach(([name, check]) => {
          serviceList.push({
            name: name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
            status: check.status === 'healthy' || check.status === 'ok' ? 'operational' :
                    check.status === 'degraded' ? 'degraded' : 'outage',
            latency: check.latency_ms,
            lastChecked: now,
            details: check.error,
          });
        });
      }

      setServices(serviceList);
      setLastUpdated(now);
      setError(null);

      // Calculate overall status
      const hasOutage = serviceList.some(s => s.status === 'outage');
      const hasDegraded = serviceList.some(s => s.status === 'degraded');
      setOverallStatus(hasOutage ? 'outage' : hasDegraded ? 'degraded' : 'operational');

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to check status');
      setOverallStatus('unknown');
      setServices([{
        name: 'API Server',
        status: 'unknown',
        lastChecked: new Date().toISOString(),
        details: 'Unable to reach API',
      }]);
    } finally {
      setLoading(false);
    }
  }, [API_URL]);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, [checkHealth]);

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white">
      {/* Header */}
      <header className="border-b border-[#333] bg-[#111]">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="text-2xl font-mono text-[#00ff88]">ARAGORA</div>
              <div className="text-sm text-gray-500">System Status</div>
            </div>
            <a
              href="https://aragora.ai"
              className="text-sm text-[#00ff88] hover:underline"
            >
              ← Back to Aragora
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Overall Status Banner */}
        <div className={`p-6 rounded-lg mb-8 ${
          overallStatus === 'operational' ? 'bg-green-900/20 border border-green-500/30' :
          overallStatus === 'degraded' ? 'bg-yellow-900/20 border border-yellow-500/30' :
          overallStatus === 'outage' ? 'bg-red-900/20 border border-red-500/30' :
          'bg-gray-900/20 border border-gray-500/30'
        }`}>
          <div className="flex items-center gap-4">
            <div className={`w-4 h-4 rounded-full ${STATUS_COLORS[overallStatus]} ${
              overallStatus !== 'operational' ? 'animate-pulse' : ''
            }`} />
            <div>
              <div className={`text-xl font-mono ${
                overallStatus === 'operational' ? 'text-green-400' :
                overallStatus === 'degraded' ? 'text-yellow-400' :
                overallStatus === 'outage' ? 'text-red-400' : 'text-gray-400'
              }`}>
                {loading ? 'Checking...' :
                 overallStatus === 'operational' ? 'All Systems Operational' :
                 overallStatus === 'degraded' ? 'Partial System Degradation' :
                 overallStatus === 'outage' ? 'System Outage Detected' :
                 'Status Unknown'}
              </div>
              {lastUpdated && (
                <div className="text-sm text-gray-500">
                  Last updated: {new Date(lastUpdated).toLocaleString()}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="p-4 mb-6 bg-red-900/20 border border-red-500/30 rounded-lg">
            <div className="text-red-400 font-mono text-sm">{error}</div>
          </div>
        )}

        {/* Services Grid */}
        <div className="mb-8">
          <h2 className="text-lg font-mono text-gray-400 mb-4">Services</h2>
          <div className="space-y-3">
            {loading ? (
              <div className="p-4 bg-[#1a1a1a] rounded-lg border border-[#333] animate-pulse">
                <div className="h-4 bg-gray-700 rounded w-1/4 mb-2" />
                <div className="h-3 bg-gray-700 rounded w-1/6" />
              </div>
            ) : (
              services.map((service, index) => (
                <ServiceCard key={index} service={service} />
              ))
            )}
          </div>
        </div>

        {/* Refresh Button */}
        <div className="flex justify-center">
          <button
            onClick={() => { setLoading(true); checkHealth(); }}
            disabled={loading}
            className="px-4 py-2 bg-[#1a1a1a] border border-[#333] rounded-lg text-sm font-mono text-gray-400 hover:border-[#00ff88] hover:text-[#00ff88] transition-colors disabled:opacity-50"
          >
            {loading ? 'Checking...' : 'Refresh Status'}
          </button>
        </div>

        {/* Footer Info */}
        <div className="mt-12 pt-8 border-t border-[#333] text-center">
          <div className="text-sm text-gray-500 space-y-2">
            <p>Status updates every 30 seconds</p>
            <p>
              For incident reports, contact{' '}
              <a href="mailto:support@aragora.ai" className="text-[#00ff88] hover:underline">
                support@aragora.ai
              </a>
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
