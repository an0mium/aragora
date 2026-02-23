'use client';

import useSWR from 'swr';
import { API_BASE_URL } from '@/config';

// --- Types ---

interface CircuitBreaker {
  name: string;
  state: 'closed' | 'open' | 'half_open';
  failure_count: number;
  last_failure: string | null;
  success_rate: number;
}

interface SLOEntry {
  name: string;
  target: number;
  current: number;
  compliant: boolean;
  burn_rate: number;
}

interface AgentEntry {
  agent_id: string;
  agent_type: string;
  status: 'active' | 'idle' | 'failed';
  last_heartbeat: string;
}

interface BudgetInfo {
  total_budget: number;
  spent: number;
  utilization: number;
  forecast: { eom: number; trend: 'increasing' | 'stable' | 'decreasing' };
}

interface SystemHealthResponse {
  data: {
    overall_status: 'healthy' | 'degraded' | 'critical';
    subsystems: Record<string, { status: string; detail: string }>;
    last_check: string;
  };
}

// --- Hooks ---

const fetcher = (url: string) => fetch(url).then(r => r.json());

export function useSystemHealth() {
  const { data, error, isLoading, mutate } = useSWR<SystemHealthResponse>(
    `${API_BASE_URL}/api/admin/system-health`,
    fetcher,
    { refreshInterval: 30000 }
  );
  return { overallStatus: data?.data?.overall_status ?? 'healthy', subsystems: data?.data?.subsystems ?? {}, lastCheck: data?.data?.last_check ?? '', loading: isLoading, error, refresh: mutate };
}

export function useCircuitBreakers() {
  const { data, error, isLoading } = useSWR<{ data: { breakers: CircuitBreaker[] } }>(
    `${API_BASE_URL}/api/admin/system-health/circuit-breakers`,
    fetcher,
    { refreshInterval: 15000 }
  );
  return { breakers: data?.data?.breakers ?? [], loading: isLoading, error };
}

export function useSLOStatus() {
  const { data, error, isLoading } = useSWR<{ data: { slos: SLOEntry[] } }>(
    `${API_BASE_URL}/api/admin/system-health/slos`,
    fetcher,
    { refreshInterval: 30000 }
  );
  return { slos: data?.data?.slos ?? [], loading: isLoading, error };
}

export function useAgentPoolHealth() {
  const { data, error, isLoading } = useSWR<{ data: { agents: AgentEntry[]; total: number; active: number } }>(
    `${API_BASE_URL}/api/admin/system-health/agents`,
    fetcher,
    { refreshInterval: 10000 }
  );
  return { agents: data?.data?.agents ?? [], total: data?.data?.total ?? 0, active: data?.data?.active ?? 0, loading: isLoading, error };
}

export function useBudgetStatus() {
  const { data, error, isLoading } = useSWR<{ data: BudgetInfo }>(
    `${API_BASE_URL}/api/admin/system-health/budget`,
    fetcher,
    { refreshInterval: 60000 }
  );
  return { budget: data?.data ?? null, loading: isLoading, error };
}
