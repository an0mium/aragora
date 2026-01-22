'use client';

import { useState } from 'react';

interface UsageData {
  period: string;
  apiCalls: number;
  tokens: number;
  debates: number;
  storage: number;
}

interface TenantUsage {
  tenantId: string;
  tenantName: string;
  tier: string;
  currentPeriod: UsageData;
  history: UsageData[];
  projectedCost: number;
}

function UsageChart({ data, label }: { data: number[]; label: string }) {
  const max = Math.max(...data, 1);
  return (
    <div className="space-y-1">
      <div className="text-sm font-medium text-gray-700">{label}</div>
      <div className="flex items-end gap-1 h-20">
        {data.map((value, i) => (
          <div key={i} className="flex-1 bg-blue-500 rounded-t" style={{ height: (value / max * 100) + '%' }} title={String(value)} />
        ))}
      </div>
    </div>
  );
}

function MetricCard({ title, value, change, unit }: { title: string; value: number; change?: number; unit?: string }) {
  const changeClass = change !== undefined && change >= 0 ? 'text-green-600' : 'text-red-600';
  const changeText = change !== undefined ? (change >= 0 ? '+' : '') + change + '% from last period' : '';
  return (
    <div className="bg-white p-4 rounded-lg border">
      <div className="text-sm text-gray-500">{title}</div>
      <div className="text-2xl font-bold">{value.toLocaleString()}{unit && <span className="text-sm font-normal text-gray-500 ml-1">{unit}</span>}</div>
      {change !== undefined && <div className={'text-sm ' + changeClass}>{changeText}</div>}
    </div>
  );
}

export default function UsageDashboard() {
  const [period, setPeriod] = useState<'day' | 'week' | 'month'>('month');
  const tenantUsage: TenantUsage[] = [
    { tenantId: 'tenant-001', tenantName: 'Acme Corporation', tier: 'enterprise', currentPeriod: { period: '2026-01', apiCalls: 85000, tokens: 2500000, debates: 450, storage: 5368709120 }, history: [{ period: '2025-10', apiCalls: 72000, tokens: 2100000, debates: 380, storage: 4294967296 }, { period: '2025-11', apiCalls: 78000, tokens: 2300000, debates: 410, storage: 4831838208 }, { period: '2025-12', apiCalls: 82000, tokens: 2400000, debates: 430, storage: 5100000000 }, { period: '2026-01', apiCalls: 85000, tokens: 2500000, debates: 450, storage: 5368709120 }], projectedCost: 4250 },
    { tenantId: 'tenant-002', tenantName: 'Startup Inc', tier: 'standard', currentPeriod: { period: '2026-01', apiCalls: 4500, tokens: 150000, debates: 25, storage: 268435456 }, history: [{ period: '2025-10', apiCalls: 2000, tokens: 80000, debates: 12, storage: 134217728 }, { period: '2025-11', apiCalls: 3000, tokens: 100000, debates: 18, storage: 180000000 }, { period: '2025-12', apiCalls: 3800, tokens: 130000, debates: 22, storage: 220000000 }, { period: '2026-01', apiCalls: 4500, tokens: 150000, debates: 25, storage: 268435456 }], projectedCost: 450 },
  ];

  const totalApiCalls = tenantUsage.reduce((sum, t) => sum + t.currentPeriod.apiCalls, 0);
  const totalTokens = tenantUsage.reduce((sum, t) => sum + t.currentPeriod.tokens, 0);
  const totalDebates = tenantUsage.reduce((sum, t) => sum + t.currentPeriod.debates, 0);
  const totalRevenue = tenantUsage.reduce((sum, t) => sum + t.projectedCost, 0);

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="flex justify-between items-center mb-6">
        <div><h1 className="text-2xl font-bold">Usage Dashboard</h1><p className="text-gray-500">Monitor platform usage and billing</p></div>
        <select value={period} onChange={(e) => setPeriod(e.target.value as typeof period)} className="px-4 py-2 border rounded-md">
          <option value="day">Last 24 Hours</option><option value="week">Last 7 Days</option><option value="month">Last 30 Days</option>
        </select>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <MetricCard title="Total API Calls" value={totalApiCalls} change={12} />
        <MetricCard title="Total Tokens" value={totalTokens} change={8} />
        <MetricCard title="Total Debates" value={totalDebates} change={15} />
        <MetricCard title="Projected Revenue" value={totalRevenue} unit="USD" change={10} />
      </div>
      <div className="bg-white rounded-lg border p-6 mb-8">
        <h2 className="text-lg font-semibold mb-4">Usage by Tenant</h2>
        <table className="w-full">
          <thead><tr className="border-b text-left text-sm text-gray-500"><th className="pb-3">Tenant</th><th className="pb-3">Tier</th><th className="pb-3">API Calls</th><th className="pb-3">Tokens</th><th className="pb-3">Debates</th><th className="pb-3">Cost</th></tr></thead>
          <tbody>
            {tenantUsage.map(tenant => (
              <tr key={tenant.tenantId} className="border-b">
                <td className="py-4 font-medium">{tenant.tenantName}</td>
                <td className="py-4"><span className={'px-2 py-1 rounded text-xs ' + (tenant.tier === 'enterprise' ? 'bg-purple-100' : 'bg-blue-100')}>{tenant.tier}</span></td>
                <td className="py-4">{tenant.currentPeriod.apiCalls.toLocaleString()}</td>
                <td className="py-4">{tenant.currentPeriod.tokens.toLocaleString()}</td>
                <td className="py-4">{tenant.currentPeriod.debates}</td>
                <td className="py-4 font-medium">${tenant.projectedCost}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg border p-6"><h2 className="text-lg font-semibold mb-4">API Calls Trend</h2><UsageChart data={[72000, 78000, 82000, 85000, 89000, 92000]} label="" /></div>
        <div className="bg-white rounded-lg border p-6"><h2 className="text-lg font-semibold mb-4">Token Usage Trend</h2><UsageChart data={[2100000, 2300000, 2400000, 2500000, 2600000, 2700000]} label="" /></div>
      </div>
    </div>
  );
}
