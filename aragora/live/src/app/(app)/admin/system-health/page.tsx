'use client';

import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { SystemHealthSummary } from '@/components/system-health/SystemHealthSummary';
import { CircuitBreakerGrid } from '@/components/system-health/CircuitBreakerGrid';
import { SLOStatusCards } from '@/components/system-health/SLOStatusCards';
import { AgentPoolHealth } from '@/components/system-health/AgentPoolHealth';
import { BudgetGauge } from '@/components/system-health/BudgetGauge';

export default function SystemHealthPage() {
  return (
    <div className="relative min-h-screen p-6 space-y-6">
      <Scanlines />
      <CRTVignette />

      <div className="relative z-10">
        <h1 className="text-xl font-mono text-[var(--acid-green)] mb-6">
          System Health Dashboard
        </h1>

        {/* Overall status banner + subsystem cards */}
        <SystemHealthSummary />

        {/* Circuit breakers + SLO compliance */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <CircuitBreakerGrid />
          <SLOStatusCards />
        </div>

        {/* Agent pool + Budget gauge */}
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <AgentPoolHealth />
          <BudgetGauge />
        </div>
      </div>
    </div>
  );
}
