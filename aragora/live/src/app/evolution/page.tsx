'use client';

import { EvolutionPanel } from '@/components/EvolutionPanel';
import { useBackendConfig } from '@/hooks/useBackendConfig';

export default function EvolutionPage() {
  const backendConfig = useBackendConfig();

  return (
    <div className="min-h-screen bg-bg text-text p-6">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-mono text-acid-green mb-2">
            {'>'} EVOLUTION_DASHBOARD
          </h1>
          <p className="text-text-muted font-mono text-sm">
            Monitor genetic evolution, agent breeding, and prompt optimization across the system.
          </p>
        </header>

        <EvolutionPanel backendConfig={backendConfig} />
      </div>
    </div>
  );
}
