'use client';

import { ProofVisualizerPanel } from '@/components/ProofVisualizerPanel';
import { useBackend } from '@/components/BackendSelector';

export default function VerificationPage() {
  const { config } = useBackend();

  const backendConfig = {
    apiUrl: config.api,
    wsUrl: config.ws,
  };

  return (
    <div className="min-h-screen bg-bg text-text p-6">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-mono text-acid-green mb-2">
            {'>'} FORMAL_VERIFICATION
          </h1>
          <p className="text-text-muted font-mono text-sm">
            Verify claims using Z3 SMT solver or Lean4 theorem prover. Translate natural language to formal proofs.
          </p>
        </header>

        <ProofVisualizerPanel backendConfig={backendConfig} />
      </div>
    </div>
  );
}
