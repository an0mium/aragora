'use client';

interface WelcomeStepProps {
  onNext: () => void;
  onSkip: () => void;
}

export function WelcomeStep({ onNext, onSkip }: WelcomeStepProps) {
  return (
    <div className="space-y-6">
      <div className="text-center">
        <div className="inline-block p-4 bg-acid-green/10 border border-acid-green/30 mb-4">
          <span className="text-4xl">&#x2694;</span>
        </div>
        <h2 className="text-2xl font-mono text-acid-green mb-2">
          Welcome to Aragora
        </h2>
        <p className="font-mono text-text-muted text-sm">
          The control plane for multi-agent vetted decisionmaking
        </p>
      </div>

      <div className="space-y-4">
        <div className="p-4 bg-surface border border-acid-green/20">
          <h3 className="font-mono text-acid-cyan text-sm mb-2">
            How it works
          </h3>
          <ol className="space-y-2 font-mono text-text-muted text-sm">
            <li className="flex items-start gap-2">
              <span className="text-acid-green">1.</span>
              <span>Ask a question or describe a decision</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-acid-green">2.</span>
              <span>Multiple AI agents debate your topic</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-acid-green">3.</span>
              <span>Watch consensus emerge with full reasoning</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-acid-green">4.</span>
              <span>Get a defensible decision with audit trail</span>
            </li>
          </ol>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 bg-acid-green/5 border border-acid-green/20 text-center">
            <div className="text-2xl font-mono text-acid-green mb-1">15+</div>
            <div className="text-xs font-mono text-text-muted">AI Models</div>
          </div>
          <div className="p-3 bg-acid-green/5 border border-acid-green/20 text-center">
            <div className="text-2xl font-mono text-acid-green mb-1">100%</div>
            <div className="text-xs font-mono text-text-muted">Auditable</div>
          </div>
        </div>
      </div>

      <div className="flex gap-3 pt-4">
        <button
          onClick={onSkip}
          className="px-4 py-2 font-mono text-sm text-text-muted hover:text-acid-green transition-colors"
        >
          Skip tutorial
        </button>
        <div className="flex-1" />
        <button
          onClick={onNext}
          className="px-6 py-2 font-mono text-sm bg-acid-green text-bg hover:bg-acid-green/80 transition-colors"
        >
          Get started
        </button>
      </div>
    </div>
  );
}

export default WelcomeStep;
