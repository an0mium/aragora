'use client';

import { usePromptEngine } from '@/hooks/usePromptEngine';
import { StageNav } from '@/components/prompt-engine/StageNav';
import { PromptInput } from '@/components/prompt-engine/PromptInput';
import { RefinementFlow } from '@/components/prompt-engine/RefinementFlow';
import { ResearchGrid } from '@/components/prompt-engine/ResearchGrid';
import { SpecEditor } from '@/components/prompt-engine/SpecEditor';
import { ProvenanceBar } from '@/components/prompt-engine/ProvenanceBar';

export default function PromptEnginePage() {
  const engine = usePromptEngine();

  return (
    <div className="flex h-full">
      <StageNav currentStage={engine.stage} />
      <main className="flex-1 p-6 overflow-y-auto">
        {!engine.session && (
          <PromptInput onSubmit={engine.createSession} />
        )}
        {engine.stage === 'interrogate' && engine.questions.length > 0 && (
          <RefinementFlow
            questions={engine.questions}
            answers={engine.session?.answers ?? {}}
            onAnswer={engine.answerQuestion}
          />
        )}
        {engine.stage === 'research' && (
          <ResearchGrid progress={engine.researchProgress} />
        )}
        {engine.stage === 'spec' && engine.spec && (
          <SpecEditor
            spec={engine.spec}
            confidence={engine.confidence}
            onApprove={engine.approveSpec}
            onSkipValidation={engine.skipValidation}
          />
        )}
        {engine.error && (
          <div className="mt-4 p-3 border border-[var(--crimson)] text-[var(--crimson)] font-mono text-sm">
            {engine.error}
          </div>
        )}
      </main>
      <ProvenanceBar hash={engine.session?.provenance_hash ?? null} stage={engine.stage} />
    </div>
  );
}
