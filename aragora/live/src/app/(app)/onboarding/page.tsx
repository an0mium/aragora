'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import {
  useOnboardingStore,
  selectCurrentStepIndex,
  selectIsOnboardingNeeded,
  type OnboardingStep,
} from '@/store/onboardingStore';
import { UseCaseSelector } from '@/components/onboarding/UseCaseSelector';
import { QuickDebatePanel } from '@/components/onboarding/QuickDebatePanel';

const VISIBLE_STEPS: { key: OnboardingStep; label: string }[] = [
  { key: 'welcome', label: 'Welcome' },
  { key: 'organization', label: 'Setup' },
  { key: 'template-select', label: 'Template' },
  { key: 'first-debate', label: 'Debate' },
  { key: 'completion', label: 'Done' },
];

export default function OnboardingPage() {
  const router = useRouter();
  const currentStep = useOnboardingStore((s) => s.currentStep);
  const isComplete = useOnboardingStore((s) => s.isComplete);
  const isSkipped = useOnboardingStore((s) => s.isSkipped);
  const organizationName = useOnboardingStore((s) => s.organizationName);
  const teamSize = useOnboardingStore((s) => s.teamSize);
  const selectedTemplate = useOnboardingStore((s) => s.selectedTemplate);
  const useCase = useOnboardingStore((s) => s.useCase);

  const setOrganizationName = useOnboardingStore((s) => s.setOrganizationName);
  const setTeamSize = useOnboardingStore((s) => s.setTeamSize);
  const nextStep = useOnboardingStore((s) => s.nextStep);
  const previousStep = useOnboardingStore((s) => s.previousStep);
  const skipOnboarding = useOnboardingStore((s) => s.skipOnboarding);
  const completeOnboarding = useOnboardingStore((s) => s.completeOnboarding);
  const setCurrentStep = useOnboardingStore((s) => s.setCurrentStep);

  const stepIndex = useOnboardingStore(selectCurrentStepIndex);

  // If already done, go to dashboard
  useEffect(() => {
    if (isComplete || isSkipped) {
      router.push('/');
    }
  }, [isComplete, isSkipped, router]);

  const handleSkip = () => {
    skipOnboarding();
    router.push('/');
  };

  // Map current store step to visible step index
  const visibleIndex = VISIBLE_STEPS.findIndex((s) => s.key === currentStep);
  const progressPercent =
    visibleIndex >= 0 ? ((visibleIndex + 1) / VISIBLE_STEPS.length) * 100 : 0;

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] relative z-10">
        <div className="container mx-auto px-4 py-6 max-w-2xl">
          {/* Progress Bar */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono text-[var(--acid-green)]">
                {'>'} ONBOARDING
              </span>
              <button
                onClick={handleSkip}
                className="text-xs font-mono text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
              >
                SKIP
              </button>
            </div>
            <div className="h-1 bg-[var(--surface)] border border-[var(--border)]">
              <div
                className="h-full bg-[var(--acid-green)] transition-all duration-500"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
            <div className="flex justify-between mt-2">
              {VISIBLE_STEPS.map((step, i) => (
                <span
                  key={step.key}
                  className={`text-[10px] font-mono ${
                    i <= visibleIndex
                      ? 'text-[var(--acid-green)]'
                      : 'text-[var(--text-muted)]'
                  }`}
                >
                  {step.label}
                </span>
              ))}
            </div>
          </div>

          {/* Step Content */}
          {currentStep === 'welcome' && (
            <div className="space-y-6">
              <div className="bg-[var(--surface)] border border-[var(--acid-green)]/40 p-6">
                <h1 className="text-xl font-mono text-[var(--acid-green)] mb-4">
                  {'>'} WELCOME TO ARAGORA
                </h1>
                <p className="text-sm font-mono text-[var(--text)] mb-4 leading-relaxed">
                  Aragora orchestrates AI agents to adversarially vet your decisions.
                  Multiple models debate, critique, and converge on the best answer --
                  then deliver an audit-ready receipt.
                </p>
                <p className="text-xs font-mono text-[var(--text-muted)]">
                  Let us set you up in under 2 minutes.
                </p>
              </div>

              <UseCaseSelector />

              <div className="flex justify-end">
                <button
                  onClick={nextStep}
                  className="px-6 py-2 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/40 hover:bg-[var(--acid-green)]/20 transition-colors"
                >
                  CONTINUE
                </button>
              </div>
            </div>
          )}

          {currentStep === 'organization' && (
            <div className="space-y-6">
              <div className="bg-[var(--surface)] border border-[var(--border)] p-6">
                <h2 className="text-lg font-mono text-[var(--acid-green)] mb-4">
                  {'>'} YOUR ORGANIZATION
                </h2>
                <div className="space-y-4">
                  <div>
                    <label className="block text-xs font-mono text-[var(--text-muted)] mb-2">
                      ORGANIZATION NAME
                    </label>
                    <input
                      type="text"
                      value={organizationName}
                      onChange={(e) => setOrganizationName(e.target.value)}
                      placeholder="Acme Corp"
                      className="w-full px-3 py-2 text-sm font-mono bg-[var(--bg)] text-[var(--text)] border border-[var(--border)] focus:border-[var(--acid-green)]/60 focus:outline-none transition-colors"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-mono text-[var(--text-muted)] mb-2">
                      TEAM SIZE
                    </label>
                    <div className="grid grid-cols-4 gap-2">
                      {(['1-5', '6-15', '16-50', '50+'] as const).map((size) => (
                        <button
                          key={size}
                          onClick={() => setTeamSize(size)}
                          className={`px-3 py-2 text-xs font-mono border transition-colors ${
                            teamSize === size
                              ? 'bg-[var(--acid-green)]/20 text-[var(--acid-green)] border-[var(--acid-green)]/40'
                              : 'bg-[var(--surface)] text-[var(--text-muted)] border-[var(--border)] hover:border-[var(--acid-green)]/30'
                          }`}
                        >
                          {size}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex justify-between">
                <button
                  onClick={previousStep}
                  className="px-4 py-2 text-xs font-mono text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
                >
                  BACK
                </button>
                <button
                  onClick={() => {
                    // Skip team-invite, go directly to template-select
                    nextStep(); // org -> team-invite
                    // Use setTimeout to allow the first state update
                    setTimeout(() => {
                      setCurrentStep('template-select');
                    }, 0);
                  }}
                  disabled={organizationName.length < 3 || !teamSize}
                  className="px-6 py-2 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/40 hover:bg-[var(--acid-green)]/20 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  CONTINUE
                </button>
              </div>
            </div>
          )}

          {(currentStep === 'team-invite' || currentStep === 'template-select') && (
            <div className="space-y-6">
              <div className="bg-[var(--surface)] border border-[var(--border)] p-6">
                <h2 className="text-lg font-mono text-[var(--acid-green)] mb-4">
                  {'>'} PICK A TEMPLATE
                </h2>
                <p className="text-xs font-mono text-[var(--text-muted)] mb-4">
                  Choose a debate template to start with. You can customize it later.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {SME_TEMPLATES.map((tmpl) => (
                    <button
                      key={tmpl.id}
                      onClick={() =>
                        useOnboardingStore.getState().setSelectedTemplate(tmpl)
                      }
                      className={`p-4 text-left border transition-colors ${
                        selectedTemplate?.id === tmpl.id
                          ? 'bg-[var(--acid-green)]/10 border-[var(--acid-green)]/40'
                          : 'bg-[var(--surface)] border-[var(--border)] hover:border-[var(--acid-green)]/30'
                      }`}
                    >
                      <div className="text-sm font-mono text-[var(--acid-green)] mb-1">
                        {tmpl.name}
                      </div>
                      <div className="text-xs font-mono text-[var(--text-muted)]">
                        {tmpl.description}
                      </div>
                      <div className="text-[10px] font-mono text-[var(--text-muted)] mt-2">
                        {tmpl.agentsCount} agents / {tmpl.rounds} rounds / ~{tmpl.estimatedDurationMinutes}min
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex justify-between">
                <button
                  onClick={() => setCurrentStep('organization')}
                  className="px-4 py-2 text-xs font-mono text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
                >
                  BACK
                </button>
                <button
                  onClick={() => setCurrentStep('first-debate')}
                  disabled={!selectedTemplate}
                  className="px-6 py-2 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/40 hover:bg-[var(--acid-green)]/20 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  CONTINUE
                </button>
              </div>
            </div>
          )}

          {currentStep === 'first-debate' && (
            <div className="space-y-6">
              <QuickDebatePanel />

              <div className="flex justify-between">
                <button
                  onClick={() => setCurrentStep('template-select')}
                  className="px-4 py-2 text-xs font-mono text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
                >
                  BACK
                </button>
                <button
                  onClick={() => {
                    completeOnboarding();
                    router.push('/');
                  }}
                  className="px-6 py-2 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/40 hover:bg-[var(--acid-green)]/20 transition-colors"
                >
                  FINISH SETUP
                </button>
              </div>
            </div>
          )}

          {currentStep === 'completion' && (
            <div className="space-y-6">
              <div className="bg-[var(--surface)] border border-[var(--acid-green)]/40 p-6 text-center">
                <div className="text-2xl font-mono text-[var(--acid-green)] mb-4">
                  {'>'} SETUP COMPLETE
                </div>
                <p className="text-sm font-mono text-[var(--text)] mb-2">
                  Your workspace is ready. Start making better decisions with AI adversarial debate.
                </p>
                <p className="text-xs font-mono text-[var(--text-muted)] mb-6">
                  You can always adjust settings from the dashboard.
                </p>
                <Link
                  href="/"
                  className="inline-block px-6 py-2 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/40 hover:bg-[var(--acid-green)]/20 transition-colors"
                >
                  GO TO DASHBOARD
                </Link>
              </div>
            </div>
          )}
        </div>
      </main>
    </>
  );
}

// Built-in SME templates
const SME_TEMPLATES = [
  {
    id: 'hiring',
    name: 'Hiring Decision',
    description: 'Evaluate candidates with structured debate',
    agentsCount: 5,
    rounds: 5,
    estimatedDurationMinutes: 3,
  },
  {
    id: 'contract-review',
    name: 'Contract Review',
    description: 'Identify risks and negotiate terms',
    agentsCount: 4,
    rounds: 5,
    estimatedDurationMinutes: 3,
  },
  {
    id: 'budget',
    name: 'Budget Allocation',
    description: 'Optimize spend across departments',
    agentsCount: 5,
    rounds: 7,
    estimatedDurationMinutes: 4,
  },
  {
    id: 'feature-priority',
    name: 'Feature Prioritization',
    description: 'Rank features by impact and effort',
    agentsCount: 5,
    rounds: 5,
    estimatedDurationMinutes: 3,
  },
  {
    id: 'vendor-selection',
    name: 'Vendor Selection',
    description: 'Compare vendors on criteria that matter',
    agentsCount: 4,
    rounds: 5,
    estimatedDurationMinutes: 3,
  },
  {
    id: 'compliance',
    name: 'Compliance Check',
    description: 'Assess regulatory compliance gaps',
    agentsCount: 4,
    rounds: 5,
    estimatedDurationMinutes: 3,
  },
];
