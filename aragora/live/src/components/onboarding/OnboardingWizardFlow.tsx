'use client';

import { useCallback } from 'react';
import Link from 'next/link';
import { useOnboardingStore, useOnboardingStep } from '@/store';
import { useAuth } from '@/context/AuthContext';
import { IndustryStep } from './steps/IndustryStep';
import { TryDebateStep } from './steps/TryDebateStep';
import { IntegrationSelector } from './IntegrationSelector';
import { ProgressBar } from './ProgressBar';

interface OnboardingWizardFlowProps {
  onComplete?: () => void;
  onSkip?: () => void;
}

const STEP_LABELS: Record<string, string> = {
  industry: '1. INDUSTRY',
  'try-debate': '2. TRY IT',
  'create-account': '3. ACCOUNT',
  'connect-tools': '4. CONNECT',
  launch: '5. LAUNCH',
};

/**
 * Progressive commitment onboarding flow.
 * Steps 1-2 require no auth. Step 3 is the auth transition.
 * Steps 4-5 require auth.
 */
export function OnboardingWizardFlow({ onComplete, onSkip }: OnboardingWizardFlowProps) {
  const {
    currentStep,
    isFirstStep,
    isLastStep,
    canProceed,
    stepIndex,
    totalSteps,
  } = useOnboardingStep();

  const { isAuthenticated } = useAuth();

  const {
    selectedIndustry,
    trialDebateResult,
    nextStep,
    previousStep,
    skipOnboarding,
    completeOnboarding,
  } = useOnboardingStore();

  const handleNext = useCallback(() => {
    if (isLastStep) {
      completeOnboarding();
      onComplete?.();
    } else {
      nextStep();
    }
  }, [isLastStep, completeOnboarding, onComplete, nextStep]);

  const handleBack = useCallback(() => {
    previousStep();
  }, [previousStep]);

  const handleSkip = useCallback(() => {
    skipOnboarding();
    onSkip?.();
  }, [skipOnboarding, onSkip]);

  const renderStep = () => {
    switch (currentStep) {
      case 'industry':
        return <IndustryStep />;
      case 'try-debate':
        return <TryDebateStep />;
      case 'create-account':
        return <CreateAccountStep isAuthenticated={isAuthenticated} />;
      case 'connect-tools':
        return (
          <IntegrationSelector
            onComplete={() => nextStep()}
            onSkip={() => nextStep()}
          />
        );
      case 'launch':
        return (
          <LaunchStep
            selectedIndustry={selectedIndustry}
            trialDebateResult={trialDebateResult}
            onComplete={onComplete}
          />
        );
      default:
        return <IndustryStep />;
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-bg/95 backdrop-blur-sm">
      <div className="w-full max-w-2xl mx-4 border border-acid-green/30 bg-surface rounded-lg overflow-hidden">
        {/* Header */}
        <div className="border-b border-acid-green/20 px-6 py-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-mono text-acid-green">GET STARTED WITH ARAGORA</h2>
            <button
              onClick={handleSkip}
              className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
            >
              [SKIP FOR NOW]
            </button>
          </div>

          {/* Step indicator pills */}
          <div className="flex items-center gap-1 mb-2">
            {Object.entries(STEP_LABELS).map(([key, label], i) => (
              <div
                key={key}
                className={`flex-1 text-center text-[10px] font-mono py-1 border-b-2 transition-colors ${
                  i < stepIndex
                    ? 'text-acid-green border-acid-green'
                    : i === stepIndex
                    ? 'text-acid-green border-acid-green'
                    : 'text-text-muted border-border'
                }`}
              >
                {label}
              </div>
            ))}
          </div>

          <ProgressBar current={stepIndex + 1} total={totalSteps} />
        </div>

        {/* Content */}
        <div className="p-6 min-h-[300px] max-h-[60vh] overflow-y-auto">
          {renderStep()}
        </div>

        {/* Footer */}
        {currentStep !== 'launch' && (
          <div className="border-t border-acid-green/20 px-6 py-4 flex items-center justify-between">
            <button
              onClick={handleBack}
              disabled={isFirstStep}
              className="px-4 py-2 text-sm font-mono text-text-muted hover:text-acid-green disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              ← BACK
            </button>

            {/* For create-account step, show login link instead of continue when not authed */}
            {currentStep === 'create-account' && !isAuthenticated ? (
              <Link
                href="/auth/register"
                className="px-6 py-2 bg-acid-green text-bg font-mono text-sm hover:bg-acid-green/90 transition-colors inline-block text-center"
              >
                CREATE ACCOUNT →
              </Link>
            ) : (
              <button
                onClick={handleNext}
                disabled={!canProceed}
                className="px-6 py-2 bg-acid-green text-bg font-mono text-sm hover:bg-acid-green/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isLastStep ? 'FINISH' : 'CONTINUE →'}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Step 3: Create Account -- auth transition.
 * Shows value proposition and links to register/login.
 */
function CreateAccountStep({ isAuthenticated }: { isAuthenticated: boolean }) {
  if (isAuthenticated) {
    return (
      <div className="space-y-4 text-center py-6">
        <div className="text-3xl">&#10003;</div>
        <h2 className="text-lg font-mono text-[var(--acid-green)]">
          You&apos;re Signed In
        </h2>
        <p className="text-sm font-mono text-[var(--text-muted)]">
          Great -- you&apos;re ready to connect your tools and launch a real debate.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-mono text-[var(--acid-green)] mb-2">
          Save Your Results
        </h2>
        <p className="text-sm font-mono text-[var(--text-muted)]">
          Create a free account to unlock real AI models and save your debate history.
        </p>
      </div>

      <div className="space-y-3">
        {[
          'Real AI models (Claude, GPT, Mistral, Gemini)',
          'Multi-round adversarial debates',
          'Audit-ready decision receipts',
          'Knowledge Mound integration',
          'Team collaboration and sharing',
        ].map((feature) => (
          <div key={feature} className="flex items-center gap-2 text-sm font-mono">
            <span className="text-[var(--acid-green)]">+</span>
            <span className="text-[var(--text)]">{feature}</span>
          </div>
        ))}
      </div>

      <div className="flex flex-col sm:flex-row gap-3 pt-2">
        <Link
          href="/auth/register"
          className="flex-1 px-6 py-3 bg-[var(--acid-green)] text-[var(--bg)] font-mono font-bold text-sm text-center hover:opacity-90 transition-opacity"
        >
          CREATE FREE ACCOUNT
        </Link>
        <Link
          href="/auth/login"
          className="flex-1 px-6 py-3 border border-[var(--acid-green)]/30 text-[var(--acid-green)] font-mono text-sm text-center hover:border-[var(--acid-green)] transition-colors"
        >
          SIGN IN
        </Link>
      </div>
    </div>
  );
}

/**
 * Step 5: Launch -- start a real debate with the same topic.
 */
function LaunchStep({
  selectedIndustry,
  trialDebateResult,
  onComplete,
}: {
  selectedIndustry: string | null;
  trialDebateResult: Record<string, unknown> | null;
  onComplete?: () => void;
}) {
  const completeOnboarding = useOnboardingStore((s) => s.completeOnboarding);

  const trialTopic = trialDebateResult
    ? String((trialDebateResult as Record<string, unknown>).topic || '')
    : '';

  const handleLaunch = () => {
    completeOnboarding();
    onComplete?.();
  };

  const arenaUrl = trialTopic
    ? `/arena?topic=${encodeURIComponent(trialTopic)}${selectedIndustry ? `&vertical=${selectedIndustry}` : ''}`
    : `/arena${selectedIndustry ? `?vertical=${selectedIndustry}` : ''}`;

  return (
    <div className="space-y-6 text-center py-4">
      <div>
        <h2 className="text-xl font-mono text-[var(--acid-green)] mb-2">
          Ready to Launch
        </h2>
        <p className="text-sm font-mono text-[var(--text-muted)]">
          Everything is set up. Start a real multi-agent debate with your topic.
        </p>
      </div>

      {trialTopic && (
        <div className="border border-[var(--border)] bg-[var(--surface)] p-4 text-left">
          <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase mb-1">
            Your Topic
          </div>
          <p className="text-sm font-mono text-[var(--text)]">{trialTopic}</p>
        </div>
      )}

      <div className="flex flex-col gap-3">
        <Link
          href={arenaUrl}
          onClick={handleLaunch}
          className="px-8 py-4 bg-[var(--acid-green)] text-[var(--bg)] font-mono font-bold text-sm hover:opacity-90 transition-opacity"
        >
          LAUNCH REAL DEBATE
        </Link>
        <button
          onClick={handleLaunch}
          className="px-4 py-2 text-xs font-mono text-[var(--text-muted)] hover:text-[var(--acid-green)] transition-colors"
        >
          Skip to dashboard
        </button>
      </div>
    </div>
  );
}

export default OnboardingWizardFlow;
