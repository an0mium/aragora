'use client';

import { useCallback } from 'react';
import { useOnboardingStore, useOnboardingStep } from '@/store';
import { WelcomeStep } from './WelcomeStep';
import { OrgSetupStep } from './OrgSetupStep';
import { TeamInviteStep } from './TeamInviteStep';
import { TemplateSelectStep } from './TemplateSelectStep';
import { FirstDebateStep } from './FirstDebateStep';
import { CompletionStep } from './CompletionStep';
import { ProgressBar } from './ProgressBar';

interface OnboardingWizardFlowProps {
  onComplete?: () => void;
  onSkip?: () => void;
}

export function OnboardingWizardFlow({ onComplete, onSkip }: OnboardingWizardFlowProps) {
  const {
    currentStep,
    isFirstStep,
    isLastStep,
    canProceed,
    stepIndex,
    totalSteps,
  } = useOnboardingStep();

  const {
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
      case 'welcome':
        return <WelcomeStep />;
      case 'organization':
        return <OrgSetupStep />;
      case 'team-invite':
        return <TeamInviteStep />;
      case 'template-select':
        return <TemplateSelectStep />;
      case 'first-debate':
        return <FirstDebateStep />;
      case 'completion':
        return <CompletionStep onComplete={onComplete} />;
      default:
        return <WelcomeStep />;
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-bg/95 backdrop-blur-sm">
      <div className="w-full max-w-2xl mx-4 border border-acid-green/30 bg-surface rounded-lg overflow-hidden">
        {/* Header */}
        <div className="border-b border-acid-green/20 px-6 py-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-mono text-acid-green">GET STARTED WITH ARAGORA</h2>
            {currentStep !== 'completion' && (
              <button
                onClick={handleSkip}
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [SKIP FOR NOW]
              </button>
            )}
          </div>
          <ProgressBar current={stepIndex + 1} total={totalSteps} />
        </div>

        {/* Content */}
        <div className="p-6 min-h-[300px]">
          {renderStep()}
        </div>

        {/* Footer */}
        {currentStep !== 'completion' && (
          <div className="border-t border-acid-green/20 px-6 py-4 flex items-center justify-between">
            <button
              onClick={handleBack}
              disabled={isFirstStep}
              className="px-4 py-2 text-sm font-mono text-text-muted hover:text-acid-green disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              ← BACK
            </button>
            <button
              onClick={handleNext}
              disabled={!canProceed}
              className="px-6 py-2 bg-acid-green text-bg font-mono text-sm hover:bg-acid-green/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLastStep ? 'FINISH' : 'CONTINUE →'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default OnboardingWizardFlow;
