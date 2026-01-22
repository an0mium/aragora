'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

interface OnboardingStep {
  id: string;
  title: string;
  description: string;
  action?: string;
  actionUrl?: string;
  tip?: string;
}

const ONBOARDING_STEPS: OnboardingStep[] = [
  {
    id: 'welcome',
    title: 'Welcome to Aragora',
    description: 'Aragora is the control plane for multi-agent deliberation. Multiple AI agents debate your questions and deliver defensible decisions with full audit trails.',
    tip: 'Think of it as decision assurance for high-stakes teams.',
  },
  {
    id: 'first-debate',
    title: 'Start Your First Debate',
    description: 'Enter a question or decision you want to stress-test. For example: "Should we use microservices or a monolith for our new project?"',
    action: 'Try it now',
    actionUrl: '/',
    tip: 'The more specific your question, the better the analysis.',
  },
  {
    id: 'agents',
    title: 'Meet the Agents',
    description: 'Each AI agent brings different perspectives. Claude excels at nuance, GPT at breadth, Gemini at creative solutions. They\'ll challenge each other\'s reasoning.',
    action: 'View agents',
    actionUrl: '/agents',
    tip: 'You can customize which agents participate in your debates.',
  },
  {
    id: 'consensus',
    title: 'Understand Consensus',
    description: 'After multiple rounds of debate, agents work toward consensus. Areas of agreement highlight robust conclusions, while disagreements reveal genuine uncertainty.',
    tip: 'Pay attention to what agents disagree on - these are often the most valuable insights.',
  },
  {
    id: 'templates',
    title: 'Use Templates',
    description: 'Not sure where to start? Use pre-built templates for common decisions: architecture reviews, code reviews, security assessments, and more.',
    action: 'Browse templates',
    actionUrl: '/templates',
    tip: 'Templates include optimized prompts and agent configurations.',
  },
  {
    id: 'done',
    title: 'You\'re Ready!',
    description: 'You now know the basics of Aragora. Start your first debate and let AI agents stress-test your thinking.',
    action: 'Start debating',
    actionUrl: '/',
  },
];

interface OnboardingFlowProps {
  onComplete?: () => void;
  onSkip?: () => void;
}

export function OnboardingFlow({ onComplete, onSkip }: OnboardingFlowProps) {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(0);
  const [isVisible, setIsVisible] = useState(true);

  const step = ONBOARDING_STEPS[currentStep];
  const isLastStep = currentStep === ONBOARDING_STEPS.length - 1;
  const progress = ((currentStep + 1) / ONBOARDING_STEPS.length) * 100;

  const handleNext = () => {
    if (isLastStep) {
      setIsVisible(false);
      localStorage.setItem('aragora_onboarding_complete', 'true');
      onComplete?.();
    } else {
      setCurrentStep((prev) => prev + 1);
    }
  };

  const handleSkip = () => {
    setIsVisible(false);
    localStorage.setItem('aragora_onboarding_complete', 'true');
    onSkip?.();
  };

  const handleAction = () => {
    if (step.actionUrl) {
      router.push(step.actionUrl);
      if (isLastStep) {
        setIsVisible(false);
        localStorage.setItem('aragora_onboarding_complete', 'true');
        onComplete?.();
      }
    }
  };

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-[100] bg-bg/95 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="max-w-lg w-full border border-acid-green/50 bg-surface p-6">
        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs font-mono text-text-muted">
              STEP {currentStep + 1} OF {ONBOARDING_STEPS.length}
            </span>
            <button
              onClick={handleSkip}
              className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
            >
              [SKIP TUTORIAL]
            </button>
          </div>
          <div className="h-1 bg-surface border border-acid-green/20">
            <div
              className="h-full bg-acid-green transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Step Content */}
        <div className="mb-8">
          <h2 className="text-xl font-mono text-acid-green mb-4">
            {step.title}
          </h2>
          <p className="font-mono text-text-muted mb-4 leading-relaxed">
            {step.description}
          </p>
          {step.tip && (
            <div className="p-3 bg-acid-green/5 border-l-2 border-acid-green">
              <p className="text-sm font-mono text-acid-cyan">
                <span className="text-acid-green">TIP:</span> {step.tip}
              </p>
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="flex gap-3">
          {currentStep > 0 && (
            <button
              onClick={() => setCurrentStep((prev) => prev - 1)}
              className="px-4 py-2 font-mono text-sm border border-acid-green/30 text-text-muted hover:border-acid-green hover:text-acid-green transition-colors"
            >
              [BACK]
            </button>
          )}
          <div className="flex-1" />
          {step.action && (
            <button
              onClick={handleAction}
              className="px-4 py-2 font-mono text-sm border border-acid-cyan/50 text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
            >
              [{step.action.toUpperCase()}]
            </button>
          )}
          <button
            onClick={handleNext}
            className="px-6 py-2 font-mono text-sm bg-acid-green text-bg hover:bg-acid-green/80 transition-colors"
          >
            {isLastStep ? '[FINISH]' : '[NEXT]'}
          </button>
        </div>

        {/* Step Indicators */}
        <div className="flex justify-center gap-2 mt-6">
          {ONBOARDING_STEPS.map((_, idx) => (
            <button
              key={idx}
              onClick={() => setCurrentStep(idx)}
              className={`w-2 h-2 transition-colors ${
                idx === currentStep
                  ? 'bg-acid-green'
                  : idx < currentStep
                  ? 'bg-acid-green/50'
                  : 'bg-surface border border-acid-green/30'
              }`}
              aria-label={`Go to step ${idx + 1}`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

export function useOnboarding() {
  const [showOnboarding, setShowOnboarding] = useState(false);

  useEffect(() => {
    const completed = localStorage.getItem('aragora_onboarding_complete');
    if (!completed) {
      setShowOnboarding(true);
    }
  }, []);

  const resetOnboarding = () => {
    localStorage.removeItem('aragora_onboarding_complete');
    setShowOnboarding(true);
  };

  return {
    showOnboarding,
    setShowOnboarding,
    resetOnboarding,
  };
}

export default OnboardingFlow;
