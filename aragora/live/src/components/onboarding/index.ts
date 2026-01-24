/**
 * Onboarding Components
 *
 * Multi-step onboarding wizard for new users.
 * Guides users through organization setup, team invitations,
 * template selection, and running their first debate.
 */

// Main onboarding flows
export { OnboardingFlow, useOnboarding } from './OnboardingFlow';
export { OnboardingWizardFlow } from './OnboardingWizardFlow';

// Shared components
export { ProgressBar } from './ProgressBar';

// Individual step components
export { WelcomeStep } from './WelcomeStep';
export { OrgSetupStep } from './OrgSetupStep';
export { TeamInviteStep } from './TeamInviteStep';
export { TemplateSelectStep } from './TemplateSelectStep';
export { FirstDebateStep } from './FirstDebateStep';
export { CompletionStep } from './CompletionStep';

// New enhanced step components (in steps/ subdirectory)
export {
  WelcomeStep as EnhancedWelcomeStep,
  UseCaseStep,
  OrganizationStep,
  TemplateStep,
  CompletionStep as EnhancedCompletionStep,
} from './steps';
