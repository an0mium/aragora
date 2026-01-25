/**
 * Tests for onboardingStore
 *
 * Tests cover:
 * - Step navigation (setCurrentStep, nextStep, previousStep)
 * - Organization setup (name, slug, team size, use case)
 * - Team member management (add, remove, update role)
 * - Template selection
 * - First debate tracking
 * - Progress tracking
 * - Completion and skip functionality
 * - Selectors (isOnboardingNeeded, canProceed, progressPercentage)
 */

import { act } from '@testing-library/react';
import {
  useOnboardingStore,
  selectIsOnboardingNeeded,
  selectCurrentStepIndex,
  selectTotalSteps,
  selectIsFirstStep,
  selectIsLastStep,
  selectCanProceed,
  selectProgressPercentage,
  SelectedTemplate,
} from '../onboardingStore';

// Sample test data
const mockTemplate: SelectedTemplate = {
  id: 'express-2v2',
  name: 'Express Debate',
  description: 'Quick 2-agent debate',
  agentsCount: 2,
  rounds: 2,
  estimatedDurationMinutes: 5,
};

const mockTemplate2: SelectedTemplate = {
  id: 'deep-4v4',
  name: 'Deep Dive',
  description: 'Comprehensive 4-agent debate',
  agentsCount: 4,
  rounds: 6,
  estimatedDurationMinutes: 30,
};

describe('onboardingStore', () => {
  beforeEach(() => {
    act(() => {
      useOnboardingStore.getState().resetOnboarding();
    });
  });

  describe('Initial State', () => {
    it('starts at welcome step', () => {
      expect(useOnboardingStore.getState().currentStep).toBe('welcome');
    });

    it('has empty organization data', () => {
      const state = useOnboardingStore.getState();
      expect(state.organizationName).toBe('');
      expect(state.organizationSlug).toBe('');
      expect(state.teamSize).toBeNull();
      expect(state.useCase).toBeNull();
    });

    it('is not complete or skipped', () => {
      const state = useOnboardingStore.getState();
      expect(state.isComplete).toBe(false);
      expect(state.isSkipped).toBe(false);
    });
  });

  describe('Step Navigation', () => {
    it('setCurrentStep changes step', () => {
      act(() => {
        useOnboardingStore.getState().setCurrentStep('organization');
      });

      expect(useOnboardingStore.getState().currentStep).toBe('organization');
    });

    it('nextStep advances to next step', () => {
      act(() => {
        useOnboardingStore.getState().nextStep();
      });

      expect(useOnboardingStore.getState().currentStep).toBe('organization');
    });

    it('nextStep marks current step as completed', () => {
      act(() => {
        useOnboardingStore.getState().nextStep();
      });

      expect(useOnboardingStore.getState().stepsCompleted.has('welcome')).toBe(true);
    });

    it('nextStep does not go past completion', () => {
      act(() => {
        useOnboardingStore.getState().setCurrentStep('completion');
        useOnboardingStore.getState().nextStep();
      });

      expect(useOnboardingStore.getState().currentStep).toBe('completion');
    });

    it('previousStep goes back', () => {
      act(() => {
        useOnboardingStore.getState().setCurrentStep('organization');
        useOnboardingStore.getState().previousStep();
      });

      expect(useOnboardingStore.getState().currentStep).toBe('welcome');
    });

    it('previousStep does not go before welcome', () => {
      act(() => {
        useOnboardingStore.getState().previousStep();
      });

      expect(useOnboardingStore.getState().currentStep).toBe('welcome');
    });

    it('markStepComplete adds step to completed set', () => {
      act(() => {
        useOnboardingStore.getState().markStepComplete('organization');
      });

      expect(useOnboardingStore.getState().stepsCompleted.has('organization')).toBe(true);
    });
  });

  describe('Organization Setup', () => {
    it('setOrganizationName updates name', () => {
      act(() => {
        useOnboardingStore.getState().setOrganizationName('Acme Corp');
      });

      expect(useOnboardingStore.getState().organizationName).toBe('Acme Corp');
    });

    it('setOrganizationSlug updates slug', () => {
      act(() => {
        useOnboardingStore.getState().setOrganizationSlug('acme-corp');
      });

      expect(useOnboardingStore.getState().organizationSlug).toBe('acme-corp');
    });

    it('setTeamSize updates team size', () => {
      act(() => {
        useOnboardingStore.getState().setTeamSize('6-15');
      });

      expect(useOnboardingStore.getState().teamSize).toBe('6-15');
    });

    it('setUseCase updates use case', () => {
      act(() => {
        useOnboardingStore.getState().setUseCase('technical_decisions');
      });

      expect(useOnboardingStore.getState().useCase).toBe('technical_decisions');
    });
  });

  describe('Team Member Management', () => {
    it('addTeamMember adds member with default role', () => {
      act(() => {
        useOnboardingStore.getState().addTeamMember('alice@example.com');
      });

      const members = useOnboardingStore.getState().teamMembers;
      expect(members).toHaveLength(1);
      expect(members[0].email).toBe('alice@example.com');
      expect(members[0].role).toBe('member');
      expect(members[0].invitedAt).toBeDefined();
    });

    it('addTeamMember adds member with custom role', () => {
      act(() => {
        useOnboardingStore.getState().addTeamMember('bob@example.com', 'admin');
      });

      const members = useOnboardingStore.getState().teamMembers;
      expect(members[0].role).toBe('admin');
    });

    it('removeTeamMember removes member by email', () => {
      act(() => {
        useOnboardingStore.getState().addTeamMember('alice@example.com');
        useOnboardingStore.getState().addTeamMember('bob@example.com');
        useOnboardingStore.getState().removeTeamMember('alice@example.com');
      });

      const members = useOnboardingStore.getState().teamMembers;
      expect(members).toHaveLength(1);
      expect(members[0].email).toBe('bob@example.com');
    });

    it('updateTeamMemberRole updates role', () => {
      act(() => {
        useOnboardingStore.getState().addTeamMember('alice@example.com', 'member');
        useOnboardingStore.getState().updateTeamMemberRole('alice@example.com', 'admin');
      });

      const members = useOnboardingStore.getState().teamMembers;
      expect(members[0].role).toBe('admin');
    });
  });

  describe('Template Selection', () => {
    it('setSelectedTemplate updates template', () => {
      act(() => {
        useOnboardingStore.getState().setSelectedTemplate(mockTemplate);
      });

      expect(useOnboardingStore.getState().selectedTemplate).toEqual(mockTemplate);
    });

    it('setSelectedTemplate can clear template', () => {
      act(() => {
        useOnboardingStore.getState().setSelectedTemplate(mockTemplate);
        useOnboardingStore.getState().setSelectedTemplate(null);
      });

      expect(useOnboardingStore.getState().selectedTemplate).toBeNull();
    });

    it('setAvailableTemplates updates template list', () => {
      act(() => {
        useOnboardingStore.getState().setAvailableTemplates([mockTemplate, mockTemplate2]);
      });

      expect(useOnboardingStore.getState().availableTemplates).toHaveLength(2);
    });
  });

  describe('First Debate Tracking', () => {
    it('setFirstDebateId updates debate ID', () => {
      act(() => {
        useOnboardingStore.getState().setFirstDebateId('debate-123');
      });

      expect(useOnboardingStore.getState().firstDebateId).toBe('debate-123');
    });

    it('setFirstDebateTopic updates topic', () => {
      act(() => {
        useOnboardingStore.getState().setFirstDebateTopic('Should we use TypeScript?');
      });

      expect(useOnboardingStore.getState().firstDebateTopic).toBe('Should we use TypeScript?');
    });

    it('setDebateStatus updates status', () => {
      act(() => {
        useOnboardingStore.getState().setDebateStatus('running');
      });

      expect(useOnboardingStore.getState().debateStatus).toBe('running');
    });

    it('setDebateError updates error', () => {
      act(() => {
        useOnboardingStore.getState().setDebateError('Connection lost');
      });

      expect(useOnboardingStore.getState().debateError).toBe('Connection lost');
    });

    it('setDebateError can clear error', () => {
      act(() => {
        useOnboardingStore.getState().setDebateError('Some error');
        useOnboardingStore.getState().setDebateError(null);
      });

      expect(useOnboardingStore.getState().debateError).toBeNull();
    });
  });

  describe('Progress Tracking', () => {
    it('updateProgress updates partial progress', () => {
      act(() => {
        useOnboardingStore.getState().updateProgress({ signupComplete: true });
      });

      const progress = useOnboardingStore.getState().progress;
      expect(progress.signupComplete).toBe(true);
      expect(progress.organizationCreated).toBe(false);
    });

    it('updateProgress merges multiple updates', () => {
      act(() => {
        useOnboardingStore.getState().updateProgress({ signupComplete: true });
        useOnboardingStore.getState().updateProgress({ organizationCreated: true });
      });

      const progress = useOnboardingStore.getState().progress;
      expect(progress.signupComplete).toBe(true);
      expect(progress.organizationCreated).toBe(true);
    });
  });

  describe('Completion', () => {
    it('completeOnboarding sets completion state', () => {
      act(() => {
        useOnboardingStore.getState().completeOnboarding();
      });

      const state = useOnboardingStore.getState();
      expect(state.isComplete).toBe(true);
      expect(state.completedAt).toBeDefined();
      expect(state.currentStep).toBe('completion');
    });

    it('completeOnboarding marks all steps complete', () => {
      act(() => {
        useOnboardingStore.getState().completeOnboarding();
      });

      const state = useOnboardingStore.getState();
      expect(state.stepsCompleted.size).toBe(6); // All 6 steps
    });

    it('skipOnboarding sets skipped state', () => {
      act(() => {
        useOnboardingStore.getState().skipOnboarding();
      });

      const state = useOnboardingStore.getState();
      expect(state.isSkipped).toBe(true);
      expect(state.completedAt).toBeDefined();
    });

    it('resetOnboarding resets all state', () => {
      act(() => {
        useOnboardingStore.getState().setOrganizationName('Test Org');
        useOnboardingStore.getState().addTeamMember('test@example.com');
        useOnboardingStore.getState().setDebateStatus('running');
        useOnboardingStore.getState().completeOnboarding();
        useOnboardingStore.getState().resetOnboarding();
      });

      const state = useOnboardingStore.getState();
      expect(state.organizationName).toBe('');
      expect(state.teamMembers).toHaveLength(0);
      expect(state.debateStatus).toBe('idle');
      expect(state.isComplete).toBe(false);
      expect(state.currentStep).toBe('welcome');
    });
  });

  describe('Selectors', () => {
    describe('selectIsOnboardingNeeded', () => {
      it('returns true when not complete and not skipped', () => {
        expect(selectIsOnboardingNeeded(useOnboardingStore.getState())).toBe(true);
      });

      it('returns false when complete', () => {
        act(() => {
          useOnboardingStore.getState().completeOnboarding();
        });

        expect(selectIsOnboardingNeeded(useOnboardingStore.getState())).toBe(false);
      });

      it('returns false when skipped', () => {
        act(() => {
          useOnboardingStore.getState().skipOnboarding();
        });

        expect(selectIsOnboardingNeeded(useOnboardingStore.getState())).toBe(false);
      });
    });

    describe('selectCurrentStepIndex', () => {
      it('returns correct index for each step', () => {
        expect(selectCurrentStepIndex(useOnboardingStore.getState())).toBe(0);

        act(() => {
          useOnboardingStore.getState().setCurrentStep('first-debate');
        });

        expect(selectCurrentStepIndex(useOnboardingStore.getState())).toBe(4);
      });
    });

    describe('selectTotalSteps', () => {
      it('returns 6 total steps', () => {
        expect(selectTotalSteps()).toBe(6);
      });
    });

    describe('selectIsFirstStep', () => {
      it('returns true on welcome step', () => {
        expect(selectIsFirstStep(useOnboardingStore.getState())).toBe(true);
      });

      it('returns false on other steps', () => {
        act(() => {
          useOnboardingStore.getState().setCurrentStep('organization');
        });

        expect(selectIsFirstStep(useOnboardingStore.getState())).toBe(false);
      });
    });

    describe('selectIsLastStep', () => {
      it('returns false on first step', () => {
        expect(selectIsLastStep(useOnboardingStore.getState())).toBe(false);
      });

      it('returns true on completion step', () => {
        act(() => {
          useOnboardingStore.getState().setCurrentStep('completion');
        });

        expect(selectIsLastStep(useOnboardingStore.getState())).toBe(true);
      });
    });

    describe('selectCanProceed', () => {
      it('returns true for welcome step', () => {
        expect(selectCanProceed(useOnboardingStore.getState())).toBe(true);
      });

      it('returns false for organization step without required fields', () => {
        act(() => {
          useOnboardingStore.getState().setCurrentStep('organization');
        });

        expect(selectCanProceed(useOnboardingStore.getState())).toBe(false);
      });

      it('returns true for organization step with required fields', () => {
        act(() => {
          useOnboardingStore.getState().setCurrentStep('organization');
          useOnboardingStore.getState().setOrganizationName('Test Organization');
          useOnboardingStore.getState().setTeamSize('1-5');
        });

        expect(selectCanProceed(useOnboardingStore.getState())).toBe(true);
      });

      it('returns true for team-invite step (optional)', () => {
        act(() => {
          useOnboardingStore.getState().setCurrentStep('team-invite');
        });

        expect(selectCanProceed(useOnboardingStore.getState())).toBe(true);
      });

      it('returns false for template-select without selection', () => {
        act(() => {
          useOnboardingStore.getState().setCurrentStep('template-select');
        });

        expect(selectCanProceed(useOnboardingStore.getState())).toBe(false);
      });

      it('returns true for template-select with selection', () => {
        act(() => {
          useOnboardingStore.getState().setCurrentStep('template-select');
          useOnboardingStore.getState().setSelectedTemplate(mockTemplate);
        });

        expect(selectCanProceed(useOnboardingStore.getState())).toBe(true);
      });

      it('returns false for first-debate step when not completed', () => {
        act(() => {
          useOnboardingStore.getState().setCurrentStep('first-debate');
          useOnboardingStore.getState().setDebateStatus('running');
        });

        expect(selectCanProceed(useOnboardingStore.getState())).toBe(false);
      });

      it('returns true for first-debate step when completed', () => {
        act(() => {
          useOnboardingStore.getState().setCurrentStep('first-debate');
          useOnboardingStore.getState().setDebateStatus('completed');
        });

        expect(selectCanProceed(useOnboardingStore.getState())).toBe(true);
      });
    });

    describe('selectProgressPercentage', () => {
      it('returns 0 when no steps completed', () => {
        expect(selectProgressPercentage(useOnboardingStore.getState())).toBe(0);
      });

      it('returns correct percentage after steps', () => {
        act(() => {
          useOnboardingStore.getState().markStepComplete('welcome');
          useOnboardingStore.getState().markStepComplete('organization');
        });

        // 2/6 = 33%
        expect(selectProgressPercentage(useOnboardingStore.getState())).toBe(33);
      });

      it('returns 100 when all steps completed', () => {
        act(() => {
          useOnboardingStore.getState().completeOnboarding();
        });

        expect(selectProgressPercentage(useOnboardingStore.getState())).toBe(100);
      });
    });
  });
});
