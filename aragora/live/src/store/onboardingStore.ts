'use client';

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export type OnboardingStep =
  | 'welcome'
  | 'organization'
  | 'team-invite'
  | 'template-select'
  | 'first-debate'
  | 'completion';

export interface TeamMember {
  email: string;
  role: 'admin' | 'member' | 'viewer';
  invitedAt?: string;
}

export interface SelectedTemplate {
  id: string;
  name: string;
  description: string;
  agentsCount: number;
  rounds: number;
  estimatedDurationMinutes: number;
}

export interface OnboardingProgress {
  signupComplete: boolean;
  organizationCreated: boolean;
  firstDebateStarted: boolean;
  firstDebateCompleted: boolean;
  receiptViewed: boolean;
}

// ============================================================================
// Store State
// ============================================================================

interface OnboardingState {
  // Navigation
  currentStep: OnboardingStep;
  stepsCompleted: Set<OnboardingStep>;

  // Organization setup
  organizationName: string;
  organizationSlug: string;
  teamSize: '1-5' | '6-15' | '16-50' | '50+' | null;
  useCase: 'team_decisions' | 'project_planning' | 'vendor_selection' | 'policy_review' | 'technical_decisions' | 'general' | null;

  // Team invitations
  teamMembers: TeamMember[];

  // Template selection
  selectedTemplate: SelectedTemplate | null;
  availableTemplates: SelectedTemplate[];

  // First debate
  firstDebateId: string | null;
  firstDebateTopic: string;
  debateStatus: 'idle' | 'creating' | 'running' | 'completed' | 'error';
  debateError: string | null;

  // Progress tracking
  progress: OnboardingProgress;
  startedAt: string | null;
  completedAt: string | null;

  // Overall status
  isComplete: boolean;
  isSkipped: boolean;
}

interface OnboardingActions {
  // Navigation
  setCurrentStep: (step: OnboardingStep) => void;
  nextStep: () => void;
  previousStep: () => void;
  markStepComplete: (step: OnboardingStep) => void;

  // Organization
  setOrganizationName: (name: string) => void;
  setOrganizationSlug: (slug: string) => void;
  setTeamSize: (size: OnboardingState['teamSize']) => void;
  setUseCase: (useCase: OnboardingState['useCase']) => void;

  // Team
  addTeamMember: (email: string, role?: TeamMember['role']) => void;
  removeTeamMember: (email: string) => void;
  updateTeamMemberRole: (email: string, role: TeamMember['role']) => void;

  // Template
  setSelectedTemplate: (template: SelectedTemplate | null) => void;
  setAvailableTemplates: (templates: SelectedTemplate[]) => void;

  // First debate
  setFirstDebateId: (id: string | null) => void;
  setFirstDebateTopic: (topic: string) => void;
  setDebateStatus: (status: OnboardingState['debateStatus']) => void;
  setDebateError: (error: string | null) => void;

  // Progress
  updateProgress: (updates: Partial<OnboardingProgress>) => void;

  // Completion
  completeOnboarding: () => void;
  skipOnboarding: () => void;
  resetOnboarding: () => void;
}

// ============================================================================
// Step Order
// ============================================================================

const STEP_ORDER: OnboardingStep[] = [
  'welcome',
  'organization',
  'team-invite',
  'template-select',
  'first-debate',
  'completion',
];

function getNextStep(current: OnboardingStep): OnboardingStep {
  const currentIndex = STEP_ORDER.indexOf(current);
  const nextIndex = Math.min(currentIndex + 1, STEP_ORDER.length - 1);
  return STEP_ORDER[nextIndex];
}

function getPreviousStep(current: OnboardingStep): OnboardingStep {
  const currentIndex = STEP_ORDER.indexOf(current);
  const prevIndex = Math.max(currentIndex - 1, 0);
  return STEP_ORDER[prevIndex];
}

// ============================================================================
// Initial State
// ============================================================================

const initialState: OnboardingState = {
  currentStep: 'welcome',
  stepsCompleted: new Set(),

  organizationName: '',
  organizationSlug: '',
  teamSize: null,
  useCase: null,

  teamMembers: [],

  selectedTemplate: null,
  availableTemplates: [],

  firstDebateId: null,
  firstDebateTopic: '',
  debateStatus: 'idle',
  debateError: null,

  progress: {
    signupComplete: false,
    organizationCreated: false,
    firstDebateStarted: false,
    firstDebateCompleted: false,
    receiptViewed: false,
  },
  startedAt: null,
  completedAt: null,

  isComplete: false,
  isSkipped: false,
};

// ============================================================================
// Store
// ============================================================================

export const useOnboardingStore = create<OnboardingState & OnboardingActions>()(
  devtools(
    persist(
      (set, get) => ({
        ...initialState,

        // Navigation
        setCurrentStep: (step) => set({ currentStep: step }),

        nextStep: () => {
          const { currentStep } = get();
          const next = getNextStep(currentStep);
          set((state) => ({
            currentStep: next,
            stepsCompleted: new Set([...state.stepsCompleted, currentStep]),
          }));
        },

        previousStep: () => {
          const { currentStep } = get();
          set({ currentStep: getPreviousStep(currentStep) });
        },

        markStepComplete: (step) =>
          set((state) => ({
            stepsCompleted: new Set([...state.stepsCompleted, step]),
          })),

        // Organization
        setOrganizationName: (name) => set({ organizationName: name }),
        setOrganizationSlug: (slug) => set({ organizationSlug: slug }),
        setTeamSize: (size) => set({ teamSize: size }),
        setUseCase: (useCase) => set({ useCase }),

        // Team
        addTeamMember: (email, role = 'member') =>
          set((state) => ({
            teamMembers: [
              ...state.teamMembers,
              { email, role, invitedAt: new Date().toISOString() },
            ],
          })),

        removeTeamMember: (email) =>
          set((state) => ({
            teamMembers: state.teamMembers.filter((m) => m.email !== email),
          })),

        updateTeamMemberRole: (email, role) =>
          set((state) => ({
            teamMembers: state.teamMembers.map((m) =>
              m.email === email ? { ...m, role } : m
            ),
          })),

        // Template
        setSelectedTemplate: (template) => set({ selectedTemplate: template }),
        setAvailableTemplates: (templates) => set({ availableTemplates: templates }),

        // First debate
        setFirstDebateId: (id) => set({ firstDebateId: id }),
        setFirstDebateTopic: (topic) => set({ firstDebateTopic: topic }),
        setDebateStatus: (status) => set({ debateStatus: status }),
        setDebateError: (error) => set({ debateError: error }),

        // Progress
        updateProgress: (updates) =>
          set((state) => ({
            progress: { ...state.progress, ...updates },
          })),

        // Completion
        completeOnboarding: () =>
          set({
            isComplete: true,
            completedAt: new Date().toISOString(),
            currentStep: 'completion',
            stepsCompleted: new Set(STEP_ORDER),
          }),

        skipOnboarding: () =>
          set({
            isSkipped: true,
            completedAt: new Date().toISOString(),
          }),

        resetOnboarding: () => set(initialState),
      }),
      {
        name: 'aragora-onboarding',
        // Only persist essential data
        partialize: (state) => ({
          isComplete: state.isComplete,
          isSkipped: state.isSkipped,
          organizationName: state.organizationName,
          organizationSlug: state.organizationSlug,
          startedAt: state.startedAt,
          completedAt: state.completedAt,
        }),
      }
    ),
    { name: 'OnboardingStore' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectIsOnboardingNeeded = (state: OnboardingState) =>
  !state.isComplete && !state.isSkipped;

export const selectCurrentStepIndex = (state: OnboardingState) =>
  STEP_ORDER.indexOf(state.currentStep);

export const selectTotalSteps = () => STEP_ORDER.length;

export const selectIsFirstStep = (state: OnboardingState) =>
  state.currentStep === STEP_ORDER[0];

export const selectIsLastStep = (state: OnboardingState) =>
  state.currentStep === STEP_ORDER[STEP_ORDER.length - 1];

export const selectCanProceed = (state: OnboardingState): boolean => {
  switch (state.currentStep) {
    case 'welcome':
      return true;
    case 'organization':
      return state.organizationName.length >= 3 && state.teamSize !== null;
    case 'team-invite':
      return true; // Optional step
    case 'template-select':
      return state.selectedTemplate !== null;
    case 'first-debate':
      return state.debateStatus === 'completed';
    case 'completion':
      return true;
    default:
      return false;
  }
};

export const selectProgressPercentage = (state: OnboardingState): number => {
  const completed = state.stepsCompleted.size;
  return Math.round((completed / STEP_ORDER.length) * 100);
};

// ============================================================================
// Hooks
// ============================================================================

export function useOnboardingStep() {
  return useOnboardingStore((state) => ({
    currentStep: state.currentStep,
    isFirstStep: selectIsFirstStep(state),
    isLastStep: selectIsLastStep(state),
    canProceed: selectCanProceed(state),
    stepIndex: selectCurrentStepIndex(state),
    totalSteps: STEP_ORDER.length,
  }));
}

export function useOnboardingProgress() {
  return useOnboardingStore((state) => ({
    progress: state.progress,
    percentage: selectProgressPercentage(state),
    isComplete: state.isComplete,
    isSkipped: state.isSkipped,
  }));
}
