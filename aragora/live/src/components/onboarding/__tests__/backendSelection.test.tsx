import { fireEvent, render, screen, waitFor } from '@testing-library/react';

let mockStoreState: Record<string, unknown>;

const mockPush = jest.fn();
const mockFetch = jest.fn();

global.fetch = mockFetch as typeof fetch;

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

jest.mock('@/store/onboardingStore', () => ({
  useOnboardingStore: (selector?: (state: Record<string, unknown>) => unknown) =>
    selector ? selector(mockStoreState) : mockStoreState,
  useOnboardingStep: () => ({ stepIndex: 0, totalSteps: 5 }),
  useOnboardingProgress: () => ({ percentage: 20 }),
  selectIsOnboardingNeeded: () => true,
}));

jest.mock('../steps', () => ({
  WelcomeStep: () => <div>Welcome</div>,
  UseCaseStep: () => <div>Use case</div>,
  OrganizationStep: () => <div>Organization</div>,
  TemplateStep: ({ onNext }: { onNext: (template: { id: string }) => void }) => (
    <button onClick={() => onNext({ id: 'hiring' })}>Use template</button>
  ),
  CompletionStep: () => <div>Complete</div>,
}));

import { OnboardingFlow } from '../OnboardingFlow';
import { QuickDebatePanel } from '../QuickDebatePanel';
import { TryDebateStep } from '../steps/TryDebateStep';

describe('Onboarding backend selection', () => {
  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem('aragora-backend', 'production');
    mockFetch.mockReset();
    mockPush.mockReset();
    mockStoreState = {
      currentStep: 'template-select',
      nextStep: jest.fn(),
      previousStep: jest.fn(),
      completeOnboarding: jest.fn(),
      skipOnboarding: jest.fn(),
      setFirstDebateId: jest.fn(),
      setDebateStatus: jest.fn(),
      selectedTemplate: { id: 'hiring', rounds: 5 },
      debateStatus: 'idle',
      debateError: null,
      firstDebateTopic: '',
      setFirstDebateTopic: jest.fn(),
      setDebateError: jest.fn(),
      updateProgress: jest.fn(),
      selectedIndustry: 'general',
      trialDebateResult: null,
      setTrialDebateResult: jest.fn(),
    };
  });

  it('OnboardingFlow creates the first debate against the selected backend', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ debate_id: 'debate-123' }),
    });

    render(<OnboardingFlow />);
    fireEvent.click(screen.getByText('Use template'));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.aragora.ai/api/v1/onboarding/first-debate',
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });

  it('QuickDebatePanel starts debates against the selected backend', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ id: 'debate-quick' }),
    });

    render(<QuickDebatePanel />);
    fireEvent.click(screen.getByText('START DEBATE'));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenNthCalledWith(
        1,
        'https://api.aragora.ai/api/v1/debates',
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });

  it('TryDebateStep runs playground debates against the selected backend', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        consensus_reached: true,
        final_answer: 'archive',
        confidence: 0.92,
        rounds_used: 2,
        participants: ['a', 'b', 'c'],
      }),
    });

    render(<TryDebateStep />);
    fireEvent.click(screen.getByText('RUN FREE DEBATE'));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.aragora.ai/api/v1/playground/debate',
        expect.objectContaining({ method: 'POST' }),
      );
    });
  });
});
