import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import PolicyPage from '../page';

// Mock next/link
jest.mock('next/link', () => {
  return function MockLink({ children, href }: { children: React.ReactNode; href: string }) {
    return <a href={href}>{children}</a>;
  };
});

// Mock visual components
jest.mock('@/components/MatrixRain', () => ({
  Scanlines: () => <div data-testid="scanlines" />,
  CRTVignette: () => <div data-testid="crt-vignette" />,
}));

jest.mock('@/components/AsciiBanner', () => ({
  AsciiBannerCompact: () => <div data-testid="ascii-banner">ARAGORA</div>,
}));

jest.mock('@/components/ThemeToggle', () => ({
  ThemeToggle: () => <button data-testid="theme-toggle">Theme</button>,
}));

// Mock BackendSelector with context
const mockBackendConfig = { api: 'http://localhost:8080' };
jest.mock('@/components/BackendSelector', () => ({
  BackendSelector: () => <div data-testid="backend-selector">Backend</div>,
  useBackend: () => ({ config: mockBackendConfig }),
}));

// Mock ErrorWithRetry
jest.mock('@/components/ErrorWithRetry', () => ({
  ErrorWithRetry: ({ error, onRetry }: { error: string; onRetry: () => void }) => (
    <div data-testid="error-display">
      <span>{error}</span>
      <button onClick={onRetry} data-testid="retry-button">Retry</button>
    </div>
  ),
}));

// Mock ToastContext
const mockShowToast = jest.fn();
jest.mock('@/context/ToastContext', () => ({
  useToastContext: () => ({ showToast: mockShowToast }),
}));

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Mock window.confirm
const mockConfirm = jest.fn();
window.confirm = mockConfirm;

describe('PolicyPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockConfirm.mockReturnValue(true);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  const mockPolicies = [
    {
      id: 'policy-1',
      name: 'No Profanity',
      description: 'Block profane language in outputs',
      type: 'content',
      severity: 'medium',
      enabled: true,
      rules: [
        { id: 'rule-1', pattern: '\\b(badword)\\b', action: 'block', message: 'Profanity detected' },
      ],
      framework_id: 'default',
      vertical_id: 'general',
      created_at: '2024-01-15T10:00:00Z',
      violation_count: 5,
    },
    {
      id: 'policy-2',
      name: 'PII Protection',
      description: 'Redact personally identifiable information',
      type: 'output',
      severity: 'critical',
      enabled: true,
      rules: [
        { id: 'rule-2', pattern: '\\d{3}-\\d{2}-\\d{4}', action: 'redact', message: 'SSN detected' },
      ],
      framework_id: 'hipaa',
      vertical_id: 'healthcare',
      created_at: '2024-01-10T08:00:00Z',
      violation_count: 12,
    },
  ];

  const mockViolations = [
    {
      id: 'violation-1',
      policy_id: 'policy-1',
      policy_name: 'No Profanity',
      rule_id: 'rule-1',
      content_snippet: 'This contains badword text',
      severity: 'medium',
      status: 'open',
      description: 'Profanity detected in agent output',
      created_at: '2024-01-16T12:00:00Z',
    },
    {
      id: 'violation-2',
      policy_id: 'policy-2',
      policy_name: 'PII Protection',
      rule_id: 'rule-2',
      content_snippet: 'SSN: 123-45-6789',
      severity: 'critical',
      status: 'resolved',
      description: 'SSN found in output',
      created_at: '2024-01-14T09:00:00Z',
      resolved_at: '2024-01-14T10:00:00Z',
      resolution_notes: 'Redacted and notified user',
    },
  ];

  const mockStats = {
    policies: {
      total: 2,
      enabled: 2,
      disabled: 0,
    },
    violations: {
      total: 17,
      open: 5,
      by_severity: {
        critical: 2,
        high: 3,
        medium: 8,
        low: 4,
      },
    },
    risk_score: 25,
  };

  const setupSuccessfulFetch = () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/api/policies')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ policies: mockPolicies }),
        });
      }
      if (url.includes('/api/compliance/violations')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ violations: mockViolations }),
        });
      }
      if (url.includes('/api/compliance/stats')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockStats),
        });
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({}),
      });
    });
  };

  describe('initial render', () => {
    it('renders visual effects', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      expect(screen.getByTestId('scanlines')).toBeInTheDocument();
      expect(screen.getByTestId('crt-vignette')).toBeInTheDocument();
    });

    it('renders header elements', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      expect(screen.getByTestId('ascii-banner')).toBeInTheDocument();
      expect(screen.getByTestId('theme-toggle')).toBeInTheDocument();
      expect(screen.getByTestId('backend-selector')).toBeInTheDocument();
    });

    it('renders page title', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      expect(screen.getByText('[POLICY_ADMIN]')).toBeInTheDocument();
      expect(screen.getByText('Compliance policies and violation tracking')).toBeInTheDocument();
    });

    it('shows loading state initially', () => {
      mockFetch.mockReturnValue(new Promise(() => {})); // Never resolves

      render(<PolicyPage />);

      expect(screen.getByText('Loading policy data...')).toBeInTheDocument();
    });

    it('renders tab navigation', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      expect(screen.getByRole('button', { name: /POLICIES/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /VIOLATIONS/i })).toBeInTheDocument();
    });

    it('renders action buttons', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      expect(screen.getByRole('button', { name: '[CHECK CONTENT]' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: '[+ NEW POLICY]' })).toBeInTheDocument();
    });
  });

  describe('data fetching', () => {
    it('fetches policies, violations, and stats on mount', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith('http://localhost:8080/api/policies');
        expect(mockFetch).toHaveBeenCalledWith('http://localhost:8080/api/compliance/violations?limit=100');
        expect(mockFetch).toHaveBeenCalledWith('http://localhost:8080/api/compliance/stats');
      });
    });

    it('displays policies when fetched successfully', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.getByText('No Profanity')).toBeInTheDocument();
        expect(screen.getByText('PII Protection')).toBeInTheDocument();
      });
    });

    it('shows empty state when no policies', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/policies')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ policies: [] }),
          });
        }
        if (url.includes('/api/compliance/violations')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ violations: [] }),
          });
        }
        if (url.includes('/api/compliance/stats')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              policies: { total: 0, enabled: 0, disabled: 0 },
              violations: { total: 0, open: 0, by_severity: { critical: 0, high: 0, medium: 0, low: 0 } },
              risk_score: 0,
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.getByText('No policies defined. Create your first compliance policy.')).toBeInTheDocument();
      });
    });

    it('displays error when fetch fails', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.getByTestId('error-display')).toBeInTheDocument();
      });
    });
  });

  describe('compliance stats', () => {
    it('displays compliance score', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.getByText('75%')).toBeInTheDocument();
        expect(screen.getByText('Compliance Score')).toBeInTheDocument();
      });
    });

    it('displays active policies count', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.getByText('2/2')).toBeInTheDocument();
        expect(screen.getByText('Active Policies')).toBeInTheDocument();
      });
    });

    it('displays open violations count', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        // Multiple elements may have "5" - just verify the label exists
        const openViolationsLabel = screen.getByText('Open Violations');
        expect(openViolationsLabel).toBeInTheDocument();
        // Get the parent card and verify it has a "5" in it
        const card = openViolationsLabel.closest('.card');
        expect(card).toBeTruthy();
        expect(card?.textContent).toContain('5');
      });
    });

    it('displays critical/high count', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        // Critical (2) + High (3) = 5
        const criticalHighElements = screen.getAllByText('5');
        expect(criticalHighElements.length).toBeGreaterThan(0);
        expect(screen.getByText('Critical/High')).toBeInTheDocument();
      });
    });

    it('displays resolved count', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        // Total (17) - Open (5) = 12
        expect(screen.getByText('12')).toBeInTheDocument();
        expect(screen.getByText('Resolved')).toBeInTheDocument();
      });
    });
  });

  describe('policies tab', () => {
    it('displays policy cards with details', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.getByText('No Profanity')).toBeInTheDocument();
        expect(screen.getByText('Block profane language in outputs')).toBeInTheDocument();
        expect(screen.getByText('[content]')).toBeInTheDocument();
      });
    });

    it('displays severity badges', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      // Wait for policies to be displayed
      await waitFor(() => {
        expect(screen.getByText('No Profanity')).toBeInTheDocument();
      });

      // Severity badges display lowercase text (CSS uppercase is visual only)
      // The badges should be present for both policies
      const mediumBadges = screen.getAllByText('medium');
      const criticalBadges = screen.getAllByText('critical');
      expect(mediumBadges.length).toBeGreaterThan(0);
      expect(criticalBadges.length).toBeGreaterThan(0);
    });

    it('displays violation counts', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.getByText('5 violations')).toBeInTheDocument();
        expect(screen.getByText('12 violations')).toBeInTheDocument();
      });
    });

    it('displays toggle buttons for policies', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        const toggleButtons = screen.getAllByRole('button', { name: '[ON]' });
        expect(toggleButtons.length).toBe(2);
      });
    });

    it('displays edit buttons for policies', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        const editButtons = screen.getAllByRole('button', { name: '[EDIT]' });
        expect(editButtons.length).toBe(2);
      });
    });

    it('displays delete buttons for policies', async () => {
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        const deleteButtons = screen.getAllByRole('button', { name: '[DEL]' });
        expect(deleteButtons.length).toBe(2);
      });
    });

    it('expands policy details when clicked', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.getByText('No Profanity')).toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByText('No Profanity'));
      });

      await waitFor(() => {
        expect(screen.getByText('Rules (1):')).toBeInTheDocument();
        expect(screen.getByText('[BLOCK]')).toBeInTheDocument();
        expect(screen.getByText('Profanity detected')).toBeInTheDocument();
      });
    });

    it('displays framework and vertical in expanded view', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.getByText('No Profanity')).toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByText('No Profanity'));
      });

      await waitFor(() => {
        expect(screen.getByText(/Framework: default/)).toBeInTheDocument();
        expect(screen.getByText(/Vertical: general/)).toBeInTheDocument();
      });
    });
  });

  describe('violations tab', () => {
    it('switches to violations tab', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /VIOLATIONS/i }));
      });

      // Table headers should be visible
      expect(screen.getByText('Policy')).toBeInTheDocument();
      expect(screen.getByText('Content')).toBeInTheDocument();
      expect(screen.getByText('Severity')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
    });

    it('displays violations in table', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /VIOLATIONS/i }));
      });

      expect(screen.getByText('No Profanity')).toBeInTheDocument();
      expect(screen.getByText('PII Protection')).toBeInTheDocument();
    });

    it('displays violation status badges', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /VIOLATIONS/i }));
      });

      // Status badges - OPEN appears in both filter button and table row
      // Use getAllByText since there can be multiple
      const openElements = screen.getAllByText('OPEN');
      const resolvedElements = screen.getAllByText('RESOLVED');
      expect(openElements.length).toBeGreaterThan(0);
      expect(resolvedElements.length).toBeGreaterThan(0);
    });

    it('displays view button for violations', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /VIOLATIONS/i }));
      });

      const viewButtons = screen.getAllByText('[VIEW]');
      expect(viewButtons.length).toBe(2);
    });

    it('filters violations by status', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /VIOLATIONS/i }));
      });

      // Click on OPEN filter
      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'OPEN' }));
      });

      // Only open violations should be visible
      expect(screen.getByText('No Profanity')).toBeInTheDocument();
      expect(screen.queryByText('PII Protection')).not.toBeInTheDocument();
    });

    it('filters violations by severity', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /VIOLATIONS/i }));
      });

      // Click on CRITICAL filter
      const criticalButtons = screen.getAllByRole('button', { name: 'CRITICAL' });
      await act(async () => {
        await user.click(criticalButtons[criticalButtons.length - 1]); // Click the filter button
      });

      // Only critical violations should be visible
      expect(screen.getByText('PII Protection')).toBeInTheDocument();
      expect(screen.queryByText('No Profanity')).not.toBeInTheDocument();
    });

    it('shows empty state when no violations match filters', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/api/policies')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ policies: mockPolicies }),
          });
        }
        if (url.includes('/api/compliance/violations')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ violations: [] }),
          });
        }
        if (url.includes('/api/compliance/stats')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockStats),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /VIOLATIONS/i }));
      });

      expect(screen.getByText('No violations recorded. Your content is compliant.')).toBeInTheDocument();
    });
  });

  describe('policy modal', () => {
    it('opens new policy modal when clicking new policy button', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[+ NEW POLICY]' }));
      });

      expect(screen.getByText('[NEW POLICY]')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Policy name')).toBeInTheDocument();
    });

    it('opens edit policy modal when clicking edit button', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      const editButtons = screen.getAllByRole('button', { name: '[EDIT]' });
      await act(async () => {
        await user.click(editButtons[0]);
      });

      expect(screen.getByText('[EDIT POLICY]')).toBeInTheDocument();
    });

    it('closes modal when clicking cancel', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[+ NEW POLICY]' }));
      });

      expect(screen.getByText('[NEW POLICY]')).toBeInTheDocument();

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[CANCEL]' }));
      });

      expect(screen.queryByText('[NEW POLICY]')).not.toBeInTheDocument();
    });

    it('allows adding rules', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[+ NEW POLICY]' }));
      });

      expect(screen.getByText('No rules defined. Add rules to define policy behavior.')).toBeInTheDocument();

      await act(async () => {
        await user.click(screen.getByText('[+ ADD RULE]'));
      });

      expect(screen.getByText('Rule 1')).toBeInTheDocument();
    });

    it('creates policy on submit', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes('/api/policies') && options?.method === 'POST') {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ id: 'new-policy' }),
          });
        }
        if (url.includes('/api/policies')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ policies: mockPolicies }),
          });
        }
        if (url.includes('/api/compliance/violations')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ violations: mockViolations }),
          });
        }
        if (url.includes('/api/compliance/stats')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockStats),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[+ NEW POLICY]' }));
      });

      await act(async () => {
        await user.type(screen.getByPlaceholderText('Policy name'), 'Test Policy');
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[CREATE POLICY]' }));
      });

      await waitFor(() => {
        expect(mockShowToast).toHaveBeenCalledWith('Policy created successfully', 'success');
      });
    });
  });

  describe('violation modal', () => {
    it('opens violation modal when clicking view', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /VIOLATIONS/i }));
      });

      const viewButtons = screen.getAllByText('[VIEW]');
      await act(async () => {
        await user.click(viewButtons[0]);
      });

      expect(screen.getByText('[VIOLATION DETAILS]')).toBeInTheDocument();
    });

    it('displays violation details in modal', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /VIOLATIONS/i }));
      });

      const viewButtons = screen.getAllByText('[VIEW]');
      await act(async () => {
        await user.click(viewButtons[0]);
      });

      // The content should appear in a modal, wait for the modal content
      await waitFor(() => {
        expect(screen.getByText('[VIOLATION DETAILS]')).toBeInTheDocument();
      });
      // The modal should show the content - verify modal is open by checking for modal-specific element
      expect(screen.getByText('Content:')).toBeInTheDocument();
    });

    it('closes violation modal when clicking close', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /VIOLATIONS/i }));
      });

      const viewButtons = screen.getAllByText('[VIEW]');
      await act(async () => {
        await user.click(viewButtons[0]);
      });

      expect(screen.getByText('[VIOLATION DETAILS]')).toBeInTheDocument();

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[CLOSE]' }));
      });

      expect(screen.queryByText('[VIOLATION DETAILS]')).not.toBeInTheDocument();
    });

    it('displays action buttons for open violations', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: /VIOLATIONS/i }));
      });

      // View the open violation (first one)
      const viewButtons = screen.getAllByText('[VIEW]');
      await act(async () => {
        await user.click(viewButtons[0]);
      });

      expect(screen.getByRole('button', { name: '[INVESTIGATE]' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: '[RESOLVE]' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: '[FALSE POSITIVE]' })).toBeInTheDocument();
    });
  });

  describe('compliance check modal', () => {
    it('opens compliance check modal', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[CHECK CONTENT]' }));
      });

      expect(screen.getByText('[COMPLIANCE CHECK]')).toBeInTheDocument();
    });

    it('displays content input in compliance check modal', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[CHECK CONTENT]' }));
      });

      expect(screen.getByPlaceholderText('Enter content to check against compliance policies...')).toBeInTheDocument();
    });

    it('disables check button when no content', async () => {
      const user = userEvent.setup();
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[CHECK CONTENT]' }));
      });

      expect(screen.getByRole('button', { name: '[CHECK]' })).toBeDisabled();
    });

    it('performs compliance check when content is entered', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes('/api/compliance/check') && options?.method === 'POST') {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              compliant: true,
              score: 95,
              issue_count: 0,
            }),
          });
        }
        if (url.includes('/api/policies')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ policies: mockPolicies }),
          });
        }
        if (url.includes('/api/compliance/violations')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ violations: mockViolations }),
          });
        }
        if (url.includes('/api/compliance/stats')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockStats),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[CHECK CONTENT]' }));
      });

      await act(async () => {
        await user.type(
          screen.getByPlaceholderText('Enter content to check against compliance policies...'),
          'This is safe content'
        );
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[CHECK]' }));
      });

      await waitFor(() => {
        expect(screen.getByText('COMPLIANT')).toBeInTheDocument();
        expect(screen.getByText('Score: 95%')).toBeInTheDocument();
      });
    });

    it('displays non-compliant result', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes('/api/compliance/check') && options?.method === 'POST') {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              compliant: false,
              score: 45,
              issue_count: 3,
            }),
          });
        }
        if (url.includes('/api/policies')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ policies: mockPolicies }),
          });
        }
        if (url.includes('/api/compliance/violations')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ violations: mockViolations }),
          });
        }
        if (url.includes('/api/compliance/stats')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockStats),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[CHECK CONTENT]' }));
      });

      await act(async () => {
        await user.type(
          screen.getByPlaceholderText('Enter content to check against compliance policies...'),
          'This is bad content with violations'
        );
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[CHECK]' }));
      });

      await waitFor(() => {
        expect(screen.getByText('NON-COMPLIANT')).toBeInTheDocument();
        expect(screen.getByText('Score: 45%')).toBeInTheDocument();
        expect(screen.getByText('3 issues found')).toBeInTheDocument();
      });
    });
  });

  describe('policy actions', () => {
    it('toggles policy when toggle button is clicked', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes('/toggle') && options?.method === 'POST') {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({}),
          });
        }
        if (url.includes('/api/policies')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ policies: mockPolicies }),
          });
        }
        if (url.includes('/api/compliance/violations')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ violations: mockViolations }),
          });
        }
        if (url.includes('/api/compliance/stats')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockStats),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      const toggleButtons = screen.getAllByRole('button', { name: '[ON]' });
      await act(async () => {
        await user.click(toggleButtons[0]);
      });

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/policies/policy-1/toggle',
          expect.objectContaining({ method: 'POST' })
        );
      });
    });

    it('deletes policy when delete button is clicked and confirmed', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes('/api/policies/policy-1') && options?.method === 'DELETE') {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({}),
          });
        }
        if (url.includes('/api/policies')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ policies: mockPolicies }),
          });
        }
        if (url.includes('/api/compliance/violations')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ violations: mockViolations }),
          });
        }
        if (url.includes('/api/compliance/stats')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockStats),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      const deleteButtons = screen.getAllByRole('button', { name: '[DEL]' });
      await act(async () => {
        await user.click(deleteButtons[0]);
      });

      await waitFor(() => {
        expect(mockConfirm).toHaveBeenCalledWith('Are you sure you want to delete this policy?');
        expect(mockShowToast).toHaveBeenCalledWith('Policy deleted successfully', 'success');
      });
    });

    it('does not delete policy when confirmation is cancelled', async () => {
      const user = userEvent.setup();
      mockConfirm.mockReturnValue(false);
      setupSuccessfulFetch();

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      const deleteButtons = screen.getAllByRole('button', { name: '[DEL]' });
      await act(async () => {
        await user.click(deleteButtons[0]);
      });

      expect(mockConfirm).toHaveBeenCalled();
      // Should not call delete endpoint
      expect(mockFetch).not.toHaveBeenCalledWith(
        expect.stringContaining('/api/policies/policy-1'),
        expect.objectContaining({ method: 'DELETE' })
      );
    });
  });

  describe('retry functionality', () => {
    it('retries loading data when retry button is clicked', async () => {
      const user = userEvent.setup();
      let callCount = 0;
      mockFetch.mockImplementation((url: string) => {
        callCount++;
        if (callCount <= 3) {
          return Promise.reject(new Error('Network error'));
        }
        // Return proper data for each endpoint
        if (url.includes('/api/policies')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ policies: [] }),
          });
        }
        if (url.includes('/api/compliance/violations')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ violations: [] }),
          });
        }
        if (url.includes('/api/compliance/stats')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              policies: { total: 0, enabled: 0, disabled: 0 },
              violations: { total: 0, open: 0, by_severity: { critical: 0, high: 0, medium: 0, low: 0 } },
              risk_score: 0,
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.getByTestId('error-display')).toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByTestId('retry-button'));
      });

      await waitFor(() => {
        expect(screen.queryByTestId('error-display')).not.toBeInTheDocument();
      });
    });
  });

  describe('error handling', () => {
    it('shows error toast when policy creation fails', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes('/api/policies') && options?.method === 'POST') {
          return Promise.resolve({
            ok: false,
            status: 500,
          });
        }
        if (url.includes('/api/policies')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ policies: mockPolicies }),
          });
        }
        if (url.includes('/api/compliance/violations')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ violations: mockViolations }),
          });
        }
        if (url.includes('/api/compliance/stats')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockStats),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[+ NEW POLICY]' }));
      });

      await act(async () => {
        await user.type(screen.getByPlaceholderText('Policy name'), 'Test Policy');
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: '[CREATE POLICY]' }));
      });

      await waitFor(() => {
        expect(mockShowToast).toHaveBeenCalledWith('Failed to create policy', 'error');
      });
    });

    it('shows error toast when policy deletion fails', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes('/api/policies/policy-1') && options?.method === 'DELETE') {
          return Promise.resolve({
            ok: false,
            status: 500,
          });
        }
        if (url.includes('/api/policies')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ policies: mockPolicies }),
          });
        }
        if (url.includes('/api/compliance/violations')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ violations: mockViolations }),
          });
        }
        if (url.includes('/api/compliance/stats')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockStats),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<PolicyPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading policy data...')).not.toBeInTheDocument();
      });

      const deleteButtons = screen.getAllByRole('button', { name: '[DEL]' });
      await act(async () => {
        await user.click(deleteButtons[0]);
      });

      await waitFor(() => {
        expect(mockShowToast).toHaveBeenCalledWith('Failed to delete policy', 'error');
      });
    });
  });
});
