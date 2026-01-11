/**
 * Tests for Billing Page
 *
 * Tests cover:
 * - Loading state
 * - Usage data display
 * - Subscription info
 * - Invoice history
 * - Usage forecast
 * - Tab navigation
 * - Manage billing button
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import BillingPage from '../src/app/billing/page';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
  }),
}));

// Mock AuthContext
jest.mock('../src/context/AuthContext', () => ({
  useAuth: () => ({
    user: { id: 'user-1', email: 'test@example.com', name: 'Test User' },
    tokens: { access_token: 'test-token' },
    isLoading: false,
    isAuthenticated: true,
    organization: { tier: 'starter' },
  }),
}));

// Mock ProtectedRoute
jest.mock('../src/components/auth/ProtectedRoute', () => ({
  ProtectedRoute: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// Mock MatrixRain
jest.mock('../src/components/MatrixRain', () => ({
  Scanlines: () => null,
  CRTVignette: () => null,
}));

// Mock AsciiBanner
jest.mock('../src/components/AsciiBanner', () => ({
  AsciiBannerCompact: () => <div>ARAGORA</div>,
}));

const mockUsage = {
  debates_used: 15,
  debates_limit: 50,
  debates_remaining: 35,
  tokens_used: 125000,
  estimated_cost_usd: 2.50,
  period_start: '2026-01-01T00:00:00Z',
};

const mockSubscription = {
  tier: 'starter',
  status: 'active',
  is_active: true,
  current_period_end: '2026-02-01T00:00:00Z',
  cancel_at_period_end: false,
  limits: {
    debates_per_month: 50,
    users_per_org: 5,
    api_access: true,
  },
};

const mockInvoices = [
  {
    id: 'inv-1',
    number: 'INV-001',
    status: 'paid',
    amount_due: 2900,
    amount_paid: 29.00,
    currency: 'usd',
    created: '2026-01-01T00:00:00Z',
    period_start: '2026-01-01T00:00:00Z',
    period_end: '2026-02-01T00:00:00Z',
    hosted_invoice_url: 'https://stripe.com/invoice/1',
    invoice_pdf: 'https://stripe.com/invoice/1.pdf',
  },
  {
    id: 'inv-2',
    number: 'INV-002',
    status: 'open',
    amount_due: 2900,
    amount_paid: 0,
    currency: 'usd',
    created: '2026-01-10T00:00:00Z',
    period_start: null,
    period_end: null,
    hosted_invoice_url: 'https://stripe.com/invoice/2',
    invoice_pdf: null,
  },
];

const mockForecast = {
  projected_debates: 45,
  cost_end_of_cycle_usd: 7.50,
  days_remaining: 21,
  will_hit_limit: false,
  tier_recommendation: null,
};

describe('BillingPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  const setupMocks = (options: {
    usageOk?: boolean;
    subscriptionOk?: boolean;
    invoicesOk?: boolean;
    forecastOk?: boolean;
  } = {}) => {
    const { usageOk = true, subscriptionOk = true, invoicesOk = true, forecastOk = true } = options;

    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/usage') && !url.includes('/forecast')) {
        return Promise.resolve({
          ok: usageOk,
          json: () => Promise.resolve({ usage: mockUsage }),
        });
      }
      if (url.includes('/subscription')) {
        return Promise.resolve({
          ok: subscriptionOk,
          json: () => Promise.resolve({ subscription: mockSubscription }),
        });
      }
      if (url.includes('/invoices')) {
        return Promise.resolve({
          ok: invoicesOk,
          json: () => Promise.resolve({ invoices: mockInvoices }),
        });
      }
      if (url.includes('/forecast')) {
        return Promise.resolve({
          ok: forecastOk,
          json: () => Promise.resolve({ forecast: mockForecast }),
        });
      }
      return Promise.resolve({ ok: false });
    });
  };

  describe('Loading State', () => {
    it('shows loading state initially', () => {
      mockFetch.mockImplementation(() => new Promise(() => {}));

      render(<BillingPage />);

      expect(screen.getByText(/loading billing data/i)).toBeInTheDocument();
    });
  });

  describe('Overview Tab', () => {
    it('displays current plan', async () => {
      setupMocks();

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText('STARTER')).toBeInTheDocument();
        expect(screen.getByText(/active/i)).toBeInTheDocument();
      });
    });

    it('displays usage data', async () => {
      setupMocks();

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText(/15 \/ 50/)).toBeInTheDocument();
        expect(screen.getByText(/35 remaining/i)).toBeInTheDocument();
      });
    });

    it('displays token usage and cost', async () => {
      setupMocks();

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText(/125,000/)).toBeInTheDocument();
        expect(screen.getByText(/\$2.50/)).toBeInTheDocument();
      });
    });

    it('displays usage forecast', async () => {
      setupMocks();

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText(/45/)).toBeInTheDocument(); // Projected debates
        expect(screen.getByText(/21/)).toBeInTheDocument(); // Days remaining
      });
    });

    it('displays plan features', async () => {
      setupMocks();

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText(/50/)).toBeInTheDocument(); // Debates per month
        expect(screen.getByText(/enabled/i)).toBeInTheDocument(); // API access
      });
    });
  });

  describe('Tab Navigation', () => {
    it('switches to invoices tab', async () => {
      setupMocks();

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText('OVERVIEW')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText(/invoices/i));

      await waitFor(() => {
        expect(screen.getByText('INVOICE HISTORY')).toBeInTheDocument();
      });
    });
  });

  describe('Invoices Tab', () => {
    it('displays invoice list', async () => {
      setupMocks();

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText('OVERVIEW')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText(/invoices/i));

      await waitFor(() => {
        expect(screen.getByText('INV-001')).toBeInTheDocument();
        expect(screen.getByText('INV-002')).toBeInTheDocument();
      });
    });

    it('shows invoice status badges', async () => {
      setupMocks();

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText('OVERVIEW')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText(/invoices/i));

      await waitFor(() => {
        expect(screen.getByText('PAID')).toBeInTheDocument();
        expect(screen.getByText('OPEN')).toBeInTheDocument();
      });
    });

    it('shows view and PDF links for invoices', async () => {
      setupMocks();

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText('OVERVIEW')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText(/invoices/i));

      await waitFor(() => {
        const viewLinks = screen.getAllByText('[VIEW]');
        expect(viewLinks.length).toBeGreaterThan(0);

        const pdfLinks = screen.getAllByText('[PDF]');
        expect(pdfLinks.length).toBeGreaterThan(0);
      });
    });

    it('shows empty message when no invoices', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/invoices')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ invoices: [] }),
          });
        }
        return setupMocks() as never;
      });

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText('OVERVIEW')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText(/invoices/i));

      await waitFor(() => {
        expect(screen.getByText(/no invoices found/i)).toBeInTheDocument();
      });
    });
  });

  describe('Forecast Warnings', () => {
    it('shows warning when will hit limit', async () => {
      const limitForecast = {
        ...mockForecast,
        will_hit_limit: true,
        tier_recommendation: 'professional',
      };

      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/forecast')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ forecast: limitForecast }),
          });
        }
        if (url.includes('/usage') && !url.includes('/forecast')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ usage: mockUsage }),
          });
        }
        if (url.includes('/subscription')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ subscription: mockSubscription }),
          });
        }
        if (url.includes('/invoices')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ invoices: mockInvoices }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText(/may exceed your limit/i)).toBeInTheDocument();
        expect(screen.getByText(/professional/i)).toBeInTheDocument();
      });
    });
  });

  describe('Manage Subscription', () => {
    it('shows manage button for paid tiers', async () => {
      setupMocks();

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText('MANAGE SUBSCRIPTION')).toBeInTheDocument();
      });
    });

    it('opens billing portal on click', async () => {
      setupMocks();

      // Mock portal response
      mockFetch.mockImplementation((url: string, options?: RequestInit) => {
        if (options?.method === 'POST' && url.includes('/portal')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              portal: { url: 'https://billing.stripe.com/session' },
            }),
          });
        }
        // Default responses for GET requests
        if (url.includes('/usage') && !url.includes('/forecast')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ usage: mockUsage }),
          });
        }
        if (url.includes('/subscription')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ subscription: mockSubscription }),
          });
        }
        if (url.includes('/invoices')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ invoices: mockInvoices }),
          });
        }
        if (url.includes('/forecast')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ forecast: mockForecast }),
          });
        }
        return Promise.resolve({ ok: false });
      });

      render(<BillingPage />);

      await waitFor(() => {
        expect(screen.getByText('MANAGE SUBSCRIPTION')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText('MANAGE SUBSCRIPTION'));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/billing/portal'),
          expect.objectContaining({ method: 'POST' })
        );
      });
    });
  });
});
