import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import IntegrationsPage from '../page';

jest.mock('next/link', () => {
  return function MockLink({
    children,
    href,
    target,
    className,
  }: {
    children: React.ReactNode;
    href: string;
    target?: string;
    className?: string;
  }) {
    return (
      <a href={href} target={target} className={className}>
        {children}
      </a>
    );
  };
});

jest.mock('@/components/MatrixRain', () => ({
  Scanlines: () => <div data-testid="scanlines" />,
  CRTVignette: () => <div data-testid="crt-vignette" />,
}));

const mockBackendConfig = { api: 'http://localhost:8080' };
jest.mock('@/components/BackendSelector', () => ({
  useBackend: () => ({ config: mockBackendConfig }),
}));

jest.mock('@/context/AuthContext', () => ({
  useAuth: () => ({
    tokens: { access_token: 'test-token' },
  }),
}));

jest.mock('@/components/integrations', () => ({
  IntegrationSetupWizard: () => <div data-testid="integration-setup-wizard" />,
  IntegrationStatusDashboard: () => <div data-testid="integration-status-dashboard" />,
  INTEGRATION_CONFIGS: {
    slack: { title: 'Slack' },
    discord: { title: 'Discord' },
    telegram: { title: 'Telegram' },
    email: { title: 'Email' },
    teams: { title: 'Teams' },
    whatsapp: { title: 'WhatsApp' },
    matrix: { title: 'Matrix' },
  },
}));

const mockFetch = jest.fn();
global.fetch = mockFetch as unknown as typeof fetch;

describe('IntegrationsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockFetch.mockImplementation((url: string) => {
      if (url.endsWith('/api/v1/integrations/health')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ integrations: [] }),
        } as Response);
      }

      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({}),
      } as Response);
    });
  });

  it('checks training export availability through the real stats endpoint', async () => {
    const user = userEvent.setup();

    render(<IntegrationsPage />);

    await user.click(screen.getByRole('button', { name: '[SYSTEM]' }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/v1/training/stats',
        expect.objectContaining({ method: 'GET' }),
      );
    });

    expect(screen.getByText('/api/v1/training/stats')).toBeInTheDocument();
  });
});
