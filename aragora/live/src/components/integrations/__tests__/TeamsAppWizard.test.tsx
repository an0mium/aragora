import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { TeamsAppWizard } from '../TeamsAppWizard';

describe('TeamsAppWizard', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  it('refreshes the connected tenant from the live v1 routes after admin consent closes', async () => {
    const popup = { closed: false } as Window;
    const openSpy = jest.spyOn(window, 'open').mockReturnValue(popup);
    const fetchMock = global.fetch as jest.MockedFunction<typeof fetch>;
    let hasTenant = false;

    fetchMock.mockImplementation((input) => {
      const url = String(input);

      if (url === 'https://api.example.com/api/v1/integrations/teams/status') {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            app_id_configured: true,
            password_configured: true,
          }),
        } as Response);
      }

      if (url === 'https://api.example.com/api/v1/sme/teams/tenants') {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            workspaces: hasTenant
              ? [
                  {
                    tenant_id: 'tenant-123',
                    tenant_name: 'Contoso',
                    installed_at_iso: '2026-03-31T18:00:00Z',
                    is_active: true,
                  },
                ]
              : [],
          }),
        } as Response);
      }

      return Promise.reject(new Error(`Unexpected fetch: ${url}`));
    });

    render(
      <TeamsAppWizard
        onClose={jest.fn()}
        onComplete={jest.fn()}
        apiBaseUrl="https://api.example.com"
      />
    );

    expect(await screen.findByRole('button', { name: '[START ADMIN CONSENT]' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '[START ADMIN CONSENT]' }));

    expect(openSpy).toHaveBeenCalledWith(
      expect.stringContaining('https://api.example.com/api/v1/sme/teams/oauth/start?host='),
      'teams-oauth',
      expect.stringContaining('popup=yes')
    );

    hasTenant = true;
    popup.closed = true;

    await act(async () => {
      jest.advanceTimersByTime(1000);
    });

    await waitFor(() => {
      expect(screen.getByText('Contoso')).toBeInTheDocument();
    });

    expect(screen.getByRole('button', { name: '[RUN CONNECTION TEST]' })).toBeInTheDocument();
  });

  it('runs the tenant health check against the workspace test endpoint', async () => {
    const fetchMock = global.fetch as jest.MockedFunction<typeof fetch>;

    fetchMock.mockImplementation((input, init) => {
      const url = String(input);

      if (url === 'https://api.example.com/api/v1/integrations/teams/status') {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            app_id_configured: true,
            password_configured: true,
          }),
        } as Response);
      }

      if (url === 'https://api.example.com/api/v1/sme/teams/tenants') {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            workspaces: [
              {
                tenant_id: 'tenant-123',
                tenant_name: 'Contoso',
                installed_at_iso: '2026-03-31T18:00:00Z',
                is_active: true,
              },
            ],
          }),
        } as Response);
      }

      if (url === 'https://api.example.com/api/v1/sme/teams/tenants/tenant-123/test') {
        expect(init).toEqual(expect.objectContaining({ method: 'POST' }));
        return Promise.resolve({
          ok: true,
          json: async () => ({
            status: 'connected',
          }),
        } as Response);
      }

      return Promise.reject(new Error(`Unexpected fetch: ${url}`));
    });

    render(
      <TeamsAppWizard
        onClose={jest.fn()}
        onComplete={jest.fn()}
        apiBaseUrl="https://api.example.com"
      />
    );

    fireEvent.click(await screen.findByRole('button', { name: '[RUN CONNECTION TEST]' }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        'https://api.example.com/api/v1/sme/teams/tenants/tenant-123/test',
        expect.objectContaining({ method: 'POST' })
      );
    });

    expect(await screen.findByText('Teams credentials validated successfully.')).toBeInTheDocument();
  });
});
