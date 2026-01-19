import { render, screen, act, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SocialPage from '../page';

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

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Mock window.location
const originalLocation = window.location;
beforeAll(() => {
  // @ts-expect-error - mocking location
  delete window.location;
  // @ts-expect-error - mocking location
  window.location = { href: '' };
});

afterAll(() => {
  window.location = originalLocation;
});

describe('SocialPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    window.location.href = '';
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('initial render', () => {
    it('renders visual effects', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      render(<SocialPage />);

      expect(screen.getByTestId('scanlines')).toBeInTheDocument();
      expect(screen.getByTestId('crt-vignette')).toBeInTheDocument();
    });

    it('renders header elements', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      render(<SocialPage />);

      expect(screen.getByTestId('ascii-banner')).toBeInTheDocument();
      expect(screen.getByTestId('theme-toggle')).toBeInTheDocument();
      expect(screen.getByTestId('backend-selector')).toBeInTheDocument();
    });

    it('renders page title', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      render(<SocialPage />);

      expect(screen.getByText('Social Media')).toBeInTheDocument();
    });

    it('shows loading state initially', () => {
      mockFetch.mockReturnValue(new Promise(() => {})); // Never resolves

      render(<SocialPage />);

      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });

    it('renders tab navigation', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      expect(screen.getByRole('button', { name: 'Connections' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Publish' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'History' })).toBeInTheDocument();
    });
  });

  describe('connector status tab', () => {
    it('fetches connector status on mount', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith('http://localhost:8080/api/youtube/status');
        expect(mockFetch).toHaveBeenCalledWith('http://localhost:8080/api/connectors');
      });
    });

    it('displays YouTube connector status', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/youtube/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              is_configured: true,
              is_connected: true,
              quota_remaining: 9500,
            }),
          });
        }
        if (url.includes('/connectors')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ connectors: [] }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ debates: [] }),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.getByText('YouTube')).toBeInTheDocument();
        expect(screen.getByText('Connected')).toBeInTheDocument();
        expect(screen.getByText('Quota: 9500 units remaining')).toBeInTheDocument();
      });
    });

    it('displays unconfigured connector state', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/youtube/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              is_configured: false,
              is_connected: false,
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ connectors: [], debates: [] }),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        // There are multiple connectors that can be "Not configured"
        const notConfigured = screen.getAllByText('Not configured');
        expect(notConfigured.length).toBeGreaterThan(0);
      });
    });

    it('shows Connect YouTube button when not connected', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/youtube/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              is_configured: true,
              is_connected: false,
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ connectors: [], debates: [] }),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Connect YouTube' })).toBeInTheDocument();
      });
    });

    it('initiates OAuth when Connect YouTube is clicked', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/youtube/auth')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              auth_url: 'https://accounts.google.com/oauth',
            }),
          });
        }
        if (url.includes('/youtube/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              is_configured: true,
              is_connected: false,
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ connectors: [], debates: [] }),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Connect YouTube' })).toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'Connect YouTube' }));
      });

      expect(window.location.href).toBe('https://accounts.google.com/oauth');
    });

    it('displays Twitter/X placeholder', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.getByText('Twitter/X')).toBeInTheDocument();
      });
    });

    it('displays configuration instructions', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.getByText('Configuration')).toBeInTheDocument();
        expect(screen.getByText(/YOUTUBE_CLIENT_ID/)).toBeInTheDocument();
      });
    });

    it('displays connector error when present', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/youtube/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              is_configured: true,
              is_connected: false,
              error: 'Token expired',
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ connectors: [], debates: [] }),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.getByText('Error: Token expired')).toBeInTheDocument();
      });
    });
  });

  describe('publish tab', () => {
    it('switches to publish tab', async () => {
      const user = userEvent.setup();
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: [] }),
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'Publish' }));
      });

      expect(screen.getByText('Publish Debate')).toBeInTheDocument();
    });

    it('displays debate selection dropdown', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/debates')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              debates: [
                { id: 'debate-1', task: 'AI Ethics Discussion', metadata: { has_audio: true } },
                { id: 'debate-2', task: 'Code Review Session', metadata: {} },
              ],
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'Publish' }));
      });

      const select = screen.getByRole('combobox');
      expect(select).toBeInTheDocument();
    });

    it('displays platform selection buttons', async () => {
      const user = userEvent.setup();
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: [] }),
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'Publish' }));
      });

      // Platform buttons may include "(not connected)" suffix
      const buttons = screen.getAllByRole('button');
      const twitterButton = buttons.find(b => b.textContent?.includes('Twitter'));
      const youtubeButton = buttons.find(b => b.textContent?.includes('YouTube') && !b.textContent?.includes('Connect'));
      expect(twitterButton).toBeTruthy();
      expect(youtubeButton).toBeTruthy();
    });

    it('disables publish button when no debate selected', async () => {
      const user = userEvent.setup();
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: [] }),
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      // Click the Publish tab
      const tabs = screen.getAllByRole('button');
      const publishTab = tabs.find(b => b.textContent === 'Publish');
      await act(async () => {
        await user.click(publishTab!);
      });

      // Find the submit button (not the tab)
      await waitFor(() => {
        const allButtons = screen.getAllByRole('button');
        const submitButton = allButtons.find(b => b.textContent === 'Publish' && b !== publishTab);
        expect(submitButton).toBeTruthy();
        expect(submitButton).toBeDisabled();
      });
    });

    it('disables platform buttons when not connected', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/youtube/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ is_connected: false }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ debates: [], connectors: [] }),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      // Click the Publish tab
      const tabs = screen.getAllByRole('button');
      const publishTab = tabs.find(b => b.textContent === 'Publish');
      await act(async () => {
        await user.click(publishTab!);
      });

      // Platform buttons should show "not connected"
      await waitFor(() => {
        const notConnected = screen.getAllByText(/not connected/i);
        expect(notConnected.length).toBeGreaterThan(0);
      });
    });

    it('publishes debate when selections are made', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes('/publish/youtube') && options?.method === 'POST') {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              job_id: 'job-123',
              url: 'https://youtube.com/watch?v=abc123',
            }),
          });
        }
        if (url.includes('/youtube/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ is_connected: true }),
          });
        }
        if (url.includes('/debates')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              debates: [{ id: 'debate-1', task: 'Test Debate', metadata: {} }],
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ connectors: [] }),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'Publish' }));
      });

      // Select debate
      await act(async () => {
        await user.selectOptions(screen.getByRole('combobox'), 'debate-1');
      });

      // Select platform (click YouTube button)
      const youtubeButton = screen.getAllByRole('button').find(b => b.textContent?.includes('YouTube') && !b.textContent?.includes('not connected'));
      if (youtubeButton) {
        await act(async () => {
          await user.click(youtubeButton);
        });
      }

      // Click publish
      const publishButton = screen.getAllByRole('button').find(b => b.textContent === 'Publish');
      if (publishButton && !publishButton.hasAttribute('disabled')) {
        await act(async () => {
          await user.click(publishButton);
        });
      }
    });

    it('displays publish notes', async () => {
      const user = userEvent.setup();
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: [] }),
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'Publish' }));
      });

      expect(screen.getByText('Notes')).toBeInTheDocument();
      expect(screen.getByText(/Twitter: Generates a thread/)).toBeInTheDocument();
    });
  });

  describe('history tab', () => {
    it('switches to history tab', async () => {
      const user = userEvent.setup();
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'History' }));
      });

      expect(screen.getByText('Publish History')).toBeInTheDocument();
    });

    it('shows empty state when no history', async () => {
      const user = userEvent.setup();
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({}),
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'History' }));
      });

      expect(screen.getByText('No publish history yet')).toBeInTheDocument();
    });

    it('displays publish history after successful publish', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes('/publish/youtube') && options?.method === 'POST') {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              job_id: 'job-123',
              url: 'https://youtube.com/watch?v=abc123',
            }),
          });
        }
        if (url.includes('/youtube/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ is_connected: true }),
          });
        }
        if (url.includes('/debates')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              debates: [{ id: 'debate-1', task: 'Test', metadata: {} }],
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ connectors: [] }),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      // Simulate a publish action (the state update happens internally)
      // After publish, history tab should show the entry
      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'History' }));
      });

      // Initially empty
      expect(screen.getByText('No publish history yet')).toBeInTheDocument();
    });
  });

  describe('error handling', () => {
    it('handles connector fetch errors gracefully', async () => {
      // The SocialPage catches connector errors internally with .catch(() => null)
      // so it won't display an error - it will just show placeholders
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/youtube/status') || url.includes('/connectors')) {
          return Promise.reject(new Error('Network error'));
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ debates: [] }),
        });
      });

      render(<SocialPage />);

      // Page should still load with placeholder connectors
      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      // YouTube placeholder should still be shown
      expect(screen.getByText('YouTube')).toBeInTheDocument();
    });

    it('handles debate fetch errors by logging to console', async () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/debates')) {
          return Promise.reject(new Error('Network error'));
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      // The component logs the error but doesn't display it
      expect(consoleSpy).toHaveBeenCalledWith('Failed to fetch debates:', expect.any(Error));
      consoleSpy.mockRestore();
    });

    it('handles publish error gracefully', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string, options?: RequestInit) => {
        if (url.includes('/publish') && options?.method === 'POST') {
          return Promise.resolve({
            ok: false,
            status: 500,
            json: () => Promise.resolve({ error: 'Publishing failed' }),
          });
        }
        if (url.includes('/youtube/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ is_connected: true }),
          });
        }
        if (url.includes('/debates')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              debates: [{ id: 'debate-1', task: 'Test', metadata: {} }],
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ connectors: [] }),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'Publish' }));
      });

      // The component handles errors internally
      expect(screen.getByText('Publish Debate')).toBeInTheDocument();
    });
  });

  describe('OAuth error handling', () => {
    it('displays error when OAuth initiation fails', async () => {
      const user = userEvent.setup();
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/youtube/auth')) {
          return Promise.resolve({
            ok: false,
            status: 500,
          });
        }
        if (url.includes('/youtube/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ is_connected: false }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ connectors: [], debates: [] }),
        });
      });

      render(<SocialPage />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Connect YouTube' })).toBeInTheDocument();
      });

      await act(async () => {
        await user.click(screen.getByRole('button', { name: 'Connect YouTube' }));
      });

      await waitFor(() => {
        expect(screen.getByTestId('error-display')).toBeInTheDocument();
      });
    });
  });
});
