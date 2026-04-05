import { fireEvent, render, screen } from '@testing-library/react';
import DocsPage from '../page';
import { getRuntimeBackendConfig } from '@/lib/runtimeBackend';

jest.mock('next/link', () => {
  return function MockLink({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) {
    return <a href={href}>{children}</a>;
  };
});

jest.mock('@/lib/runtimeBackend', () => ({
  getRuntimeBackendConfig: jest.fn(),
}));

const mockGetRuntimeBackendConfig = getRuntimeBackendConfig as jest.MockedFunction<
  typeof getRuntimeBackendConfig
>;

describe('DocsPage', () => {
  beforeEach(() => {
    mockGetRuntimeBackendConfig.mockReset();
  });

  it('loads docs through the same-origin proxy when the runtime backend uses a relative API base', () => {
    mockGetRuntimeBackendConfig.mockReturnValue({
      backend: 'development',
      config: {
        api: '',
        ws: 'ws://localhost:8765/ws',
        label: 'DEV',
        description: 'Local host',
      },
    });

    render(<DocsPage />);

    expect(screen.queryByText(/set .*NEXT_PUBLIC_API_URL/i)).not.toBeInTheDocument();
    expect(screen.getByTitle('API Documentation - swagger')).toHaveAttribute('src', '/api/v2/docs');
  });

  it('switches the iframe between swagger and redoc using the runtime backend API base', () => {
    mockGetRuntimeBackendConfig.mockReturnValue({
      backend: 'production',
      config: {
        api: 'https://api.aragora.ai',
        ws: 'wss://api.aragora.ai/ws',
        label: 'PROD',
        description: 'Production',
      },
    });

    render(<DocsPage />);

    expect(screen.getByTitle('API Documentation - swagger')).toHaveAttribute(
      'src',
      'https://api.aragora.ai/api/v2/docs',
    );

    fireEvent.click(screen.getByRole('button', { name: 'REDOC' }));

    expect(screen.getByTitle('API Documentation - redoc')).toHaveAttribute(
      'src',
      'https://api.aragora.ai/api/v2/redoc',
    );
  });
});
