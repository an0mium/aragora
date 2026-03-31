import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { ReceiptShareModal } from '../ReceiptShareModal';

jest.mock('@/context/AuthContext', () => ({
  useAuth: () => ({
    tokens: { access_token: 'share-token' },
  }),
}));

const mockFetch = jest.fn();
global.fetch = mockFetch as unknown as typeof fetch;

const originalClipboard = navigator.clipboard;
const mockClipboardWriteText = jest.fn().mockResolvedValue(undefined);

beforeAll(() => {
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: mockClipboardWriteText,
    },
  });
});

afterAll(() => {
  Object.defineProperty(navigator, 'clipboard', {
    configurable: true,
    value: originalClipboard,
  });
});

function jsonResponse(data: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: async () => data,
  } as Response;
}

describe('ReceiptShareModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('creates a share link with the v2 receipt endpoint and copies the absolute URL', async () => {
    mockFetch.mockResolvedValue(
      jsonResponse({
        share_url: '/api/v2/receipts/share/test-token',
        expires_at: '2026-03-02T00:00:00Z',
      })
    );

    const user = userEvent.setup();

    render(
      <ReceiptShareModal
        isOpen
        onClose={jest.fn()}
        receiptId="receipt-123"
        receiptSummary="Deployment rollback receipt"
        apiUrl="http://localhost:8080"
      />
    );

    await user.type(screen.getByLabelText(/max opens/i), '5');
    await user.click(screen.getByRole('button', { name: /create share link/i }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/v2/receipts/receipt-123/share',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            Authorization: 'Bearer share-token',
          }),
          body: JSON.stringify({
            expires_in_hours: 24,
            max_accesses: 5,
          }),
        })
      );
    });

    await waitFor(() => {
      expect(screen.getByLabelText('Receipt share link')).toHaveValue(
        'http://localhost/api/v2/receipts/share/test-token'
      );
      expect(screen.getByText('Copied share link')).toBeInTheDocument();
      expect(screen.getByText(/Expires/)).toBeInTheDocument();
    });
  });

  it('validates max opens before calling the backend', async () => {
    const user = userEvent.setup();

    render(
      <ReceiptShareModal
        isOpen
        onClose={jest.fn()}
        receiptId="receipt-123"
        apiUrl="http://localhost:8080"
      />
    );

    await user.type(screen.getByLabelText(/max opens/i), '0');
    await user.click(screen.getByRole('button', { name: /create share link/i }));

    expect(mockFetch).not.toHaveBeenCalled();
    expect(screen.getByText('Max accesses must be a positive whole number')).toBeInTheDocument();
  });
});
