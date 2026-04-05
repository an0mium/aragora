import { renderWithProviders, screen } from '@/test-utils';
import QuickstartPage from '../page';

jest.mock('@/components/landing/Header', () => ({
  Header: () => <div data-testid="landing-header">Header</div>,
}));

jest.mock('@/components/landing/Footer', () => ({
  Footer: () => <div data-testid="landing-footer">Footer</div>,
}));

jest.mock('@/components/openrouter/ConnectOpenRouterButton', () => ({
  ConnectOpenRouterButton: () => (
    <button type="button">Connect OpenRouter</button>
  ),
}));

describe('QuickstartPage', () => {
  it('documents the current CLI-first onboarding flow', () => {
    renderWithProviders(<QuickstartPage />);

    expect(screen.getByText('pip install aragora-debate')).toBeInTheDocument();
    expect(
      screen.getByText('python -m aragora quickstart --demo --no-browser'),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'python -m aragora quickstart --question "Should we rewrite this service in Go?" --no-browser',
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'python -m aragora receipt inspect .aragora/receipts/quickstart-live-receipt.json',
      ),
    ).toBeInTheDocument();

    expect(screen.queryByText('aragora debate "your question"')).not.toBeInTheDocument();
    expect(screen.queryByText(/from aragora_debate\.arena import Arena/)).not.toBeInTheDocument();
  });
});
