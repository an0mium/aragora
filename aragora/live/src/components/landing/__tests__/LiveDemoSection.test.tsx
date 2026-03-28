import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LiveDemoSection } from '../LiveDemoSection';

jest.mock('@/context/ThemeContext', () => ({
  useTheme: () => ({ theme: 'dark' }),
}));

describe('LiveDemoSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Object.defineProperty(window, 'scrollTo', {
      value: jest.fn(),
      writable: true,
    });
  });

  it('shows the landing-page live debate framing', () => {
    render(<LiveDemoSection />);

    expect(screen.getByText(/see it in action/i)).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: /watch a live debate unfold turn by turn/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/visitors can watch agents argue back and forth in real time/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/live debate streaming/i)).toBeInTheDocument();
  });

  it('renders a transcript-style stream with multiple agent turns', () => {
    render(<LiveDemoSection />);

    expect(screen.getByLabelText(/live debate transcript/i)).toBeInTheDocument();
    expect(screen.getByText('Strategic Analyst')).toBeInTheDocument();
    expect(screen.getByText("Devil's Advocate")).toBeInTheDocument();
    expect(screen.getByText('Implementation Expert')).toBeInTheDocument();
    expect(screen.getAllByTestId('stream-message')).toHaveLength(4);
    expect(screen.getByText(/approved with conditions/i)).toBeInTheDocument();
    expect(screen.getByText(/78% confidence/i)).toBeInTheDocument();
  });

  it('scrolls visitors back to the top when the CTA is clicked', async () => {
    const user = userEvent.setup();
    render(<LiveDemoSection />);

    await user.click(screen.getByRole('button', { name: /run your own debate/i }));

    expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'smooth' });
  });
});
