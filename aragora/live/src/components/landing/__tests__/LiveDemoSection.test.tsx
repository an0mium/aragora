import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LiveDemoSection } from '../LiveDemoSection';

jest.mock('@/context/ThemeContext', () => ({
  useTheme: () => ({ theme: 'dark', setTheme: jest.fn() }),
}));

describe('LiveDemoSection', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    window.scrollTo = jest.fn();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  it('renders the live debate preview and opening turn', () => {
    render(<LiveDemoSection />);

    expect(
      screen.getByRole('heading', { name: /watch agents argue in real time/i })
    ).toBeInTheDocument();
    expect(screen.getByText(/live now/i)).toBeInTheDocument();
    expect(
      screen.getByText(/should we split our monolith into services before the next product launch/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/claude \| strategic analyst/i)).toBeInTheDocument();
    expect(
      screen.getByText(/split billing and notifications first/i)
    ).toBeInTheDocument();
  });

  it('advances through the transcript over time and loops back to the opening turn', () => {
    render(<LiveDemoSection />);

    act(() => {
      jest.advanceTimersByTime(2600);
    });
    expect(screen.getByText(/gpt-4 \| skeptical operator/i)).toBeInTheDocument();
    expect(
      screen.getByText(/fix release discipline and test isolation/i)
    ).toBeInTheDocument();

    act(() => {
      jest.advanceTimersByTime(2600);
    });
    expect(screen.getByText(/gemini \| systems synthesizer/i)).toBeInTheDocument();
    expect(
      screen.getByText(/extract only the order ingest path/i)
    ).toBeInTheDocument();

    act(() => {
      jest.advanceTimersByTime(2600);
    });
    expect(screen.getByText(/approved with conditions/i)).toBeInTheDocument();

    act(() => {
      jest.advanceTimersByTime(2600);
    });
    expect(
      screen.queryByText(/gpt-4 \| skeptical operator/i)
    ).not.toBeInTheDocument();
    expect(screen.getByText(/claude \| strategic analyst/i)).toBeInTheDocument();
  });

  it('scrolls visitors back to the hero when the CTA is clicked', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });

    render(<LiveDemoSection />);

    await user.click(screen.getByRole('button', { name: /run your own debate/i }));

    expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'smooth' });
  });
});
