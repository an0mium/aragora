/**
 * Tests for the BriefPanel component.
 */
import { render, screen } from '@testing-library/react';
import { BriefPanel } from '../src/components/review-queue/BriefPanel';

describe('BriefPanel', () => {
  it('renders loading state', () => {
    render(<BriefPanel brief={null} loading />);
    expect(screen.getByTestId('brief-panel-loading')).toBeInTheDocument();
  });

  it('renders error state', () => {
    render(<BriefPanel brief={null} error="boom" />);
    expect(screen.getByTestId('brief-panel-error')).toHaveTextContent('boom');
  });

  it('renders empty state when brief is null', () => {
    render(<BriefPanel brief={null} />);
    expect(screen.getByTestId('brief-panel-empty')).toBeInTheDocument();
  });

  it('renders all role sections when populated', () => {
    render(
      <BriefPanel
        brief={{
          pr_number: 42,
          head_sha: 'abcdef1234567890',
          verdict: 'approve_candidate',
          confidence: 4,
          logic: 'flow looks fine',
          security: 'no surface changes',
          maintainability: 'small diff',
          skeptic: 'watch for flaky CI',
        }}
      />,
    );
    expect(screen.getByTestId('brief-verdict')).toHaveTextContent('approve_candidate');
    expect(screen.getByTestId('brief-confidence')).toHaveTextContent('confidence 4/5');
    expect(screen.getByTestId('brief-section-logic')).toHaveTextContent('flow looks fine');
    expect(screen.getByTestId('brief-section-security')).toHaveTextContent('no surface changes');
    expect(screen.getByTestId('brief-section-maintainability')).toHaveTextContent('small diff');
    expect(screen.getByTestId('brief-section-skeptic')).toHaveTextContent('watch for flaky CI');
  });

  it('shows em-dash for missing sections', () => {
    render(
      <BriefPanel
        brief={{
          pr_number: 42,
          head_sha: 'xxxxxxxxxxxxxxxx',
          verdict: 'needs_human_attention',
          confidence: null,
          logic: '',
          security: null,
          maintainability: '   ',
          skeptic: undefined as unknown as string,
        }}
      />,
    );
    expect(screen.getByTestId('brief-section-logic')).toHaveTextContent('—');
    expect(screen.getByTestId('brief-section-security')).toHaveTextContent('—');
    expect(screen.getByTestId('brief-section-maintainability')).toHaveTextContent('—');
    expect(screen.getByTestId('brief-section-skeptic')).toHaveTextContent('—');
    expect(screen.queryByTestId('brief-confidence')).toBeNull();
  });
});
