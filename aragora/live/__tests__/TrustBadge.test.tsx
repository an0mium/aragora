import { render, screen } from '@testing-library/react';
import { TrustBadge, type CalibrationData } from '../src/components/TrustBadge';

const EXCELLENT_CALIBRATION: CalibrationData = {
  brier_score: 0.05,
  ece: 0.02,
  trust_tier: 'excellent',
  prediction_count: 100,
};

const POOR_CALIBRATION: CalibrationData = {
  brier_score: 0.45,
  ece: 0.38,
  trust_tier: 'poor',
  prediction_count: 12,
};

const UNRATED_CALIBRATION: CalibrationData = {
  brier_score: 0,
  ece: 0,
  trust_tier: 'unrated',
  prediction_count: 0,
};

describe('TrustBadge', () => {
  it('returns null when calibration is null', () => {
    const { container } = render(<TrustBadge calibration={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('returns null when calibration is undefined', () => {
    const { container } = render(<TrustBadge />);
    expect(container.firstChild).toBeNull();
  });

  describe('size="md" (default)', () => {
    it('renders tier label', () => {
      render(<TrustBadge calibration={EXCELLENT_CALIBRATION} />);
      expect(screen.getByText('Excellent')).toBeInTheDocument();
    });

    it('does not render brier score text', () => {
      render(<TrustBadge calibration={EXCELLENT_CALIBRATION} />);
      expect(screen.queryByText('0.05')).not.toBeInTheDocument();
    });

    it('shows tooltip with calibration details', () => {
      render(<TrustBadge calibration={EXCELLENT_CALIBRATION} />);
      const badge = screen.getByText('Excellent').closest('span[title]');
      expect(badge).toHaveAttribute('title', 'Brier: 0.050 | ECE: 0.020 | 100 predictions');
    });
  });

  describe('size="sm"', () => {
    it('renders only the dot (no label text)', () => {
      render(<TrustBadge calibration={EXCELLENT_CALIBRATION} size="sm" />);
      expect(screen.queryByText('Excellent')).not.toBeInTheDocument();
    });

    it('still has a tooltip', () => {
      const { container } = render(
        <TrustBadge calibration={EXCELLENT_CALIBRATION} size="sm" />
      );
      const badge = container.querySelector('span[title]');
      expect(badge).toHaveAttribute('title', expect.stringContaining('Brier: 0.050'));
    });
  });

  describe('size="lg"', () => {
    it('renders tier label and brier score', () => {
      render(<TrustBadge calibration={EXCELLENT_CALIBRATION} size="lg" />);
      expect(screen.getByText('Excellent')).toBeInTheDocument();
      expect(screen.getByText('0.05')).toBeInTheDocument();
    });
  });

  describe('tier color mapping', () => {
    it.each([
      ['excellent', 'Excellent'],
      ['good', 'Good'],
      ['moderate', 'Moderate'],
      ['poor', 'Poor'],
      ['unrated', 'Unrated'],
    ] as const)('renders %s tier with correct label', (tier, expectedLabel) => {
      const data: CalibrationData = { ...EXCELLENT_CALIBRATION, trust_tier: tier };
      render(<TrustBadge calibration={data} />);
      expect(screen.getByText(expectedLabel)).toBeInTheDocument();
    });
  });

  describe('tooltip formatting', () => {
    it('formats poor calibration tooltip correctly', () => {
      render(<TrustBadge calibration={POOR_CALIBRATION} />);
      const badge = screen.getByText('Poor').closest('span[title]');
      expect(badge).toHaveAttribute('title', 'Brier: 0.450 | ECE: 0.380 | 12 predictions');
    });

    it('formats unrated calibration tooltip correctly', () => {
      render(<TrustBadge calibration={UNRATED_CALIBRATION} />);
      const badge = screen.getByText('Unrated').closest('span[title]');
      expect(badge).toHaveAttribute('title', 'Brier: 0.000 | ECE: 0.000 | 0 predictions');
    });
  });
});
