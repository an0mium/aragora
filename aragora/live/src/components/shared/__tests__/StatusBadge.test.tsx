/**
 * Tests for StatusBadge component
 *
 * Tests cover:
 * - Rendering with different variants
 * - Size variations
 * - Custom className
 * - Default props
 */

import { render, screen } from '@testing-library/react';
import { StatusBadge, type BadgeVariant } from '../StatusBadge';

describe('StatusBadge', () => {
  describe('rendering', () => {
    it('renders label text', () => {
      render(<StatusBadge label="Active" />);
      expect(screen.getByText('Active')).toBeInTheDocument();
    });

    it('renders as a span element', () => {
      render(<StatusBadge label="Test" />);
      expect(screen.getByText('Test').tagName).toBe('SPAN');
    });
  });

  describe('variants', () => {
    const variants: BadgeVariant[] = [
      'success',
      'warning',
      'error',
      'info',
      'neutral',
      'purple',
      'orange',
    ];

    variants.forEach((variant) => {
      it(`applies ${variant} variant styles`, () => {
        render(<StatusBadge label="Test" variant={variant} />);
        const badge = screen.getByText('Test');
        expect(badge).toHaveClass('rounded');
        expect(badge).toHaveClass('border');
      });
    });

    it('defaults to neutral variant', () => {
      render(<StatusBadge label="Default" />);
      const badge = screen.getByText('Default');
      expect(badge).toHaveClass('bg-surface');
      expect(badge).toHaveClass('text-text-muted');
    });

    it('applies success variant colors', () => {
      render(<StatusBadge label="Success" variant="success" />);
      const badge = screen.getByText('Success');
      expect(badge).toHaveClass('text-green-400');
    });

    it('applies error variant colors', () => {
      render(<StatusBadge label="Error" variant="error" />);
      const badge = screen.getByText('Error');
      expect(badge).toHaveClass('text-red-400');
    });
  });

  describe('sizes', () => {
    it('applies small size by default', () => {
      render(<StatusBadge label="Small" />);
      const badge = screen.getByText('Small');
      expect(badge).toHaveClass('text-xs');
      expect(badge).toHaveClass('px-2');
      expect(badge).toHaveClass('py-0.5');
    });

    it('applies small size explicitly', () => {
      render(<StatusBadge label="Small" size="sm" />);
      const badge = screen.getByText('Small');
      expect(badge).toHaveClass('text-xs');
    });

    it('applies medium size', () => {
      render(<StatusBadge label="Medium" size="md" />);
      const badge = screen.getByText('Medium');
      expect(badge).toHaveClass('text-sm');
      expect(badge).toHaveClass('px-3');
      expect(badge).toHaveClass('py-1');
    });
  });

  describe('custom className', () => {
    it('applies custom className', () => {
      render(<StatusBadge label="Custom" className="my-custom-class" />);
      const badge = screen.getByText('Custom');
      expect(badge).toHaveClass('my-custom-class');
    });

    it('merges custom className with default classes', () => {
      render(<StatusBadge label="Merged" className="extra-class" />);
      const badge = screen.getByText('Merged');
      expect(badge).toHaveClass('rounded');
      expect(badge).toHaveClass('border');
      expect(badge).toHaveClass('extra-class');
    });
  });
});
