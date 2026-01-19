/**
 * Tests for SkipLink components
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import { SkipLink, SkipLinkTarget } from '@/components/shared/SkipLink';

describe('SkipLink', () => {
  it('renders with correct href', () => {
    render(<SkipLink targetId="main-content" />);

    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', '#main-content');
  });

  it('renders default text', () => {
    render(<SkipLink targetId="main" />);

    expect(screen.getByText('Skip to main content')).toBeInTheDocument();
  });

  it('renders custom text', () => {
    render(<SkipLink targetId="main">Skip navigation</SkipLink>);

    expect(screen.getByText('Skip navigation')).toBeInTheDocument();
  });

  it('is visually hidden by default', () => {
    render(<SkipLink targetId="main" />);

    const link = screen.getByRole('link');
    expect(link).toHaveClass('sr-only');
  });

  it('becomes visible on focus', () => {
    render(<SkipLink targetId="main" />);

    const link = screen.getByRole('link');
    // Check that focus:not-sr-only class is present
    expect(link.className).toContain('focus:not-sr-only');
  });
});

describe('SkipLinkTarget', () => {
  it('renders with correct id', () => {
    render(<SkipLinkTarget id="main-content" />);

    const target = document.getElementById('main-content');
    expect(target).toBeInTheDocument();
  });

  it('is not focusable by default', () => {
    render(<SkipLinkTarget id="main" />);

    const target = document.getElementById('main');
    expect(target).toHaveAttribute('tabIndex', '-1');
  });

  it('is hidden from screen readers', () => {
    render(<SkipLinkTarget id="main" />);

    const target = document.getElementById('main');
    expect(target).toHaveAttribute('aria-hidden', 'true');
  });

  it('is visually hidden', () => {
    render(<SkipLinkTarget id="main" />);

    const target = document.getElementById('main');
    expect(target).toHaveClass('sr-only');
  });
});
