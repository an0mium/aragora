/**
 * Tests for ConnectionStatus component
 */

import { render, screen } from '@testing-library/react';
import { ConnectionStatus } from '../src/components/ConnectionStatus';

describe('ConnectionStatus', () => {
  describe('Connected State', () => {
    it('displays "Connected" text when connected', () => {
      render(<ConnectionStatus connected={true} />);
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    it('shows green indicator when connected', () => {
      const { container } = render(<ConnectionStatus connected={true} />);
      const indicator = container.querySelector('.bg-success');
      expect(indicator).toBeInTheDocument();
    });

    it('has pulse animation when connected', () => {
      const { container } = render(<ConnectionStatus connected={true} />);
      const indicator = container.querySelector('.animate-pulse');
      expect(indicator).toBeInTheDocument();
    });
  });

  describe('Disconnected State', () => {
    it('displays "Disconnected" text when not connected', () => {
      render(<ConnectionStatus connected={false} />);
      expect(screen.getByText('Disconnected')).toBeInTheDocument();
    });

    it('shows warning indicator when disconnected', () => {
      const { container } = render(<ConnectionStatus connected={false} />);
      const indicator = container.querySelector('.bg-warning');
      expect(indicator).toBeInTheDocument();
    });

    it('does not have pulse animation when disconnected', () => {
      const { container } = render(<ConnectionStatus connected={false} />);
      const indicator = container.querySelector('.animate-pulse');
      expect(indicator).toBeNull();
    });
  });

  describe('Indicator Shape', () => {
    it('has rounded indicator dot', () => {
      const { container } = render(<ConnectionStatus connected={true} />);
      const indicator = container.querySelector('.rounded-full');
      expect(indicator).toBeInTheDocument();
    });

    it('indicator has correct size', () => {
      const { container } = render(<ConnectionStatus connected={true} />);
      const indicator = container.querySelector('.w-2.h-2');
      expect(indicator).toBeInTheDocument();
    });
  });

  describe('Layout', () => {
    it('uses flexbox layout', () => {
      const { container } = render(<ConnectionStatus connected={true} />);
      const wrapper = container.firstChild;
      expect(wrapper).toHaveClass('flex');
    });

    it('centers items vertically', () => {
      const { container } = render(<ConnectionStatus connected={true} />);
      const wrapper = container.firstChild;
      expect(wrapper).toHaveClass('items-center');
    });
  });
});
