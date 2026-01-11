/**
 * Tests for RoleBadge and RoleIcon components
 *
 * Tests cover:
 * - Debate role rendering (proposer, critic, synthesizer, judge, etc.)
 * - Cognitive role rendering (analyst, skeptic, lateral_thinker, etc.)
 * - Size variants (sm, md)
 * - Fallback for unknown roles
 * - RoleIcon compact display
 */

import { render, screen } from '@testing-library/react';
import { RoleBadge, RoleIcon } from '../src/components/RoleBadge';

describe('RoleBadge', () => {
  describe('Debate Roles', () => {
    it('renders proposer role with correct icon and label', () => {
      render(<RoleBadge role="proposer" />);

      expect(screen.getByText('💡')).toBeInTheDocument();
      expect(screen.getByText('Proposer')).toBeInTheDocument();
    });

    it('renders critic role with correct icon and label', () => {
      render(<RoleBadge role="critic" />);

      expect(screen.getByText('🔍')).toBeInTheDocument();
      expect(screen.getByText('Critic')).toBeInTheDocument();
    });

    it('renders synthesizer role with correct icon and label', () => {
      render(<RoleBadge role="synthesizer" />);

      expect(screen.getByText('🔮')).toBeInTheDocument();
      expect(screen.getByText('Synthesizer')).toBeInTheDocument();
    });

    it('renders judge role with correct icon and label', () => {
      render(<RoleBadge role="judge" />);

      expect(screen.getByText('⚖️')).toBeInTheDocument();
      expect(screen.getByText('Judge')).toBeInTheDocument();
    });

    it('renders reviewer role with correct icon and label', () => {
      render(<RoleBadge role="reviewer" />);

      expect(screen.getByText('📋')).toBeInTheDocument();
      expect(screen.getByText('Reviewer')).toBeInTheDocument();
    });

    it('renders implementer role with correct icon and label', () => {
      render(<RoleBadge role="implementer" />);

      expect(screen.getByText('🔧')).toBeInTheDocument();
      expect(screen.getByText('Implementer')).toBeInTheDocument();
    });

    it('handles case-insensitive role names', () => {
      render(<RoleBadge role="PROPOSER" />);

      expect(screen.getByText('Proposer')).toBeInTheDocument();
    });
  });

  describe('Cognitive Roles', () => {
    it('renders analyst cognitive role', () => {
      render(<RoleBadge role="agent" cognitiveRole="analyst" />);

      expect(screen.getByText('🔬')).toBeInTheDocument();
      expect(screen.getByText('Analyst')).toBeInTheDocument();
    });

    it('renders skeptic cognitive role', () => {
      render(<RoleBadge role="agent" cognitiveRole="skeptic" />);

      expect(screen.getByText('🤔')).toBeInTheDocument();
      expect(screen.getByText('Skeptic')).toBeInTheDocument();
    });

    it('renders lateral_thinker cognitive role', () => {
      render(<RoleBadge role="agent" cognitiveRole="lateral_thinker" />);

      expect(screen.getByText('💡')).toBeInTheDocument();
      expect(screen.getByText('Lateral')).toBeInTheDocument();
    });

    it('renders devil_advocate cognitive role', () => {
      render(<RoleBadge role="agent" cognitiveRole="devil_advocate" />);

      expect(screen.getByText('😈')).toBeInTheDocument();
      expect(screen.getByText("Devil's Advocate")).toBeInTheDocument();
    });

    it('displays both cognitive and debate roles when both provided', () => {
      render(<RoleBadge role="proposer" cognitiveRole="analyst" />);

      // Both badges should be present
      expect(screen.getByText('🔬')).toBeInTheDocument();
      expect(screen.getByText('Analyst')).toBeInTheDocument();
      expect(screen.getByText('💡')).toBeInTheDocument();
      expect(screen.getByText('Proposer')).toBeInTheDocument();
    });
  });

  describe('Size Variants', () => {
    it('applies medium size classes by default', () => {
      const { container } = render(<RoleBadge role="proposer" />);

      const badge = container.querySelector('span.inline-flex');
      expect(badge).toHaveClass('text-sm');
      expect(badge).toHaveClass('px-2');
      expect(badge).toHaveClass('py-1');
    });

    it('applies small size classes when size="sm"', () => {
      const { container } = render(<RoleBadge role="proposer" size="sm" />);

      const badge = container.querySelector('span.inline-flex');
      expect(badge).toHaveClass('text-xs');
      expect(badge).toHaveClass('px-1.5');
      expect(badge).toHaveClass('py-0.5');
    });
  });

  describe('Fallback Behavior', () => {
    it('renders default icon for unknown role', () => {
      render(<RoleBadge role="unknown-role" />);

      expect(screen.getByText('🤖')).toBeInTheDocument();
      expect(screen.getByText('unknown-role')).toBeInTheDocument();
    });

    it('renders "Agent" label when role is empty', () => {
      render(<RoleBadge role="" />);

      expect(screen.getByText('Agent')).toBeInTheDocument();
    });
  });
});

describe('RoleIcon', () => {
  it('renders debate role icon only', () => {
    const { container } = render(<RoleIcon role="proposer" />);

    expect(container).toHaveTextContent('💡');
  });

  it('renders cognitive role icon when provided', () => {
    const { container } = render(<RoleIcon role="agent" cognitiveRole="analyst" />);

    expect(container).toHaveTextContent('🔬');
  });

  it('prefers cognitive role over debate role', () => {
    const { container } = render(<RoleIcon role="proposer" cognitiveRole="skeptic" />);

    // Should show skeptic icon, not proposer
    expect(container).toHaveTextContent('🤔');
  });

  it('has tooltip with role label', () => {
    render(<RoleIcon role="judge" />);

    const icon = screen.getByTitle('Judge');
    expect(icon).toBeInTheDocument();
  });

  it('renders default robot icon for unknown role', () => {
    const { container } = render(<RoleIcon role="unknown" />);

    expect(container).toHaveTextContent('🤖');
  });
});
