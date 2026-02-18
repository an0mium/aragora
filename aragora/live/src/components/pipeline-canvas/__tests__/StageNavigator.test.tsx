/**
 * Tests for StageNavigator component.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { StageNavigator } from '../StageNavigator';
import type { PipelineStageType } from '../types';

describe('StageNavigator', () => {
  const defaultStatus: Record<PipelineStageType, string> = {
    ideas: 'complete',
    goals: 'complete',
    actions: 'pending',
    orchestration: 'pending',
  };

  const defaultProps = {
    stageStatus: defaultStatus,
    activeStage: 'ideas' as PipelineStageType,
    onStageSelect: jest.fn(),
  };

  it('renders all 4 stage labels', () => {
    render(<StageNavigator {...defaultProps} />);
    expect(screen.getByText('Ideas')).toBeInTheDocument();
    expect(screen.getByText('Goals')).toBeInTheDocument();
    expect(screen.getByText('Actions')).toBeInTheDocument();
    expect(screen.getByText('Orchestration')).toBeInTheDocument();
  });

  it('calls onStageSelect when stage clicked', () => {
    const onSelect = jest.fn();
    render(<StageNavigator {...defaultProps} onStageSelect={onSelect} />);
    fireEvent.click(screen.getByText('Goals'));
    expect(onSelect).toHaveBeenCalledWith('goals');
  });

  it('shows ADVANCE button when onAdvance provided and stages pending', () => {
    const onAdvance = jest.fn();
    render(<StageNavigator {...defaultProps} onAdvance={onAdvance} />);
    const advanceBtn = screen.getByText('ADVANCE');
    expect(advanceBtn).toBeInTheDocument();
    fireEvent.click(advanceBtn);
    expect(onAdvance).toHaveBeenCalledWith('actions');
  });

  it('hides ADVANCE button in readOnly mode', () => {
    render(<StageNavigator {...defaultProps} onAdvance={jest.fn()} readOnly />);
    expect(screen.queryByText('ADVANCE')).not.toBeInTheDocument();
  });

  it('does not show ADVANCE when all stages complete', () => {
    const allComplete: Record<PipelineStageType, string> = {
      ideas: 'complete',
      goals: 'complete',
      actions: 'complete',
      orchestration: 'complete',
    };
    render(<StageNavigator {...defaultProps} stageStatus={allComplete} onAdvance={jest.fn()} />);
    expect(screen.queryByText('ADVANCE')).not.toBeInTheDocument();
  });
});
