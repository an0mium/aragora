/**
 * Tests for PipelinePropertyEditor component.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { PipelinePropertyEditor } from '../editors/PipelinePropertyEditor';
import type { PipelineStageType } from '../types';

describe('PipelinePropertyEditor', () => {
  const baseProps = {
    onUpdate: jest.fn(),
    onDelete: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  /* ---------------------------------------------------------------------- */
  /*  1. Empty state when node is null                                       */
  /* ---------------------------------------------------------------------- */

  it('renders empty state when node is null', () => {
    render(
      <PipelinePropertyEditor
        {...baseProps}
        node={null}
        stage={'ideas' as PipelineStageType}
      />,
    );
    expect(
      screen.getByText('Select a node to edit its properties.'),
    ).toBeInTheDocument();
  });

  /* ---------------------------------------------------------------------- */
  /*  2. Ideas stage fields                                                  */
  /* ---------------------------------------------------------------------- */

  it('renders Ideas stage fields', () => {
    const node: Record<string, unknown> = {
      label: 'My Idea',
      ideaType: 'concept',
      fullContent: 'Some detailed content',
      agent: 'claude',
    };

    render(
      <PipelinePropertyEditor
        {...baseProps}
        node={node}
        stage={'ideas' as PipelineStageType}
      />,
    );

    // Header
    expect(screen.getByText('Ideas Properties')).toBeInTheDocument();

    // Idea Type select
    const ideaTypeSelect = screen.getByDisplayValue('Concept');
    expect(ideaTypeSelect).toBeInTheDocument();
    expect(ideaTypeSelect.tagName).toBe('SELECT');

    // Full Content textarea
    const fullContentTextarea = screen.getByDisplayValue('Some detailed content');
    expect(fullContentTextarea).toBeInTheDocument();
    expect(fullContentTextarea.tagName).toBe('TEXTAREA');

    // Agent input
    const agentInput = screen.getByDisplayValue('claude');
    expect(agentInput).toBeInTheDocument();
    expect(agentInput.tagName).toBe('INPUT');
  });

  /* ---------------------------------------------------------------------- */
  /*  3. Goals stage fields                                                  */
  /* ---------------------------------------------------------------------- */

  it('renders Goals stage fields', () => {
    const node: Record<string, unknown> = {
      label: 'My Goal',
      goalType: 'strategy',
      description: 'Achieve product-market fit',
      priority: 'high',
      confidence: 75,
      tags: ['ux', 'backend'],
    };

    render(
      <PipelinePropertyEditor
        {...baseProps}
        node={node}
        stage={'goals' as PipelineStageType}
      />,
    );

    // Header
    expect(screen.getByText('Goals Properties')).toBeInTheDocument();

    // Goal Type select
    const goalTypeSelect = screen.getByDisplayValue('Strategy');
    expect(goalTypeSelect).toBeInTheDocument();
    expect(goalTypeSelect.tagName).toBe('SELECT');

    // Description textarea
    const descTextarea = screen.getByDisplayValue('Achieve product-market fit');
    expect(descTextarea).toBeInTheDocument();

    // Priority select
    const prioritySelect = screen.getByDisplayValue('High');
    expect(prioritySelect).toBeInTheDocument();

    // Confidence slider -- label shows the formatted value
    expect(screen.getByText('Confidence: 75%')).toBeInTheDocument();
    const slider = screen.getByRole('slider');
    expect(slider).toHaveValue('75');

    // Tags input (comma-separated string)
    const tagsInput = screen.getByDisplayValue('ux, backend');
    expect(tagsInput).toBeInTheDocument();
  });

  /* ---------------------------------------------------------------------- */
  /*  4. Actions stage fields                                                */
  /* ---------------------------------------------------------------------- */

  it('renders Actions stage fields', () => {
    const node: Record<string, unknown> = {
      label: 'Build API',
      stepType: 'task',
      description: 'Implement REST endpoints',
      optional: true,
      timeoutSeconds: 300,
    };

    render(
      <PipelinePropertyEditor
        {...baseProps}
        node={node}
        stage={'actions' as PipelineStageType}
      />,
    );

    // Header
    expect(screen.getByText('Actions Properties')).toBeInTheDocument();

    // Step Type select
    const stepTypeSelect = screen.getByDisplayValue('Task');
    expect(stepTypeSelect).toBeInTheDocument();
    expect(stepTypeSelect.tagName).toBe('SELECT');

    // Description textarea
    const descTextarea = screen.getByDisplayValue('Implement REST endpoints');
    expect(descTextarea).toBeInTheDocument();

    // Optional checkbox
    const optionalCheckbox = screen.getByRole('checkbox');
    expect(optionalCheckbox).toBeChecked();

    // Timeout input
    const timeoutInput = screen.getByDisplayValue('300');
    expect(timeoutInput).toBeInTheDocument();
  });

  /* ---------------------------------------------------------------------- */
  /*  5. Orchestration stage fields                                          */
  /* ---------------------------------------------------------------------- */

  it('renders Orchestration stage fields', () => {
    const node: Record<string, unknown> = {
      label: 'Review Task',
      orchType: 'debate',
      assignedAgent: 'gpt-4',
      agentType: 'reviewer',
      capabilities: ['code_review', 'testing'],
    };

    render(
      <PipelinePropertyEditor
        {...baseProps}
        node={node}
        stage={'orchestration' as PipelineStageType}
      />,
    );

    // Header
    expect(screen.getByText('Orchestration Properties')).toBeInTheDocument();

    // Orchestration Type select
    const orchTypeSelect = screen.getByDisplayValue('Debate');
    expect(orchTypeSelect).toBeInTheDocument();
    expect(orchTypeSelect.tagName).toBe('SELECT');

    // Assigned Agent input
    const agentInput = screen.getByDisplayValue('gpt-4');
    expect(agentInput).toBeInTheDocument();

    // Agent Type input
    const agentTypeInput = screen.getByDisplayValue('reviewer');
    expect(agentTypeInput).toBeInTheDocument();

    // Capabilities input (comma-separated string)
    const capsInput = screen.getByDisplayValue('code_review, testing');
    expect(capsInput).toBeInTheDocument();
  });

  /* ---------------------------------------------------------------------- */
  /*  6. Calls onUpdate when a field changes                                 */
  /* ---------------------------------------------------------------------- */

  it('calls onUpdate when a field changes', () => {
    const onUpdate = jest.fn();
    const node: Record<string, unknown> = {
      label: 'Original',
      ideaType: 'concept',
      fullContent: '',
      agent: '',
    };

    render(
      <PipelinePropertyEditor
        {...baseProps}
        node={node}
        stage={'ideas' as PipelineStageType}
        onUpdate={onUpdate}
      />,
    );

    // Change the label input
    const labelInput = screen.getByDisplayValue('Original');
    fireEvent.change(labelInput, { target: { value: 'Updated Label' } });
    expect(onUpdate).toHaveBeenCalledWith({ label: 'Updated Label' });

    // Change the Idea Type select
    const ideaTypeSelect = screen.getByDisplayValue('Concept');
    fireEvent.change(ideaTypeSelect, { target: { value: 'insight' } });
    expect(onUpdate).toHaveBeenCalledWith({ ideaType: 'insight' });
  });

  /* ---------------------------------------------------------------------- */
  /*  7. Calls onDelete when delete button is clicked                        */
  /* ---------------------------------------------------------------------- */

  it('calls onDelete when delete button is clicked', () => {
    const onDelete = jest.fn();
    const node: Record<string, unknown> = {
      label: 'Deletable',
      ideaType: 'concept',
    };

    render(
      <PipelinePropertyEditor
        {...baseProps}
        node={node}
        stage={'ideas' as PipelineStageType}
        onDelete={onDelete}
      />,
    );

    const deleteBtn = screen.getByText('Delete Node');
    fireEvent.click(deleteBtn);
    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  /* ---------------------------------------------------------------------- */
  /*  8. readOnly mode hides delete and shows read-only fields               */
  /* ---------------------------------------------------------------------- */

  it('in readOnly mode, does not show delete button and shows read-only fields', () => {
    const node: Record<string, unknown> = {
      label: 'Read Only Node',
      ideaType: 'insight',
      fullContent: 'Locked content',
      agent: 'gemini',
    };

    render(
      <PipelinePropertyEditor
        {...baseProps}
        node={node}
        stage={'ideas' as PipelineStageType}
        readOnly
      />,
    );

    // Delete button should not be present
    expect(screen.queryByText('Delete Node')).not.toBeInTheDocument();

    // Read-only fields render as <p> text, not inputs/selects
    expect(screen.queryAllByRole('textbox')).toHaveLength(0);
    expect(screen.queryAllByRole('combobox')).toHaveLength(0);

    // Values should still be visible as plain text
    expect(screen.getByText('Read Only Node')).toBeInTheDocument();
    expect(screen.getByText('insight')).toBeInTheDocument();
    expect(screen.getByText('Locked content')).toBeInTheDocument();
    expect(screen.getByText('gemini')).toBeInTheDocument();
  });
});
