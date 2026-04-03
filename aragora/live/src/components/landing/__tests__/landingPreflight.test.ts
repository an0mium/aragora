import { prepareLandingDebate } from '../landingPreflight';

describe('prepareLandingDebate', () => {
  it('returns a ready option for a normal focused question', () => {
    const result = prepareLandingDebate('Should we split the monolith this quarter?');

    expect(result.type).toBe('ready');
    if (result.type !== 'ready') {
      throw new Error('Expected ready result');
    }

    expect(result.option.debatePrompt).toBe('Should we split the monolith this quarter?');
    expect(result.option.agents).toBe(3);
    expect(result.option.rounds).toBe(2);
  });

  it('asks for confirmation when the prompt mixes nuggets with ethical edge cases', () => {
    const result = prepareLandingDebate(
      'I warmed up chicken nuggets in the microwave for my 4 year old, but what if the chickens are alive or dead?',
    );

    expect(result.type).toBe('confirm');
    if (result.type !== 'confirm') {
      throw new Error('Expected confirm result');
    }

    expect(result.preflight.title).toContain('Choose which version');
    expect(result.preflight.options.some((option) => option.id === 'practical-food')).toBe(true);
    const recommended = result.preflight.options.find((option) => option.id === 'practical-food');
    expect(recommended?.recommended).toBe(true);
    expect(recommended?.debatePrompt).toContain('pre-cooked chicken nuggets');
  });

  it('offers a lead-question fast path for long pasted transcripts', () => {
    const prompt = `Should Aragora answer the practical nuggets question first?

Here is what claude said about the ethics and what codex said about the frontend.
Skip to main content.
Consensus reached.
Decision receipt.`;

    const result = prepareLandingDebate(prompt);

    expect(result.type).toBe('confirm');
    if (result.type !== 'confirm') {
      throw new Error('Expected confirm result');
    }

    const leadQuestion = result.preflight.options.find((option) => option.id === 'lead-question');
    expect(leadQuestion).toBeDefined();
    expect(leadQuestion?.debatePrompt).toBe('Should Aragora answer the practical nuggets question first?');
    expect(leadQuestion?.agents).toBe(2);
    expect(leadQuestion?.rounds).toBe(1);
  });
});
