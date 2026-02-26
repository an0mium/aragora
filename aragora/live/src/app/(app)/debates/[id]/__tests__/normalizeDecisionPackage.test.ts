import { normalizeDecisionPackage } from '../normalizeDecisionPackage';

describe('normalizeDecisionPackage', () => {
  it('fills missing array fields with safe defaults', () => {
    const normalized = normalizeDecisionPackage(
      {
        id: 'debate-1',
        question: 'Q',
        confidence: 0.5,
      },
      'fallback-id'
    );

    expect(normalized.id).toBe('debate-1');
    expect(normalized.agents).toEqual([]);
    expect(normalized.arguments).toEqual([]);
    expect(normalized.cost_breakdown).toEqual([]);
    expect(normalized.next_steps).toEqual([]);
    expect(normalized.receipt).toBeNull();
  });

  it('uses fallback id when payload id is missing', () => {
    const normalized = normalizeDecisionPackage({}, 'fallback-id');
    expect(normalized.id).toBe('fallback-id');
  });

  it('filters malformed entries in string arrays', () => {
    const normalized = normalizeDecisionPackage(
      {
        id: 'debate-2',
        agents: ['claude', 7, null, 'gpt-5'],
        next_steps: ['step one', { bad: true }, 'step two'],
      },
      'fallback-id'
    );

    expect(normalized.agents).toEqual(['claude', 'gpt-5']);
    expect(normalized.next_steps).toEqual(['step one', 'step two']);
  });

  it('normalizes malformed receipt to null', () => {
    const normalized = normalizeDecisionPackage(
      {
        id: 'debate-3',
        receipt: {
          signers: ['a', 'b'],
        },
      },
      'fallback-id'
    );

    expect(normalized.receipt).toBeNull();
  });
});
