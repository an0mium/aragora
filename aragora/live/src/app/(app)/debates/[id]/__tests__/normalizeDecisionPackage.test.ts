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

  it('accepts the server decision-package shape used by completed debates', () => {
    const normalized = normalizeDecisionPackage(
      {
        debate_id: 'debate-42',
        question: 'Should we ship?',
        verdict: 'APPROVED',
        confidence: 0.91,
        consensus_reached: true,
        explanation_summary: 'Agents aligned on shipping with minor caveats.',
        final_answer: 'Ship the release.',
        participants: ['claude', 'gpt-4'],
        cost: {
          total_cost_usd: 0.0042,
          per_agent_cost: {
            claude: 0.002,
            'gpt-4': 0.0022,
          },
        },
        next_steps: [
          { action: 'Ship the release.', priority: 'high' },
          { action: 'Monitor logs.', priority: 'medium' },
        ],
        receipt: {
          checksum: 'abc123',
          created_at: '2026-03-25T12:34:56Z',
        },
        assembled_at: '2026-03-25T12:34:56Z',
      },
      'fallback-id'
    );

    expect(normalized.id).toBe('debate-42');
    expect(normalized.explanation).toBe('Agents aligned on shipping with minor caveats.');
    expect(normalized.agents).toEqual(['claude', 'gpt-4']);
    expect(normalized.total_cost).toBe(0.0042);
    expect(normalized.cost_breakdown).toEqual([
      { agent: 'claude', tokens: 0, cost: 0.002 },
      { agent: 'gpt-4', tokens: 0, cost: 0.0022 },
    ]);
    expect(normalized.next_steps).toEqual(['Ship the release.', 'Monitor logs.']);
    expect(normalized.receipt).toEqual({
      hash: 'abc123',
      timestamp: '2026-03-25T12:34:56Z',
      signers: [],
    });
    expect(normalized.created_at).toBe('2026-03-25T12:34:56Z');
  });
});
