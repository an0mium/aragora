import { fetchDebateClient } from '../[[...id]]/fetchDebate';

describe('fetchDebateClient', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  afterAll(() => {
    global.fetch = originalFetch;
  });

  it('falls back to the next candidate when the public endpoint returns malformed JSON', async () => {
    const fetchMock = global.fetch as jest.MockedFunction<typeof fetch>;

    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: { unexpected: 'shape' } }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          id: 'fallback-debate',
          topic: 'Recovered via playground fallback',
          status: 'completed',
          consensus_reached: true,
          confidence: 0.88,
          verdict: 'Approve',
          duration_seconds: 4.2,
          participants: ['analyst', 'critic'],
          proposals: { analyst: 'Use fallback.', critic: 'Validate the payload first.' },
          critiques: [],
          votes: [],
          final_answer: 'Recovered debate payload.',
          receipt_hash: 'sha256:test',
        }),
      } as Response);

    const result = await fetchDebateClient('fallback-debate');

    expect(result).not.toBeNull();
    expect(result?.id).toBe('fallback-debate');
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('accepts public debate payloads when receipt_hash is null', async () => {
    const fetchMock = global.fetch as jest.MockedFunction<typeof fetch>;

    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        data: {
          id: 'debate-with-null-receipt',
          topic: 'Receipt hashes can be pending',
          status: 'completed',
          consensus_reached: true,
          confidence: 0.91,
          verdict: 'Ship the public viewer fix.',
          duration_seconds: 3.4,
          participants: ['analyst', 'critic'],
          proposals: { analyst: 'Allow null.', critic: 'Keep other fields strict.' },
          critiques: [],
          votes: [],
          final_answer: 'Public viewer payload parsed successfully.',
          receipt_hash: null,
        },
      }),
    } as Response);

    const result = await fetchDebateClient('debate-with-null-receipt');

    expect(result).not.toBeNull();
    expect(result?.receipt_hash).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('normalizes persisted debate payloads from the primary debate store', async () => {
    const fetchMock = global.fetch as jest.MockedFunction<typeof fetch>;

    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'debate-123',
        question: 'Should this shared debate be readable without login?',
        status: 'completed',
        messages: [
          {
            agent: 'analyst',
            role: 'proposal',
            content: 'Yes. Public links should resolve to the full argument.',
            round: 1,
          },
          {
            agent: 'critic',
            role: 'critique',
            target: 'analyst',
            content: 'Only after the backend serves public debate payloads.',
            round: 1,
          },
        ],
        result: {
          consensus_reached: true,
          confidence: 0.89,
          final_answer: 'Make the shared link anonymous and keep the full transcript.',
          participants: ['analyst', 'critic'],
        },
      }),
    } as Response);

    const result = await fetchDebateClient('debate-123');

    expect(result).not.toBeNull();
    expect(result?.topic).toBe('Should this shared debate be readable without login?');
    expect(result?.messages).toHaveLength(2);
    expect(result?.proposals.analyst).toContain('Public links should resolve');
    expect(result?.critiques[0]).toEqual({
      agent: 'critic',
      target: 'analyst',
      text: 'Only after the backend serves public debate payloads.',
    });
  });
});
