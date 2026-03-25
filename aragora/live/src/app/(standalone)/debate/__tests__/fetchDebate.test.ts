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
});
