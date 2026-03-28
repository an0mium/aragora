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

  it('normalizes archived public debate payloads from the primary debates API', async () => {
    const fetchMock = global.fetch as jest.MockedFunction<typeof fetch>;

    fetchMock.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'debate-123',
        task: 'Should we expose archived debates publicly?',
        status: 'completed',
        consensus_reached: true,
        confidence: 0.73,
        winning_proposal: 'Yes, publish the archived debate link.',
        agents: ['analyst', 'critic'],
        critiques: [
          {
            agent: 'critic',
            target_agent: 'analyst',
            text: 'Verify the anonymous access path before shipping.',
          },
        ],
        messages: [
          {
            agent: 'analyst',
            role: 'proposer',
            round: 1,
            content: 'Public links increase the utility of archived debates.',
          },
          {
            agent: 'critic',
            role: 'critic',
            round: 1,
            content: 'Public links must not depend on an authenticated session.',
          },
        ],
        receipt_hash: null,
      }),
    } as Response);

    const result = await fetchDebateClient('debate-123');

    expect(result).not.toBeNull();
    expect(result?.id).toBe('debate-123');
    expect(result?.topic).toBe('Should we expose archived debates publicly?');
    expect(result?.final_answer).toBe('Yes, publish the archived debate link.');
    expect(result?.proposals.analyst).toContain('Public links increase the utility');
    expect(result?.critiques[0]).toEqual({
      agent: 'critic',
      target: 'analyst',
      text: 'Verify the anonymous access path before shipping.',
    });
    expect(result?.messages).toHaveLength(2);
  });
});
