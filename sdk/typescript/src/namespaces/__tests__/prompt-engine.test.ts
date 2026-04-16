import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { PromptEngineAPI } from '../prompt-engine';

interface MockClient {
  request: Mock;
}

describe('PromptEngineAPI', () => {
  let api: PromptEngineAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new PromptEngineAPI(mockClient as any);
  });

  it('maps prompt-engine run and action routes', async () => {
    mockClient.request.mockResolvedValue({ data: {} });

    await api.listRuns({ status: 'spec_ready', limit: 10 });
    await api.getRun('run/demo');
    await api.run({ prompt: 'Build the thing', profile: 'autonomous' });
    await api.decompose({ prompt: 'Build the thing', context: { repo: 'aragora' } });
    await api.interrogate({ intent: { raw_prompt: 'Build' }, depth: 'quick' });
    await api.research({ intent: { raw_prompt: 'Build' }, context: { repo: 'aragora' } });
    await api.specify({
      intent: { raw_prompt: 'Build' },
      questions: [],
      research: { summary: 'ok' },
    });
    await api.validate({ specification: { title: 'Spec' } });

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'GET', '/api/prompt-engine/runs', {
      params: { status: 'spec_ready', limit: 10 },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(2, 'GET', '/api/prompt-engine/runs/run%2Fdemo');
    expect(mockClient.request).toHaveBeenNthCalledWith(3, 'POST', '/api/prompt-engine/run', {
      body: { prompt: 'Build the thing', profile: 'autonomous' },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(4, 'POST', '/api/prompt-engine/decompose', {
      body: { prompt: 'Build the thing', context: { repo: 'aragora' } },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(5, 'POST', '/api/prompt-engine/interrogate', {
      body: { intent: { raw_prompt: 'Build' }, depth: 'quick' },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(6, 'POST', '/api/prompt-engine/research', {
      body: { intent: { raw_prompt: 'Build' }, context: { repo: 'aragora' } },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(7, 'POST', '/api/prompt-engine/specify', {
      body: { intent: { raw_prompt: 'Build' }, questions: [], research: { summary: 'ok' } },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(8, 'POST', '/api/prompt-engine/validate', {
      body: { specification: { title: 'Spec' } },
    });
  });
});
