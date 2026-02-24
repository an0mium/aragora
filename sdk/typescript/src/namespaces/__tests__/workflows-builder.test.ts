import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { WorkflowsAPI } from '../workflows';

interface MockClient {
  request: Mock;
}

describe('WorkflowsAPI visual builder routes', () => {
  let api: WorkflowsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new WorkflowsAPI(mockClient as any);
  });

  it('auto-layout posts to the builder endpoint', async () => {
    mockClient.request.mockResolvedValue({ positions: [] });

    await api.autoLayout({ steps: [], transitions: [] });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/workflows/auto-layout', {
      body: { steps: [], transitions: [] },
    });
  });

  it('fromPattern posts to the pattern endpoint', async () => {
    mockClient.request.mockResolvedValue({ id: 'wf-1' });

    await api.fromPattern({ pattern: 'debate' });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/workflows/from-pattern', {
      body: { pattern: 'debate' },
    });
  });

  it('generate posts to the generation endpoint', async () => {
    mockClient.request.mockResolvedValue({ id: 'wf-2' });

    await api.generate({ description: 'Route reviews to security team' });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/workflows/generate', {
      body: { description: 'Route reviews to security team' },
    });
  });

  it('listStepTypes uses the versioned step catalog endpoint', async () => {
    mockClient.request.mockResolvedValue({ step_types: [] });

    await api.listStepTypes({ category: 'analysis' });

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/workflows/step-types', {
      params: { category: 'analysis' },
    });
  });

  it('validate posts workflow definitions to the validation endpoint', async () => {
    mockClient.request.mockResolvedValue({ valid: true });

    await api.validate({ name: 'wf' });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/workflows/validate', {
      body: { name: 'wf' },
    });
  });

  it('replay posts to the versioned replay endpoint', async () => {
    mockClient.request.mockResolvedValue({ run_id: 'run-1' });

    await api.replay('wf-123', { inputs: { approved: true } });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/workflows/wf-123/replay', {
      body: { inputs: { approved: true } },
    });
  });
});
