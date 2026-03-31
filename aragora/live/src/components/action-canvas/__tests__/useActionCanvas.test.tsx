import { act, renderHook, waitFor } from '@testing-library/react';
import { useActionCanvas } from '../useActionCanvas';

const mockSetNodes = jest.fn();
const mockSetEdges = jest.fn();
const mockOnNodesChange = jest.fn();
const mockOnEdgesChange = jest.fn();

jest.mock('@xyflow/react', () => ({
  useNodesState: (initial: unknown[]) => [initial, mockSetNodes, mockOnNodesChange],
  useEdgesState: (initial: unknown[]) => [initial, mockSetEdges, mockOnEdgesChange],
  addEdge: jest.fn((connection: unknown, edges: unknown[]) => [
    ...edges,
    { id: 'edge-new', ...(connection as object) },
  ]),
}));

const mockFetch = jest.fn();
global.fetch = mockFetch;

class MockWebSocket {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onopen: (() => void) | null = null;
  send = jest.fn();
  close = jest.fn();
}

describe('useActionCanvas', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    localStorage.setItem('aragora-backend', 'production');
    mockFetch.mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);

      if (url === 'https://api.aragora.ai/api/v1/actions/canvas-1') {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: 'canvas-1',
            name: 'Test action canvas',
            metadata: { pipeline_id: 'pipe-123' },
            nodes: [
              {
                id: 'action-1',
                label: 'Ship the fix',
                position: { x: 0, y: 0 },
                data: { action_type: 'task', description: 'Ship the backend routing fix' },
              },
            ],
            edges: [],
          }),
        });
      }

      if (url === 'https://api.aragora.ai/api/v1/canvas/pipeline/advance') {
        return Promise.resolve({
          ok: true,
          json: async () => ({ status: 'advanced' }),
        });
      }

      return Promise.reject(new Error(`Unexpected fetch: ${url}`));
    });

    (global as typeof globalThis & { WebSocket: typeof WebSocket }).WebSocket =
      MockWebSocket as unknown as typeof WebSocket;
  });

  it('uses the selected backend for load and advance requests', async () => {
    const { result } = renderHook(() => useActionCanvas('canvas-1'));

    await waitFor(() => {
      expect(result.current.canvasMeta?.metadata?.pipeline_id).toBe('pipe-123');
    });

    expect(mockFetch).toHaveBeenCalledWith('https://api.aragora.ai/api/v1/actions/canvas-1');

    act(() => {
      result.current.setSelectedNodeId('action-1');
    });

    await act(async () => {
      await result.current.advanceToOrchestration();
    });

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.aragora.ai/api/v1/canvas/pipeline/advance',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      }),
    );
  });
});
