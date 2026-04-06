import { act, renderHook } from '@testing-library/react';

import { useSharing } from '../useSharing';
import { useApi } from '../useApi';
import { useBackend } from '@/components/BackendSelector';

jest.mock('../useApi', () => ({
  useApi: jest.fn(),
}));

jest.mock('@/components/BackendSelector', () => ({
  useBackend: jest.fn(),
}));

describe('useSharing', () => {
  const get = jest.fn();
  const post = jest.fn();
  const put = jest.fn();
  const request = jest.fn();
  const reset = jest.fn();
  const del = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    (useBackend as jest.Mock).mockReturnValue({
      config: { api: 'https://api.test.com' },
    });
    (useApi as jest.Mock).mockReturnValue({
      data: null,
      loading: false,
      error: null,
      get,
      post,
      put,
      request,
      reset,
      delete: del,
    });
  });

  it('normalizes the live shared-with-me payload into the explorer shape', async () => {
    get.mockResolvedValueOnce({
      items: [
        {
          id: 'item-1',
          title: 'Receipt context',
          content: 'Decision receipt context',
          shared_by: 'user-456',
          shared_by_name: 'Casey',
          shared_by_type: 'user',
          shared_at: '2026-04-06T12:30:00Z',
          expires_at: '2026-04-09T12:30:00Z',
          permissions: ['read', 'write'],
          source_workspace_id: 'ws-source',
          source_workspace_name: 'Source Workspace',
        },
      ],
    });

    const { result } = renderHook(() => useSharing({ workspaceId: 'ws-target' }));

    let items;
    await act(async () => {
      items = await result.current.loadSharedWithMe(25);
    });

    expect(get).toHaveBeenCalledWith(
      '/api/knowledge/mound/shared-with-me?workspace_id=ws-target&limit=25'
    );
    expect(items).toEqual([
      {
        id: 'item-1',
        title: 'Receipt context',
        content: 'Decision receipt context',
        sharedBy: {
          id: 'user-456',
          name: 'Casey',
          type: 'user',
        },
        sharedAt: new Date('2026-04-06T12:30:00Z'),
        expiresAt: new Date('2026-04-09T12:30:00Z'),
        permissions: ['read', 'write'],
        sourceWorkspace: {
          id: 'ws-source',
          name: 'Source Workspace',
        },
      },
    ]);
  });

  it('sends the backend share contract instead of legacy to_user_id/to_workspace_id fields', async () => {
    post.mockResolvedValueOnce({
      share: {
        id: 'grant-1',
        item_id: 'item-1',
        target_type: 'workspace',
        target_id: 'ws-destination',
        permissions: ['read'],
        shared_at: '2026-04-06T13:00:00Z',
      },
    });

    const { result } = renderHook(() => useSharing({ workspaceId: 'ws-source' }));

    let share;
    await act(async () => {
      share = await result.current.shareItem({
        itemId: 'item-1',
        toWorkspaceId: 'ws-destination',
        permissions: ['read'],
      });
    });

    expect(post).toHaveBeenCalledWith('/api/knowledge/mound/share', {
      item_id: 'item-1',
      from_workspace_id: 'ws-source',
      target_type: 'workspace',
      target_id: 'ws-destination',
      permissions: ['read'],
      expires_at: undefined,
    });
    expect(share).toEqual({
      grantId: 'grant-1',
      itemId: 'item-1',
      targetType: 'workspace',
      targetId: 'ws-destination',
      permissions: ['read'],
      sharedAt: new Date('2026-04-06T13:00:00Z'),
      expiresAt: undefined,
    });
  });

  it('revokes shares through DELETE /share with a JSON body', async () => {
    request.mockResolvedValueOnce({ success: true });

    const { result } = renderHook(() => useSharing({ workspaceId: 'ws-source' }));

    await act(async () => {
      await result.current.revokeShare('item-1', 'user-456');
    });

    expect(request).toHaveBeenCalledWith('/api/knowledge/mound/share', {
      method: 'DELETE',
      body: JSON.stringify({
        item_id: 'item-1',
        grantee_id: 'user-456',
      }),
    });
  });
});
