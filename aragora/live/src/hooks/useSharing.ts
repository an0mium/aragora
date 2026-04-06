'use client';

/**
 * Sharing hook for cross-workspace knowledge sharing.
 *
 * Provides:
 * - Share items with workspaces/users
 * - View items shared with me
 * - Revoke shares
 */

import { useState, useCallback } from 'react';
import { useApi } from './useApi';
import { useBackend } from '@/components/BackendSelector';
import type { SharedItem } from '@/components/control-plane/KnowledgeExplorer/SharedWithMeTab';

export interface ShareRequest {
  itemId: string;
  targetType?: 'workspace' | 'user';
  targetId?: string;
  toWorkspaceId?: string;
  toUserId?: string;
  permissions: string[];
  expiresAt?: Date;
  fromWorkspaceId?: string;
}

export interface ShareResponse {
  grantId: string;
  itemId: string;
  targetType: string;
  targetId: string;
  permissions: string[];
  sharedAt: Date;
  expiresAt?: Date;
}

export interface UseSharingOptions {
  /** Current workspace ID */
  workspaceId?: string;
}

export interface UseSharingReturn {
  // State
  sharedItems: SharedItem[];
  isLoading: boolean;
  error: string | null;

  // Operations
  shareItem: (request: ShareRequest) => Promise<ShareResponse>;
  loadSharedWithMe: (limit?: number) => Promise<SharedItem[]>;
  revokeShare: (itemId: string, granteeId: string) => Promise<void>;
  getMyShares: (itemId: string) => Promise<ShareResponse[]>;
}

type SharedItemApiResponse = {
  id: string;
  title?: string;
  content?: string;
  metadata?: Record<string, unknown>;
  shared_by?: string;
  shared_by_name?: string;
  shared_by_type?: 'user' | 'workspace';
  shared_at?: string;
  expires_at?: string | null;
  permissions?: string[];
  source_workspace_id?: string;
  source_workspace_name?: string;
};

type ShareGrantApiResponse = {
  id?: string;
  item_id?: string;
  target_type?: string;
  target_id?: string;
  grantee_type?: string;
  grantee_id?: string;
  permissions?: string[];
  shared_at?: string;
  granted_at?: string;
  expires_at?: string | null;
};

function parseDate(value?: string | null): Date | undefined {
  return value ? new Date(value) : undefined;
}

function normalizeSharedItem(item: SharedItemApiResponse, fallbackWorkspaceId: string): SharedItem {
  const metadata = item.metadata ?? {};
  const sourceWorkspaceId =
    item.source_workspace_id ??
    (typeof metadata.workspace_id === 'string' ? metadata.workspace_id : undefined) ??
    fallbackWorkspaceId;
  const sourceWorkspaceName =
    item.source_workspace_name ??
    (typeof metadata.source_workspace_name === 'string'
      ? metadata.source_workspace_name
      : undefined) ??
    sourceWorkspaceId;
  const sharedById = item.shared_by ?? 'unknown';
  const sharedByName = item.shared_by_name ?? sharedById;

  return {
    id: item.id,
    title: item.title ?? item.content?.slice(0, 80) ?? item.id,
    content: item.content ?? '',
    sharedBy: {
      id: sharedById,
      name: sharedByName,
      type: item.shared_by_type ?? 'user',
    },
    sharedAt: parseDate(item.shared_at) ?? new Date(0),
    expiresAt: parseDate(item.expires_at),
    permissions: item.permissions ?? ['read'],
    sourceWorkspace: {
      id: sourceWorkspaceId,
      name: sourceWorkspaceName,
    },
  };
}

function resolveShareTarget(shareRequest: ShareRequest): {
  targetType: 'workspace' | 'user';
  targetId: string;
} {
  if (shareRequest.targetType && shareRequest.targetId) {
    return { targetType: shareRequest.targetType, targetId: shareRequest.targetId };
  }
  if (shareRequest.toWorkspaceId) {
    return { targetType: 'workspace', targetId: shareRequest.toWorkspaceId };
  }
  if (shareRequest.toUserId) {
    return { targetType: 'user', targetId: shareRequest.toUserId };
  }
  throw new Error('shareItem requires a target workspace or user');
}

function normalizeShareGrant(
  grant: ShareGrantApiResponse,
  fallback: { itemId: string; targetType: string; targetId: string }
): ShareResponse {
  return {
    grantId: grant.id ?? `${fallback.itemId}:${fallback.targetType}:${fallback.targetId}`,
    itemId: grant.item_id ?? fallback.itemId,
    targetType: grant.target_type ?? grant.grantee_type ?? fallback.targetType,
    targetId: grant.target_id ?? grant.grantee_id ?? fallback.targetId,
    permissions: grant.permissions ?? ['read'],
    sharedAt: parseDate(grant.shared_at ?? grant.granted_at) ?? new Date(0),
    expiresAt: parseDate(grant.expires_at),
  };
}

export function useSharing(options: UseSharingOptions = {}): UseSharingReturn {
  const { workspaceId = 'default' } = options;
  const { config: backendConfig } = useBackend();
  const api = useApi(backendConfig?.api);
  const [sharedItems, setSharedItems] = useState<SharedItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const shareItem = useCallback(
    async (shareRequest: ShareRequest): Promise<ShareResponse> => {
      setIsLoading(true);
      setError(null);
      try {
        const { targetType, targetId } = resolveShareTarget(shareRequest);
        const response = (await api.post('/api/knowledge/mound/share', {
          item_id: shareRequest.itemId,
          from_workspace_id: shareRequest.fromWorkspaceId ?? workspaceId,
          target_type: targetType,
          target_id: targetId,
          permissions: shareRequest.permissions,
          expires_at: shareRequest.expiresAt?.toISOString(),
        })) as { share?: ShareGrantApiResponse };
        return normalizeShareGrant(response.share ?? {}, {
          itemId: shareRequest.itemId,
          targetType,
          targetId,
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to share item';
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [api, workspaceId]
  );

  const loadSharedWithMe = useCallback(
    async (limit = 50): Promise<SharedItem[]> => {
      setIsLoading(true);
      setError(null);
      try {
        const response = (await api.get(
          `/api/knowledge/mound/shared-with-me?workspace_id=${workspaceId}&limit=${limit}`
        )) as { items: SharedItemApiResponse[] };
        const items = (response.items ?? []).map((item) =>
          normalizeSharedItem(item, workspaceId)
        );
        setSharedItems(items);
        return items;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load shared items';
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [api, workspaceId]
  );

  const revokeShare = useCallback(
    async (itemId: string, granteeId: string): Promise<void> => {
      setIsLoading(true);
      setError(null);
      try {
        await api.request('/api/knowledge/mound/share', {
          method: 'DELETE',
          body: JSON.stringify({
            item_id: itemId,
            grantee_id: granteeId,
          }),
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to revoke share';
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [api]
  );

  const getMyShares = useCallback(
    async (itemId: string): Promise<ShareResponse[]> => {
      setIsLoading(true);
      setError(null);
      try {
        const response = (await api.get(
          `/api/knowledge/mound/my-shares?workspace_id=${workspaceId}&limit=200`
        )) as { grants?: ShareGrantApiResponse[] };
        return (response.grants ?? [])
          .filter((grant) => !itemId || grant.item_id === itemId)
          .map((grant) =>
            normalizeShareGrant(grant, {
              itemId: grant.item_id ?? itemId,
              targetType: grant.grantee_type ?? 'workspace',
              targetId: grant.grantee_id ?? '',
            })
          );
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to get shares';
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [api]
  );

  return {
    sharedItems,
    isLoading,
    error,
    shareItem,
    loadSharedWithMe,
    revokeShare,
    getMyShares,
  };
}

export default useSharing;
