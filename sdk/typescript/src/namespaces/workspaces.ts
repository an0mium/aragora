/**
 * Workspaces Namespace API
 *
 * Provides a namespaced interface for workspace management.
 * Essential for multi-tenancy and team collaboration.
 */

/**
 * Workspace definition
 */
export interface Workspace {
  id: string;
  name: string;
  slug: string;
  description?: string;
  owner_id: string;
  tier: 'free' | 'pro' | 'enterprise';
  settings: WorkspaceSettings;
  member_count: number;
  debate_count: number;
  created_at: string;
  updated_at: string;
}

/**
 * Workspace settings
 */
export interface WorkspaceSettings {
  allow_public_debates: boolean;
  default_debate_visibility: 'private' | 'workspace' | 'public';
  require_approval_for_debates: boolean;
  max_debate_duration_minutes: number;
  enabled_features: string[];
  retention_days: number;
  custom_branding?: {
    logo_url?: string;
    primary_color?: string;
    accent_color?: string;
  };
}

/**
 * Workspace member
 */
export interface WorkspaceMember {
  user_id: string;
  workspace_id: string;
  role: 'viewer' | 'member' | 'admin' | 'owner';
  email: string;
  name?: string;
  avatar_url?: string;
  joined_at: string;
  last_active_at?: string;
}

/**
 * Workspace profile
 */
export interface WorkspaceProfile {
  id: string;
  workspace_id: string;
  name: string;
  description?: string;
  permissions: string[];
  is_default: boolean;
}

/**
 * Create workspace request
 */
export interface CreateWorkspaceRequest {
  name: string;
  slug?: string;
  description?: string;
  settings?: Partial<WorkspaceSettings>;
}

/**
 * Update workspace request
 */
export interface UpdateWorkspaceRequest {
  name?: string;
  description?: string;
  settings?: Partial<WorkspaceSettings>;
}

/**
 * Add member request
 */
export interface AddMemberRequest {
  user_id?: string;
  email?: string;
  role: 'viewer' | 'member' | 'admin';
}

/**
 * Update member request
 */
export interface UpdateMemberRequest {
  role: 'viewer' | 'member' | 'admin';
}

/**
 * Interface for the internal client used by WorkspacesAPI.
 */
interface WorkspacesClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  put<T>(path: string, body?: unknown): Promise<T>;
  delete<T>(path: string): Promise<T>;
}

/**
 * Workspaces API namespace.
 *
 * Provides methods for managing workspaces:
 * - Create and configure workspaces
 * - Manage workspace members and roles
 * - Configure workspace settings and profiles
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Create a workspace
 * const workspace = await client.workspaces.create({
 *   name: 'Engineering Team',
 *   description: 'Technical discussions and decision-making',
 * });
 *
 * // Add members
 * await client.workspaces.addMember(workspace.id, { email: 'dev@company.com', role: 'member' });
 *
 * // List members
 * const { members } = await client.workspaces.listMembers(workspace.id);
 *
 * // Update settings
 * await client.workspaces.update(workspace.id, {
 *   settings: { allow_public_debates: false },
 * });
 * ```
 */
export class WorkspacesAPI {
  constructor(private client: WorkspacesClientInterface) {}

  // ===========================================================================
  // Workspace Management
  // ===========================================================================

  /**
   * List all workspaces.
   */
  async list(): Promise<{ workspaces: Workspace[]; total: number }> {
    return this.client.listWorkspaces();
  }

  /**
   * Get a specific workspace by ID.
   */
  async get(id: string): Promise<Workspace> {
    return this.client.getWorkspace(id);
  }

  /**
   * Create a new workspace.
   */
  async create(body: CreateWorkspaceRequest): Promise<Workspace> {
    return this.client.createWorkspace(body);
  }

  /**
   * Update an existing workspace.
   */
  async update(id: string, body: UpdateWorkspaceRequest): Promise<Workspace> {
    return this.client.updateWorkspace(id, body);
  }

  /**
   * Delete a workspace.
   */
  async delete(id: string): Promise<{ deleted: boolean }> {
    return this.client.deleteWorkspace(id);
  }

  // ===========================================================================
  // Member Management
  // ===========================================================================

  /**
   * List workspace members.
   */
  async listMembers(workspaceId: string): Promise<{ members: WorkspaceMember[]; total: number }> {
    return this.client.listWorkspaceMembers(workspaceId);
  }

  /**
   * Add a member to the workspace.
   */
  async addMember(workspaceId: string, body: AddMemberRequest): Promise<WorkspaceMember> {
    return this.client.addWorkspaceMember(workspaceId, body);
  }

  /**
   * Update a member's role.
   */
  async updateMember(workspaceId: string, userId: string, body: UpdateMemberRequest): Promise<WorkspaceMember> {
    return this.client.updateWorkspaceMember(workspaceId, userId, body);
  }

  /**
   * Remove a member from the workspace.
   */
  async removeMember(workspaceId: string, userId: string): Promise<{ removed: boolean }> {
    return this.client.removeWorkspaceMember(workspaceId, userId);
  }

  // ===========================================================================
  // Profiles
  // ===========================================================================

  /**
   * List workspace profiles (permission sets).
   */
  async listProfiles(workspaceId: string): Promise<{ profiles: WorkspaceProfile[] }> {
    return this.client.listWorkspaceProfiles(workspaceId);
  }
}
