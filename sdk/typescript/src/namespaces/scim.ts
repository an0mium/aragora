/**
 * SCIM 2.0 Namespace API
 *
 * Provides methods for SCIM 2.0 user and group provisioning (RFC 7643/7644):
 * - User CRUD operations
 * - Group CRUD operations
 * - Filtering and pagination
 *
 * Endpoints:
 *   GET    /scim/v2/Users           - List users
 *   POST   /scim/v2/Users           - Create user
 *   GET    /scim/v2/Users/:id       - Get user
 *   PUT    /scim/v2/Users/:id       - Replace user
 *   PATCH  /scim/v2/Users/:id       - Patch user
 *   DELETE /scim/v2/Users/:id       - Delete user
 *   GET    /scim/v2/Groups          - List groups
 *   POST   /scim/v2/Groups          - Create group
 *   GET    /scim/v2/Groups/:id      - Get group
 *   PUT    /scim/v2/Groups/:id      - Replace group
 *   PATCH  /scim/v2/Groups/:id      - Patch group
 *   DELETE /scim/v2/Groups/:id      - Delete group
 */

/**
 * SCIM User name component.
 */
export interface ScimUserName {
  formatted?: string;
  familyName?: string;
  givenName?: string;
  middleName?: string;
  honorificPrefix?: string;
  honorificSuffix?: string;
}

/**
 * SCIM email entry.
 */
export interface ScimEmail {
  value: string;
  type?: string;
  primary?: boolean;
}

/**
 * SCIM phone number entry.
 */
export interface ScimPhoneNumber {
  value: string;
  type?: string;
  primary?: boolean;
}

/**
 * SCIM User resource.
 */
export interface ScimUser {
  schemas: string[];
  id?: string;
  externalId?: string;
  userName: string;
  name?: ScimUserName;
  displayName?: string;
  nickName?: string;
  profileUrl?: string;
  title?: string;
  userType?: string;
  preferredLanguage?: string;
  locale?: string;
  timezone?: string;
  active?: boolean;
  emails?: ScimEmail[];
  phoneNumbers?: ScimPhoneNumber[];
  meta?: {
    resourceType?: string;
    created?: string;
    lastModified?: string;
    location?: string;
    version?: string;
  };
}

/**
 * SCIM Group member.
 */
export interface ScimGroupMember {
  value: string;
  $ref?: string;
  display?: string;
  type?: string;
}

/**
 * SCIM Group resource.
 */
export interface ScimGroup {
  schemas: string[];
  id?: string;
  externalId?: string;
  displayName: string;
  members?: ScimGroupMember[];
  meta?: {
    resourceType?: string;
    created?: string;
    lastModified?: string;
    location?: string;
    version?: string;
  };
}

/**
 * SCIM ListResponse.
 */
export interface ScimListResponse<T> {
  schemas: string[];
  totalResults: number;
  startIndex: number;
  itemsPerPage: number;
  Resources: T[];
}

/**
 * SCIM PatchOp.
 */
export interface ScimPatchOp {
  schemas: string[];
  Operations: Array<{
    op: 'add' | 'remove' | 'replace';
    path?: string;
    value?: unknown;
  }>;
}

/**
 * Options for listing SCIM resources.
 */
export interface ScimListOptions {
  /** 1-based starting index (default 1) */
  startIndex?: number;
  /** Maximum number of results (default 100) */
  count?: number;
  /** SCIM filter expression */
  filter?: string;
}

/**
 * Client interface for making HTTP requests.
 */
interface ScimClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * SCIM 2.0 API namespace.
 *
 * Provides RFC 7643/7644 compliant user and group provisioning.
 * Typically used by IdP (Identity Provider) integrations like
 * Okta, Azure AD, OneLogin, etc.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // List users with filtering
 * const users = await client.scim.listUsers({
 *   filter: 'userName eq "john@example.com"',
 * });
 *
 * // Create a user
 * const user = await client.scim.createUser({
 *   schemas: ['urn:ietf:params:scim:schemas:core:2.0:User'],
 *   userName: 'john@example.com',
 *   name: { givenName: 'John', familyName: 'Doe' },
 *   emails: [{ value: 'john@example.com', primary: true }],
 *   active: true,
 * });
 *
 * // Update a user
 * await client.scim.patchUser(user.id!, {
 *   schemas: ['urn:ietf:params:scim:api:messages:2.0:PatchOp'],
 *   Operations: [{ op: 'replace', path: 'active', value: false }],
 * });
 *
 * // Delete a user
 * await client.scim.deleteUser(user.id!);
 * ```
 */
export class SCIMAPI {
  constructor(private client: ScimClientInterface) {}

  // =========================================================================
  // Users
  // =========================================================================

  /**
   * List users with optional filtering and pagination.
   *
   * @param options - Filtering and pagination options
   * @returns SCIM ListResponse with user resources
   */
  async listUsers(options?: ScimListOptions): Promise<ScimListResponse<ScimUser>> {
    const params: Record<string, unknown> = {
      startIndex: options?.startIndex ?? 1,
      count: options?.count ?? 100,
    };
    if (options?.filter) params.filter = options.filter;

    return this.client.request('GET', '/scim/v2/Users', {
      params,
    });
  }

  /**
   * Get a user by ID.
   *
   * @param userId - SCIM user ID
   * @returns SCIM User resource
   */
  async getUser(userId: string): Promise<ScimUser> {
    return this.client.request('GET', `/scim/v2/Users/${userId}`);
  }

  /**
   * Create a new user.
   *
   * @param user - SCIM User resource with schemas, userName, etc.
   * @returns Created SCIM User resource
   */
  async createUser(user: ScimUser): Promise<ScimUser> {
    return this.client.request('POST', '/scim/v2/Users', {
      json: user as unknown as Record<string, unknown>,
    });
  }

  /**
   * Replace a user (full update).
   *
   * @param userId - SCIM user ID
   * @param user - Complete SCIM User resource
   * @returns Updated SCIM User resource
   */
  async replaceUser(userId: string, user: ScimUser): Promise<ScimUser> {
    return this.client.request('PUT', `/scim/v2/Users/${userId}`, {
      json: user as unknown as Record<string, unknown>,
    });
  }

  /**
   * Partially update a user.
   *
   * @param userId - SCIM user ID
   * @param operations - SCIM PatchOp with Operations array
   * @returns Updated SCIM User resource
   */
  async patchUser(userId: string, operations: ScimPatchOp): Promise<ScimUser> {
    return this.client.request('PATCH', `/scim/v2/Users/${userId}`, {
      json: operations as unknown as Record<string, unknown>,
    });
  }

  /**
   * Delete a user.
   *
   * @param userId - SCIM user ID
   * @returns void (204 No Content on success)
   */
  async deleteUser(userId: string): Promise<void> {
    await this.client.request('DELETE', `/scim/v2/Users/${userId}`);
  }

  // =========================================================================
  // Groups
  // =========================================================================

  /**
   * List groups with optional filtering and pagination.
   *
   * @param options - Filtering and pagination options
   * @returns SCIM ListResponse with group resources
   */
  async listGroups(options?: ScimListOptions): Promise<ScimListResponse<ScimGroup>> {
    const params: Record<string, unknown> = {
      startIndex: options?.startIndex ?? 1,
      count: options?.count ?? 100,
    };
    if (options?.filter) params.filter = options.filter;

    return this.client.request('GET', '/scim/v2/Groups', {
      params,
    });
  }

  /**
   * Get a group by ID.
   *
   * @param groupId - SCIM group ID
   * @returns SCIM Group resource
   */
  async getGroup(groupId: string): Promise<ScimGroup> {
    return this.client.request('GET', `/scim/v2/Groups/${groupId}`);
  }

  /**
   * Create a new group.
   *
   * @param group - SCIM Group resource with schemas, displayName, members, etc.
   * @returns Created SCIM Group resource
   */
  async createGroup(group: ScimGroup): Promise<ScimGroup> {
    return this.client.request('POST', '/scim/v2/Groups', {
      json: group as unknown as Record<string, unknown>,
    });
  }

  /**
   * Replace a group (full update).
   *
   * @param groupId - SCIM group ID
   * @param group - Complete SCIM Group resource
   * @returns Updated SCIM Group resource
   */
  async replaceGroup(groupId: string, group: ScimGroup): Promise<ScimGroup> {
    return this.client.request('PUT', `/scim/v2/Groups/${groupId}`, {
      json: group as unknown as Record<string, unknown>,
    });
  }

  /**
   * Partially update a group.
   *
   * @param groupId - SCIM group ID
   * @param operations - SCIM PatchOp with Operations array
   * @returns Updated SCIM Group resource
   */
  async patchGroup(groupId: string, operations: ScimPatchOp): Promise<ScimGroup> {
    return this.client.request('PATCH', `/scim/v2/Groups/${groupId}`, {
      json: operations as unknown as Record<string, unknown>,
    });
  }

  /**
   * Delete a group.
   *
   * @param groupId - SCIM group ID
   * @returns void (204 No Content on success)
   */
  async deleteGroup(groupId: string): Promise<void> {
    await this.client.request('DELETE', `/scim/v2/Groups/${groupId}`);
  }
}
