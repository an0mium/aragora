/**
 * Enterprise Authentication Example
 *
 * Demonstrates enterprise authentication capabilities in Aragora:
 * - OIDC/OAuth setup and configuration
 * - MFA (Multi-Factor Authentication) configuration
 * - SCIM 2.0 user provisioning
 * - SSO integration with identity providers
 *
 * Usage:
 *   npx ts-node examples/auth-enterprise.ts
 *
 * Environment:
 *   ARAGORA_API_KEY - Your API key (admin role required)
 *   ARAGORA_API_URL - API URL (default: https://api.aragora.ai)
 */

import { createClient, AragoraError } from '@aragora/sdk';

// Configuration
const API_URL = process.env.ARAGORA_API_URL || 'https://api.aragora.ai';
const API_KEY = process.env.ARAGORA_API_KEY || 'your-admin-api-key-here';

async function main() {
  // Initialize the client with admin credentials
  const client = createClient({
    baseUrl: API_URL,
    apiKey: API_KEY,
    timeout: 30000,
  });

  try {
    // =========================================================================
    // 1. OAuth Provider Configuration
    // =========================================================================
    console.log('=== OAuth Provider Setup ===\n');

    // List available OAuth providers
    const { providers } = await client.oauth.getProviders();
    console.log('Available OAuth providers:');
    for (const provider of providers) {
      console.log(`  - ${provider.name} (${provider.provider}): ${provider.enabled ? 'Enabled' : 'Disabled'}`);
    }
    console.log('');

    // Get Google OAuth authorization URL
    const googleAuth = await client.oauth.getGoogleAuthUrl({
      redirect_uri: 'https://your-app.com/auth/callback',
      state: 'random-state-string-for-csrf-protection',
    });
    console.log('Google OAuth URL:', googleAuth.authorization_url);
    console.log('State:', googleAuth.state);
    console.log('');

    // Get GitHub OAuth authorization URL
    const githubAuth = await client.oauth.getGitHubAuthUrl({
      redirect_uri: 'https://your-app.com/auth/callback/github',
    });
    console.log('GitHub OAuth URL:', githubAuth.authorization_url);
    console.log('');

    // Get Microsoft OAuth for enterprise Azure AD integration
    const microsoftAuth = await client.oauth.getMicrosoftAuthUrl({
      redirect_uri: 'https://your-app.com/auth/callback/microsoft',
    });
    console.log('Microsoft (Azure AD) OAuth URL:', microsoftAuth.authorization_url);
    console.log('');

    // =========================================================================
    // 2. OIDC Configuration for Enterprise SSO
    // =========================================================================
    console.log('=== OIDC Configuration ===\n');

    // Get OIDC authorization URL for custom identity provider
    const oidcAuth = await client.oauth.getOIDCAuthUrl({
      provider_id: 'okta-enterprise', // Your configured OIDC provider
      redirect_uri: 'https://your-app.com/auth/callback/oidc',
    });
    console.log('OIDC Authorization URL:', oidcAuth.authorization_url);
    console.log('');

    // List user's linked OAuth providers
    const { providers: linkedProviders } = await client.oauth.getLinkedProviders();
    console.log('User linked OAuth accounts:');
    for (const linked of linkedProviders) {
      console.log(`  - ${linked.provider}: ${linked.email} (linked: ${linked.linked_at})`);
    }
    console.log('');

    // =========================================================================
    // 3. MFA (Multi-Factor Authentication) Configuration
    // =========================================================================
    console.log('=== MFA Configuration ===\n');

    // Setup MFA - generates secret and QR code
    console.log('Setting up MFA...');
    const mfaSetup = await client.rbac.setupMfa();
    console.log('MFA Secret:', mfaSetup.secret);
    console.log('QR Code URL:', mfaSetup.qr_code);
    console.log('');

    // After user scans QR code and enters code:
    // const mfaCode = '123456'; // From authenticator app
    // const enableResult = await client.rbac.enableMfa(mfaCode);
    // console.log('MFA Enabled:', enableResult.enabled);
    // console.log('Backup Codes:', enableResult.backup_codes);

    // Verify MFA during login
    console.log('To verify MFA during login:');
    console.log('  const result = await client.rbac.verifyMfa(code);');
    console.log('  if (result.verified) { /* proceed with login */ }');
    console.log('');

    // Regenerate backup codes (requires current MFA code)
    console.log('To regenerate backup codes:');
    console.log('  const { backup_codes } = await client.rbac.regenerateBackupCodes(currentCode);');
    console.log('');

    // =========================================================================
    // 4. API Key Management
    // =========================================================================
    console.log('=== API Key Management ===\n');

    // Generate a new API key with specific permissions
    console.log('Generating API key...');
    const newKey = await client.rbac.generateApiKey(
      'deployment-service',
      ['debates:read', 'debates:create', 'workflows:execute'],
      '2025-12-31T23:59:59Z' // Expiration date
    );
    console.log('New API Key:', newKey.key);
    console.log('Key ID:', newKey.key_id);
    console.log('(Store this key securely - it cannot be retrieved again!)');
    console.log('');

    // List existing API keys
    const { keys } = await client.rbac.listApiKeys();
    console.log('Existing API keys:');
    for (const key of keys) {
      console.log(`  - ${(key as { name?: string }).name || 'Unnamed'} (${(key as { key_id?: string }).key_id})`);
    }
    console.log('');

    // =========================================================================
    // 5. Session Management
    // =========================================================================
    console.log('=== Session Management ===\n');

    // List active sessions
    const { sessions } = await client.rbac.listSessions();
    console.log(`Active sessions: ${sessions.length}`);
    for (const session of sessions) {
      const s = session as { id?: string; last_active?: string; ip_address?: string; user_agent?: string };
      console.log(`  - Session ${s.id}`);
      console.log(`    Last active: ${s.last_active || 'Unknown'}`);
      console.log(`    IP: ${s.ip_address || 'Unknown'}`);
      console.log(`    User Agent: ${s.user_agent || 'Unknown'}`);
    }
    console.log('');

    // Revoke a specific session (uncomment to use)
    // await client.rbac.revokeSession('session-id-here');
    // console.log('Session revoked');

    // Logout from all devices (uncomment to use)
    // const { logged_out } = await client.rbac.logoutAll();
    // console.log('Logged out from all devices:', logged_out);

    // =========================================================================
    // 6. Role-Based Access Control (RBAC)
    // =========================================================================
    console.log('=== RBAC Configuration ===\n');

    // List all roles
    const { roles } = await client.rbac.listRoles();
    console.log('Available roles:');
    for (const role of roles) {
      console.log(`  - ${role.name} (${role.id})`);
      console.log(`    Permissions: ${role.permissions.slice(0, 3).join(', ')}${role.permissions.length > 3 ? '...' : ''}`);
      console.log(`    System role: ${role.is_system}`);
    }
    console.log('');

    // Create a custom role
    const customRole = await client.rbac.createRole({
      name: 'Debate Analyst',
      description: 'Can view debates and analytics but not create or modify',
      permissions: [
        'debates:read',
        'analytics:read',
        'explainability:read',
        'reports:read',
      ],
    });
    console.log(`Created role: ${customRole.name} (${customRole.id})`);
    console.log('');

    // List all permissions
    const { permissions } = await client.rbac.listPermissions();
    console.log('Sample permissions:');
    for (const perm of permissions.slice(0, 10)) {
      console.log(`  - ${perm.name}: ${perm.description || 'No description'}`);
    }
    console.log(`  ... and ${permissions.length - 10} more`);
    console.log('');

    // Assign role to user
    const userId = 'user-123'; // Replace with actual user ID
    await client.rbac.assignRole(userId, customRole.id);
    console.log(`Assigned role '${customRole.name}' to user ${userId}`);

    // Check user permissions
    const { allowed } = await client.rbac.checkPermission(userId, 'debates:read');
    console.log(`User can read debates: ${allowed}`);
    console.log('');

    // =========================================================================
    // 7. User Management
    // =========================================================================
    console.log('=== User Management ===\n');

    // List users in organization
    const { users, total } = await client.rbac.listUsers({ limit: 10 });
    console.log(`Users in organization: ${total}`);
    for (const user of users.slice(0, 5)) {
      const u = user as { email?: string; name?: string; role?: string };
      console.log(`  - ${u.email} (${u.name || 'No name'}) - Role: ${u.role || 'N/A'}`);
    }
    console.log('');

    // Invite a new user
    console.log('To invite a new user:');
    console.log('  const { invitation_id } = await client.rbac.inviteUser(');
    console.log("    'newuser@example.com',");
    console.log("    'analyst' // role");
    console.log('  );');
    console.log('');

    // =========================================================================
    // 8. Audit and Compliance
    // =========================================================================
    console.log('=== Audit and Compliance ===\n');

    // Query audit entries
    const { entries } = await client.rbac.queryAudit({
      action: 'login',
      since: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(), // Last 7 days
      limit: 10,
    });
    console.log(`Recent login events: ${entries.length}`);
    for (const entry of entries.slice(0, 5)) {
      const e = entry as { user_id?: string; action?: string; timestamp?: string; ip_address?: string };
      console.log(`  - ${e.timestamp}: User ${e.user_id} - ${e.action}`);
    }
    console.log('');

    // Get user activity history
    const { activities } = await client.rbac.getUserActivityHistory(userId, { limit: 10 });
    console.log(`User activity history: ${activities.length} events`);
    console.log('');

    // Verify audit log integrity
    const { valid, issues } = await client.rbac.verifyAuditIntegrity();
    console.log(`Audit log integrity: ${valid ? 'Valid' : 'Issues found'}`);
    if (issues.length > 0) {
      console.log('Issues:', issues);
    }
    console.log('');

    // Get denied access attempts (security monitoring)
    const { denied } = await client.rbac.getDeniedAccess({ limit: 10 });
    console.log(`Recent denied access attempts: ${denied.length}`);
    console.log('');

    // =========================================================================
    // 9. Workspace Roles (For multi-workspace environments)
    // =========================================================================
    console.log('=== Workspace Roles ===\n');

    const workspaceId = 'workspace-123'; // Replace with actual workspace ID

    // Get workspace-specific roles
    console.log('Getting workspace roles...');
    console.log('  const { roles, profile } = await client.rbac.getWorkspaceRoles(workspaceId);');
    console.log('');

    // Add member to workspace
    console.log('To add member to workspace:');
    console.log('  await client.rbac.addWorkspaceMember(workspaceId, userId, "editor");');
    console.log('');

    // Update member role in workspace
    console.log('To update member role:');
    console.log('  await client.rbac.updateMemberRole(workspaceId, userId, "admin");');
    console.log('');

    console.log('Enterprise authentication example completed successfully!');

  } catch (error) {
    handleError(error);
    process.exit(1);
  }
}

// =========================================================================
// OAuth Callback Handler Example
// =========================================================================
async function handleOAuthCallback(
  client: ReturnType<typeof createClient>,
  provider: 'google' | 'github' | 'microsoft' | 'apple' | 'oidc',
  code: string,
  state: string
): Promise<void> {
  console.log(`Handling ${provider} OAuth callback...`);

  // Exchange authorization code for tokens
  const result = await client.oauth.handleCallback(provider, {
    code,
    state,
    redirect_uri: 'https://your-app.com/auth/callback',
  });

  if (result.success) {
    console.log('OAuth successful!');
    console.log(`User ID: ${result.user_id}`);
    console.log(`Email: ${result.email}`);
    console.log(`Name: ${result.name}`);
    console.log(`Access Token: ${result.access_token}`);
    if (result.refresh_token) {
      console.log(`Refresh Token: ${result.refresh_token}`);
    }
  } else {
    console.error('OAuth failed:', result.error);
    console.error('Description:', result.error_description);
  }
}

// =========================================================================
// Link OAuth Account Example
// =========================================================================
async function linkOAuthAccount(
  client: ReturnType<typeof createClient>,
  provider: 'google' | 'github' | 'microsoft' | 'apple' | 'oidc',
  code: string
): Promise<void> {
  console.log(`Linking ${provider} account...`);

  const result = await client.oauth.linkAccount({
    provider,
    code,
    redirect_uri: 'https://your-app.com/auth/link',
  });

  if (result.success) {
    console.log('Account linked successfully!');
    console.log(`Provider: ${result.provider}`);
    console.log(`Email: ${result.email}`);
  } else {
    console.error('Linking failed:', result.message);
  }
}

// =========================================================================
// Helper Functions
// =========================================================================

function handleError(error: unknown): void {
  if (error instanceof AragoraError) {
    console.error('\n--- Aragora Error ---');
    console.error(`Message: ${error.message}`);
    console.error(`Code: ${error.code || 'N/A'}`);
    console.error(`Status: ${error.status || 'N/A'}`);

    // Common enterprise auth errors
    if (error.code === 'FORBIDDEN') {
      console.error('\nNote: This operation requires admin privileges.');
      console.error('Ensure your API key has the required permissions.');
    } else if (error.code === 'AUTH_REQUIRED') {
      console.error('\nNote: Authentication is required for this operation.');
      console.error('Check that your API key is valid and not expired.');
    }

    if (error.traceId) {
      console.error(`Trace ID: ${error.traceId}`);
    }
  } else if (error instanceof Error) {
    console.error('\n--- Error ---');
    console.error(`Message: ${error.message}`);
  } else {
    console.error('\n--- Unknown Error ---');
    console.error(error);
  }
}

// Run the example
main();

// Export functions for use as a module
export { handleOAuthCallback, linkOAuthAccount };
