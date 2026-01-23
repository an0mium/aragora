#!/usr/bin/env node
/**
 * Sync documentation from main docs/ directory to Docusaurus structure.
 *
 * This script copies and transforms markdown files from the main docs/
 * directory to the Docusaurus docs/ directory with proper structure.
 *
 * Usage:
 *   node scripts/sync-docs.js
 */

const fs = require('fs');
const path = require('path');

// Source and destination directories
const SOURCE_DIR = path.join(__dirname, '../../docs');
const DEST_DIR = path.join(__dirname, '../docs');

// Document mapping: source -> destination with category organization
const DOC_MAP = {
  // =========================================================================
  // Getting Started
  // =========================================================================
  'GETTING_STARTED.md': 'getting-started/overview.md',
  'DEVELOPER_QUICKSTART.md': 'getting-started/quickstart.md',
  'CONFIGURATION.md': 'getting-started/configuration.md',
  'ENVIRONMENT.md': 'getting-started/environment.md',

  // =========================================================================
  // Core Concepts
  // =========================================================================
  'DEBATE_PHASES.md': 'core-concepts/debates.md',
  'DEBATE_INTERNALS.md': 'core-concepts/debate-internals.md',
  'CUSTOM_AGENTS.md': 'core-concepts/agents.md',
  'AGENT_DEVELOPMENT.md': 'core-concepts/agent-development.md',
  'AGENT_SELECTION.md': 'core-concepts/agent-selection.md',
  'algorithms/CONSENSUS.md': 'core-concepts/consensus.md',
  'MEMORY_TIERS.md': 'core-concepts/memory.md',
  'MEMORY_STRATEGY.md': 'core-concepts/memory-strategy.md',
  'KNOWLEDGE_MOUND.md': 'core-concepts/knowledge-mound.md',
  'ARCHITECTURE.md': 'core-concepts/architecture.md',
  'REASONING.md': 'core-concepts/reasoning.md',

  // =========================================================================
  // Guides
  // =========================================================================
  'SDK_GUIDE.md': 'guides/sdk.md',
  'API_QUICK_START.md': 'guides/api-quickstart.md',
  'API_USAGE.md': 'guides/api-usage.md',
  'WORKFLOWS.md': 'guides/workflows.md',
  'INTEGRATIONS.md': 'guides/integrations.md',
  'BOT_INTEGRATIONS.md': 'guides/bot-integrations.md',
  'CHAT_CONNECTOR_GUIDE.md': 'guides/chat-connector.md',
  'CONNECTORS.md': 'guides/connectors.md',
  'CONNECTORS_SETUP.md': 'guides/connectors-setup.md',
  'CONNECTOR_TROUBLESHOOTING.md': 'guides/connector-troubleshooting.md',
  'EVIDENCE.md': 'guides/evidence.md',
  'GRAPH_DEBATES.md': 'guides/graph-debates.md',
  'MATRIX_DEBATES.md': 'guides/matrix-debates.md',
  'GAUNTLET.md': 'guides/gauntlet.md',
  'AUTOMATION_INTEGRATIONS.md': 'guides/automation.md',
  'EMAIL_PRIORITIZATION.md': 'guides/email-prioritization.md',
  'SHARED_INBOX.md': 'guides/shared-inbox.md',
  'COST_VISIBILITY.md': 'guides/cost-visibility.md',
  'CODING_ASSISTANCE.md': 'guides/coding-assistance.md',
  'BROADCAST.md': 'guides/broadcast.md',
  'PULSE.md': 'guides/pulse.md',
  'WEBSOCKET_EVENTS.md': 'guides/websocket-events.md',

  // =========================================================================
  // API Reference
  // =========================================================================
  'API_REFERENCE.md': 'api/reference.md',
  'API_ENDPOINTS.md': 'api/endpoints.md',
  'API_EXAMPLES.md': 'api/examples.md',
  'API_VERSIONING.md': 'api/versioning.md',
  'API_RATE_LIMITS.md': 'api/rate-limits.md',
  'API_STABILITY.md': 'api/stability.md',
  'API_DISCOVERY.md': 'api/discovery.md',
  'CLI_REFERENCE.md': 'api/cli.md',
  'GITHUB_PR_REVIEW.md': 'api/github-pr-review.md',

  // =========================================================================
  // Deployment
  // =========================================================================
  'DEPLOYMENT.md': 'deployment/overview.md',
  'SECURITY_DEPLOYMENT.md': 'deployment/security.md',
  'SCALING.md': 'deployment/scaling.md',
  'CAPACITY_PLANNING.md': 'deployment/capacity-planning.md',
  'REDIS_HA.md': 'deployment/redis-ha.md',
  'DATABASE_SETUP.md': 'deployment/database-setup.md',
  'DATABASE.md': 'deployment/database.md',
  'DATABASE_SCHEMA.md': 'deployment/database-schema.md',
  'DISASTER_RECOVERY.md': 'deployment/disaster-recovery.md',
  'DR_DRILL_PROCEDURES.md': 'deployment/dr-drills.md',
  'OBSERVABILITY.md': 'deployment/observability-setup.md',

  // =========================================================================
  // Operations / Runbooks
  // =========================================================================
  'runbooks/RUNBOOK_DEPLOYMENT.md': 'operations/runbook-deployment.md',
  'runbooks/RUNBOOK_INCIDENT.md': 'operations/runbook-incident.md',
  'runbooks/RUNBOOK_DATABASE_ISSUES.md': 'operations/runbook-database.md',
  'runbooks/RUNBOOK_PROVIDER_FAILURE.md': 'operations/runbook-provider.md',
  'ALERT_RUNBOOKS.md': 'operations/alert-runbooks.md',

  // =========================================================================
  // Enterprise
  // =========================================================================
  'GOVERNANCE.md': 'enterprise/governance.md',
  'CONTROL_PLANE.md': 'enterprise/control-plane-overview.md',
  'CONTROL_PLANE_GUIDE.md': 'enterprise/control-plane.md',
  'ENTERPRISE_FEATURES.md': 'enterprise/features.md',
  'ENTERPRISE_SUPPORT.md': 'enterprise/support.md',
  'COMMERCIAL_OVERVIEW.md': 'enterprise/commercial-overview.md',
  'COMMERCIAL_POSITIONING.md': 'enterprise/positioning.md',
  'BILLING.md': 'enterprise/billing.md',
  'BILLING_UNITS.md': 'enterprise/billing-units.md',

  // =========================================================================
  // Security & Compliance
  // =========================================================================
  'SECURITY.md': 'security/overview.md',
  'AUTH_GUIDE.md': 'security/authentication.md',
  'COMPLIANCE.md': 'security/compliance.md',
  'DATA_CLASSIFICATION.md': 'security/data-classification.md',
  'DATA_RESIDENCY.md': 'security/data-residency.md',
  'PRIVACY_POLICY.md': 'security/privacy-policy.md',
  'BREACH_NOTIFICATION_SLA.md': 'security/breach-notification.md',
  'CI_CD_SECURITY.md': 'security/ci-cd.md',
  'REMOTE_WORK_SECURITY.md': 'security/remote-work.md',
  'DSAR_WORKFLOW.md': 'security/dsar.md',

  // =========================================================================
  // Admin & Management
  // =========================================================================
  'ADMIN.md': 'admin/overview.md',
  'A_B_TESTING.md': 'admin/ab-testing.md',
  'NOMIC_LOOP.md': 'admin/nomic-loop.md',

  // =========================================================================
  // Advanced Topics
  // =========================================================================
  'RLM_GUIDE.md': 'advanced/rlm.md',
  'RLM_USER_GUIDE.md': 'advanced/rlm-user.md',
  'RLM_DEVELOPER_GUIDE.md': 'advanced/rlm-developer.md',
  'INTEGRATION_RLM.md': 'advanced/rlm-integration.md',
  'CROSS_POLLINATION.md': 'advanced/cross-pollination.md',
  'CROSS_FUNCTIONAL_FEATURES.md': 'advanced/cross-functional.md',
  'TRICKSTER.md': 'advanced/trickster.md',
  'FORMAL_VERIFICATION.md': 'advanced/formal-verification.md',

  // =========================================================================
  // Analysis & Metrics
  // =========================================================================
  'ANALYSIS.md': 'analysis/overview.md',
  'CODEBASE_ANALYSIS.md': 'analysis/codebase.md',
  'BENCHMARK_RESULTS.md': 'analysis/benchmarks.md',

  // =========================================================================
  // Contributing
  // =========================================================================
  'CONTRIBUTING.md': 'contributing/guide.md',
  'DEPRECATION_POLICY.md': 'contributing/deprecation.md',
  'STATUS.md': 'contributing/status.md',
  'DEPENDENCIES.md': 'contributing/dependencies.md',

  // =========================================================================
  // Additional Missing Files (commonly referenced)
  // =========================================================================
  // Core
  'TROUBLESHOOTING.md': 'operations/troubleshooting.md',
  'QUEUE.md': 'guides/queue.md',
  'RATE_LIMITING.md': 'api/rate-limits.md',
  'SECRETS_MANAGEMENT.md': 'deployment/secrets-management.md',
  'MEMORY_ANALYTICS.md': 'core-concepts/memory-analytics.md',

  // API
  'MCP_INTEGRATION.md': 'guides/mcp-integration.md',
  'MCP_ADVANCED.md': 'guides/mcp-advanced.md',

  // Operations
  'PERFORMANCE_TARGETS.md': 'operations/performance-targets.md',
  'PRODUCTION_READINESS.md': 'operations/production-readiness.md',

  // Advanced
  'GENESIS.md': 'advanced/genesis.md',
  'EVOLUTION_PATTERNS.md': 'advanced/evolution-patterns.md',

  // Admin

  // Security

  // Integration / Enterprise
  'POSTGRESQL_MIGRATION.md': 'deployment/postgresql-migration.md',

  // Algorithms
  'algorithms/CONVERGENCE.md': 'core-concepts/convergence-algorithm.md',
  'algorithms/ELO_CALIBRATION.md': 'core-concepts/elo-calibration.md',

  // Documents
  'FEATURES.md': 'guides/features.md',
  'VERTICALS.md': 'guides/verticals.md',
  'OPERATIONS.md': 'operations/overview.md',
};

// Add frontmatter to markdown files
function addFrontmatter(content, title, description) {
  // Check if already has frontmatter
  if (content.startsWith('---')) {
    return content;
  }

  // Escape title for YAML (quote if contains special chars)
  const escapeYaml = (str) => {
    if (str.includes(':') || str.includes('#') || str.includes("'") || str.includes('"') || str.includes('\n')) {
      // Double-quote and escape internal double quotes
      return `"${str.replace(/"/g, '\\"')}"`;
    }
    return str;
  };

  const safeTitle = escapeYaml(title);
  const safeDesc = escapeYaml(description || title);

  const frontmatter = `---
title: ${safeTitle}
description: ${safeDesc}
---

`;

  return frontmatter + content;
}

// Extract title from markdown
function extractTitle(content) {
  const match = content.match(/^#\s+(.+)$/m);
  return match ? match[1].replace(/[`*_]/g, '') : 'Documentation';
}

// Build reverse lookup from source file to destination path
const REVERSE_LOOKUP = {};
for (const [src, dest] of Object.entries(DOC_MAP)) {
  // Normalize source path variations
  const srcBase = src.replace(/^\.\//, '').replace(/^\//, '');
  const srcName = path.basename(srcBase);

  // Store both with and without .md extension
  REVERSE_LOOKUP[srcBase] = dest;
  REVERSE_LOOKUP[srcName] = dest;
  REVERSE_LOOKUP[srcBase.replace('.md', '')] = dest.replace('.md', '');
  REVERSE_LOOKUP[srcName.replace('.md', '')] = dest.replace('.md', '');
}

// Fix content for Docusaurus compatibility
function fixContent(content, destPath) {
  // Fix escaped backticks (common in generated docs)
  content = content.replace(/\\`\\`\\`/g, '```');
  content = content.replace(/\\`([^`\\]+)\\`/g, '`$1`');

  // Escape curly braces in URL patterns (e.g., {id} -> \{id\})
  // Only escape braces that look like URL params (word chars inside)
  content = content.replace(/\{(\w+)\}/g, '\\{$1\\}');

  // Escape angle brackets in comparisons (e.g., <0.3 -> &lt;0.3)
  content = content.replace(/<(\d)/g, '&lt;$1');

  // Get the current doc's directory for relative path calculation
  const currentDir = path.dirname(destPath);

  // Transform internal doc links to Docusaurus paths
  // Match links like [text](./FILE.md), [text](../FILE.md), [text](FILE.md)
  content = content.replace(
    /\]\((?:\.\.\/|\.\/)?([A-Z_\/]+\.md)(#[^)]+)?\)/gi,
    (match, filePath, anchor) => {
      // Try to find the destination path in our mapping
      const normalized = filePath.replace(/^\.\.\//, '').replace(/^\.\//, '');
      const newPath = REVERSE_LOOKUP[normalized] || REVERSE_LOOKUP[path.basename(normalized)];

      if (newPath) {
        // Calculate relative path from current doc to target doc
        const targetDir = path.dirname(newPath);
        const targetFile = path.basename(newPath, '.md');

        // If same directory, use just the filename
        if (targetDir === currentDir) {
          return `](./${targetFile}${anchor || ''})`;
        }

        // Calculate relative path
        const relativePath = path.relative(currentDir, targetDir);
        const relativeLink = relativePath ? `${relativePath}/${targetFile}` : targetFile;
        return `](${relativeLink}${anchor || ''})`;
      }

      // If not found, keep original but log it
      return match;
    }
  );

  // Also fix links without .md extension
  content = content.replace(
    /\]\((?:\.\.\/|\.\/)?([A-Z_\/]+)(#[^)]+)?\)(?!\.md)/gi,
    (match, filePath, anchor) => {
      // Check if this looks like an internal doc link
      if (!filePath.includes('/') && filePath === filePath.toUpperCase() && filePath.length > 3) {
        const newPath = REVERSE_LOOKUP[filePath] || REVERSE_LOOKUP[filePath + '.md'];
        if (newPath) {
          const targetDir = path.dirname(newPath);
          const targetFile = path.basename(newPath, '.md');

          if (targetDir === currentDir) {
            return `](./${targetFile}${anchor || ''})`;
          }

          const relativePath = path.relative(currentDir, targetDir);
          const relativeLink = relativePath ? `${relativePath}/${targetFile}` : targetFile;
          return `](${relativeLink}${anchor || ''})`;
        }
      }
      return match;
    }
  );

  return content;
}

// Process a single file
function processFile(srcPath, destPath) {
  if (!fs.existsSync(srcPath)) {
    console.log(`  Skipping (not found): ${path.basename(srcPath)}`);
    return false;
  }

  let content = fs.readFileSync(srcPath, 'utf8');
  const title = extractTitle(content);

  // Add frontmatter
  content = addFrontmatter(content, title);

  // Fix content for compatibility (pass relative dest path)
  const relDestPath = destPath.replace(DEST_DIR + '/', '');
  content = fixContent(content, relDestPath);

  // Ensure destination directory exists
  const destDir = path.dirname(destPath);
  if (!fs.existsSync(destDir)) {
    fs.mkdirSync(destDir, { recursive: true });
  }

  fs.writeFileSync(destPath, content);
  console.log(`  ✓ ${path.basename(srcPath)} -> ${destPath.replace(DEST_DIR + '/', '')}`);
  return true;
}

// Create index file for a category
function createIndexFile(category, title, description, items = []) {
  const indexPath = path.join(DEST_DIR, category, 'index.md');

  let itemsList = '';
  if (items.length > 0) {
    itemsList = '\n\n## In This Section\n\n' + items.map(item => `- [${item.title}](${item.path})`).join('\n');
  }

  const content = `---
title: ${title}
description: ${description}
sidebar_position: 1
---

# ${title}

${description}

Explore the documentation in this section to learn more.${itemsList}
`;

  if (!fs.existsSync(path.dirname(indexPath))) {
    fs.mkdirSync(path.dirname(indexPath), { recursive: true });
  }

  fs.writeFileSync(indexPath, content);
  console.log(`  ✓ Created index: ${category}/index.md`);
}

// Main sync function
function syncDocs() {
  console.log('\\n📚 Syncing documentation...\\n');

  // Ensure destination directory exists
  if (!fs.existsSync(DEST_DIR)) {
    fs.mkdirSync(DEST_DIR, { recursive: true });
  }

  let synced = 0;
  let skipped = 0;

  // Process each mapped file
  for (const [src, dest] of Object.entries(DOC_MAP)) {
    const srcPath = path.join(SOURCE_DIR, src);
    const destPath = path.join(DEST_DIR, dest);
    if (processFile(srcPath, destPath)) {
      synced++;
    } else {
      skipped++;
    }
  }

  // Create index files for each category
  console.log('\\n📁 Creating category index files...\\n');

  const categories = [
    { path: 'getting-started', title: 'Getting Started', desc: 'Learn how to get started with Aragora' },
    { path: 'core-concepts', title: 'Core Concepts', desc: 'Understand the key concepts of Aragora' },
    { path: 'guides', title: 'Guides', desc: 'Step-by-step guides for common tasks' },
    { path: 'api', title: 'API Reference', desc: 'Complete API documentation' },
    { path: 'deployment', title: 'Deployment', desc: 'Deploy Aragora in production' },
    { path: 'operations', title: 'Operations', desc: 'Runbooks and operational procedures' },
    { path: 'enterprise', title: 'Enterprise', desc: 'Enterprise features and compliance' },
    { path: 'security', title: 'Security & Compliance', desc: 'Security, authentication, and compliance' },
    { path: 'admin', title: 'Administration', desc: 'Administrative features and management' },
    { path: 'advanced', title: 'Advanced Topics', desc: 'Advanced features and internals' },
    { path: 'analysis', title: 'Analysis & Metrics', desc: 'Performance analysis and benchmarks' },
    { path: 'contributing', title: 'Contributing', desc: 'How to contribute to Aragora' },
  ];

  for (const cat of categories) {
    createIndexFile(cat.path, cat.title, cat.desc);
  }

  console.log(`\\n✅ Done! Synced ${synced} files, skipped ${skipped} (not found)\\n`);
}

// Run sync
syncDocs();
