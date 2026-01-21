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

// Document mapping: source -> destination
const DOC_MAP = {
  // Getting Started
  'QUICKSTART.md': 'getting-started/quickstart.md',
  'INSTALLATION.md': 'getting-started/installation.md',

  // Core Concepts
  'DEBATE_PHASES.md': 'core-concepts/debates.md',
  'CUSTOM_AGENTS.md': 'core-concepts/agents.md',
  'algorithms/CONSENSUS.md': 'core-concepts/consensus.md',
  'MEMORY_TIERS.md': 'core-concepts/memory.md',

  // Guides
  'SDK_GUIDE.md': 'guides/sdk.md',
  'AGENT_DEVELOPMENT.md': 'guides/custom-agents.md',
  'WORKFLOWS.md': 'guides/workflows.md',
  'INTEGRATIONS.md': 'guides/integrations.md',
  'EVIDENCE.md': 'guides/evidence.md',

  // Deployment
  'SECURITY_DEPLOYMENT.md': 'deployment/security.md',
  'SCALING.md': 'deployment/scaling.md',
  'REDIS_HA.md': 'deployment/redis.md',

  // Enterprise
  'GOVERNANCE.md': 'enterprise/multi-tenancy.md',
  'integration/control-plane-setup.md': 'enterprise/control-plane.md',
  'COMPLIANCE.md': 'enterprise/compliance.md',

  // Runbooks
  'runbooks/RUNBOOK_DEPLOYMENT.md': 'deployment/runbook.md',
  'runbooks/RUNBOOK_INCIDENT.md': 'enterprise/incident-response.md',
  'DISASTER_RECOVERY.md': 'deployment/disaster-recovery.md',
};

// Add frontmatter to markdown files
function addFrontmatter(content, title, description) {
  // Check if already has frontmatter
  if (content.startsWith('---')) {
    return content;
  }

  const frontmatter = `---
title: ${title}
description: ${description || title}
---

`;

  return frontmatter + content;
}

// Extract title from markdown
function extractTitle(content) {
  const match = content.match(/^#\s+(.+)$/m);
  return match ? match[1] : 'Documentation';
}

// Process a single file
function processFile(srcPath, destPath) {
  if (!fs.existsSync(srcPath)) {
    console.log(`  Skipping (not found): ${srcPath}`);
    return;
  }

  let content = fs.readFileSync(srcPath, 'utf8');
  const title = extractTitle(content);

  // Add frontmatter
  content = addFrontmatter(content, title);

  // Fix relative links
  content = content.replace(
    /\]\(\.\.\/([^)]+)\)/g,
    ']($1)'
  );

  // Ensure destination directory exists
  const destDir = path.dirname(destPath);
  if (!fs.existsSync(destDir)) {
    fs.mkdirSync(destDir, { recursive: true });
  }

  fs.writeFileSync(destPath, content);
  console.log(`  Synced: ${path.basename(srcPath)} -> ${destPath}`);
}

// Main sync function
function syncDocs() {
  console.log('Syncing documentation...\n');

  // Ensure destination directory exists
  if (!fs.existsSync(DEST_DIR)) {
    fs.mkdirSync(DEST_DIR, { recursive: true });
  }

  // Process each mapped file
  for (const [src, dest] of Object.entries(DOC_MAP)) {
    const srcPath = path.join(SOURCE_DIR, src);
    const destPath = path.join(DEST_DIR, dest);
    processFile(srcPath, destPath);
  }

  // Create index files
  createIndexFile('getting-started', 'Getting Started', 'Learn how to get started with Aragora');
  createIndexFile('core-concepts', 'Core Concepts', 'Understand the key concepts of Aragora');
  createIndexFile('guides', 'Guides', 'Step-by-step guides for common tasks');
  createIndexFile('deployment', 'Deployment', 'Deploy Aragora in production');
  createIndexFile('enterprise', 'Enterprise', 'Enterprise features and compliance');

  console.log('\nDone!');
}

function createIndexFile(category, title, description) {
  const indexPath = path.join(DEST_DIR, category, 'index.md');
  const content = `---
title: ${title}
description: ${description}
---

# ${title}

${description}

Explore the documentation in this section to learn more.
`;

  if (!fs.existsSync(path.dirname(indexPath))) {
    fs.mkdirSync(path.dirname(indexPath), { recursive: true });
  }

  fs.writeFileSync(indexPath, content);
}

// Run sync
syncDocs();
