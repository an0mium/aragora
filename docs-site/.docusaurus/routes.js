import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/search',
    component: ComponentCreator('/search', '5de'),
    exact: true
  },
  {
    path: '/docs',
    component: ComponentCreator('/docs', '5d2'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', 'e14'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '0c7'),
            routes: [
              {
                path: '/docs/admin/',
                component: ComponentCreator('/docs/admin/', '314'),
                exact: true
              },
              {
                path: '/docs/admin/ab-testing',
                component: ComponentCreator('/docs/admin/ab-testing', 'a2e'),
                exact: true
              },
              {
                path: '/docs/admin/nomic-loop',
                component: ComponentCreator('/docs/admin/nomic-loop', '995'),
                exact: true
              },
              {
                path: '/docs/admin/overview',
                component: ComponentCreator('/docs/admin/overview', '6dc'),
                exact: true
              },
              {
                path: '/docs/advanced/',
                component: ComponentCreator('/docs/advanced/', '9e2'),
                exact: true
              },
              {
                path: '/docs/advanced/cross-functional',
                component: ComponentCreator('/docs/advanced/cross-functional', '43e'),
                exact: true
              },
              {
                path: '/docs/advanced/cross-pollination',
                component: ComponentCreator('/docs/advanced/cross-pollination', '11b'),
                exact: true
              },
              {
                path: '/docs/advanced/evolution-patterns',
                component: ComponentCreator('/docs/advanced/evolution-patterns', 'c58'),
                exact: true
              },
              {
                path: '/docs/advanced/formal-verification',
                component: ComponentCreator('/docs/advanced/formal-verification', '67e'),
                exact: true
              },
              {
                path: '/docs/advanced/genesis',
                component: ComponentCreator('/docs/advanced/genesis', '3a6'),
                exact: true
              },
              {
                path: '/docs/advanced/rlm',
                component: ComponentCreator('/docs/advanced/rlm', '1d3'),
                exact: true
              },
              {
                path: '/docs/advanced/rlm-developer',
                component: ComponentCreator('/docs/advanced/rlm-developer', '47c'),
                exact: true
              },
              {
                path: '/docs/advanced/rlm-integration',
                component: ComponentCreator('/docs/advanced/rlm-integration', '5a6'),
                exact: true
              },
              {
                path: '/docs/advanced/rlm-user',
                component: ComponentCreator('/docs/advanced/rlm-user', 'df5'),
                exact: true
              },
              {
                path: '/docs/advanced/trickster',
                component: ComponentCreator('/docs/advanced/trickster', '5d5'),
                exact: true
              },
              {
                path: '/docs/analysis/',
                component: ComponentCreator('/docs/analysis/', '0c0'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/',
                component: ComponentCreator('/docs/analysis/adr/', '52a'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/001-phase-based-debate-execution',
                component: ComponentCreator('/docs/analysis/adr/001-phase-based-debate-execution', 'c9a'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/002-agent-fallback-openrouter',
                component: ComponentCreator('/docs/analysis/adr/002-agent-fallback-openrouter', 'e26'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/003-multi-tier-memory-system',
                component: ComponentCreator('/docs/analysis/adr/003-multi-tier-memory-system', '857'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/004-incremental-type-safety',
                component: ComponentCreator('/docs/analysis/adr/004-incremental-type-safety', '1c5'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/005-composition-over-inheritance',
                component: ComponentCreator('/docs/analysis/adr/005-composition-over-inheritance', '5e7'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/006-api-versioning-strategy',
                component: ComponentCreator('/docs/analysis/adr/006-api-versioning-strategy', '214'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/007-selection-plugin-architecture',
                component: ComponentCreator('/docs/analysis/adr/007-selection-plugin-architecture', 'ace'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/008-rlm-semantic-compression',
                component: ComponentCreator('/docs/analysis/adr/008-rlm-semantic-compression', '641'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/009-control-plane-architecture',
                component: ComponentCreator('/docs/analysis/adr/009-control-plane-architecture', '0be'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/010-debate-orchestration-pattern',
                component: ComponentCreator('/docs/analysis/adr/010-debate-orchestration-pattern', '125'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/011-multi-tier-memory-comparison',
                component: ComponentCreator('/docs/analysis/adr/011-multi-tier-memory-comparison', 'bc0'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/012-agent-fallback-strategy',
                component: ComponentCreator('/docs/analysis/adr/012-agent-fallback-strategy', '3b6'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/013-workflow-dag-design',
                component: ComponentCreator('/docs/analysis/adr/013-workflow-dag-design', 'eda'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/014-knowledge-mound-architecture',
                component: ComponentCreator('/docs/analysis/adr/014-knowledge-mound-architecture', 'b24'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/015-lazy-import-patterns',
                component: ComponentCreator('/docs/analysis/adr/015-lazy-import-patterns', '443'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/016-marketplace-architecture',
                component: ComponentCreator('/docs/analysis/adr/016-marketplace-architecture', '1e4'),
                exact: true
              },
              {
                path: '/docs/analysis/benchmarks',
                component: ComponentCreator('/docs/analysis/benchmarks', '106'),
                exact: true
              },
              {
                path: '/docs/analysis/case-studies/',
                component: ComponentCreator('/docs/analysis/case-studies/', '756'),
                exact: true
              },
              {
                path: '/docs/analysis/case-studies/architecture-stress-test',
                component: ComponentCreator('/docs/analysis/case-studies/architecture-stress-test', '11c'),
                exact: true
              },
              {
                path: '/docs/analysis/case-studies/epic-strategic-debate',
                component: ComponentCreator('/docs/analysis/case-studies/epic-strategic-debate', 'f56'),
                exact: true
              },
              {
                path: '/docs/analysis/case-studies/gdpr-compliance-audit',
                component: ComponentCreator('/docs/analysis/case-studies/gdpr-compliance-audit', '93e'),
                exact: true
              },
              {
                path: '/docs/analysis/case-studies/security-api-review',
                component: ComponentCreator('/docs/analysis/case-studies/security-api-review', '760'),
                exact: true
              },
              {
                path: '/docs/analysis/codebase',
                component: ComponentCreator('/docs/analysis/codebase', 'd0c'),
                exact: true
              },
              {
                path: '/docs/analysis/overview',
                component: ComponentCreator('/docs/analysis/overview', '99e'),
                exact: true
              },
              {
                path: '/docs/api-reference/',
                component: ComponentCreator('/docs/api-reference/', 'd75'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/agents/list',
                component: ComponentCreator('/docs/api-reference/agents/list', '23a'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/agents/overview',
                component: ComponentCreator('/docs/api-reference/agents/overview', 'e88'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/agents/stats',
                component: ComponentCreator('/docs/api-reference/agents/stats', 'e8d'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/consensus',
                component: ComponentCreator('/docs/api-reference/debates/consensus', '145'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/create',
                component: ComponentCreator('/docs/api-reference/debates/create', '449'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/get',
                component: ComponentCreator('/docs/api-reference/debates/get', '55e'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/list',
                component: ComponentCreator('/docs/api-reference/debates/list', 'ab4'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/overview',
                component: ComponentCreator('/docs/api-reference/debates/overview', '69c'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/knowledge/overview',
                component: ComponentCreator('/docs/api-reference/knowledge/overview', '220'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/knowledge/query',
                component: ComponentCreator('/docs/api-reference/knowledge/query', '19a'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/knowledge/store',
                component: ComponentCreator('/docs/api-reference/knowledge/store', 'a0b'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/workflows/execute',
                component: ComponentCreator('/docs/api-reference/workflows/execute', 'a4f'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/workflows/overview',
                component: ComponentCreator('/docs/api-reference/workflows/overview', '870'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api/',
                component: ComponentCreator('/docs/api/', '077'),
                exact: true
              },
              {
                path: '/docs/api/cli',
                component: ComponentCreator('/docs/api/cli', 'e7b'),
                exact: true
              },
              {
                path: '/docs/api/discovery',
                component: ComponentCreator('/docs/api/discovery', 'aea'),
                exact: true
              },
              {
                path: '/docs/api/endpoints',
                component: ComponentCreator('/docs/api/endpoints', '281'),
                exact: true
              },
              {
                path: '/docs/api/evidence',
                component: ComponentCreator('/docs/api/evidence', '5d0'),
                exact: true
              },
              {
                path: '/docs/api/examples',
                component: ComponentCreator('/docs/api/examples', 'a20'),
                exact: true
              },
              {
                path: '/docs/api/github-pr-review',
                component: ComponentCreator('/docs/api/github-pr-review', '7ed'),
                exact: true
              },
              {
                path: '/docs/api/rate-limits',
                component: ComponentCreator('/docs/api/rate-limits', '99f'),
                exact: true
              },
              {
                path: '/docs/api/reference',
                component: ComponentCreator('/docs/api/reference', '690'),
                exact: true
              },
              {
                path: '/docs/api/stability',
                component: ComponentCreator('/docs/api/stability', 'e7e'),
                exact: true
              },
              {
                path: '/docs/api/versioning',
                component: ComponentCreator('/docs/api/versioning', '19d'),
                exact: true
              },
              {
                path: '/docs/api/webhooks',
                component: ComponentCreator('/docs/api/webhooks', '58b'),
                exact: true
              },
              {
                path: '/docs/contributing/',
                component: ComponentCreator('/docs/contributing/', '13a'),
                exact: true
              },
              {
                path: '/docs/contributing/dependencies',
                component: ComponentCreator('/docs/contributing/dependencies', '209'),
                exact: true
              },
              {
                path: '/docs/contributing/deprecation',
                component: ComponentCreator('/docs/contributing/deprecation', 'eb4'),
                exact: true
              },
              {
                path: '/docs/contributing/documentation-index',
                component: ComponentCreator('/docs/contributing/documentation-index', 'be8'),
                exact: true
              },
              {
                path: '/docs/contributing/documentation-map',
                component: ComponentCreator('/docs/contributing/documentation-map', '630'),
                exact: true
              },
              {
                path: '/docs/contributing/first-contribution',
                component: ComponentCreator('/docs/contributing/first-contribution', '9b3'),
                exact: true
              },
              {
                path: '/docs/contributing/frontend-development',
                component: ComponentCreator('/docs/contributing/frontend-development', 'eab'),
                exact: true
              },
              {
                path: '/docs/contributing/frontend-routes',
                component: ComponentCreator('/docs/contributing/frontend-routes', '64a'),
                exact: true
              },
              {
                path: '/docs/contributing/guide',
                component: ComponentCreator('/docs/contributing/guide', '24c'),
                exact: true
              },
              {
                path: '/docs/contributing/handler-development',
                component: ComponentCreator('/docs/contributing/handler-development', '8fe'),
                exact: true
              },
              {
                path: '/docs/contributing/handlers',
                component: ComponentCreator('/docs/contributing/handlers', 'ad2'),
                exact: true
              },
              {
                path: '/docs/contributing/status',
                component: ComponentCreator('/docs/contributing/status', 'fa5'),
                exact: true
              },
              {
                path: '/docs/contributing/testing',
                component: ComponentCreator('/docs/contributing/testing', 'e35'),
                exact: true
              },
              {
                path: '/docs/core-concepts/',
                component: ComponentCreator('/docs/core-concepts/', 'f69'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agent-catalog',
                component: ComponentCreator('/docs/core-concepts/agent-catalog', '118'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agent-development',
                component: ComponentCreator('/docs/core-concepts/agent-development', 'd5c'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agent-selection',
                component: ComponentCreator('/docs/core-concepts/agent-selection', '777'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agents',
                component: ComponentCreator('/docs/core-concepts/agents', '5e1'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/architecture',
                component: ComponentCreator('/docs/core-concepts/architecture', 'c35'),
                exact: true
              },
              {
                path: '/docs/core-concepts/consensus',
                component: ComponentCreator('/docs/core-concepts/consensus', '135'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/convergence-algorithm',
                component: ComponentCreator('/docs/core-concepts/convergence-algorithm', 'c7c'),
                exact: true
              },
              {
                path: '/docs/core-concepts/debate-internals',
                component: ComponentCreator('/docs/core-concepts/debate-internals', 'd8f'),
                exact: true
              },
              {
                path: '/docs/core-concepts/debates',
                component: ComponentCreator('/docs/core-concepts/debates', '48c'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/elo-calibration',
                component: ComponentCreator('/docs/core-concepts/elo-calibration', '854'),
                exact: true
              },
              {
                path: '/docs/core-concepts/knowledge-mound',
                component: ComponentCreator('/docs/core-concepts/knowledge-mound', '86d'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/memory',
                component: ComponentCreator('/docs/core-concepts/memory', 'b6e'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/memory-analytics',
                component: ComponentCreator('/docs/core-concepts/memory-analytics', '70e'),
                exact: true
              },
              {
                path: '/docs/core-concepts/memory-overview',
                component: ComponentCreator('/docs/core-concepts/memory-overview', '426'),
                exact: true
              },
              {
                path: '/docs/core-concepts/memory-strategy',
                component: ComponentCreator('/docs/core-concepts/memory-strategy', '6c8'),
                exact: true
              },
              {
                path: '/docs/core-concepts/reasoning',
                component: ComponentCreator('/docs/core-concepts/reasoning', '6a3'),
                exact: true
              },
              {
                path: '/docs/core-concepts/workflow-engine',
                component: ComponentCreator('/docs/core-concepts/workflow-engine', '9ad'),
                exact: true
              },
              {
                path: '/docs/deployment/',
                component: ComponentCreator('/docs/deployment/', '9db'),
                exact: true
              },
              {
                path: '/docs/deployment/async-gateway',
                component: ComponentCreator('/docs/deployment/async-gateway', 'ea0'),
                exact: true
              },
              {
                path: '/docs/deployment/capacity-planning',
                component: ComponentCreator('/docs/deployment/capacity-planning', '40f'),
                exact: true
              },
              {
                path: '/docs/deployment/container-volumes',
                component: ComponentCreator('/docs/deployment/container-volumes', 'c4b'),
                exact: true
              },
              {
                path: '/docs/deployment/database',
                component: ComponentCreator('/docs/deployment/database', '9cb'),
                exact: true
              },
              {
                path: '/docs/deployment/database-consolidation',
                component: ComponentCreator('/docs/deployment/database-consolidation', 'e5d'),
                exact: true
              },
              {
                path: '/docs/deployment/database-schema',
                component: ComponentCreator('/docs/deployment/database-schema', '807'),
                exact: true
              },
              {
                path: '/docs/deployment/database-setup',
                component: ComponentCreator('/docs/deployment/database-setup', '217'),
                exact: true
              },
              {
                path: '/docs/deployment/disaster-recovery',
                component: ComponentCreator('/docs/deployment/disaster-recovery', '112'),
                exact: true
              },
              {
                path: '/docs/deployment/docker',
                component: ComponentCreator('/docs/deployment/docker', '9a4'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/deployment/dr-drills',
                component: ComponentCreator('/docs/deployment/dr-drills', 'd5d'),
                exact: true
              },
              {
                path: '/docs/deployment/kubernetes',
                component: ComponentCreator('/docs/deployment/kubernetes', 'a94'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/deployment/observability',
                component: ComponentCreator('/docs/deployment/observability', 'b0c'),
                exact: true
              },
              {
                path: '/docs/deployment/observability-setup',
                component: ComponentCreator('/docs/deployment/observability-setup', '41e'),
                exact: true
              },
              {
                path: '/docs/deployment/overview',
                component: ComponentCreator('/docs/deployment/overview', '370'),
                exact: true
              },
              {
                path: '/docs/deployment/postgresql-migration',
                component: ComponentCreator('/docs/deployment/postgresql-migration', 'cb0'),
                exact: true
              },
              {
                path: '/docs/deployment/production-deployment',
                component: ComponentCreator('/docs/deployment/production-deployment', '25c'),
                exact: true
              },
              {
                path: '/docs/deployment/rate-limiting',
                component: ComponentCreator('/docs/deployment/rate-limiting', 'f65'),
                exact: true
              },
              {
                path: '/docs/deployment/redis',
                component: ComponentCreator('/docs/deployment/redis', '5f0'),
                exact: true
              },
              {
                path: '/docs/deployment/redis-ha',
                component: ComponentCreator('/docs/deployment/redis-ha', 'd7d'),
                exact: true
              },
              {
                path: '/docs/deployment/runbook',
                component: ComponentCreator('/docs/deployment/runbook', 'c8f'),
                exact: true
              },
              {
                path: '/docs/deployment/scaling',
                component: ComponentCreator('/docs/deployment/scaling', '68a'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/deployment/secrets-management',
                component: ComponentCreator('/docs/deployment/secrets-management', 'f91'),
                exact: true
              },
              {
                path: '/docs/deployment/secrets-migration',
                component: ComponentCreator('/docs/deployment/secrets-migration', '125'),
                exact: true
              },
              {
                path: '/docs/deployment/security',
                component: ComponentCreator('/docs/deployment/security', '5c9'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/deployment/streaming',
                component: ComponentCreator('/docs/deployment/streaming', 'a2f'),
                exact: true
              },
              {
                path: '/docs/deployment/tls',
                component: ComponentCreator('/docs/deployment/tls', '963'),
                exact: true
              },
              {
                path: '/docs/enterprise/',
                component: ComponentCreator('/docs/enterprise/', '583'),
                exact: true
              },
              {
                path: '/docs/enterprise/audit-logs',
                component: ComponentCreator('/docs/enterprise/audit-logs', 'b78'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/enterprise/billing',
                component: ComponentCreator('/docs/enterprise/billing', '1d6'),
                exact: true
              },
              {
                path: '/docs/enterprise/billing-units',
                component: ComponentCreator('/docs/enterprise/billing-units', '7a5'),
                exact: true
              },
              {
                path: '/docs/enterprise/commercial-overview',
                component: ComponentCreator('/docs/enterprise/commercial-overview', '184'),
                exact: true
              },
              {
                path: '/docs/enterprise/compliance',
                component: ComponentCreator('/docs/enterprise/compliance', '731'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/enterprise/control-plane',
                component: ComponentCreator('/docs/enterprise/control-plane', '8a3'),
                exact: true
              },
              {
                path: '/docs/enterprise/control-plane-feasibility',
                component: ComponentCreator('/docs/enterprise/control-plane-feasibility', '1ef'),
                exact: true
              },
              {
                path: '/docs/enterprise/control-plane-overview',
                component: ComponentCreator('/docs/enterprise/control-plane-overview', '2d6'),
                exact: true
              },
              {
                path: '/docs/enterprise/features',
                component: ComponentCreator('/docs/enterprise/features', '170'),
                exact: true
              },
              {
                path: '/docs/enterprise/governance',
                component: ComponentCreator('/docs/enterprise/governance', '987'),
                exact: true
              },
              {
                path: '/docs/enterprise/incident-response',
                component: ComponentCreator('/docs/enterprise/incident-response', '360'),
                exact: true
              },
              {
                path: '/docs/enterprise/multi-tenancy',
                component: ComponentCreator('/docs/enterprise/multi-tenancy', 'b8e'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/enterprise/positioning',
                component: ComponentCreator('/docs/enterprise/positioning', '6ee'),
                exact: true
              },
              {
                path: '/docs/enterprise/sla',
                component: ComponentCreator('/docs/enterprise/sla', 'f65'),
                exact: true
              },
              {
                path: '/docs/enterprise/sso',
                component: ComponentCreator('/docs/enterprise/sso', '9d0'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/enterprise/stripe-setup',
                component: ComponentCreator('/docs/enterprise/stripe-setup', '079'),
                exact: true
              },
              {
                path: '/docs/enterprise/support',
                component: ComponentCreator('/docs/enterprise/support', 'b2a'),
                exact: true
              },
              {
                path: '/docs/getting-started/',
                component: ComponentCreator('/docs/getting-started/', '7d4'),
                exact: true
              },
              {
                path: '/docs/getting-started/configuration',
                component: ComponentCreator('/docs/getting-started/configuration', 'cbc'),
                exact: true
              },
              {
                path: '/docs/getting-started/environment',
                component: ComponentCreator('/docs/getting-started/environment', 'fab'),
                exact: true
              },
              {
                path: '/docs/getting-started/first-debate',
                component: ComponentCreator('/docs/getting-started/first-debate', '53d'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/getting-started/installation',
                component: ComponentCreator('/docs/getting-started/installation', '40d'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/getting-started/introduction',
                component: ComponentCreator('/docs/getting-started/introduction', 'a43'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/getting-started/overview',
                component: ComponentCreator('/docs/getting-started/overview', '466'),
                exact: true
              },
              {
                path: '/docs/getting-started/quickstart',
                component: ComponentCreator('/docs/getting-started/quickstart', '5bc'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/guides/',
                component: ComponentCreator('/docs/guides/', '39b'),
                exact: true
              },
              {
                path: '/docs/guides/accounting',
                component: ComponentCreator('/docs/guides/accounting', 'c25'),
                exact: true
              },
              {
                path: '/docs/guides/api-quickstart',
                component: ComponentCreator('/docs/guides/api-quickstart', '186'),
                exact: true
              },
              {
                path: '/docs/guides/api-usage',
                component: ComponentCreator('/docs/guides/api-usage', 'd1e'),
                exact: true
              },
              {
                path: '/docs/guides/automation',
                component: ComponentCreator('/docs/guides/automation', '66a'),
                exact: true
              },
              {
                path: '/docs/guides/bot-integrations',
                component: ComponentCreator('/docs/guides/bot-integrations', '7c1'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/broadcast',
                component: ComponentCreator('/docs/guides/broadcast', '60e'),
                exact: true
              },
              {
                path: '/docs/guides/channels',
                component: ComponentCreator('/docs/guides/channels', '337'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/chat-connector',
                component: ComponentCreator('/docs/guides/chat-connector', 'ddd'),
                exact: true
              },
              {
                path: '/docs/guides/coding-assistance',
                component: ComponentCreator('/docs/guides/coding-assistance', 'ea6'),
                exact: true
              },
              {
                path: '/docs/guides/connector-troubleshooting',
                component: ComponentCreator('/docs/guides/connector-troubleshooting', '4ea'),
                exact: true
              },
              {
                path: '/docs/guides/connectors',
                component: ComponentCreator('/docs/guides/connectors', 'b8f'),
                exact: true
              },
              {
                path: '/docs/guides/connectors-setup',
                component: ComponentCreator('/docs/guides/connectors-setup', '8c8'),
                exact: true
              },
              {
                path: '/docs/guides/cost-visibility',
                component: ComponentCreator('/docs/guides/cost-visibility', 'b32'),
                exact: true
              },
              {
                path: '/docs/guides/custom-agents',
                component: ComponentCreator('/docs/guides/custom-agents', 'fc2'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/documents',
                component: ComponentCreator('/docs/guides/documents', 'fd7'),
                exact: true
              },
              {
                path: '/docs/guides/email-prioritization',
                component: ComponentCreator('/docs/guides/email-prioritization', '48a'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/evidence',
                component: ComponentCreator('/docs/guides/evidence', '846'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/features',
                component: ComponentCreator('/docs/guides/features', '620'),
                exact: true
              },
              {
                path: '/docs/guides/gauntlet',
                component: ComponentCreator('/docs/guides/gauntlet', '470'),
                exact: true
              },
              {
                path: '/docs/guides/gauntlet-architecture',
                component: ComponentCreator('/docs/guides/gauntlet-architecture', 'b2d'),
                exact: true
              },
              {
                path: '/docs/guides/graph-debates',
                component: ComponentCreator('/docs/guides/graph-debates', 'd98'),
                exact: true
              },
              {
                path: '/docs/guides/harnesses',
                component: ComponentCreator('/docs/guides/harnesses', 'e5a'),
                exact: true
              },
              {
                path: '/docs/guides/inbox-guide',
                component: ComponentCreator('/docs/guides/inbox-guide', '648'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/integrations',
                component: ComponentCreator('/docs/guides/integrations', '801'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/library-usage',
                component: ComponentCreator('/docs/guides/library-usage', 'b1a'),
                exact: true
              },
              {
                path: '/docs/guides/matrix-debates',
                component: ComponentCreator('/docs/guides/matrix-debates', '4dd'),
                exact: true
              },
              {
                path: '/docs/guides/mcp-advanced',
                component: ComponentCreator('/docs/guides/mcp-advanced', '5ff'),
                exact: true
              },
              {
                path: '/docs/guides/mcp-integration',
                component: ComponentCreator('/docs/guides/mcp-integration', 'ef8'),
                exact: true
              },
              {
                path: '/docs/guides/modes',
                component: ComponentCreator('/docs/guides/modes', '8cc'),
                exact: true
              },
              {
                path: '/docs/guides/modes-reference',
                component: ComponentCreator('/docs/guides/modes-reference', 'cca'),
                exact: true
              },
              {
                path: '/docs/guides/plugin-guide',
                component: ComponentCreator('/docs/guides/plugin-guide', 'c6f'),
                exact: true
              },
              {
                path: '/docs/guides/probe-strategies',
                component: ComponentCreator('/docs/guides/probe-strategies', 'bb2'),
                exact: true
              },
              {
                path: '/docs/guides/pulse',
                component: ComponentCreator('/docs/guides/pulse', '9b2'),
                exact: true
              },
              {
                path: '/docs/guides/queue',
                component: ComponentCreator('/docs/guides/queue', '058'),
                exact: true
              },
              {
                path: '/docs/guides/sdk',
                component: ComponentCreator('/docs/guides/sdk', '942'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/sdk-consolidation',
                component: ComponentCreator('/docs/guides/sdk-consolidation', '066'),
                exact: true
              },
              {
                path: '/docs/guides/sdk-parity',
                component: ComponentCreator('/docs/guides/sdk-parity', '3e2'),
                exact: true
              },
              {
                path: '/docs/guides/sdk-typescript',
                component: ComponentCreator('/docs/guides/sdk-typescript', '7aa'),
                exact: true
              },
              {
                path: '/docs/guides/shared-inbox',
                component: ComponentCreator('/docs/guides/shared-inbox', 'ade'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/templates',
                component: ComponentCreator('/docs/guides/templates', 'bc9'),
                exact: true
              },
              {
                path: '/docs/guides/user-onboarding',
                component: ComponentCreator('/docs/guides/user-onboarding', 'fc3'),
                exact: true
              },
              {
                path: '/docs/guides/verticals',
                component: ComponentCreator('/docs/guides/verticals', 'fba'),
                exact: true
              },
              {
                path: '/docs/guides/websocket-events',
                component: ComponentCreator('/docs/guides/websocket-events', 'd70'),
                exact: true
              },
              {
                path: '/docs/guides/workflows',
                component: ComponentCreator('/docs/guides/workflows', 'a24'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/operations/',
                component: ComponentCreator('/docs/operations/', '424'),
                exact: true
              },
              {
                path: '/docs/operations/alert-runbooks',
                component: ComponentCreator('/docs/operations/alert-runbooks', 'be6'),
                exact: true
              },
              {
                path: '/docs/operations/incident-communication',
                component: ComponentCreator('/docs/operations/incident-communication', '5d9'),
                exact: true
              },
              {
                path: '/docs/operations/incident-response',
                component: ComponentCreator('/docs/operations/incident-response', '3e6'),
                exact: true
              },
              {
                path: '/docs/operations/incident-response-playbooks',
                component: ComponentCreator('/docs/operations/incident-response-playbooks', 'bc5'),
                exact: true
              },
              {
                path: '/docs/operations/overview',
                component: ComponentCreator('/docs/operations/overview', 'c1d'),
                exact: true
              },
              {
                path: '/docs/operations/performance-targets',
                component: ComponentCreator('/docs/operations/performance-targets', 'dd0'),
                exact: true
              },
              {
                path: '/docs/operations/production-readiness',
                component: ComponentCreator('/docs/operations/production-readiness', 'd5e'),
                exact: true
              },
              {
                path: '/docs/operations/production-runbook',
                component: ComponentCreator('/docs/operations/production-runbook', '5ba'),
                exact: true
              },
              {
                path: '/docs/operations/runbook',
                component: ComponentCreator('/docs/operations/runbook', '176'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-database',
                component: ComponentCreator('/docs/operations/runbook-database', '4ce'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-deployment',
                component: ComponentCreator('/docs/operations/runbook-deployment', 'a05'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-incident',
                component: ComponentCreator('/docs/operations/runbook-incident', '849'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-metrics',
                component: ComponentCreator('/docs/operations/runbook-metrics', '8fe'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-provider',
                component: ComponentCreator('/docs/operations/runbook-provider', 'a36'),
                exact: true
              },
              {
                path: '/docs/operations/troubleshooting',
                component: ComponentCreator('/docs/operations/troubleshooting', '6dd'),
                exact: true
              },
              {
                path: '/docs/security/',
                component: ComponentCreator('/docs/security/', '52c'),
                exact: true
              },
              {
                path: '/docs/security/authentication',
                component: ComponentCreator('/docs/security/authentication', '3ed'),
                exact: true
              },
              {
                path: '/docs/security/breach-notification',
                component: ComponentCreator('/docs/security/breach-notification', '4e4'),
                exact: true
              },
              {
                path: '/docs/security/ci-cd',
                component: ComponentCreator('/docs/security/ci-cd', 'b43'),
                exact: true
              },
              {
                path: '/docs/security/compliance',
                component: ComponentCreator('/docs/security/compliance', '6f5'),
                exact: true
              },
              {
                path: '/docs/security/compliance-presets',
                component: ComponentCreator('/docs/security/compliance-presets', 'eea'),
                exact: true
              },
              {
                path: '/docs/security/data-classification',
                component: ComponentCreator('/docs/security/data-classification', '913'),
                exact: true
              },
              {
                path: '/docs/security/data-residency',
                component: ComponentCreator('/docs/security/data-residency', '1cc'),
                exact: true
              },
              {
                path: '/docs/security/dsar',
                component: ComponentCreator('/docs/security/dsar', '213'),
                exact: true
              },
              {
                path: '/docs/security/oauth-guide',
                component: ComponentCreator('/docs/security/oauth-guide', '1a7'),
                exact: true
              },
              {
                path: '/docs/security/oauth-setup',
                component: ComponentCreator('/docs/security/oauth-setup', 'd4b'),
                exact: true
              },
              {
                path: '/docs/security/overview',
                component: ComponentCreator('/docs/security/overview', 'e37'),
                exact: true
              },
              {
                path: '/docs/security/patterns',
                component: ComponentCreator('/docs/security/patterns', 'f96'),
                exact: true
              },
              {
                path: '/docs/security/privacy-policy',
                component: ComponentCreator('/docs/security/privacy-policy', '085'),
                exact: true
              },
              {
                path: '/docs/security/remote-work',
                component: ComponentCreator('/docs/security/remote-work', '958'),
                exact: true
              },
              {
                path: '/docs/security/runtime',
                component: ComponentCreator('/docs/security/runtime', '25c'),
                exact: true
              },
              {
                path: '/docs/security/session-management',
                component: ComponentCreator('/docs/security/session-management', '478'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2bc'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
