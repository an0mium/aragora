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
    component: ComponentCreator('/docs', '1e7'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', 'ba2'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '2f1'),
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
                component: ComponentCreator('/docs/admin/nomic-loop', 'ee6'),
                exact: true
              },
              {
                path: '/docs/admin/overview',
                component: ComponentCreator('/docs/admin/overview', 'fe3'),
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
                component: ComponentCreator('/docs/advanced/cross-pollination', '041'),
                exact: true
              },
              {
                path: '/docs/advanced/evolution-patterns',
                component: ComponentCreator('/docs/advanced/evolution-patterns', 'c58'),
                exact: true
              },
              {
                path: '/docs/advanced/formal-verification',
                component: ComponentCreator('/docs/advanced/formal-verification', '06d'),
                exact: true
              },
              {
                path: '/docs/advanced/genesis',
                component: ComponentCreator('/docs/advanced/genesis', '3a6'),
                exact: true
              },
              {
                path: '/docs/advanced/rlm',
                component: ComponentCreator('/docs/advanced/rlm', '646'),
                exact: true
              },
              {
                path: '/docs/advanced/rlm-developer',
                component: ComponentCreator('/docs/advanced/rlm-developer', '8af'),
                exact: true
              },
              {
                path: '/docs/advanced/rlm-integration',
                component: ComponentCreator('/docs/advanced/rlm-integration', 'dec'),
                exact: true
              },
              {
                path: '/docs/advanced/rlm-user',
                component: ComponentCreator('/docs/advanced/rlm-user', '781'),
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
                component: ComponentCreator('/docs/analysis/adr/', '408'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/001-phase-based-debate-execution',
                component: ComponentCreator('/docs/analysis/adr/001-phase-based-debate-execution', 'bcc'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/002-agent-fallback-openrouter',
                component: ComponentCreator('/docs/analysis/adr/002-agent-fallback-openrouter', '69a'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/003-multi-tier-memory-system',
                component: ComponentCreator('/docs/analysis/adr/003-multi-tier-memory-system', '631'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/004-incremental-type-safety',
                component: ComponentCreator('/docs/analysis/adr/004-incremental-type-safety', '618'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/005-composition-over-inheritance',
                component: ComponentCreator('/docs/analysis/adr/005-composition-over-inheritance', '10d'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/006-api-versioning-strategy',
                component: ComponentCreator('/docs/analysis/adr/006-api-versioning-strategy', '807'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/007-selection-plugin-architecture',
                component: ComponentCreator('/docs/analysis/adr/007-selection-plugin-architecture', '317'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/008-rlm-semantic-compression',
                component: ComponentCreator('/docs/analysis/adr/008-rlm-semantic-compression', '386'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/009-control-plane-architecture',
                component: ComponentCreator('/docs/analysis/adr/009-control-plane-architecture', '241'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/010-debate-orchestration-pattern',
                component: ComponentCreator('/docs/analysis/adr/010-debate-orchestration-pattern', 'fbc'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/011-multi-tier-memory-comparison',
                component: ComponentCreator('/docs/analysis/adr/011-multi-tier-memory-comparison', '40e'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/012-agent-fallback-strategy',
                component: ComponentCreator('/docs/analysis/adr/012-agent-fallback-strategy', 'f20'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/013-workflow-dag-design',
                component: ComponentCreator('/docs/analysis/adr/013-workflow-dag-design', '0de'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/014-knowledge-mound-architecture',
                component: ComponentCreator('/docs/analysis/adr/014-knowledge-mound-architecture', '2bc'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/015-lazy-import-patterns',
                component: ComponentCreator('/docs/analysis/adr/015-lazy-import-patterns', 'c07'),
                exact: true
              },
              {
                path: '/docs/analysis/adr/016-marketplace-architecture',
                component: ComponentCreator('/docs/analysis/adr/016-marketplace-architecture', '402'),
                exact: true
              },
              {
                path: '/docs/analysis/benchmarks',
                component: ComponentCreator('/docs/analysis/benchmarks', '78f'),
                exact: true
              },
              {
                path: '/docs/analysis/case-studies/',
                component: ComponentCreator('/docs/analysis/case-studies/', 'a3d'),
                exact: true
              },
              {
                path: '/docs/analysis/case-studies/architecture-stress-test',
                component: ComponentCreator('/docs/analysis/case-studies/architecture-stress-test', '15b'),
                exact: true
              },
              {
                path: '/docs/analysis/case-studies/epic-strategic-debate',
                component: ComponentCreator('/docs/analysis/case-studies/epic-strategic-debate', '395'),
                exact: true
              },
              {
                path: '/docs/analysis/case-studies/gdpr-compliance-audit',
                component: ComponentCreator('/docs/analysis/case-studies/gdpr-compliance-audit', 'df2'),
                exact: true
              },
              {
                path: '/docs/analysis/case-studies/security-api-review',
                component: ComponentCreator('/docs/analysis/case-studies/security-api-review', 'db8'),
                exact: true
              },
              {
                path: '/docs/analysis/codebase',
                component: ComponentCreator('/docs/analysis/codebase', '150'),
                exact: true
              },
              {
                path: '/docs/analysis/overview',
                component: ComponentCreator('/docs/analysis/overview', '99e'),
                exact: true
              },
              {
                path: '/docs/api-reference/',
                component: ComponentCreator('/docs/api-reference/', '106'),
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
                component: ComponentCreator('/docs/api/cli', '290'),
                exact: true
              },
              {
                path: '/docs/api/discovery',
                component: ComponentCreator('/docs/api/discovery', 'd58'),
                exact: true
              },
              {
                path: '/docs/api/endpoints',
                component: ComponentCreator('/docs/api/endpoints', '2df'),
                exact: true
              },
              {
                path: '/docs/api/evidence',
                component: ComponentCreator('/docs/api/evidence', 'a70'),
                exact: true
              },
              {
                path: '/docs/api/examples',
                component: ComponentCreator('/docs/api/examples', '8f3'),
                exact: true
              },
              {
                path: '/docs/api/github-pr-review',
                component: ComponentCreator('/docs/api/github-pr-review', 'aa2'),
                exact: true
              },
              {
                path: '/docs/api/rate-limits',
                component: ComponentCreator('/docs/api/rate-limits', 'fbd'),
                exact: true
              },
              {
                path: '/docs/api/reference',
                component: ComponentCreator('/docs/api/reference', '489'),
                exact: true
              },
              {
                path: '/docs/api/stability',
                component: ComponentCreator('/docs/api/stability', 'e7e'),
                exact: true
              },
              {
                path: '/docs/api/versioning',
                component: ComponentCreator('/docs/api/versioning', '2da'),
                exact: true
              },
              {
                path: '/docs/api/webhooks',
                component: ComponentCreator('/docs/api/webhooks', '9bf'),
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
                component: ComponentCreator('/docs/contributing/documentation-index', '3ad'),
                exact: true
              },
              {
                path: '/docs/contributing/documentation-map',
                component: ComponentCreator('/docs/contributing/documentation-map', '940'),
                exact: true
              },
              {
                path: '/docs/contributing/first-contribution',
                component: ComponentCreator('/docs/contributing/first-contribution', '481'),
                exact: true
              },
              {
                path: '/docs/contributing/frontend-development',
                component: ComponentCreator('/docs/contributing/frontend-development', '45d'),
                exact: true
              },
              {
                path: '/docs/contributing/frontend-routes',
                component: ComponentCreator('/docs/contributing/frontend-routes', '07f'),
                exact: true
              },
              {
                path: '/docs/contributing/guide',
                component: ComponentCreator('/docs/contributing/guide', '24c'),
                exact: true
              },
              {
                path: '/docs/contributing/handler-development',
                component: ComponentCreator('/docs/contributing/handler-development', '0e6'),
                exact: true
              },
              {
                path: '/docs/contributing/handlers',
                component: ComponentCreator('/docs/contributing/handlers', '97e'),
                exact: true
              },
              {
                path: '/docs/contributing/status',
                component: ComponentCreator('/docs/contributing/status', '3ef'),
                exact: true
              },
              {
                path: '/docs/contributing/testing',
                component: ComponentCreator('/docs/contributing/testing', '186'),
                exact: true
              },
              {
                path: '/docs/core-concepts/',
                component: ComponentCreator('/docs/core-concepts/', 'f69'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agent-catalog',
                component: ComponentCreator('/docs/core-concepts/agent-catalog', '2de'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agent-development',
                component: ComponentCreator('/docs/core-concepts/agent-development', '3be'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agent-selection',
                component: ComponentCreator('/docs/core-concepts/agent-selection', '777'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agents',
                component: ComponentCreator('/docs/core-concepts/agents', '069'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/architecture',
                component: ComponentCreator('/docs/core-concepts/architecture', '846'),
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
                component: ComponentCreator('/docs/core-concepts/knowledge-mound', '97f'),
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
                component: ComponentCreator('/docs/core-concepts/memory-overview', '550'),
                exact: true
              },
              {
                path: '/docs/core-concepts/memory-strategy',
                component: ComponentCreator('/docs/core-concepts/memory-strategy', '4dc'),
                exact: true
              },
              {
                path: '/docs/core-concepts/reasoning',
                component: ComponentCreator('/docs/core-concepts/reasoning', '6a3'),
                exact: true
              },
              {
                path: '/docs/core-concepts/workflow-engine',
                component: ComponentCreator('/docs/core-concepts/workflow-engine', '707'),
                exact: true
              },
              {
                path: '/docs/deployment/',
                component: ComponentCreator('/docs/deployment/', '9db'),
                exact: true
              },
              {
                path: '/docs/deployment/capacity-planning',
                component: ComponentCreator('/docs/deployment/capacity-planning', '40f'),
                exact: true
              },
              {
                path: '/docs/deployment/database',
                component: ComponentCreator('/docs/deployment/database', 'ce4'),
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
                component: ComponentCreator('/docs/deployment/disaster-recovery', 'c67'),
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
                component: ComponentCreator('/docs/deployment/kubernetes', '003'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/deployment/observability',
                component: ComponentCreator('/docs/deployment/observability', '684'),
                exact: true
              },
              {
                path: '/docs/deployment/observability-setup',
                component: ComponentCreator('/docs/deployment/observability-setup', '23e'),
                exact: true
              },
              {
                path: '/docs/deployment/overview',
                component: ComponentCreator('/docs/deployment/overview', '370'),
                exact: true
              },
              {
                path: '/docs/deployment/postgresql-migration',
                component: ComponentCreator('/docs/deployment/postgresql-migration', 'd3c'),
                exact: true
              },
              {
                path: '/docs/deployment/production-deployment',
                component: ComponentCreator('/docs/deployment/production-deployment', 'da1'),
                exact: true
              },
              {
                path: '/docs/deployment/rate-limiting',
                component: ComponentCreator('/docs/deployment/rate-limiting', 'e13'),
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
                component: ComponentCreator('/docs/deployment/runbook', '94b'),
                exact: true
              },
              {
                path: '/docs/deployment/scaling',
                component: ComponentCreator('/docs/deployment/scaling', '622'),
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
                component: ComponentCreator('/docs/deployment/secrets-migration', '4c2'),
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
                component: ComponentCreator('/docs/deployment/streaming', 'e34'),
                exact: true
              },
              {
                path: '/docs/deployment/tls',
                component: ComponentCreator('/docs/deployment/tls', '60d'),
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
                component: ComponentCreator('/docs/enterprise/commercial-overview', '490'),
                exact: true
              },
              {
                path: '/docs/enterprise/compliance',
                component: ComponentCreator('/docs/enterprise/compliance', 'f75'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/enterprise/control-plane',
                component: ComponentCreator('/docs/enterprise/control-plane', '0dd'),
                exact: true
              },
              {
                path: '/docs/enterprise/control-plane-feasibility',
                component: ComponentCreator('/docs/enterprise/control-plane-feasibility', '1ef'),
                exact: true
              },
              {
                path: '/docs/enterprise/control-plane-overview',
                component: ComponentCreator('/docs/enterprise/control-plane-overview', '622'),
                exact: true
              },
              {
                path: '/docs/enterprise/features',
                component: ComponentCreator('/docs/enterprise/features', 'b0d'),
                exact: true
              },
              {
                path: '/docs/enterprise/governance',
                component: ComponentCreator('/docs/enterprise/governance', 'eb7'),
                exact: true
              },
              {
                path: '/docs/enterprise/incident-response',
                component: ComponentCreator('/docs/enterprise/incident-response', '20a'),
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
                component: ComponentCreator('/docs/enterprise/positioning', '312'),
                exact: true
              },
              {
                path: '/docs/enterprise/sla',
                component: ComponentCreator('/docs/enterprise/sla', '443'),
                exact: true
              },
              {
                path: '/docs/enterprise/sso',
                component: ComponentCreator('/docs/enterprise/sso', 'b7f'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/enterprise/stripe-setup',
                component: ComponentCreator('/docs/enterprise/stripe-setup', 'd0d'),
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
                component: ComponentCreator('/docs/getting-started/configuration', 'e73'),
                exact: true
              },
              {
                path: '/docs/getting-started/environment',
                component: ComponentCreator('/docs/getting-started/environment', '52b'),
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
                component: ComponentCreator('/docs/getting-started/introduction', 'b8b'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/getting-started/overview',
                component: ComponentCreator('/docs/getting-started/overview', '5e0'),
                exact: true
              },
              {
                path: '/docs/getting-started/quickstart',
                component: ComponentCreator('/docs/getting-started/quickstart', '1c2'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/guides/',
                component: ComponentCreator('/docs/guides/', '39b'),
                exact: true
              },
              {
                path: '/docs/guides/api-quickstart',
                component: ComponentCreator('/docs/guides/api-quickstart', '186'),
                exact: true
              },
              {
                path: '/docs/guides/api-usage',
                component: ComponentCreator('/docs/guides/api-usage', 'e94'),
                exact: true
              },
              {
                path: '/docs/guides/automation',
                component: ComponentCreator('/docs/guides/automation', '4b2'),
                exact: true
              },
              {
                path: '/docs/guides/bot-integrations',
                component: ComponentCreator('/docs/guides/bot-integrations', 'd14'),
                exact: true
              },
              {
                path: '/docs/guides/broadcast',
                component: ComponentCreator('/docs/guides/broadcast', '60e'),
                exact: true
              },
              {
                path: '/docs/guides/channels',
                component: ComponentCreator('/docs/guides/channels', 'e84'),
                exact: true
              },
              {
                path: '/docs/guides/chat-connector',
                component: ComponentCreator('/docs/guides/chat-connector', 'd1a'),
                exact: true
              },
              {
                path: '/docs/guides/coding-assistance',
                component: ComponentCreator('/docs/guides/coding-assistance', '748'),
                exact: true
              },
              {
                path: '/docs/guides/connector-troubleshooting',
                component: ComponentCreator('/docs/guides/connector-troubleshooting', '6a2'),
                exact: true
              },
              {
                path: '/docs/guides/connectors',
                component: ComponentCreator('/docs/guides/connectors', 'e96'),
                exact: true
              },
              {
                path: '/docs/guides/connectors-setup',
                component: ComponentCreator('/docs/guides/connectors-setup', '637'),
                exact: true
              },
              {
                path: '/docs/guides/cost-visibility',
                component: ComponentCreator('/docs/guides/cost-visibility', 'b32'),
                exact: true
              },
              {
                path: '/docs/guides/custom-agents',
                component: ComponentCreator('/docs/guides/custom-agents', '666'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/documents',
                component: ComponentCreator('/docs/guides/documents', '65e'),
                exact: true
              },
              {
                path: '/docs/guides/email-prioritization',
                component: ComponentCreator('/docs/guides/email-prioritization', '4a9'),
                exact: true
              },
              {
                path: '/docs/guides/evidence',
                component: ComponentCreator('/docs/guides/evidence', 'a5c'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/features',
                component: ComponentCreator('/docs/guides/features', '0c9'),
                exact: true
              },
              {
                path: '/docs/guides/gauntlet',
                component: ComponentCreator('/docs/guides/gauntlet', '504'),
                exact: true
              },
              {
                path: '/docs/guides/gauntlet-architecture',
                component: ComponentCreator('/docs/guides/gauntlet-architecture', '147'),
                exact: true
              },
              {
                path: '/docs/guides/graph-debates',
                component: ComponentCreator('/docs/guides/graph-debates', 'd98'),
                exact: true
              },
              {
                path: '/docs/guides/harnesses',
                component: ComponentCreator('/docs/guides/harnesses', 'cb5'),
                exact: true
              },
              {
                path: '/docs/guides/integrations',
                component: ComponentCreator('/docs/guides/integrations', '801'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/library-usage',
                component: ComponentCreator('/docs/guides/library-usage', '2a1'),
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
                component: ComponentCreator('/docs/guides/modes', 'ff1'),
                exact: true
              },
              {
                path: '/docs/guides/modes-reference',
                component: ComponentCreator('/docs/guides/modes-reference', '436'),
                exact: true
              },
              {
                path: '/docs/guides/plugin-guide',
                component: ComponentCreator('/docs/guides/plugin-guide', 'e2d'),
                exact: true
              },
              {
                path: '/docs/guides/probe-strategies',
                component: ComponentCreator('/docs/guides/probe-strategies', 'e3e'),
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
                path: '/docs/guides/sdk-typescript',
                component: ComponentCreator('/docs/guides/sdk-typescript', '476'),
                exact: true
              },
              {
                path: '/docs/guides/shared-inbox',
                component: ComponentCreator('/docs/guides/shared-inbox', '113'),
                exact: true
              },
              {
                path: '/docs/guides/user-onboarding',
                component: ComponentCreator('/docs/guides/user-onboarding', 'f66'),
                exact: true
              },
              {
                path: '/docs/guides/verticals',
                component: ComponentCreator('/docs/guides/verticals', '5e7'),
                exact: true
              },
              {
                path: '/docs/guides/websocket-events',
                component: ComponentCreator('/docs/guides/websocket-events', 'a95'),
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
                component: ComponentCreator('/docs/operations/alert-runbooks', '1f4'),
                exact: true
              },
              {
                path: '/docs/operations/incident-communication',
                component: ComponentCreator('/docs/operations/incident-communication', '088'),
                exact: true
              },
              {
                path: '/docs/operations/incident-response',
                component: ComponentCreator('/docs/operations/incident-response', '563'),
                exact: true
              },
              {
                path: '/docs/operations/incident-response-playbooks',
                component: ComponentCreator('/docs/operations/incident-response-playbooks', '4bc'),
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
                component: ComponentCreator('/docs/operations/production-readiness', 'c2d'),
                exact: true
              },
              {
                path: '/docs/operations/production-runbook',
                component: ComponentCreator('/docs/operations/production-runbook', '47e'),
                exact: true
              },
              {
                path: '/docs/operations/runbook',
                component: ComponentCreator('/docs/operations/runbook', '915'),
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
                component: ComponentCreator('/docs/operations/runbook-metrics', 'd91'),
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
                component: ComponentCreator('/docs/security/authentication', '231'),
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
                component: ComponentCreator('/docs/security/compliance-presets', 'b7c'),
                exact: true
              },
              {
                path: '/docs/security/data-classification',
                component: ComponentCreator('/docs/security/data-classification', '913'),
                exact: true
              },
              {
                path: '/docs/security/data-residency',
                component: ComponentCreator('/docs/security/data-residency', '6ba'),
                exact: true
              },
              {
                path: '/docs/security/dsar',
                component: ComponentCreator('/docs/security/dsar', '213'),
                exact: true
              },
              {
                path: '/docs/security/oauth-guide',
                component: ComponentCreator('/docs/security/oauth-guide', '6c7'),
                exact: true
              },
              {
                path: '/docs/security/oauth-setup',
                component: ComponentCreator('/docs/security/oauth-setup', 'f08'),
                exact: true
              },
              {
                path: '/docs/security/overview',
                component: ComponentCreator('/docs/security/overview', '771'),
                exact: true
              },
              {
                path: '/docs/security/patterns',
                component: ComponentCreator('/docs/security/patterns', 'f88'),
                exact: true
              },
              {
                path: '/docs/security/privacy-policy',
                component: ComponentCreator('/docs/security/privacy-policy', 'b4c'),
                exact: true
              },
              {
                path: '/docs/security/remote-work',
                component: ComponentCreator('/docs/security/remote-work', '695'),
                exact: true
              },
              {
                path: '/docs/security/runtime',
                component: ComponentCreator('/docs/security/runtime', 'c7b'),
                exact: true
              },
              {
                path: '/docs/security/session-management',
                component: ComponentCreator('/docs/security/session-management', '75d'),
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
