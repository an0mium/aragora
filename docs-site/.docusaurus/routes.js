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
    component: ComponentCreator('/docs', 'd2e'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '653'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '489'),
            routes: [
              {
                path: '/docs/admin/',
                component: ComponentCreator('/docs/admin/', '314'),
                exact: true
              },
              {
                path: '/docs/admin/ab-testing',
                component: ComponentCreator('/docs/admin/ab-testing', '803'),
                exact: true
              },
              {
                path: '/docs/admin/nomic-loop',
                component: ComponentCreator('/docs/admin/nomic-loop', '064'),
                exact: true
              },
              {
                path: '/docs/admin/overview',
                component: ComponentCreator('/docs/admin/overview', 'b3a'),
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
                component: ComponentCreator('/docs/advanced/cross-pollination', '9e1'),
                exact: true
              },
              {
                path: '/docs/advanced/evolution-patterns',
                component: ComponentCreator('/docs/advanced/evolution-patterns', 'd65'),
                exact: true
              },
              {
                path: '/docs/advanced/formal-verification',
                component: ComponentCreator('/docs/advanced/formal-verification', 'ff8'),
                exact: true
              },
              {
                path: '/docs/advanced/genesis',
                component: ComponentCreator('/docs/advanced/genesis', '005'),
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
                path: '/docs/analysis/benchmarks',
                component: ComponentCreator('/docs/analysis/benchmarks', '854'),
                exact: true
              },
              {
                path: '/docs/analysis/overview',
                component: ComponentCreator('/docs/analysis/overview', '3d8'),
                exact: true
              },
              {
                path: '/docs/api-reference/',
                component: ComponentCreator('/docs/api-reference/', '135'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/agents/list',
                component: ComponentCreator('/docs/api-reference/agents/list', '101'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/agents/overview',
                component: ComponentCreator('/docs/api-reference/agents/overview', '853'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/agents/stats',
                component: ComponentCreator('/docs/api-reference/agents/stats', 'ce0'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/consensus',
                component: ComponentCreator('/docs/api-reference/debates/consensus', 'b6b'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/create',
                component: ComponentCreator('/docs/api-reference/debates/create', '396'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/get',
                component: ComponentCreator('/docs/api-reference/debates/get', '1ee'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/list',
                component: ComponentCreator('/docs/api-reference/debates/list', '6ba'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/overview',
                component: ComponentCreator('/docs/api-reference/debates/overview', 'a95'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/knowledge/overview',
                component: ComponentCreator('/docs/api-reference/knowledge/overview', '615'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/knowledge/query',
                component: ComponentCreator('/docs/api-reference/knowledge/query', 'e67'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/knowledge/store',
                component: ComponentCreator('/docs/api-reference/knowledge/store', '3d3'),
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
                component: ComponentCreator('/docs/api-reference/workflows/overview', '621'),
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
                component: ComponentCreator('/docs/api/cli', 'd04'),
                exact: true
              },
              {
                path: '/docs/api/discovery',
                component: ComponentCreator('/docs/api/discovery', 'a23'),
                exact: true
              },
              {
                path: '/docs/api/endpoints',
                component: ComponentCreator('/docs/api/endpoints', '565'),
                exact: true
              },
              {
                path: '/docs/api/examples',
                component: ComponentCreator('/docs/api/examples', 'a0f'),
                exact: true
              },
              {
                path: '/docs/api/rate-limits',
                component: ComponentCreator('/docs/api/rate-limits', 'a86'),
                exact: true
              },
              {
                path: '/docs/api/reference',
                component: ComponentCreator('/docs/api/reference', '770'),
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
                path: '/docs/contributing/guide',
                component: ComponentCreator('/docs/contributing/guide', 'ae4'),
                exact: true
              },
              {
                path: '/docs/contributing/status',
                component: ComponentCreator('/docs/contributing/status', 'd87'),
                exact: true
              },
              {
                path: '/docs/core-concepts/',
                component: ComponentCreator('/docs/core-concepts/', 'f69'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agent-development',
                component: ComponentCreator('/docs/core-concepts/agent-development', '489'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agent-selection',
                component: ComponentCreator('/docs/core-concepts/agent-selection', 'ebb'),
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
                component: ComponentCreator('/docs/core-concepts/architecture', 'b11'),
                exact: true
              },
              {
                path: '/docs/core-concepts/consensus',
                component: ComponentCreator('/docs/core-concepts/consensus', '966'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/convergence-algorithm',
                component: ComponentCreator('/docs/core-concepts/convergence-algorithm', '789'),
                exact: true
              },
              {
                path: '/docs/core-concepts/debate-internals',
                component: ComponentCreator('/docs/core-concepts/debate-internals', '7a7'),
                exact: true
              },
              {
                path: '/docs/core-concepts/debates',
                component: ComponentCreator('/docs/core-concepts/debates', 'aa2'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/elo-calibration',
                component: ComponentCreator('/docs/core-concepts/elo-calibration', '822'),
                exact: true
              },
              {
                path: '/docs/core-concepts/knowledge-mound',
                component: ComponentCreator('/docs/core-concepts/knowledge-mound', '2b6'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/memory',
                component: ComponentCreator('/docs/core-concepts/memory', '134'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/memory-analytics',
                component: ComponentCreator('/docs/core-concepts/memory-analytics', '8df'),
                exact: true
              },
              {
                path: '/docs/core-concepts/memory-strategy',
                component: ComponentCreator('/docs/core-concepts/memory-strategy', '4eb'),
                exact: true
              },
              {
                path: '/docs/core-concepts/reasoning',
                component: ComponentCreator('/docs/core-concepts/reasoning', '044'),
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
                component: ComponentCreator('/docs/deployment/database', '932'),
                exact: true
              },
              {
                path: '/docs/deployment/database-consolidation',
                component: ComponentCreator('/docs/deployment/database-consolidation', 'c69'),
                exact: true
              },
              {
                path: '/docs/deployment/database-schema',
                component: ComponentCreator('/docs/deployment/database-schema', '393'),
                exact: true
              },
              {
                path: '/docs/deployment/database-setup',
                component: ComponentCreator('/docs/deployment/database-setup', '217'),
                exact: true
              },
              {
                path: '/docs/deployment/disaster-recovery',
                component: ComponentCreator('/docs/deployment/disaster-recovery', 'ef1'),
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
                component: ComponentCreator('/docs/deployment/observability', 'e01'),
                exact: true
              },
              {
                path: '/docs/deployment/observability-setup',
                component: ComponentCreator('/docs/deployment/observability-setup', '411'),
                exact: true
              },
              {
                path: '/docs/deployment/overview',
                component: ComponentCreator('/docs/deployment/overview', 'efa'),
                exact: true
              },
              {
                path: '/docs/deployment/postgresql-migration',
                component: ComponentCreator('/docs/deployment/postgresql-migration', '63b'),
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
                component: ComponentCreator('/docs/deployment/scaling', '941'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/deployment/secrets-management',
                component: ComponentCreator('/docs/deployment/secrets-management', 'c43'),
                exact: true
              },
              {
                path: '/docs/deployment/security',
                component: ComponentCreator('/docs/deployment/security', '5c9'),
                exact: true,
                sidebar: "guidesSidebar"
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
                component: ComponentCreator('/docs/enterprise/billing', '2ec'),
                exact: true
              },
              {
                path: '/docs/enterprise/billing-units',
                component: ComponentCreator('/docs/enterprise/billing-units', '7a5'),
                exact: true
              },
              {
                path: '/docs/enterprise/commercial-overview',
                component: ComponentCreator('/docs/enterprise/commercial-overview', '71e'),
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
                component: ComponentCreator('/docs/enterprise/control-plane', '2cf'),
                exact: true
              },
              {
                path: '/docs/enterprise/control-plane-feasibility',
                component: ComponentCreator('/docs/enterprise/control-plane-feasibility', '032'),
                exact: true
              },
              {
                path: '/docs/enterprise/features',
                component: ComponentCreator('/docs/enterprise/features', '091'),
                exact: true
              },
              {
                path: '/docs/enterprise/governance',
                component: ComponentCreator('/docs/enterprise/governance', '015'),
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
                component: ComponentCreator('/docs/enterprise/positioning', 'd4e'),
                exact: true
              },
              {
                path: '/docs/enterprise/sso',
                component: ComponentCreator('/docs/enterprise/sso', 'b7f'),
                exact: true,
                sidebar: "guidesSidebar"
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
                component: ComponentCreator('/docs/getting-started/environment', '855'),
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
                component: ComponentCreator('/docs/getting-started/introduction', 'ec1'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/getting-started/quickstart',
                component: ComponentCreator('/docs/getting-started/quickstart', '454'),
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
                component: ComponentCreator('/docs/guides/api-quickstart', 'd7b'),
                exact: true
              },
              {
                path: '/docs/guides/api-usage',
                component: ComponentCreator('/docs/guides/api-usage', '1b6'),
                exact: true
              },
              {
                path: '/docs/guides/automation',
                component: ComponentCreator('/docs/guides/automation', '2c0'),
                exact: true
              },
              {
                path: '/docs/guides/bot-integrations',
                component: ComponentCreator('/docs/guides/bot-integrations', 'd50'),
                exact: true
              },
              {
                path: '/docs/guides/broadcast',
                component: ComponentCreator('/docs/guides/broadcast', '60e'),
                exact: true
              },
              {
                path: '/docs/guides/chat-connector',
                component: ComponentCreator('/docs/guides/chat-connector', '9a5'),
                exact: true
              },
              {
                path: '/docs/guides/connector-troubleshooting',
                component: ComponentCreator('/docs/guides/connector-troubleshooting', 'f2b'),
                exact: true
              },
              {
                path: '/docs/guides/connectors',
                component: ComponentCreator('/docs/guides/connectors', '058'),
                exact: true
              },
              {
                path: '/docs/guides/connectors-setup',
                component: ComponentCreator('/docs/guides/connectors-setup', 'aab'),
                exact: true
              },
              {
                path: '/docs/guides/custom-agents',
                component: ComponentCreator('/docs/guides/custom-agents', '666'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/evidence',
                component: ComponentCreator('/docs/guides/evidence', 'd0c'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/features',
                component: ComponentCreator('/docs/guides/features', '07e'),
                exact: true
              },
              {
                path: '/docs/guides/gauntlet',
                component: ComponentCreator('/docs/guides/gauntlet', '24e'),
                exact: true
              },
              {
                path: '/docs/guides/graph-debates',
                component: ComponentCreator('/docs/guides/graph-debates', 'd98'),
                exact: true
              },
              {
                path: '/docs/guides/integrations',
                component: ComponentCreator('/docs/guides/integrations', '8fe'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/matrix-debates',
                component: ComponentCreator('/docs/guides/matrix-debates', '4dd'),
                exact: true
              },
              {
                path: '/docs/guides/mcp-advanced',
                component: ComponentCreator('/docs/guides/mcp-advanced', 'a64'),
                exact: true
              },
              {
                path: '/docs/guides/mcp-integration',
                component: ComponentCreator('/docs/guides/mcp-integration', '9ec'),
                exact: true
              },
              {
                path: '/docs/guides/pulse',
                component: ComponentCreator('/docs/guides/pulse', '3ed'),
                exact: true
              },
              {
                path: '/docs/guides/queue',
                component: ComponentCreator('/docs/guides/queue', '2c3'),
                exact: true
              },
              {
                path: '/docs/guides/sdk',
                component: ComponentCreator('/docs/guides/sdk', '89f'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/verticals',
                component: ComponentCreator('/docs/guides/verticals', '497'),
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
                path: '/docs/operations/overview',
                component: ComponentCreator('/docs/operations/overview', '899'),
                exact: true
              },
              {
                path: '/docs/operations/performance-targets',
                component: ComponentCreator('/docs/operations/performance-targets', '4a8'),
                exact: true
              },
              {
                path: '/docs/operations/production-readiness',
                component: ComponentCreator('/docs/operations/production-readiness', 'd17'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-database',
                component: ComponentCreator('/docs/operations/runbook-database', '964'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-deployment',
                component: ComponentCreator('/docs/operations/runbook-deployment', '1eb'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-incident',
                component: ComponentCreator('/docs/operations/runbook-incident', '0ae'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-provider',
                component: ComponentCreator('/docs/operations/runbook-provider', 'a36'),
                exact: true
              },
              {
                path: '/docs/operations/troubleshooting',
                component: ComponentCreator('/docs/operations/troubleshooting', '851'),
                exact: true
              },
              {
                path: '/docs/security/',
                component: ComponentCreator('/docs/security/', '52c'),
                exact: true
              },
              {
                path: '/docs/security/authentication',
                component: ComponentCreator('/docs/security/authentication', 'f91'),
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
                component: ComponentCreator('/docs/security/compliance', '6ac'),
                exact: true
              },
              {
                path: '/docs/security/data-classification',
                component: ComponentCreator('/docs/security/data-classification', '913'),
                exact: true
              },
              {
                path: '/docs/security/data-residency',
                component: ComponentCreator('/docs/security/data-residency', '0e9'),
                exact: true
              },
              {
                path: '/docs/security/dsar',
                component: ComponentCreator('/docs/security/dsar', '213'),
                exact: true
              },
              {
                path: '/docs/security/overview',
                component: ComponentCreator('/docs/security/overview', '3e7'),
                exact: true
              },
              {
                path: '/docs/security/privacy-policy',
                component: ComponentCreator('/docs/security/privacy-policy', '724'),
                exact: true
              },
              {
                path: '/docs/security/remote-work',
                component: ComponentCreator('/docs/security/remote-work', 'fff'),
                exact: true
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
