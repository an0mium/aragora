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
    component: ComponentCreator('/docs', 'a3d'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '537'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '173'),
            routes: [
              {
                path: '/docs/admin/',
                component: ComponentCreator('/docs/admin/', 'e56'),
                exact: true
              },
              {
                path: '/docs/admin/ab-testing',
                component: ComponentCreator('/docs/admin/ab-testing', '29b'),
                exact: true
              },
              {
                path: '/docs/admin/nomic-loop',
                component: ComponentCreator('/docs/admin/nomic-loop', 'f3c'),
                exact: true
              },
              {
                path: '/docs/admin/overview',
                component: ComponentCreator('/docs/admin/overview', 'd61'),
                exact: true
              },
              {
                path: '/docs/advanced/',
                component: ComponentCreator('/docs/advanced/', '1a5'),
                exact: true
              },
              {
                path: '/docs/advanced/cross-functional',
                component: ComponentCreator('/docs/advanced/cross-functional', '1a3'),
                exact: true
              },
              {
                path: '/docs/advanced/cross-pollination',
                component: ComponentCreator('/docs/advanced/cross-pollination', 'c3c'),
                exact: true
              },
              {
                path: '/docs/advanced/formal-verification',
                component: ComponentCreator('/docs/advanced/formal-verification', 'ae2'),
                exact: true
              },
              {
                path: '/docs/advanced/trickster',
                component: ComponentCreator('/docs/advanced/trickster', 'f04'),
                exact: true
              },
              {
                path: '/docs/analysis/',
                component: ComponentCreator('/docs/analysis/', '373'),
                exact: true
              },
              {
                path: '/docs/analysis/benchmarks',
                component: ComponentCreator('/docs/analysis/benchmarks', '198'),
                exact: true
              },
              {
                path: '/docs/analysis/overview',
                component: ComponentCreator('/docs/analysis/overview', '79c'),
                exact: true
              },
              {
                path: '/docs/api-reference/',
                component: ComponentCreator('/docs/api-reference/', 'b91'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/agents/list',
                component: ComponentCreator('/docs/api-reference/agents/list', '07a'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/agents/overview',
                component: ComponentCreator('/docs/api-reference/agents/overview', '743'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/agents/stats',
                component: ComponentCreator('/docs/api-reference/agents/stats', 'c8d'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/consensus',
                component: ComponentCreator('/docs/api-reference/debates/consensus', 'd6c'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/create',
                component: ComponentCreator('/docs/api-reference/debates/create', '85a'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/get',
                component: ComponentCreator('/docs/api-reference/debates/get', '3de'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/list',
                component: ComponentCreator('/docs/api-reference/debates/list', '8f9'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/debates/overview',
                component: ComponentCreator('/docs/api-reference/debates/overview', 'a13'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/knowledge/overview',
                component: ComponentCreator('/docs/api-reference/knowledge/overview', 'bc5'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/knowledge/query',
                component: ComponentCreator('/docs/api-reference/knowledge/query', '0ce'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/knowledge/store',
                component: ComponentCreator('/docs/api-reference/knowledge/store', '197'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/workflows/execute',
                component: ComponentCreator('/docs/api-reference/workflows/execute', '37f'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api-reference/workflows/overview',
                component: ComponentCreator('/docs/api-reference/workflows/overview', '743'),
                exact: true,
                sidebar: "apiSidebar"
              },
              {
                path: '/docs/api/',
                component: ComponentCreator('/docs/api/', 'eb9'),
                exact: true
              },
              {
                path: '/docs/api/cli',
                component: ComponentCreator('/docs/api/cli', 'a62'),
                exact: true
              },
              {
                path: '/docs/api/discovery',
                component: ComponentCreator('/docs/api/discovery', '63d'),
                exact: true
              },
              {
                path: '/docs/api/endpoints',
                component: ComponentCreator('/docs/api/endpoints', '88b'),
                exact: true
              },
              {
                path: '/docs/api/examples',
                component: ComponentCreator('/docs/api/examples', '971'),
                exact: true
              },
              {
                path: '/docs/api/rate-limits',
                component: ComponentCreator('/docs/api/rate-limits', '554'),
                exact: true
              },
              {
                path: '/docs/api/reference',
                component: ComponentCreator('/docs/api/reference', 'dee'),
                exact: true
              },
              {
                path: '/docs/api/stability',
                component: ComponentCreator('/docs/api/stability', 'ba6'),
                exact: true
              },
              {
                path: '/docs/api/versioning',
                component: ComponentCreator('/docs/api/versioning', '82c'),
                exact: true
              },
              {
                path: '/docs/contributing/',
                component: ComponentCreator('/docs/contributing/', 'e0c'),
                exact: true
              },
              {
                path: '/docs/contributing/dependencies',
                component: ComponentCreator('/docs/contributing/dependencies', '2da'),
                exact: true
              },
              {
                path: '/docs/contributing/deprecation',
                component: ComponentCreator('/docs/contributing/deprecation', '767'),
                exact: true
              },
              {
                path: '/docs/contributing/guide',
                component: ComponentCreator('/docs/contributing/guide', 'a01'),
                exact: true
              },
              {
                path: '/docs/contributing/status',
                component: ComponentCreator('/docs/contributing/status', 'e09'),
                exact: true
              },
              {
                path: '/docs/core-concepts/',
                component: ComponentCreator('/docs/core-concepts/', '61e'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agent-development',
                component: ComponentCreator('/docs/core-concepts/agent-development', 'eea'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agent-selection',
                component: ComponentCreator('/docs/core-concepts/agent-selection', 'd3b'),
                exact: true
              },
              {
                path: '/docs/core-concepts/agents',
                component: ComponentCreator('/docs/core-concepts/agents', '51b'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/architecture',
                component: ComponentCreator('/docs/core-concepts/architecture', 'ec8'),
                exact: true
              },
              {
                path: '/docs/core-concepts/consensus',
                component: ComponentCreator('/docs/core-concepts/consensus', '55a'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/debate-internals',
                component: ComponentCreator('/docs/core-concepts/debate-internals', '3a2'),
                exact: true
              },
              {
                path: '/docs/core-concepts/debates',
                component: ComponentCreator('/docs/core-concepts/debates', '991'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/knowledge-mound',
                component: ComponentCreator('/docs/core-concepts/knowledge-mound', 'e9f'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/memory',
                component: ComponentCreator('/docs/core-concepts/memory', 'ac0'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/core-concepts/memory-strategy',
                component: ComponentCreator('/docs/core-concepts/memory-strategy', '262'),
                exact: true
              },
              {
                path: '/docs/core-concepts/reasoning',
                component: ComponentCreator('/docs/core-concepts/reasoning', '3d3'),
                exact: true
              },
              {
                path: '/docs/deployment/',
                component: ComponentCreator('/docs/deployment/', 'cf8'),
                exact: true
              },
              {
                path: '/docs/deployment/capacity-planning',
                component: ComponentCreator('/docs/deployment/capacity-planning', 'b8c'),
                exact: true
              },
              {
                path: '/docs/deployment/database',
                component: ComponentCreator('/docs/deployment/database', '866'),
                exact: true
              },
              {
                path: '/docs/deployment/database-consolidation',
                component: ComponentCreator('/docs/deployment/database-consolidation', '7af'),
                exact: true
              },
              {
                path: '/docs/deployment/database-schema',
                component: ComponentCreator('/docs/deployment/database-schema', '36e'),
                exact: true
              },
              {
                path: '/docs/deployment/database-setup',
                component: ComponentCreator('/docs/deployment/database-setup', 'ca0'),
                exact: true
              },
              {
                path: '/docs/deployment/disaster-recovery',
                component: ComponentCreator('/docs/deployment/disaster-recovery', '38f'),
                exact: true
              },
              {
                path: '/docs/deployment/docker',
                component: ComponentCreator('/docs/deployment/docker', '7ce'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/deployment/dr-drills',
                component: ComponentCreator('/docs/deployment/dr-drills', 'a04'),
                exact: true
              },
              {
                path: '/docs/deployment/kubernetes',
                component: ComponentCreator('/docs/deployment/kubernetes', '3a7'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/deployment/observability',
                component: ComponentCreator('/docs/deployment/observability', '45f'),
                exact: true
              },
              {
                path: '/docs/deployment/overview',
                component: ComponentCreator('/docs/deployment/overview', '669'),
                exact: true
              },
              {
                path: '/docs/deployment/redis',
                component: ComponentCreator('/docs/deployment/redis', 'e05'),
                exact: true
              },
              {
                path: '/docs/deployment/redis-ha',
                component: ComponentCreator('/docs/deployment/redis-ha', '5dc'),
                exact: true
              },
              {
                path: '/docs/deployment/runbook',
                component: ComponentCreator('/docs/deployment/runbook', '75d'),
                exact: true
              },
              {
                path: '/docs/deployment/scaling',
                component: ComponentCreator('/docs/deployment/scaling', '9ef'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/deployment/security',
                component: ComponentCreator('/docs/deployment/security', '8ad'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/enterprise/',
                component: ComponentCreator('/docs/enterprise/', 'de8'),
                exact: true
              },
              {
                path: '/docs/enterprise/audit-logs',
                component: ComponentCreator('/docs/enterprise/audit-logs', '504'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/enterprise/billing',
                component: ComponentCreator('/docs/enterprise/billing', 'd47'),
                exact: true
              },
              {
                path: '/docs/enterprise/billing-units',
                component: ComponentCreator('/docs/enterprise/billing-units', 'd96'),
                exact: true
              },
              {
                path: '/docs/enterprise/commercial-overview',
                component: ComponentCreator('/docs/enterprise/commercial-overview', 'dbe'),
                exact: true
              },
              {
                path: '/docs/enterprise/compliance',
                component: ComponentCreator('/docs/enterprise/compliance', '6a6'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/enterprise/control-plane',
                component: ComponentCreator('/docs/enterprise/control-plane', '51f'),
                exact: true
              },
              {
                path: '/docs/enterprise/control-plane-feasibility',
                component: ComponentCreator('/docs/enterprise/control-plane-feasibility', 'f04'),
                exact: true
              },
              {
                path: '/docs/enterprise/features',
                component: ComponentCreator('/docs/enterprise/features', '587'),
                exact: true
              },
              {
                path: '/docs/enterprise/governance',
                component: ComponentCreator('/docs/enterprise/governance', '680'),
                exact: true
              },
              {
                path: '/docs/enterprise/incident-response',
                component: ComponentCreator('/docs/enterprise/incident-response', '49f'),
                exact: true
              },
              {
                path: '/docs/enterprise/multi-tenancy',
                component: ComponentCreator('/docs/enterprise/multi-tenancy', '7ac'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/enterprise/positioning',
                component: ComponentCreator('/docs/enterprise/positioning', 'd36'),
                exact: true
              },
              {
                path: '/docs/enterprise/sso',
                component: ComponentCreator('/docs/enterprise/sso', '154'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/enterprise/support',
                component: ComponentCreator('/docs/enterprise/support', '843'),
                exact: true
              },
              {
                path: '/docs/getting-started/',
                component: ComponentCreator('/docs/getting-started/', '309'),
                exact: true
              },
              {
                path: '/docs/getting-started/configuration',
                component: ComponentCreator('/docs/getting-started/configuration', '11c'),
                exact: true
              },
              {
                path: '/docs/getting-started/environment',
                component: ComponentCreator('/docs/getting-started/environment', '7fb'),
                exact: true
              },
              {
                path: '/docs/getting-started/first-debate',
                component: ComponentCreator('/docs/getting-started/first-debate', 'fd7'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/getting-started/installation',
                component: ComponentCreator('/docs/getting-started/installation', 'c80'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/getting-started/introduction',
                component: ComponentCreator('/docs/getting-started/introduction', '39d'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/getting-started/quickstart',
                component: ComponentCreator('/docs/getting-started/quickstart', 'ebb'),
                exact: true,
                sidebar: "gettingStartedSidebar"
              },
              {
                path: '/docs/guides/',
                component: ComponentCreator('/docs/guides/', '09f'),
                exact: true
              },
              {
                path: '/docs/guides/api-quickstart',
                component: ComponentCreator('/docs/guides/api-quickstart', '69d'),
                exact: true
              },
              {
                path: '/docs/guides/api-usage',
                component: ComponentCreator('/docs/guides/api-usage', 'e8b'),
                exact: true
              },
              {
                path: '/docs/guides/automation',
                component: ComponentCreator('/docs/guides/automation', '9be'),
                exact: true
              },
              {
                path: '/docs/guides/bot-integrations',
                component: ComponentCreator('/docs/guides/bot-integrations', '24e'),
                exact: true
              },
              {
                path: '/docs/guides/broadcast',
                component: ComponentCreator('/docs/guides/broadcast', '1f7'),
                exact: true
              },
              {
                path: '/docs/guides/chat-connector',
                component: ComponentCreator('/docs/guides/chat-connector', '053'),
                exact: true
              },
              {
                path: '/docs/guides/connector-troubleshooting',
                component: ComponentCreator('/docs/guides/connector-troubleshooting', '236'),
                exact: true
              },
              {
                path: '/docs/guides/connectors',
                component: ComponentCreator('/docs/guides/connectors', 'cbd'),
                exact: true
              },
              {
                path: '/docs/guides/connectors-setup',
                component: ComponentCreator('/docs/guides/connectors-setup', '9b0'),
                exact: true
              },
              {
                path: '/docs/guides/custom-agents',
                component: ComponentCreator('/docs/guides/custom-agents', 'd7b'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/evidence',
                component: ComponentCreator('/docs/guides/evidence', 'ec2'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/gauntlet',
                component: ComponentCreator('/docs/guides/gauntlet', 'dfe'),
                exact: true
              },
              {
                path: '/docs/guides/graph-debates',
                component: ComponentCreator('/docs/guides/graph-debates', 'd34'),
                exact: true
              },
              {
                path: '/docs/guides/integrations',
                component: ComponentCreator('/docs/guides/integrations', '20a'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/matrix-debates',
                component: ComponentCreator('/docs/guides/matrix-debates', '5cc'),
                exact: true
              },
              {
                path: '/docs/guides/pulse',
                component: ComponentCreator('/docs/guides/pulse', 'fe3'),
                exact: true
              },
              {
                path: '/docs/guides/sdk',
                component: ComponentCreator('/docs/guides/sdk', '6be'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/guides/websocket-events',
                component: ComponentCreator('/docs/guides/websocket-events', '40c'),
                exact: true
              },
              {
                path: '/docs/guides/workflows',
                component: ComponentCreator('/docs/guides/workflows', '39d'),
                exact: true,
                sidebar: "guidesSidebar"
              },
              {
                path: '/docs/operations/',
                component: ComponentCreator('/docs/operations/', 'cce'),
                exact: true
              },
              {
                path: '/docs/operations/alert-runbooks',
                component: ComponentCreator('/docs/operations/alert-runbooks', 'cb4'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-database',
                component: ComponentCreator('/docs/operations/runbook-database', 'b8e'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-deployment',
                component: ComponentCreator('/docs/operations/runbook-deployment', '0d4'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-incident',
                component: ComponentCreator('/docs/operations/runbook-incident', 'e6a'),
                exact: true
              },
              {
                path: '/docs/operations/runbook-provider',
                component: ComponentCreator('/docs/operations/runbook-provider', 'b9c'),
                exact: true
              },
              {
                path: '/docs/security/',
                component: ComponentCreator('/docs/security/', 'c69'),
                exact: true
              },
              {
                path: '/docs/security/authentication',
                component: ComponentCreator('/docs/security/authentication', '56a'),
                exact: true
              },
              {
                path: '/docs/security/breach-notification',
                component: ComponentCreator('/docs/security/breach-notification', 'a8a'),
                exact: true
              },
              {
                path: '/docs/security/ci-cd',
                component: ComponentCreator('/docs/security/ci-cd', 'de2'),
                exact: true
              },
              {
                path: '/docs/security/compliance',
                component: ComponentCreator('/docs/security/compliance', '013'),
                exact: true
              },
              {
                path: '/docs/security/data-classification',
                component: ComponentCreator('/docs/security/data-classification', '2e8'),
                exact: true
              },
              {
                path: '/docs/security/data-residency',
                component: ComponentCreator('/docs/security/data-residency', '2eb'),
                exact: true
              },
              {
                path: '/docs/security/dsar',
                component: ComponentCreator('/docs/security/dsar', '07e'),
                exact: true
              },
              {
                path: '/docs/security/overview',
                component: ComponentCreator('/docs/security/overview', 'd01'),
                exact: true
              },
              {
                path: '/docs/security/privacy-policy',
                component: ComponentCreator('/docs/security/privacy-policy', 'f2f'),
                exact: true
              },
              {
                path: '/docs/security/remote-work',
                component: ComponentCreator('/docs/security/remote-work', 'f9a'),
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
