// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // Getting Started sidebar
  gettingStartedSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/introduction',
        'getting-started/quickstart',
        'getting-started/installation',
        'getting-started/first-debate',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'core-concepts/debates',
        'core-concepts/agents',
        'core-concepts/consensus',
        'core-concepts/memory',
        'core-concepts/knowledge-mound',
      ],
    },
  ],

  // Guides sidebar
  guidesSidebar: [
    {
      type: 'category',
      label: 'Developer Guides',
      collapsed: false,
      items: [
        'guides/sdk',
        'guides/custom-agents',
        'guides/workflows',
        'guides/integrations',
        'guides/evidence',
      ],
    },
    {
      type: 'category',
      label: 'Deployment',
      items: [
        'deployment/docker',
        'deployment/kubernetes',
        'deployment/scaling',
        'deployment/security',
      ],
    },
    {
      type: 'category',
      label: 'Enterprise',
      items: [
        'enterprise/multi-tenancy',
        'enterprise/sso',
        'enterprise/audit-logs',
        'enterprise/compliance',
      ],
    },
  ],

  // API Reference sidebar (manual structure)
  apiSidebar: [
    'api-reference/index',
    {
      type: 'category',
      label: 'Debates',
      items: [
        'api-reference/debates/overview',
        'api-reference/debates/create',
        'api-reference/debates/get',
        'api-reference/debates/list',
        'api-reference/debates/consensus',
      ],
    },
    {
      type: 'category',
      label: 'Agents',
      items: [
        'api-reference/agents/overview',
        'api-reference/agents/list',
        'api-reference/agents/stats',
      ],
    },
    {
      type: 'category',
      label: 'Knowledge',
      items: [
        'api-reference/knowledge/overview',
        'api-reference/knowledge/query',
        'api-reference/knowledge/store',
      ],
    },
    {
      type: 'category',
      label: 'Workflows',
      items: [
        'api-reference/workflows/overview',
        'api-reference/workflows/execute',
      ],
    },
  ],
};

module.exports = sidebars;
