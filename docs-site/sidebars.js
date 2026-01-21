// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // Getting Started sidebar
  gettingStartedSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
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
      label: 'Guides',
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

  // API Reference sidebar (auto-generated from OpenAPI)
  apiSidebar: [
    {
      type: 'category',
      label: 'API Reference',
      link: {
        type: 'doc',
        id: 'api-reference/index',
      },
      items: require('./docs/api-reference/sidebar.js'),
    },
  ],
};

module.exports = sidebars;
