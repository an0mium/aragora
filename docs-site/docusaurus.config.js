// @ts-check
// Docusaurus Configuration for Aragora Documentation Portal
// See: https://docusaurus.io/docs/configuration

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Aragora Documentation',
  tagline: 'Control plane for multi-agent robust decisionmaking across org knowledge and channels',
  favicon: 'img/favicon.ico',

  // Production URL
  url: 'https://docs.aragora.ai',
  baseUrl: '/',

  // GitHub Pages deployment config (if using)
  organizationName: 'aragora',
  projectName: 'aragora',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/aragora/aragora/tree/main/docs/',
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        blog: false, // Disable blog for docs-only site
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  plugins: [
    // OpenAPI documentation plugin
    [
      'docusaurus-plugin-openapi-docs',
      {
        id: 'api',
        docsPluginId: 'classic',
        config: {
          aragora: {
            specPath: '../docs/api/openapi.json',
            outputDir: 'docs/api-reference',
            sidebarOptions: {
              groupPathsBy: 'tag',
            },
          },
        },
      },
    ],
  ],

  themes: ['docusaurus-theme-openapi-docs'],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Social card
      image: 'img/aragora-social-card.png',

      // Navbar
      navbar: {
        title: 'Aragora',
        logo: {
          alt: 'Aragora Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'gettingStartedSidebar',
            position: 'left',
            label: 'Getting Started',
          },
          {
            type: 'docSidebar',
            sidebarId: 'guidesSidebar',
            position: 'left',
            label: 'Guides',
          },
          {
            type: 'docSidebar',
            sidebarId: 'apiSidebar',
            position: 'left',
            label: 'API Reference',
          },
          {
            href: 'https://github.com/aragora/aragora',
            label: 'GitHub',
            position: 'right',
          },
          {
            href: 'https://status.aragora.ai',
            label: 'Status',
            position: 'right',
          },
        ],
      },

      // Footer
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Getting Started',
                to: '/docs/getting-started',
              },
              {
                label: 'API Reference',
                to: '/docs/api-reference',
              },
              {
                label: 'SDK Guide',
                to: '/docs/guides/sdk',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub Discussions',
                href: 'https://github.com/aragora/aragora/discussions',
              },
              {
                label: 'Discord',
                href: 'https://discord.gg/aragora',
              },
              {
                label: 'Twitter',
                href: 'https://twitter.com/aragora_ai',
              },
            ],
          },
          {
            title: 'Company',
            items: [
              {
                label: 'Blog',
                href: 'https://aragora.ai/blog',
              },
              {
                label: 'Privacy Policy',
                href: 'https://aragora.ai/privacy',
              },
              {
                label: 'Terms of Service',
                href: 'https://aragora.ai/terms',
              },
            ],
          },
        ],
        copyright: `Copyright ${new Date().getFullYear()} Aragora. Built with Docusaurus.`,
      },

      // Code blocks
      prism: {
        theme: require('prism-react-renderer').themes.github,
        darkTheme: require('prism-react-renderer').themes.dracula,
        additionalLanguages: ['bash', 'python', 'json', 'yaml', 'typescript'],
      },

      // Search (Algolia)
      algolia: {
        appId: 'YOUR_ALGOLIA_APP_ID',
        apiKey: 'YOUR_ALGOLIA_API_KEY',
        indexName: 'aragora-docs',
        contextualSearch: true,
      },

      // Announcement bar (optional)
      announcementBar: {
        id: 'v2_release',
        content:
          'Aragora v2.1 is out! Check out the <a href="/docs/migration">migration guide</a>.',
        backgroundColor: '#4F46E5',
        textColor: '#FFFFFF',
        isCloseable: true,
      },
    }),
};

module.exports = config;
