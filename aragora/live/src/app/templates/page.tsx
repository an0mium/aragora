'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';

interface Template {
  id: string;
  name: string;
  category: string;
  description: string;
  prompt: string;
  agents: string[];
  rounds: number;
  tags: string[];
  examples?: string[];
}

const TEMPLATES: Template[] = [
  // Architecture
  {
    id: 'microservices-vs-monolith',
    name: 'Microservices vs Monolith',
    category: 'Architecture',
    description: 'Evaluate whether your project should use a microservices architecture or a monolithic design.',
    prompt: 'We are building [PROJECT_DESCRIPTION]. Should we use a microservices architecture or a monolith? Consider: team size, deployment complexity, scaling needs, and long-term maintenance.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 3,
    tags: ['architecture', 'scaling', 'deployment'],
    examples: ['E-commerce platform', 'SaaS dashboard', 'Mobile app backend'],
  },
  {
    id: 'database-selection',
    name: 'Database Selection',
    category: 'Architecture',
    description: 'Choose the right database technology for your use case.',
    prompt: 'For [USE_CASE], which database should we use? Options: PostgreSQL, MongoDB, DynamoDB, Redis, or other. Consider: query patterns, scale, consistency needs, and operational complexity.',
    agents: ['claude', 'gpt-4', 'codestral'],
    rounds: 3,
    tags: ['database', 'infrastructure', 'data'],
  },
  {
    id: 'api-design',
    name: 'API Design Review',
    category: 'Architecture',
    description: 'Review and improve your API design before implementation.',
    prompt: 'Review this API design: [API_SPEC]. Evaluate: REST vs GraphQL choice, versioning strategy, authentication approach, error handling, and pagination.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 3,
    tags: ['api', 'design', 'rest', 'graphql'],
  },
  // Code Review
  {
    id: 'code-quality',
    name: 'Code Quality Assessment',
    category: 'Code Review',
    description: 'Get an objective assessment of your code quality and suggestions for improvement.',
    prompt: 'Review this code for quality: [CODE]. Evaluate: readability, maintainability, SOLID principles, error handling, and potential bugs.',
    agents: ['claude', 'codestral', 'gpt-4'],
    rounds: 2,
    tags: ['code', 'quality', 'review'],
  },
  {
    id: 'refactoring-strategy',
    name: 'Refactoring Strategy',
    category: 'Code Review',
    description: 'Develop a refactoring plan for legacy code.',
    prompt: 'This legacy code needs refactoring: [CODE_DESCRIPTION]. What should be the refactoring strategy? Consider: risk, incremental approach, testing strategy, and time investment.',
    agents: ['claude', 'gpt-4', 'codestral'],
    rounds: 3,
    tags: ['refactoring', 'legacy', 'technical-debt'],
  },
  // Security
  {
    id: 'security-review',
    name: 'Security Assessment',
    category: 'Security',
    description: 'Identify security vulnerabilities and get remediation advice.',
    prompt: 'Review this system/code for security vulnerabilities: [DESCRIPTION]. Check for: OWASP Top 10, authentication flaws, authorization issues, injection attacks, and data exposure risks.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 3,
    tags: ['security', 'vulnerabilities', 'owasp'],
  },
  {
    id: 'auth-strategy',
    name: 'Authentication Strategy',
    category: 'Security',
    description: 'Choose the right authentication approach for your application.',
    prompt: 'For [APPLICATION_TYPE], what authentication strategy should we use? Options: JWT, sessions, OAuth2, SAML, passwordless. Consider: security, UX, implementation complexity.',
    agents: ['claude', 'gpt-4', 'mistral'],
    rounds: 3,
    tags: ['auth', 'security', 'identity'],
  },
  // DevOps
  {
    id: 'deployment-strategy',
    name: 'Deployment Strategy',
    category: 'DevOps',
    description: 'Choose the right deployment strategy for your release.',
    prompt: 'For [RELEASE_DESCRIPTION], what deployment strategy should we use? Options: blue-green, canary, rolling, feature flags. Consider: risk tolerance, rollback needs, and monitoring.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 2,
    tags: ['deployment', 'devops', 'release'],
  },
  {
    id: 'cloud-provider',
    name: 'Cloud Provider Selection',
    category: 'DevOps',
    description: 'Compare cloud providers for your workload.',
    prompt: 'For [WORKLOAD_DESCRIPTION], which cloud provider is best? Compare: AWS, GCP, Azure. Consider: cost, services needed, team expertise, and vendor lock-in.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 3,
    tags: ['cloud', 'aws', 'gcp', 'azure'],
  },
  // Product
  {
    id: 'feature-prioritization',
    name: 'Feature Prioritization',
    category: 'Product',
    description: 'Prioritize features for your product roadmap.',
    prompt: 'We have these potential features: [FEATURE_LIST]. How should we prioritize them? Consider: user impact, effort, strategic alignment, and dependencies.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 3,
    tags: ['product', 'prioritization', 'roadmap'],
  },
  {
    id: 'build-vs-buy',
    name: 'Build vs Buy',
    category: 'Product',
    description: 'Decide whether to build a feature in-house or use a third-party solution.',
    prompt: 'Should we build [FEATURE] in-house or buy/integrate a third-party solution? Consider: cost, time, customization needs, and long-term maintenance.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 3,
    tags: ['build', 'buy', 'make-or-buy'],
  },
  // Testing
  {
    id: 'testing-strategy',
    name: 'Testing Strategy',
    category: 'Testing',
    description: 'Design a comprehensive testing strategy for your project.',
    prompt: 'For [PROJECT_TYPE], what should our testing strategy be? Consider: unit tests, integration tests, e2e tests, load tests, and the testing pyramid.',
    agents: ['claude', 'gpt-4', 'codestral'],
    rounds: 2,
    tags: ['testing', 'qa', 'quality'],
  },
];

const CATEGORIES = [...new Set(TEMPLATES.map((t) => t.category))];

export default function TemplatesPage() {
  const router = useRouter();
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);
  const [customPrompt, setCustomPrompt] = useState('');

  const filteredTemplates = TEMPLATES.filter((template) => {
    const matchesCategory = !selectedCategory || template.category === selectedCategory;
    const matchesSearch =
      !searchQuery ||
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.tags.some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesCategory && matchesSearch;
  });

  const handleUseTemplate = (template: Template) => {
    setSelectedTemplate(template);
    setCustomPrompt(template.prompt);
  };

  const handleStartDebate = () => {
    if (selectedTemplate && customPrompt) {
      // Navigate to home with the prompt pre-filled
      const params = new URLSearchParams({
        prompt: customPrompt,
        agents: selectedTemplate.agents.join(','),
        rounds: selectedTemplate.rounds.toString(),
      });
      router.push(`/?${params.toString()}`);
    }
  };

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <Link
                href="/gallery"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [GALLERY]
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Hero */}
        <div className="border-b border-acid-green/20 bg-surface/30">
          <div className="container mx-auto px-4 py-12 text-center">
            <h1 className="text-3xl md:text-4xl font-mono text-acid-green mb-4">
              {'>'} DEBATE TEMPLATES
            </h1>
            <p className="text-text-muted font-mono max-w-2xl mx-auto">
              Pre-built templates for common technical decisions. Choose a template,
              customize it for your context, and let AI agents stress-test your thinking.
            </p>
          </div>
        </div>

        <div className="container mx-auto px-4 py-8">
          {/* Filters */}
          <div className="flex flex-col md:flex-row gap-4 mb-8">
            {/* Category Filter */}
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setSelectedCategory(null)}
                className={`px-3 py-1 text-xs font-mono border transition-colors ${
                  !selectedCategory
                    ? 'border-acid-green bg-acid-green/20 text-acid-green'
                    : 'border-acid-green/30 text-text-muted hover:border-acid-green/60'
                }`}
              >
                [ALL]
              </button>
              {CATEGORIES.map((category) => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={`px-3 py-1 text-xs font-mono border transition-colors ${
                    selectedCategory === category
                      ? 'border-acid-green bg-acid-green/20 text-acid-green'
                      : 'border-acid-green/30 text-text-muted hover:border-acid-green/60'
                  }`}
                >
                  [{category.toUpperCase()}]
                </button>
              ))}
            </div>

            {/* Search */}
            <div className="flex-1 md:max-w-xs ml-auto">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search templates..."
                className="w-full px-4 py-2 text-sm font-mono bg-surface border border-acid-green/30
                         text-text placeholder-text-muted/50 focus:border-acid-green focus:outline-none"
              />
            </div>
          </div>

          {/* Template Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredTemplates.map((template) => (
              <div
                key={template.id}
                className="card p-6 hover:border-acid-green/60 transition-colors"
              >
                {/* Category Badge */}
                <div className="text-xs font-mono text-acid-cyan mb-2">
                  [{template.category.toUpperCase()}]
                </div>

                {/* Name */}
                <h3 className="font-mono text-acid-green mb-2">
                  {template.name}
                </h3>

                {/* Description */}
                <p className="text-sm font-mono text-text-muted mb-4">
                  {template.description}
                </p>

                {/* Tags */}
                <div className="flex flex-wrap gap-1 mb-4">
                  {template.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 text-xs font-mono bg-surface border border-acid-green/20 text-text-muted"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                {/* Meta */}
                <div className="flex items-center justify-between text-xs font-mono text-text-muted mb-4">
                  <span>{template.agents.length} agents</span>
                  <span>{template.rounds} rounds</span>
                </div>

                {/* Use Template Button */}
                <button
                  onClick={() => handleUseTemplate(template)}
                  className="w-full px-4 py-2 text-sm font-mono border border-acid-green/50
                           text-acid-green hover:bg-acid-green/10 transition-colors"
                >
                  [USE TEMPLATE]
                </button>
              </div>
            ))}
          </div>

          {filteredTemplates.length === 0 && (
            <div className="text-center py-12">
              <p className="text-text-muted font-mono">No templates match your search.</p>
            </div>
          )}
        </div>

        {/* Template Customization Modal */}
        {selectedTemplate && (
          <div className="fixed inset-0 z-[100] bg-bg/95 backdrop-blur-sm flex items-center justify-center p-4">
            <div className="max-w-2xl w-full border border-acid-green/50 bg-surface p-6 max-h-[90vh] overflow-y-auto">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <div className="text-xs font-mono text-acid-cyan mb-1">
                    [{selectedTemplate.category.toUpperCase()}]
                  </div>
                  <h2 className="text-xl font-mono text-acid-green">
                    {selectedTemplate.name}
                  </h2>
                </div>
                <button
                  onClick={() => setSelectedTemplate(null)}
                  className="text-text-muted hover:text-acid-green transition-colors"
                >
                  [X]
                </button>
              </div>

              {/* Prompt Editor */}
              <div className="mb-6">
                <label className="block text-sm font-mono text-text-muted mb-2">
                  CUSTOMIZE YOUR PROMPT
                </label>
                <p className="text-xs font-mono text-text-muted/70 mb-3">
                  Replace [PLACEHOLDERS] with your specific context.
                </p>
                <textarea
                  value={customPrompt}
                  onChange={(e) => setCustomPrompt(e.target.value)}
                  rows={6}
                  className="w-full px-4 py-3 font-mono text-sm bg-bg border border-acid-green/30
                           text-text placeholder-text-muted/50 focus:border-acid-green focus:outline-none resize-none"
                />
              </div>

              {/* Examples */}
              {selectedTemplate.examples && (
                <div className="mb-6">
                  <label className="block text-sm font-mono text-text-muted mb-2">
                    EXAMPLE CONTEXTS
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {selectedTemplate.examples.map((example) => (
                      <button
                        key={example}
                        onClick={() => {
                          // Replace first placeholder with example
                          const placeholder = customPrompt.match(/\[([^\]]+)\]/)?.[0];
                          if (placeholder) {
                            setCustomPrompt(customPrompt.replace(placeholder, example));
                          }
                        }}
                        className="px-3 py-1 text-xs font-mono border border-acid-cyan/30
                                 text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
                      >
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Configuration Summary */}
              <div className="p-4 bg-bg border border-acid-green/20 mb-6">
                <div className="grid grid-cols-2 gap-4 text-sm font-mono">
                  <div>
                    <span className="text-text-muted">Agents:</span>{' '}
                    <span className="text-acid-cyan">{selectedTemplate.agents.join(', ')}</span>
                  </div>
                  <div>
                    <span className="text-text-muted">Rounds:</span>{' '}
                    <span className="text-acid-cyan">{selectedTemplate.rounds}</span>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-3">
                <button
                  onClick={() => setSelectedTemplate(null)}
                  className="px-4 py-2 font-mono text-sm border border-acid-green/30
                           text-text-muted hover:border-acid-green hover:text-acid-green transition-colors"
                >
                  [CANCEL]
                </button>
                <button
                  onClick={handleStartDebate}
                  className="flex-1 px-6 py-2 font-mono text-sm bg-acid-green text-bg
                           hover:bg-acid-green/80 transition-colors"
                >
                  [START DEBATE]
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} {TEMPLATES.length} TEMPLATES AVAILABLE
          </p>
          <p className="text-text-muted/50 mt-2">
            Have a template suggestion?{' '}
            <a href="mailto:feedback@aragora.ai" className="text-acid-cyan hover:text-acid-green">
              Let us know
            </a>
          </p>
        </footer>
      </main>
    </>
  );
}
