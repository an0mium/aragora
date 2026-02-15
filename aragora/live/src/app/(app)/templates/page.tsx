'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { TemplateSearch } from '@/components/templates/TemplateSearch';
import { API_BASE_URL } from '@/config';

interface Template {
  id?: string;
  name: string;
  category: string;
  description: string;
  prompt?: string;
  agents: string[];
  rounds: number;
  tags?: string[];
  examples?: string[];
  example_topics?: string[];
}

// Fallback templates used when API is unavailable
const FALLBACK_TEMPLATES: Template[] = [
  {
    name: 'Microservices vs Monolith',
    category: 'Architecture',
    description: 'Evaluate whether your project should use a microservices architecture or a monolithic design.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 3,
    tags: ['architecture', 'scaling', 'deployment'],
    example_topics: ['E-commerce platform', 'SaaS dashboard', 'Mobile app backend'],
  },
  {
    name: 'Database Selection',
    category: 'Architecture',
    description: 'Choose the right database technology for your use case.',
    agents: ['claude', 'gpt-4', 'codestral'],
    rounds: 3,
    tags: ['database', 'infrastructure', 'data'],
  },
  {
    name: 'Security Assessment',
    category: 'Security',
    description: 'Identify security vulnerabilities and get remediation advice.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 3,
    tags: ['security', 'vulnerabilities', 'owasp'],
  },
  {
    name: 'Feature Prioritization',
    category: 'Product',
    description: 'Prioritize features for your product roadmap.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 3,
    tags: ['product', 'prioritization', 'roadmap'],
  },
  {
    name: 'Build vs Buy',
    category: 'Product',
    description: 'Decide whether to build a feature in-house or use a third-party solution.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 3,
    tags: ['build', 'buy', 'make-or-buy'],
  },
  {
    name: 'Deployment Strategy',
    category: 'DevOps',
    description: 'Choose the right deployment strategy for your release.',
    agents: ['claude', 'gpt-4', 'gemini'],
    rounds: 2,
    tags: ['deployment', 'devops', 'release'],
  },
];

export default function TemplatesPage() {
  const router = useRouter();
  const [templates, setTemplates] = useState<Template[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);
  const [customPrompt, setCustomPrompt] = useState('');
  const highlightRef = useRef<string | null>(null);

  // Fetch templates from API
  useEffect(() => {
    async function fetchTemplates() {
      setIsLoading(true);
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/templates`);
        if (response.ok) {
          const data = await response.json();
          const items = Array.isArray(data) ? data : data.templates ?? [];
          setTemplates(items);
        } else {
          setTemplates(FALLBACK_TEMPLATES);
          setError('Could not load templates from server. Showing defaults.');
        }
      } catch {
        setTemplates(FALLBACK_TEMPLATES);
        setError('Could not connect to server. Showing default templates.');
      } finally {
        setIsLoading(false);
      }
    }
    fetchTemplates();
  }, []);

  const categories = [...new Set(templates.map((t) => t.category))];

  const filteredTemplates = templates.filter((template) => {
    const matchesCategory = !selectedCategory || template.category === selectedCategory;
    const matchesSearch =
      !searchQuery ||
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (template.tags ?? []).some((tag) => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesCategory && matchesSearch;
  });

  const handleUseTemplate = (template: Template) => {
    router.push(`/arena?template=${encodeURIComponent(template.name)}`);
  };

  const handleSearchSelect = (templateName: string) => {
    highlightRef.current = templateName;
    const el = document.getElementById(`template-${templateName.replace(/\s+/g, '-').toLowerCase()}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el.classList.add('ring-2', 'ring-acid-green');
      setTimeout(() => {
        el.classList.remove('ring-2', 'ring-acid-green');
        highlightRef.current = null;
      }, 2000);
    }
  };

  const topics = (t: Template) => t.example_topics ?? t.examples ?? [];

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
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
          {/* Smart Search */}
          <div className="mb-6">
            <TemplateSearch onSelect={handleSearchSelect} />
          </div>

          {/* Error Banner */}
          {error && (
            <div className="mb-4 border border-acid-cyan/30 bg-acid-cyan/5 p-3 flex items-center justify-between">
              <span className="text-xs font-mono text-acid-cyan">{error}</span>
              <button onClick={() => setError(null)} className="text-acid-cyan hover:text-acid-green text-xs font-mono">
                [X]
              </button>
            </div>
          )}

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
              {categories.map((category) => (
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
                placeholder="Filter templates..."
                className="w-full px-4 py-2 text-sm font-mono bg-surface border border-acid-green/30
                         text-text placeholder-text-muted/50 focus:border-acid-green focus:outline-none"
              />
            </div>
          </div>

          {/* Loading */}
          {isLoading && (
            <div className="flex items-center justify-center py-16">
              <div className="text-center">
                <div className="w-8 h-8 border-2 border-acid-green/30 border-t-acid-green rounded-full animate-spin mx-auto mb-4" />
                <p className="text-text-muted text-sm font-mono">Loading templates...</p>
              </div>
            </div>
          )}

          {/* Template Grid */}
          {!isLoading && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredTemplates.map((template) => (
                <div
                  key={template.name}
                  id={`template-${template.name.replace(/\s+/g, '-').toLowerCase()}`}
                  className="card p-6 hover:border-acid-green/60 transition-all"
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

                  {/* Example Topics */}
                  {topics(template).length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-3">
                      {topics(template).slice(0, 4).map((topic) => (
                        <span
                          key={topic}
                          className="px-2 py-0.5 text-xs font-mono bg-acid-cyan/10 text-acid-cyan border border-acid-cyan/20"
                        >
                          {topic}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Tags */}
                  {(template.tags ?? []).length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-4">
                      {(template.tags ?? []).map((tag) => (
                        <span
                          key={tag}
                          className="px-2 py-0.5 text-xs font-mono bg-surface border border-acid-green/20 text-text-muted"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}

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
          )}

          {!isLoading && filteredTemplates.length === 0 && (
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
              {topics(selectedTemplate).length > 0 && (
                <div className="mb-6">
                  <label className="block text-sm font-mono text-text-muted mb-2">
                    EXAMPLE CONTEXTS
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {topics(selectedTemplate).map((example) => (
                      <button
                        key={example}
                        onClick={() => {
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
                  onClick={() => {
                    if (selectedTemplate && customPrompt) {
                      const params = new URLSearchParams({
                        prompt: customPrompt,
                        agents: selectedTemplate.agents.join(','),
                        rounds: selectedTemplate.rounds.toString(),
                      });
                      router.push(`/?${params.toString()}`);
                    }
                  }}
                  className="flex-1 px-6 py-2 font-mono text-sm bg-acid-green text-bg
                           hover:bg-acid-green/80 transition-colors"
                >
                  [START DEBATE]
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </>
  );
}
