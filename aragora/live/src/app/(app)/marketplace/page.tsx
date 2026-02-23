'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useRightSidebar } from '@/context/RightSidebarContext';
import { useAuth } from '@/context/AuthContext';
import {
  useMarketplaceStore,
  CATEGORY_STYLES,
  type Template,
  type TemplateCategory,
  type TemplateType,
} from '@/store/marketplaceStore';

const CATEGORIES: Array<{ value: TemplateCategory | 'all'; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'analysis', label: 'Analysis' },
  { value: 'coding', label: 'Coding' },
  { value: 'creative', label: 'Creative' },
  { value: 'debate', label: 'Debate' },
  { value: 'research', label: 'Research' },
  { value: 'decision', label: 'Decision' },
  { value: 'brainstorm', label: 'Brainstorm' },
  { value: 'review', label: 'Review' },
  { value: 'planning', label: 'Planning' },
  { value: 'custom', label: 'Custom' },
];

const TYPES: Array<{ value: TemplateType | 'all'; label: string }> = [
  { value: 'all', label: 'All Types' },
  { value: 'agent', label: 'Agents' },
  { value: 'debate', label: 'Debates' },
  { value: 'workflow', label: 'Workflows' },
];

export default function MarketplacePage() {
  const {
    featuredTemplates,
    selectedCategory,
    selectedType,
    searchQuery,
    sortBy,
    isLoading,
    error,
    selectedTemplate,
    installedTemplates,
    fetchTemplates,
    fetchFeatured,
    setCategory,
    setType,
    setSearchQuery,
    setSortBy,
    selectTemplate,
    installTemplate,
    uninstallTemplate,
    starTemplate,
    getFilteredTemplates,
  } = useMarketplaceStore();

  const router = useRouter();
  const { setContext, clearContext } = useRightSidebar();
  const [showPublish, setShowPublish] = useState(false);

  // Fetch data on mount
  useEffect(() => {
    fetchTemplates();
    fetchFeatured();
  }, [fetchTemplates, fetchFeatured]);

  // Set up right sidebar
  useEffect(() => {
    const templates = getFilteredTemplates();

    setContext({
      title: 'Marketplace',
      subtitle: 'Template library',
      statsContent: (
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Templates</span>
            <span className="text-sm font-mono text-[var(--acid-green)]">{templates.length}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Featured</span>
            <span className="text-sm font-mono text-[var(--acid-cyan)]">{featuredTemplates.length}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Installed</span>
            <span className="text-sm font-mono text-[var(--text)]">{installedTemplates.length}</span>
          </div>
        </div>
      ),
      actionsContent: (
        <div className="space-y-2">
          <button
            onClick={() => setShowPublish(true)}
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--acid-green)] text-[var(--bg)] font-bold hover:bg-[var(--acid-green)]/80 transition-colors"
          >
            PUBLISH TEMPLATE
          </button>
          <Link
            href="/marketplace/my-templates"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
          >
            MY TEMPLATES
          </Link>
        </div>
      ),
    });

    return () => clearContext();
  }, [
    featuredTemplates.length,
    installedTemplates.length,
    setContext,
    clearContext,
    getFilteredTemplates,
  ]);

  const filteredTemplates = getFilteredTemplates();

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <div className="max-w-6xl mx-auto px-4 py-8">
          {/* Header */}
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">TEMPLATE MARKETPLACE</h1>
            <p className="text-text-muted text-sm font-mono">
              Browse, install, and publish agent, debate, and workflow templates.
            </p>
          </div>

          {/* Error Banner */}
          {error && (
            <div className="mb-4 border border-warning/30 bg-warning/10 p-3">
              <p className="text-warning text-sm font-mono">{error}</p>
            </div>
          )}

          {/* Search and Filters */}
          <div className="mb-6 space-y-4">
            {/* Search Bar */}
            <div className="flex gap-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search templates..."
                className="flex-1 px-4 py-2 bg-surface border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green"
              />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as 'downloads' | 'stars' | 'recent')}
                className="px-4 py-2 bg-surface border border-acid-green/30 text-text font-mono text-sm focus:outline-none"
              >
                <option value="downloads">Most Downloads</option>
                <option value="stars">Most Stars</option>
                <option value="recent">Recent</option>
              </select>
            </div>

            {/* Type Filters */}
            <div className="flex gap-2 flex-wrap">
              {TYPES.map((type) => (
                <button
                  key={type.value}
                  onClick={() => setType(type.value)}
                  className={`px-3 py-1 text-xs font-mono border transition-colors ${
                    selectedType === type.value
                      ? 'bg-acid-cyan/20 text-acid-cyan border-acid-cyan/50'
                      : 'bg-surface text-text-muted border-border hover:border-acid-cyan/30'
                  }`}
                >
                  {type.label}
                </button>
              ))}
            </div>

            {/* Category Filters */}
            <div className="flex gap-2 flex-wrap">
              {CATEGORIES.map((cat) => (
                <button
                  key={cat.value}
                  onClick={() => setCategory(cat.value)}
                  className={`px-3 py-1 text-xs font-mono border transition-colors ${
                    selectedCategory === cat.value
                      ? 'bg-acid-green/20 text-acid-green border-acid-green/50'
                      : 'bg-surface text-text-muted border-border hover:border-acid-green/30'
                  }`}
                >
                  {cat.value !== 'all' && CATEGORY_STYLES[cat.value]?.icon} {cat.label}
                </button>
              ))}
            </div>
          </div>

          {/* Featured Section */}
          {featuredTemplates.length > 0 && selectedCategory === 'all' && selectedType === 'all' && !searchQuery && (
            <div className="mb-8">
              <h2 className="text-sm font-mono text-acid-cyan uppercase tracking-wider mb-4">
                Featured Templates
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {featuredTemplates.slice(0, 3).map((template) => (
                  <TemplateCard
                    key={template.metadata.id}
                    template={template}
                    isInstalled={installedTemplates.includes(template.metadata.id)}
                    onSelect={() => selectTemplate(template)}
                    onInstall={() => installTemplate(template.metadata.id)}
                    onUninstall={() => uninstallTemplate(template.metadata.id)}
                    onStar={() => starTemplate(template.metadata.id)}
                    onUse={() => router.push(`/arena?template=${encodeURIComponent(template.metadata.name)}`)}
                    featured
                  />
                ))}
              </div>
            </div>
          )}

          {/* Templates Grid */}
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-sm font-mono text-acid-cyan uppercase tracking-wider">
              {selectedCategory !== 'all'
                ? `${CATEGORY_STYLES[selectedCategory]?.icon} ${selectedCategory}`
                : selectedType !== 'all'
                ? `${selectedType} Templates`
                : 'All Templates'}
              {` (${filteredTemplates.length})`}
            </h2>
          </div>

          {isLoading ? (
            <div className="flex items-center justify-center py-16">
              <div className="text-center">
                <div className="w-8 h-8 border-2 border-acid-green/30 border-t-acid-green rounded-full animate-spin mx-auto mb-4" />
                <p className="text-text-muted text-sm font-mono">Loading templates...</p>
              </div>
            </div>
          ) : filteredTemplates.length === 0 ? (
            <div className="border border-acid-green/20 bg-surface/30 p-8 text-center">
              <p className="text-text-muted text-sm font-mono mb-2">No templates found</p>
              <p className="text-text-muted/60 text-xs font-mono">
                Try adjusting your filters or search query.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredTemplates.map((template) => (
                <TemplateCard
                  key={template.metadata.id}
                  template={template}
                  isInstalled={installedTemplates.includes(template.metadata.id)}
                  onSelect={() => selectTemplate(template)}
                  onInstall={() => installTemplate(template.metadata.id)}
                  onUninstall={() => uninstallTemplate(template.metadata.id)}
                  onStar={() => starTemplate(template.metadata.id)}
                  onUse={() => router.push(`/arena?template=${encodeURIComponent(template.metadata.name)}`)}
                />
              ))}
            </div>
          )}
        </div>
      </main>

      {/* Template Detail Modal */}
      {selectedTemplate && (
        <TemplateDetailModal
          template={selectedTemplate}
          isInstalled={installedTemplates.includes(selectedTemplate.metadata.id)}
          onClose={() => selectTemplate(null)}
          onInstall={() => installTemplate(selectedTemplate.metadata.id)}
          onUninstall={() => uninstallTemplate(selectedTemplate.metadata.id)}
          onStar={() => starTemplate(selectedTemplate.metadata.id)}
        />
      )}

      {/* Publish Modal */}
      {showPublish && <PublishModal onClose={() => setShowPublish(false)} />}
    </>
  );
}

// ============================================================================
// Template Card Component
// ============================================================================

interface TemplateCardProps {
  template: Template;
  isInstalled: boolean;
  onSelect: () => void;
  onInstall: () => void;
  onUninstall: () => void;
  onStar: () => void;
  onUse: () => void;
  featured?: boolean;
}

function TemplateCard({
  template,
  isInstalled,
  onSelect,
  onInstall,
  onUninstall,
  onStar,
  onUse,
  featured,
}: TemplateCardProps) {
  const { metadata } = template;
  const categoryStyle = CATEGORY_STYLES[metadata.category];
  const templateType = getTemplateType(template);

  return (
    <div
      onClick={onSelect}
      className={`border bg-surface/50 p-4 cursor-pointer transition-colors hover:bg-surface/80 ${
        featured ? 'border-acid-green/50' : 'border-acid-green/20'
      }`}
    >
      {featured && (
        <div className="text-xs font-mono text-acid-green mb-2 uppercase">Featured</div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div>
          <h3 className="text-sm font-mono text-text font-bold">{metadata.name}</h3>
          <div className="text-xs font-mono text-text-muted">by {metadata.author}</div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span className={`px-2 py-0.5 text-xs font-mono ${categoryStyle.color} ${categoryStyle.bgColor}`}>
            {categoryStyle.icon} {metadata.category}
          </span>
          <span className="text-xs font-mono text-text-muted uppercase">{templateType}</span>
        </div>
      </div>

      {/* Description */}
      <p className="text-xs font-mono text-text-muted mb-3 line-clamp-2">{metadata.description}</p>

      {/* Tags */}
      <div className="flex flex-wrap gap-1 mb-3">
        {metadata.tags.slice(0, 3).map((tag) => (
          <span
            key={tag}
            className="px-2 py-0.5 text-xs font-mono bg-bg/50 text-text-muted border border-border"
          >
            {tag}
          </span>
        ))}
      </div>

      {/* Stats */}
      <div className="flex items-center justify-between text-xs font-mono">
        <div className="flex gap-4">
          <span className="text-text-muted">
            <span className="text-acid-green">{metadata.downloads}</span> downloads
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onStar();
            }}
            className="text-text-muted hover:text-yellow-400 transition-colors"
          >
            <span className="text-yellow-400">{metadata.stars}</span> stars
          </button>
        </div>
        <div className="flex gap-1">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onUse();
            }}
            className="px-2 py-1 text-xs font-mono bg-acid-cyan/10 text-acid-cyan border border-acid-cyan/30 hover:bg-acid-cyan/20 transition-colors"
          >
            USE
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (isInstalled) {
                onUninstall();
              } else {
                onInstall();
              }
            }}
            className={`px-2 py-1 text-xs font-mono transition-colors ${
              isInstalled
                ? 'bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20'
                : 'bg-acid-green/10 text-acid-green border border-acid-green/30 hover:bg-acid-green/20'
            }`}
          >
            {isInstalled ? 'UNINSTALL' : 'INSTALL'}
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Template Detail Modal
// ============================================================================

interface TemplateDetailModalProps {
  template: Template;
  isInstalled: boolean;
  onClose: () => void;
  onInstall: () => void;
  onUninstall: () => void;
  onStar: () => void;
}

function TemplateDetailModal({
  template,
  isInstalled,
  onClose,
  onInstall,
  onUninstall,
  onStar,
}: TemplateDetailModalProps) {
  const { metadata } = template;
  const categoryStyle = CATEGORY_STYLES[metadata.category];
  const templateType = getTemplateType(template);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-bg/80 backdrop-blur-sm">
      <div className="w-full max-w-2xl max-h-[90vh] overflow-auto bg-surface border border-acid-green/30 m-4">
        {/* Header */}
        <div className="sticky top-0 bg-surface border-b border-acid-green/20 p-4 flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className={`px-2 py-0.5 text-xs font-mono ${categoryStyle.color} ${categoryStyle.bgColor}`}>
                {categoryStyle.icon} {metadata.category}
              </span>
              <span className="text-xs font-mono text-text-muted uppercase">{templateType}</span>
            </div>
            <h2 className="text-lg font-mono text-text font-bold">{metadata.name}</h2>
            <div className="text-xs font-mono text-text-muted">
              by {metadata.author} &middot; v{metadata.version}
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text transition-colors text-xl"
          >
            &times;
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Description */}
          <div>
            <h3 className="text-xs font-mono text-acid-cyan uppercase mb-2">Description</h3>
            <p className="text-sm font-mono text-text-muted">{metadata.description}</p>
          </div>

          {/* Stats */}
          <div className="flex gap-6">
            <div>
              <span className="text-xs font-mono text-text-muted">Downloads</span>
              <div className="text-lg font-mono text-acid-green">{metadata.downloads}</div>
            </div>
            <div>
              <span className="text-xs font-mono text-text-muted">Stars</span>
              <div className="text-lg font-mono text-yellow-400">{metadata.stars}</div>
            </div>
            <div>
              <span className="text-xs font-mono text-text-muted">License</span>
              <div className="text-sm font-mono text-text">{metadata.license}</div>
            </div>
          </div>

          {/* Tags */}
          <div>
            <h3 className="text-xs font-mono text-acid-cyan uppercase mb-2">Tags</h3>
            <div className="flex flex-wrap gap-1">
              {metadata.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-0.5 text-xs font-mono bg-bg/50 text-text-muted border border-border"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Template-specific content */}
          {templateType === 'agent' && 'system_prompt' in template && (
            <div>
              <h3 className="text-xs font-mono text-acid-cyan uppercase mb-2">System Prompt</h3>
              <pre className="p-3 bg-bg/50 border border-acid-green/10 text-xs font-mono text-text whitespace-pre-wrap max-h-48 overflow-auto">
                {template.system_prompt}
              </pre>
            </div>
          )}

          {templateType === 'debate' && 'protocol' in template && (
            <div>
              <h3 className="text-xs font-mono text-acid-cyan uppercase mb-2">Protocol</h3>
              <pre className="p-3 bg-bg/50 border border-acid-green/10 text-xs font-mono text-text whitespace-pre-wrap max-h-48 overflow-auto">
                {JSON.stringify(template.protocol, null, 2)}
              </pre>
            </div>
          )}

          {templateType === 'workflow' && 'nodes' in template && (
            <div>
              <h3 className="text-xs font-mono text-acid-cyan uppercase mb-2">
                Nodes ({template.nodes.length})
              </h3>
              <div className="space-y-1">
                {template.nodes.slice(0, 5).map((node, i) => (
                  <div key={i} className="text-xs font-mono text-text-muted">
                    {node.id}: {node.type}
                  </div>
                ))}
                {template.nodes.length > 5 && (
                  <div className="text-xs font-mono text-text-muted/60">
                    +{template.nodes.length - 5} more nodes
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Links */}
          {(metadata.repository_url || metadata.documentation_url) && (
            <div>
              <h3 className="text-xs font-mono text-acid-cyan uppercase mb-2">Links</h3>
              <div className="flex gap-4">
                {metadata.repository_url && (
                  <a
                    href={metadata.repository_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs font-mono text-acid-green hover:underline"
                  >
                    Repository
                  </a>
                )}
                {metadata.documentation_url && (
                  <a
                    href={metadata.documentation_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs font-mono text-acid-green hover:underline"
                  >
                    Documentation
                  </a>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="sticky bottom-0 bg-surface border-t border-acid-green/20 p-4 flex gap-2">
          <button
            onClick={isInstalled ? onUninstall : onInstall}
            className={`flex-1 py-2 font-mono font-bold transition-colors ${
              isInstalled
                ? 'bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20'
                : 'bg-acid-green text-bg hover:bg-acid-green/80'
            }`}
          >
            {isInstalled ? 'UNINSTALL' : 'INSTALL'}
          </button>
          <button
            onClick={onStar}
            className="px-4 py-2 font-mono bg-yellow-500/10 text-yellow-400 border border-yellow-500/30 hover:bg-yellow-500/20 transition-colors"
          >
            STAR
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Publish Modal
// ============================================================================

interface PublishModalProps {
  onClose: () => void;
}

function PublishModal({ onClose }: PublishModalProps) {
  const { publishTemplate, isLoading } = useMarketplaceStore();
  const { user } = useAuth();
  const [templateType, setTemplateType] = useState<TemplateType>('agent');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState<TemplateCategory>('custom');
  const [tags, setTags] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const metadata = {
      id: name.toLowerCase().replace(/\s+/g, '-'),
      name,
      description,
      version: '1.0.0',
      author: user?.name || user?.email || 'anonymous',
      category,
      tags: tags.split(',').map((t) => t.trim()).filter(Boolean),
      downloads: 0,
      stars: 0,
      license: 'MIT',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      repository_url: '',
      documentation_url: '',
    };

    // Basic template structure based on type
    const template =
      templateType === 'agent'
        ? { metadata, agent_type: 'claude', system_prompt: '', capabilities: [], constraints: [] }
        : templateType === 'debate'
        ? { metadata, task_template: '', agent_roles: [], protocol: { rounds: 3 }, evaluation_criteria: [] }
        : { metadata, nodes: [], edges: [], inputs: {}, outputs: {} };

    await publishTemplate(template);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-bg/80 backdrop-blur-sm">
      <div className="w-full max-w-lg bg-surface border border-acid-green/30 m-4">
        <div className="border-b border-acid-green/20 p-4 flex items-center justify-between">
          <h2 className="text-lg font-mono text-acid-green font-bold">PUBLISH TEMPLATE</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text transition-colors text-xl">
            &times;
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Template Type */}
          <div>
            <label className="text-xs font-mono text-acid-cyan uppercase block mb-2">Type</label>
            <div className="flex gap-2">
              {TYPES.filter((t) => t.value !== 'all').map((type) => (
                <button
                  key={type.value}
                  type="button"
                  onClick={() => setTemplateType(type.value as TemplateType)}
                  className={`px-3 py-1 text-xs font-mono border transition-colors ${
                    templateType === type.value
                      ? 'bg-acid-green/20 text-acid-green border-acid-green/50'
                      : 'bg-surface text-text-muted border-border hover:border-acid-green/30'
                  }`}
                >
                  {type.label}
                </button>
              ))}
            </div>
          </div>

          {/* Name */}
          <div>
            <label className="text-xs font-mono text-acid-cyan uppercase block mb-2">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              className="w-full px-3 py-2 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green"
              placeholder="My Awesome Template"
            />
          </div>

          {/* Description */}
          <div>
            <label className="text-xs font-mono text-acid-cyan uppercase block mb-2">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              required
              rows={3}
              className="w-full px-3 py-2 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green resize-none"
              placeholder="A brief description of what this template does..."
            />
          </div>

          {/* Category */}
          <div>
            <label className="text-xs font-mono text-acid-cyan uppercase block mb-2">Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value as TemplateCategory)}
              className="w-full px-3 py-2 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none"
            >
              {CATEGORIES.filter((c) => c.value !== 'all').map((cat) => (
                <option key={cat.value} value={cat.value}>
                  {cat.label}
                </option>
              ))}
            </select>
          </div>

          {/* Tags */}
          <div>
            <label className="text-xs font-mono text-acid-cyan uppercase block mb-2">Tags (comma-separated)</label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              className="w-full px-3 py-2 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green"
              placeholder="ai, productivity, coding"
            />
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={isLoading || !name || !description}
            className="w-full py-2 bg-acid-green text-bg font-mono font-bold hover:bg-acid-green/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'PUBLISHING...' : 'PUBLISH'}
          </button>
        </form>
      </div>
    </div>
  );
}

// ============================================================================
// Helpers
// ============================================================================

function getTemplateType(template: Template): TemplateType {
  if ('agent_type' in template) return 'agent';
  if ('task_template' in template) return 'debate';
  return 'workflow';
}
