'use client';

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { API_BASE_URL } from '@/config';

// ============================================================================
// Types - Maps to aragora/marketplace/models.py
// ============================================================================

export type TemplateCategory =
  | 'analysis'
  | 'coding'
  | 'creative'
  | 'debate'
  | 'research'
  | 'decision'
  | 'brainstorm'
  | 'review'
  | 'planning'
  | 'custom';

export type TemplateType = 'agent' | 'debate' | 'workflow';

export interface TemplateMetadata {
  id: string;
  name: string;
  description: string;
  version: string;
  author: string;
  category: TemplateCategory;
  tags: string[];
  created_at: string;
  updated_at: string;
  downloads: number;
  stars: number;
  license: string;
  repository_url: string | null;
  documentation_url: string | null;
}

export interface AgentTemplate {
  metadata: TemplateMetadata;
  agent_type: string;
  system_prompt: string;
  model_config: Record<string, unknown>;
  capabilities: string[];
  constraints: string[];
  examples: Array<{ input: string; output: string }>;
  content_hash: string;
}

export interface DebateTemplate {
  metadata: TemplateMetadata;
  task_template: string;
  agent_roles: Array<{ role: string; [key: string]: unknown }>;
  protocol: {
    rounds: number;
    consensus_mode: string;
    [key: string]: unknown;
  };
  evaluation_criteria: string[];
  success_metrics: Record<string, number>;
  content_hash: string;
}

export interface WorkflowTemplate {
  metadata: TemplateMetadata;
  nodes: Array<{ id: string; type: string; [key: string]: unknown }>;
  edges: Array<{ source: string; target: string }>;
  inputs: Record<string, { type: string; description?: string }>;
  outputs: Record<string, { type: string; description?: string }>;
  variables: Record<string, unknown>;
  content_hash: string;
}

export type Template = AgentTemplate | DebateTemplate | WorkflowTemplate;

export interface TemplateRating {
  user_id: string;
  template_id: string;
  score: number;
  review: string | null;
  created_at: string;
}

// Category styling
export const CATEGORY_STYLES: Record<TemplateCategory, { color: string; bgColor: string; icon: string }> = {
  analysis: { color: 'text-blue-400', bgColor: 'bg-blue-500/10', icon: '📊' },
  coding: { color: 'text-green-400', bgColor: 'bg-green-500/10', icon: '💻' },
  creative: { color: 'text-purple-400', bgColor: 'bg-purple-500/10', icon: '🎨' },
  debate: { color: 'text-red-400', bgColor: 'bg-red-500/10', icon: '⚔️' },
  research: { color: 'text-cyan-400', bgColor: 'bg-cyan-500/10', icon: '🔬' },
  decision: { color: 'text-yellow-400', bgColor: 'bg-yellow-500/10', icon: '⚖️' },
  brainstorm: { color: 'text-orange-400', bgColor: 'bg-orange-500/10', icon: '💡' },
  review: { color: 'text-teal-400', bgColor: 'bg-teal-500/10', icon: '🔍' },
  planning: { color: 'text-indigo-400', bgColor: 'bg-indigo-500/10', icon: '📋' },
  custom: { color: 'text-gray-400', bgColor: 'bg-gray-500/10', icon: '🔧' },
};

export const TYPE_LABELS: Record<TemplateType, string> = {
  agent: 'Agent',
  debate: 'Debate',
  workflow: 'Workflow',
};

// ============================================================================
// Store State
// ============================================================================

interface MarketplaceState {
  // Templates
  templates: Template[];
  featuredTemplates: Template[];

  // Filters
  selectedCategory: TemplateCategory | 'all';
  selectedType: TemplateType | 'all';
  searchQuery: string;
  sortBy: 'downloads' | 'stars' | 'recent';

  // UI
  isLoading: boolean;
  error: string | null;
  selectedTemplate: Template | null;

  // User's templates
  myTemplates: Template[];
  installedTemplates: string[]; // template IDs
}

interface MarketplaceActions {
  // Fetch actions
  fetchTemplates: () => Promise<void>;
  fetchFeatured: () => Promise<void>;
  fetchMyTemplates: () => Promise<void>;
  fetchTemplateById: (id: string) => Promise<Template | null>;

  // Filter actions
  setCategory: (category: TemplateCategory | 'all') => void;
  setType: (type: TemplateType | 'all') => void;
  setSearchQuery: (query: string) => void;
  setSortBy: (sort: 'downloads' | 'stars' | 'recent') => void;

  // Template actions
  installTemplate: (templateId: string) => Promise<void>;
  uninstallTemplate: (templateId: string) => Promise<void>;
  starTemplate: (templateId: string) => Promise<void>;

  // Publishing
  publishTemplate: (template: Partial<Template>) => Promise<void>;
  updateTemplate: (id: string, updates: Partial<Template>) => Promise<void>;
  deleteTemplate: (id: string) => Promise<void>;

  // UI actions
  selectTemplate: (template: Template | null) => void;

  // Computed
  getFilteredTemplates: () => Template[];

  // Reset
  reset: () => void;
}

// ============================================================================
// Initial State
// ============================================================================

const initialState: MarketplaceState = {
  templates: [],
  featuredTemplates: [],
  selectedCategory: 'all',
  selectedType: 'all',
  searchQuery: '',
  sortBy: 'downloads',
  isLoading: false,
  error: null,
  selectedTemplate: null,
  myTemplates: [],
  installedTemplates: [],
};

// ============================================================================
// API Helpers
// ============================================================================

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `API error: ${response.status}`);
  }

  return response.json();
}

// ============================================================================
// Store
// ============================================================================

export const useMarketplaceStore = create<MarketplaceState & MarketplaceActions>()(
  devtools(
    (set, get) => ({
      ...initialState,

      // Fetch actions
      fetchTemplates: async () => {
        set({ isLoading: true, error: null });
        try {
          // Try v1 browse endpoint first, then fall back to legacy
          let templates: Template[];
          try {
            const data = await fetchApi<{ templates?: Template[] } | Template[]>('/api/v1/marketplace/browse');
            templates = Array.isArray(data) ? data : (data.templates ?? []);
          } catch {
            templates = await fetchApi<Template[]>('/api/marketplace/templates');
          }
          set({ templates, isLoading: false });
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Failed to fetch templates',
            isLoading: false,
          });
        }
      },

      fetchFeatured: async () => {
        try {
          let featured: Template[];
          try {
            const data = await fetchApi<{ templates?: Template[] } | Template[]>('/api/v1/marketplace/featured');
            featured = Array.isArray(data) ? data : (data.templates ?? []);
          } catch {
            featured = await fetchApi<Template[]>('/api/marketplace/templates/featured');
          }
          set({ featuredTemplates: featured });
        } catch {
          // Featured is non-critical
        }
      },

      fetchMyTemplates: async () => {
        try {
          const myTemplates = await fetchApi<Template[]>('/api/marketplace/templates/mine');
          set({ myTemplates });
        } catch {
          // My templates is non-critical
        }
      },

      fetchTemplateById: async (id: string) => {
        try {
          const template = await fetchApi<Template>(`/api/marketplace/templates/${id}`);
          return template;
        } catch {
          return null;
        }
      },

      // Filter actions
      setCategory: (selectedCategory) => {
        set({ selectedCategory });
      },

      setType: (selectedType) => {
        set({ selectedType });
      },

      setSearchQuery: (searchQuery) => {
        set({ searchQuery });
      },

      setSortBy: (sortBy) => {
        set({ sortBy });
      },

      // Template actions
      installTemplate: async (templateId: string) => {
        try {
          await fetchApi(`/api/marketplace/templates/${templateId}/install`, {
            method: 'POST',
          });
          set((state) => ({
            installedTemplates: [...state.installedTemplates, templateId],
          }));
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Failed to install template',
          });
        }
      },

      uninstallTemplate: async (templateId: string) => {
        try {
          await fetchApi(`/api/marketplace/templates/${templateId}/uninstall`, {
            method: 'POST',
          });
          set((state) => ({
            installedTemplates: state.installedTemplates.filter((id) => id !== templateId),
          }));
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Failed to uninstall template',
          });
        }
      },

      starTemplate: async (templateId: string) => {
        try {
          await fetchApi(`/api/marketplace/templates/${templateId}/star`, {
            method: 'POST',
          });
          // Optimistically update the star count
          set((state) => ({
            templates: state.templates.map((t) =>
              t.metadata.id === templateId
                ? { ...t, metadata: { ...t.metadata, stars: t.metadata.stars + 1 } }
                : t
            ),
          }));
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Failed to star template',
          });
        }
      },

      // Publishing
      publishTemplate: async (template: Partial<Template>) => {
        set({ isLoading: true, error: null });
        try {
          const published = await fetchApi<Template>('/api/marketplace/templates', {
            method: 'POST',
            body: JSON.stringify(template),
          });
          set((state) => ({
            templates: [published, ...state.templates],
            myTemplates: [published, ...state.myTemplates],
            isLoading: false,
          }));
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Failed to publish template',
            isLoading: false,
          });
        }
      },

      updateTemplate: async (id: string, updates: Partial<Template>) => {
        try {
          const updated = await fetchApi<Template>(`/api/marketplace/templates/${id}`, {
            method: 'PUT',
            body: JSON.stringify(updates),
          });
          set((state) => ({
            templates: state.templates.map((t) => (t.metadata.id === id ? updated : t)),
            myTemplates: state.myTemplates.map((t) => (t.metadata.id === id ? updated : t)),
          }));
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Failed to update template',
          });
        }
      },

      deleteTemplate: async (id: string) => {
        try {
          await fetchApi(`/api/marketplace/templates/${id}`, {
            method: 'DELETE',
          });
          set((state) => ({
            templates: state.templates.filter((t) => t.metadata.id !== id),
            myTemplates: state.myTemplates.filter((t) => t.metadata.id !== id),
          }));
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Failed to delete template',
          });
        }
      },

      // UI actions
      selectTemplate: (selectedTemplate) => {
        set({ selectedTemplate });
      },

      // Computed
      getFilteredTemplates: () => {
        const { templates, selectedCategory, selectedType, searchQuery, sortBy } = get();

        let filtered = [...templates];

        // Filter by category
        if (selectedCategory !== 'all') {
          filtered = filtered.filter((t) => t.metadata.category === selectedCategory);
        }

        // Filter by type
        if (selectedType !== 'all') {
          filtered = filtered.filter((t) => {
            if (selectedType === 'agent') return 'agent_type' in t;
            if (selectedType === 'debate') return 'task_template' in t;
            if (selectedType === 'workflow') return 'nodes' in t;
            return true;
          });
        }

        // Filter by search query
        if (searchQuery) {
          const query = searchQuery.toLowerCase();
          filtered = filtered.filter(
            (t) =>
              t.metadata.name.toLowerCase().includes(query) ||
              t.metadata.description.toLowerCase().includes(query) ||
              t.metadata.tags.some((tag) => tag.toLowerCase().includes(query))
          );
        }

        // Sort
        filtered.sort((a, b) => {
          switch (sortBy) {
            case 'downloads':
              return b.metadata.downloads - a.metadata.downloads;
            case 'stars':
              return b.metadata.stars - a.metadata.stars;
            case 'recent':
              return new Date(b.metadata.updated_at).getTime() - new Date(a.metadata.updated_at).getTime();
            default:
              return 0;
          }
        });

        return filtered;
      },

      // Reset
      reset: () => {
        set(initialState);
      },
    }),
    { name: 'marketplace-store' }
  )
);
