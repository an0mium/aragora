'use client';

import { useCallback, useEffect, useRef } from 'react';
import { useDebounce } from './useTimers';
import { useBackend } from '@/components/BackendSelector';
import { useCommandPaletteStore } from '@/store/commandPaletteStore';
import type { SearchResult, SearchCategory, QuickAction } from '@/components/command-palette/types';
import { QUICK_ACTIONS } from '@/components/command-palette/types';

// Navigation pages for client-side search
const NAVIGATION_PAGES: SearchResult[] = [
  { id: 'page-hub', type: 'pages', title: 'Hub', subtitle: 'Main dashboard', href: '/hub', icon: '+' },
  { id: 'page-arena', type: 'pages', title: 'New Debate', subtitle: 'Start a debate', href: '/arena', icon: '!' },
  { id: 'page-debates', type: 'pages', title: 'Debates', subtitle: 'Browse past debates', href: '/debates', icon: '#' },
  { id: 'page-knowledge', type: 'pages', title: 'Knowledge', subtitle: 'Knowledge base', href: '/knowledge', icon: '?' },
  { id: 'page-leaderboard', type: 'pages', title: 'Leaderboard', subtitle: 'Agent rankings', href: '/leaderboard', icon: '^' },
  { id: 'page-agents', type: 'pages', title: 'Agents', subtitle: 'Agent recommender', href: '/agents', icon: '&' },
  { id: 'page-gallery', type: 'pages', title: 'Gallery', subtitle: 'Public debates', href: '/gallery', icon: '*' },
  { id: 'page-documents', type: 'pages', title: 'Documents', subtitle: 'Manage documents', href: '/documents', icon: ']' },
  { id: 'page-workflows', type: 'pages', title: 'Workflows', subtitle: 'Automation workflows', href: '/workflows', icon: '>' },
  { id: 'page-connectors', type: 'pages', title: 'Connectors', subtitle: 'Data connectors', href: '/connectors', icon: '<' },
  { id: 'page-analytics', type: 'pages', title: 'Analytics', subtitle: 'Usage analytics', href: '/analytics', icon: '~' },
  { id: 'page-templates', type: 'pages', title: 'Templates', subtitle: 'Debate templates', href: '/templates', icon: '[' },
  { id: 'page-settings', type: 'pages', title: 'Settings', subtitle: 'User settings', href: '/settings', icon: '*' },
  { id: 'page-integrations', type: 'pages', title: 'Integrations', subtitle: 'Platform integrations', href: '/integrations', icon: ':' },
  { id: 'page-verticals', type: 'pages', title: 'Verticals', subtitle: 'Industry solutions', href: '/verticals', icon: '/' },
  { id: 'page-memory', type: 'pages', title: 'Memory', subtitle: 'Memory explorer', href: '/memory', icon: '=' },
  { id: 'page-gauntlet', type: 'pages', title: 'Stress Test', subtitle: 'Decision gauntlet', href: '/gauntlet', icon: '%' },
  { id: 'page-reviews', type: 'pages', title: 'Code Reviews', subtitle: 'Security reviews', href: '/reviews', icon: '<' },
  { id: 'page-audit', type: 'pages', title: 'Document Audit', subtitle: 'Compliance audit', href: '/audit', icon: '|' },
  { id: 'page-security', type: 'pages', title: 'Security', subtitle: 'Security overview', href: '/security', icon: '!' },
  { id: 'page-compliance', type: 'pages', title: 'Compliance', subtitle: 'Compliance dashboard', href: '/compliance', icon: '%' },
  { id: 'page-research', type: 'pages', title: 'Research', subtitle: 'Research hub', href: '/research', icon: '?' },
  { id: 'page-decisions', type: 'pages', title: 'Decisions', subtitle: 'Decision tracker', href: '/decisions', icon: '^' },
  { id: 'page-receipts', type: 'pages', title: 'Receipts', subtitle: 'Audit trails', href: '/receipts', icon: '>' },
  { id: 'page-evidence', type: 'pages', title: 'Evidence', subtitle: 'Evidence chain', href: '/evidence', icon: '|' },
  { id: 'page-admin', type: 'pages', title: 'Admin', subtitle: 'Admin dashboard', href: '/admin', icon: '!' },
];

const DEBOUNCE_MS = 200;

/**
 * Search pages by query
 */
function searchPages(query: string): SearchResult[] {
  const lowerQuery = query.toLowerCase();
  return NAVIGATION_PAGES.filter(
    (page) =>
      page.title.toLowerCase().includes(lowerQuery) ||
      page.subtitle?.toLowerCase().includes(lowerQuery)
  ).slice(0, 8);
}

/**
 * Filter quick actions by query
 */
function filterQuickActions(query: string): QuickAction[] {
  const lowerQuery = query.toLowerCase();
  return QUICK_ACTIONS.filter(
    (action) =>
      action.label.toLowerCase().includes(lowerQuery) ||
      action.keywords.some((kw) => kw.toLowerCase().includes(lowerQuery)) ||
      action.description?.toLowerCase().includes(lowerQuery)
  ).slice(0, 5);
}

/**
 * Convert QuickAction to SearchResult format
 */
function quickActionToSearchResult(action: QuickAction): SearchResult {
  return {
    id: `action-${action.id}`,
    type: 'actions',
    title: action.label,
    subtitle: action.description,
    href: action.href,
    action: action.action,
    icon: action.icon,
    keywords: action.keywords,
  };
}

interface UseCommandPaletteSearchResult {
  search: (query: string, category: SearchCategory) => Promise<void>;
  results: SearchResult[];
  isSearching: boolean;
  error: string | null;
}

/**
 * useCommandPaletteSearch
 *
 * Hook for aggregating search results from multiple sources:
 * - Pages (client-side)
 * - Quick actions (client-side)
 * - Debates (API)
 * - Agents (API)
 * - Documents (API)
 * - Knowledge (API)
 */
export function useCommandPaletteSearch(): UseCommandPaletteSearchResult {
  const { config } = useBackend();
  const { results, isSearching, searchError, setResults, setIsSearching, setSearchError, query, activeCategory } =
    useCommandPaletteStore();

  const abortControllerRef = useRef<AbortController | null>(null);
  const debouncedQuery = useDebounce(query, DEBOUNCE_MS);

  /**
   * Search debates via API
   */
  const searchDebates = useCallback(
    async (q: string, signal: AbortSignal): Promise<SearchResult[]> => {
      try {
        const response = await fetch(
          `${config.api}/api/debates?limit=5&q=${encodeURIComponent(q)}`,
          { signal }
        );
        if (!response.ok) return [];
        const data = await response.json();
        return (data.debates || []).map((d: { id: string; task?: string; question?: string; consensus?: string }) => ({
          id: `debate-${d.id}`,
          type: 'debates' as SearchCategory,
          title: d.task || d.question || d.id,
          subtitle: d.consensus ? `Consensus: ${d.consensus}` : undefined,
          href: `/debates/${d.id}`,
          icon: '!',
        }));
      } catch {
        return [];
      }
    },
    [config.api]
  );

  /**
   * Search agents via API
   */
  const searchAgents = useCallback(
    async (q: string, signal: AbortSignal): Promise<SearchResult[]> => {
      try {
        const response = await fetch(
          `${config.api}/api/agents/configs?limit=5`,
          { signal }
        );
        if (!response.ok) return [];
        const data = await response.json();
        const agents = data.configs || data.agents || [];
        const lowerQ = q.toLowerCase();
        return agents
          .filter((a: { name?: string; display_name?: string }) =>
            a.name?.toLowerCase().includes(lowerQ) ||
            a.display_name?.toLowerCase().includes(lowerQ)
          )
          .slice(0, 5)
          .map((a: { name: string; display_name?: string; model?: string }) => ({
            id: `agent-${a.name}`,
            type: 'agents' as SearchCategory,
            title: a.display_name || a.name,
            subtitle: a.model || 'AI Agent',
            href: `/agents?agent=${a.name}`,
            icon: '&',
          }));
      } catch {
        return [];
      }
    },
    [config.api]
  );

  /**
   * Search documents via API
   */
  const searchDocuments = useCallback(
    async (q: string, signal: AbortSignal): Promise<SearchResult[]> => {
      try {
        const response = await fetch(
          `${config.api}/api/documents?limit=5&q=${encodeURIComponent(q)}`,
          { signal }
        );
        if (!response.ok) return [];
        const data = await response.json();
        return (data.documents || []).map((d: { id: string; filename?: string; name?: string; type?: string }) => ({
          id: `doc-${d.id}`,
          type: 'documents' as SearchCategory,
          title: d.filename || d.name || d.id,
          subtitle: d.type || 'Document',
          href: `/documents?id=${d.id}`,
          icon: ']',
        }));
      } catch {
        return [];
      }
    },
    [config.api]
  );

  /**
   * Search knowledge via API
   */
  const searchKnowledge = useCallback(
    async (q: string, signal: AbortSignal): Promise<SearchResult[]> => {
      try {
        const response = await fetch(`${config.api}/api/knowledge/mound/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q, limit: 5 }),
          signal,
        });
        if (!response.ok) return [];
        const data = await response.json();
        return (data.nodes || []).map((n: { id: string; content?: string; type?: string }) => ({
          id: `knowledge-${n.id}`,
          type: 'knowledge' as SearchCategory,
          title: n.content?.slice(0, 60) || n.id,
          subtitle: n.type || 'Knowledge',
          href: `/knowledge?node=${n.id}`,
          icon: '?',
        }));
      } catch {
        return [];
      }
    },
    [config.api]
  );

  /**
   * Aggregate search across all sources
   */
  const search = useCallback(
    async (q: string, category: SearchCategory) => {
      // Cancel any pending request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const trimmedQuery = q.trim();

      // If no query, show quick actions only
      if (!trimmedQuery) {
        const actions = QUICK_ACTIONS.slice(0, 6).map(quickActionToSearchResult);
        setResults(actions);
        setIsSearching(false);
        setSearchError(null);
        return;
      }

      setIsSearching(true);
      setSearchError(null);

      const controller = new AbortController();
      abortControllerRef.current = controller;

      try {
        const allResults: SearchResult[] = [];

        // Always search pages and actions (client-side, fast)
        if (category === 'all' || category === 'pages') {
          allResults.push(...searchPages(trimmedQuery));
        }
        if (category === 'all' || category === 'actions') {
          allResults.push(...filterQuickActions(trimmedQuery).map(quickActionToSearchResult));
        }

        // API searches in parallel
        const apiSearches: Promise<SearchResult[]>[] = [];

        if (category === 'all' || category === 'debates') {
          apiSearches.push(searchDebates(trimmedQuery, controller.signal));
        }
        if (category === 'all' || category === 'agents') {
          apiSearches.push(searchAgents(trimmedQuery, controller.signal));
        }
        if (category === 'all' || category === 'documents') {
          apiSearches.push(searchDocuments(trimmedQuery, controller.signal));
        }
        if (category === 'all' || category === 'knowledge') {
          apiSearches.push(searchKnowledge(trimmedQuery, controller.signal));
        }

        // Wait for all API searches
        const apiResults = await Promise.allSettled(apiSearches);

        // Collect successful results
        for (const result of apiResults) {
          if (result.status === 'fulfilled') {
            allResults.push(...result.value);
          }
        }

        // Sort by relevance (exact matches first)
        const lowerQuery = trimmedQuery.toLowerCase();
        allResults.sort((a, b) => {
          const aExact = a.title.toLowerCase() === lowerQuery ? 1 : 0;
          const bExact = b.title.toLowerCase() === lowerQuery ? 1 : 0;
          return bExact - aExact;
        });

        setResults(allResults.slice(0, 20));
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          // Request was cancelled, ignore
          return;
        }
        setSearchError('Search failed. Please try again.');
      } finally {
        setIsSearching(false);
      }
    },
    [
      searchDebates,
      searchAgents,
      searchDocuments,
      searchKnowledge,
      setResults,
      setIsSearching,
      setSearchError,
    ]
  );

  // Trigger search when debounced query changes
  useEffect(() => {
    search(debouncedQuery, activeCategory);
  }, [debouncedQuery, activeCategory, search]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    search,
    results,
    isSearching,
    error: searchError,
  };
}

export default useCommandPaletteSearch;
