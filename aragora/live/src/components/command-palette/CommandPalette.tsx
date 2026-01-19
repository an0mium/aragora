'use client';

import { useCallback, useMemo, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useFocusTrap } from '@/hooks/useFocusTrap';
import { useCommandPaletteStore } from '@/store/commandPaletteStore';
import { useCommandPaletteSearch } from '@/hooks/useCommandPaletteSearch';
import { CommandPaletteInput } from './CommandPaletteInput';
import { CommandPaletteResults } from './CommandPaletteResults';
import type { SearchResult, RecentItem, QuickAction, SearchCategory } from './types';
import { CATEGORIES, QUICK_ACTIONS } from './types';

/**
 * CommandPalette
 *
 * Global command palette modal accessible via Cmd+K.
 * Provides search across debates, agents, documents, knowledge, pages, and quick actions.
 */
export function CommandPalette() {
  const router = useRouter();
  const {
    isOpen,
    close,
    query,
    setQuery,
    activeCategory,
    setActiveCategory,
    selectedIndex,
    setSelectedIndex,
    moveUp,
    moveDown,
    recentItems,
    addRecentItem,
    results,
    isSearching,
    searchError,
  } = useCommandPaletteStore();

  // Initialize search hook
  useCommandPaletteSearch();

  // Focus trap ref
  const focusTrapRef = useFocusTrap<HTMLDivElement>({
    isActive: isOpen,
    onEscape: close,
    returnFocusOnDeactivate: true,
  });

  // Build sections for results
  const sections = useMemo(() => {
    const sectionList: {
      id: string;
      title: string;
      items: (SearchResult | RecentItem | QuickAction)[];
    }[] = [];

    // Show recent items when no query
    if (!query.trim() && recentItems.length > 0) {
      sectionList.push({
        id: 'recent',
        title: 'Recent',
        items: recentItems,
      });
    }

    // Show quick actions when no query
    if (!query.trim()) {
      sectionList.push({
        id: 'actions',
        title: 'Quick Actions',
        items: QUICK_ACTIONS.slice(0, 6),
      });
    }

    // Group results by type when searching
    if (query.trim() && results.length > 0) {
      const groupedResults: Record<string, SearchResult[]> = {};

      for (const result of results) {
        const type = result.type || 'other';
        if (!groupedResults[type]) {
          groupedResults[type] = [];
        }
        groupedResults[type].push(result);
      }

      // Add sections in category order
      const categoryOrder: SearchCategory[] = ['pages', 'actions', 'debates', 'agents', 'documents', 'knowledge'];

      for (const category of categoryOrder) {
        const items = groupedResults[category];
        if (items && items.length > 0) {
          const categoryConfig = CATEGORIES.find((c) => c.id === category);
          sectionList.push({
            id: category,
            title: categoryConfig?.label || category,
            items,
          });
        }
      }
    }

    return sectionList;
  }, [query, recentItems, results]);

  // Calculate total items across all sections
  const totalItems = sections.reduce((sum, s) => sum + s.items.length, 0);

  // Handle item selection
  const handleSelect = useCallback(
    (item: SearchResult | RecentItem | QuickAction, _index: number) => {
      // Add to recent items
      const recentItem: Omit<RecentItem, 'timestamp'> = {
        id: item.id,
        type: 'type' in item ? item.type : 'actions',
        title: 'label' in item ? item.label : item.title,
        subtitle: 'description' in item ? item.description : 'subtitle' in item ? item.subtitle : undefined,
        href: item.href,
        icon: item.icon,
      };
      addRecentItem(recentItem);

      // Execute action or navigate
      if ('action' in item && item.action) {
        item.action();
        close();
      } else if (item.href) {
        router.push(item.href);
        close();
      }
    },
    [router, close, addRecentItem]
  );

  // Get selected item
  const getSelectedItem = useCallback((): SearchResult | RecentItem | QuickAction | null => {
    let currentIndex = 0;
    for (const section of sections) {
      for (const item of section.items) {
        if (currentIndex === selectedIndex) {
          return item;
        }
        currentIndex++;
      }
    }
    return null;
  }, [sections, selectedIndex]);

  // Handle keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          moveDown();
          break;
        case 'ArrowUp':
          e.preventDefault();
          moveUp();
          break;
        case 'Enter': {
          e.preventDefault();
          const item = getSelectedItem();
          if (item) {
            handleSelect(item, selectedIndex);
          }
          break;
        }
        case 'Tab': {
          e.preventDefault();
          // Cycle through categories
          const currentIdx = CATEGORIES.findIndex((c) => c.id === activeCategory);
          const nextIdx = e.shiftKey
            ? (currentIdx - 1 + CATEGORIES.length) % CATEGORIES.length
            : (currentIdx + 1) % CATEGORIES.length;
          setActiveCategory(CATEGORIES[nextIdx].id);
          break;
        }
      }
    },
    [moveUp, moveDown, getSelectedItem, handleSelect, selectedIndex, activeCategory, setActiveCategory]
  );

  // Handle hover
  const handleHover = useCallback(
    (index: number) => {
      setSelectedIndex(index);
    },
    [setSelectedIndex]
  );

  // Click outside to close
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const modal = document.getElementById('command-palette-modal');
      if (modal && !modal.contains(target)) {
        close();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, close]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] bg-bg/80 backdrop-blur-sm">
      <div
        id="command-palette-modal"
        ref={focusTrapRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="command-palette-title"
        className="w-full max-w-2xl mx-4 border border-acid-green/40 bg-surface shadow-2xl shadow-acid-green/10 rounded-sm overflow-hidden"
      >
        {/* Hidden title for accessibility */}
        <h2 id="command-palette-title" className="sr-only">
          Command Palette
        </h2>

        {/* Search input */}
        <CommandPaletteInput
          value={query}
          onChange={setQuery}
          onKeyDown={handleKeyDown}
          isSearching={isSearching}
          resultCount={totalItems}
          selectedIndex={selectedIndex}
        />

        {/* Category tabs */}
        <div className="flex gap-1 px-4 py-2 border-b border-acid-green/10 bg-bg/30 overflow-x-auto">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              onClick={() => setActiveCategory(cat.id)}
              className={`
                px-3 py-1 text-xs font-mono transition-colors rounded-sm
                ${activeCategory === cat.id
                  ? 'bg-acid-green/20 text-acid-green border border-acid-green/40'
                  : 'text-text-muted hover:text-text border border-transparent'
                }
              `}
            >
              <span className="mr-1">{cat.icon}</span>
              {cat.label}
            </button>
          ))}
        </div>

        {/* Results */}
        <CommandPaletteResults
          sections={sections}
          selectedIndex={selectedIndex}
          onSelect={handleSelect}
          onHover={handleHover}
          isSearching={isSearching}
          searchError={searchError}
          query={query}
        />

        {/* Footer with keyboard hints */}
        <div className="px-4 py-2 border-t border-acid-green/10 bg-bg/30 flex gap-4 text-xs font-mono text-text-muted">
          <span>
            <kbd className="text-acid-green">↑↓</kbd> navigate
          </span>
          <span>
            <kbd className="text-acid-green">↵</kbd> select
          </span>
          <span>
            <kbd className="text-acid-green">tab</kbd> category
          </span>
          <span className="ml-auto">
            <kbd className="text-acid-green">⌘K</kbd> toggle
          </span>
        </div>
      </div>
    </div>
  );
}

export default CommandPalette;
