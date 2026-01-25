/**
 * Tests for commandPaletteStore
 *
 * Tests cover:
 * - Modal open/close/toggle actions
 * - Query and search state
 * - Category selection
 * - Results management
 * - Navigation (moveUp, moveDown, selectedIndex)
 * - Recent items (add, remove, clear, max limit)
 * - Reset functionality
 * - Selectors
 */

import { act } from '@testing-library/react';
import {
  useCommandPaletteStore,
  selectIsOpen,
  selectQuery,
  selectActiveCategory,
  selectSelectedIndex,
  selectResults,
  selectIsSearching,
  selectSearchError,
  selectRecentItems,
} from '../commandPaletteStore';

// Sample test data
const mockSearchResult = {
  id: 'result-1',
  type: 'debates' as const,
  title: 'Test Debate',
  subtitle: 'A test debate',
  href: '/debates/1',
};

const mockRecentItem = {
  id: 'recent-1',
  type: 'debates' as const,
  title: 'Recent Debate',
  href: '/debates/recent',
};

describe('commandPaletteStore', () => {
  beforeEach(() => {
    act(() => {
      useCommandPaletteStore.getState().reset();
      useCommandPaletteStore.getState().clearRecentItems();
    });
  });

  describe('Initial State', () => {
    it('starts closed with empty query', () => {
      const state = useCommandPaletteStore.getState();
      expect(state.isOpen).toBe(false);
      expect(state.query).toBe('');
      expect(state.activeCategory).toBe('all');
      expect(state.selectedIndex).toBe(0);
      expect(state.results).toEqual([]);
      expect(state.isSearching).toBe(false);
      expect(state.searchError).toBeNull();
    });
  });

  describe('Modal Actions', () => {
    it('open sets isOpen to true and resets state', () => {
      // First set some state
      act(() => {
        useCommandPaletteStore.getState().setQuery('test');
        useCommandPaletteStore.getState().setSelectedIndex(5);
      });

      act(() => {
        useCommandPaletteStore.getState().open();
      });

      const state = useCommandPaletteStore.getState();
      expect(state.isOpen).toBe(true);
      expect(state.query).toBe('');
      expect(state.selectedIndex).toBe(0);
      expect(state.searchError).toBeNull();
    });

    it('close sets isOpen to false and resets state', () => {
      act(() => {
        useCommandPaletteStore.getState().open();
        useCommandPaletteStore.getState().setQuery('test');
      });

      act(() => {
        useCommandPaletteStore.getState().close();
      });

      const state = useCommandPaletteStore.getState();
      expect(state.isOpen).toBe(false);
      expect(state.query).toBe('');
      expect(state.selectedIndex).toBe(0);
    });

    it('toggle switches between open and closed', () => {
      expect(useCommandPaletteStore.getState().isOpen).toBe(false);

      act(() => {
        useCommandPaletteStore.getState().toggle();
      });
      expect(useCommandPaletteStore.getState().isOpen).toBe(true);

      act(() => {
        useCommandPaletteStore.getState().toggle();
      });
      expect(useCommandPaletteStore.getState().isOpen).toBe(false);
    });
  });

  describe('Search State', () => {
    it('setQuery updates query and resets selectedIndex', () => {
      act(() => {
        useCommandPaletteStore.getState().setSelectedIndex(3);
        useCommandPaletteStore.getState().setQuery('search term');
      });

      const state = useCommandPaletteStore.getState();
      expect(state.query).toBe('search term');
      expect(state.selectedIndex).toBe(0);
    });

    it('setActiveCategory updates category and resets selectedIndex', () => {
      act(() => {
        useCommandPaletteStore.getState().setSelectedIndex(5);
        useCommandPaletteStore.getState().setActiveCategory('agents');
      });

      const state = useCommandPaletteStore.getState();
      expect(state.activeCategory).toBe('agents');
      expect(state.selectedIndex).toBe(0);
    });

    it('setSelectedIndex updates index', () => {
      act(() => {
        useCommandPaletteStore.getState().setSelectedIndex(7);
      });

      expect(useCommandPaletteStore.getState().selectedIndex).toBe(7);
    });

    it('setIsSearching updates searching state', () => {
      act(() => {
        useCommandPaletteStore.getState().setIsSearching(true);
      });

      expect(useCommandPaletteStore.getState().isSearching).toBe(true);

      act(() => {
        useCommandPaletteStore.getState().setIsSearching(false);
      });

      expect(useCommandPaletteStore.getState().isSearching).toBe(false);
    });

    it('setSearchError updates error state', () => {
      act(() => {
        useCommandPaletteStore.getState().setSearchError('Search failed');
      });

      expect(useCommandPaletteStore.getState().searchError).toBe('Search failed');

      act(() => {
        useCommandPaletteStore.getState().setSearchError(null);
      });

      expect(useCommandPaletteStore.getState().searchError).toBeNull();
    });
  });

  describe('Results', () => {
    it('setResults updates results and resets selectedIndex', () => {
      act(() => {
        useCommandPaletteStore.getState().setSelectedIndex(3);
        useCommandPaletteStore.getState().setResults([mockSearchResult]);
      });

      const state = useCommandPaletteStore.getState();
      expect(state.results).toHaveLength(1);
      expect(state.results[0]).toEqual(mockSearchResult);
      expect(state.selectedIndex).toBe(0);
    });

    it('setResults can clear results', () => {
      act(() => {
        useCommandPaletteStore.getState().setResults([mockSearchResult]);
        useCommandPaletteStore.getState().setResults([]);
      });

      expect(useCommandPaletteStore.getState().results).toEqual([]);
    });
  });

  describe('Navigation', () => {
    describe('with results', () => {
      beforeEach(() => {
        act(() => {
          useCommandPaletteStore.getState().setQuery('test');
          useCommandPaletteStore.getState().setResults([
            { ...mockSearchResult, id: '1' },
            { ...mockSearchResult, id: '2' },
            { ...mockSearchResult, id: '3' },
          ]);
        });
      });

      it('moveDown increments selectedIndex', () => {
        expect(useCommandPaletteStore.getState().selectedIndex).toBe(0);

        act(() => {
          useCommandPaletteStore.getState().moveDown();
        });

        expect(useCommandPaletteStore.getState().selectedIndex).toBe(1);
      });

      it('moveDown wraps to 0 at end', () => {
        act(() => {
          useCommandPaletteStore.getState().setSelectedIndex(2);
          useCommandPaletteStore.getState().moveDown();
        });

        expect(useCommandPaletteStore.getState().selectedIndex).toBe(0);
      });

      it('moveUp decrements selectedIndex', () => {
        act(() => {
          useCommandPaletteStore.getState().setSelectedIndex(2);
          useCommandPaletteStore.getState().moveUp();
        });

        expect(useCommandPaletteStore.getState().selectedIndex).toBe(1);
      });

      it('moveUp wraps to last at beginning', () => {
        expect(useCommandPaletteStore.getState().selectedIndex).toBe(0);

        act(() => {
          useCommandPaletteStore.getState().moveUp();
        });

        expect(useCommandPaletteStore.getState().selectedIndex).toBe(2);
      });
    });

    describe('with no results', () => {
      it('moveDown does nothing with no results', () => {
        expect(useCommandPaletteStore.getState().selectedIndex).toBe(0);

        act(() => {
          useCommandPaletteStore.getState().setQuery('test'); // No results with query
          useCommandPaletteStore.getState().moveDown();
        });

        expect(useCommandPaletteStore.getState().selectedIndex).toBe(0);
      });

      it('moveUp does nothing with no results', () => {
        expect(useCommandPaletteStore.getState().selectedIndex).toBe(0);

        act(() => {
          useCommandPaletteStore.getState().setQuery('test');
          useCommandPaletteStore.getState().moveUp();
        });

        expect(useCommandPaletteStore.getState().selectedIndex).toBe(0);
      });
    });

    describe('with recent items and no query', () => {
      beforeEach(() => {
        act(() => {
          useCommandPaletteStore.getState().addRecentItem(mockRecentItem);
          useCommandPaletteStore.getState().setResults([mockSearchResult]);
        });
      });

      it('includes recent items in total count for navigation', () => {
        // Should have 1 recent + 1 result = 2 total items
        act(() => {
          useCommandPaletteStore.getState().setSelectedIndex(1);
          useCommandPaletteStore.getState().moveDown();
        });

        // Should wrap to 0
        expect(useCommandPaletteStore.getState().selectedIndex).toBe(0);
      });
    });
  });

  describe('Recent Items', () => {
    it('addRecentItem adds item with timestamp', () => {
      const now = Date.now();
      jest.spyOn(Date, 'now').mockReturnValue(now);

      act(() => {
        useCommandPaletteStore.getState().addRecentItem(mockRecentItem);
      });

      const recentItems = useCommandPaletteStore.getState().recentItems;
      expect(recentItems).toHaveLength(1);
      expect(recentItems[0].id).toBe(mockRecentItem.id);
      expect(recentItems[0].timestamp).toBe(now);

      jest.restoreAllMocks();
    });

    it('addRecentItem moves existing item to front', () => {
      act(() => {
        useCommandPaletteStore.getState().addRecentItem({ ...mockRecentItem, id: 'item-1' });
        useCommandPaletteStore.getState().addRecentItem({ ...mockRecentItem, id: 'item-2' });
        useCommandPaletteStore.getState().addRecentItem({ ...mockRecentItem, id: 'item-1' }); // Re-add first
      });

      const recentItems = useCommandPaletteStore.getState().recentItems;
      expect(recentItems).toHaveLength(2);
      expect(recentItems[0].id).toBe('item-1');
      expect(recentItems[1].id).toBe('item-2');
    });

    it('addRecentItem respects max limit of 10', () => {
      act(() => {
        for (let i = 0; i < 15; i++) {
          useCommandPaletteStore.getState().addRecentItem({ ...mockRecentItem, id: `item-${i}` });
        }
      });

      const recentItems = useCommandPaletteStore.getState().recentItems;
      expect(recentItems).toHaveLength(10);
      // Most recent should be first
      expect(recentItems[0].id).toBe('item-14');
      // Oldest kept should be item-5 (items 0-4 were pushed out)
      expect(recentItems[9].id).toBe('item-5');
    });

    it('removeRecentItem removes specific item', () => {
      act(() => {
        useCommandPaletteStore.getState().addRecentItem({ ...mockRecentItem, id: 'keep' });
        useCommandPaletteStore.getState().addRecentItem({ ...mockRecentItem, id: 'remove' });
        useCommandPaletteStore.getState().removeRecentItem('remove');
      });

      const recentItems = useCommandPaletteStore.getState().recentItems;
      expect(recentItems).toHaveLength(1);
      expect(recentItems[0].id).toBe('keep');
    });

    it('clearRecentItems removes all items', () => {
      act(() => {
        useCommandPaletteStore.getState().addRecentItem({ ...mockRecentItem, id: 'item-1' });
        useCommandPaletteStore.getState().addRecentItem({ ...mockRecentItem, id: 'item-2' });
        useCommandPaletteStore.getState().clearRecentItems();
      });

      expect(useCommandPaletteStore.getState().recentItems).toEqual([]);
    });
  });

  describe('Reset', () => {
    it('resets all state except recent items', () => {
      act(() => {
        useCommandPaletteStore.getState().open();
        useCommandPaletteStore.getState().setQuery('test');
        useCommandPaletteStore.getState().setActiveCategory('agents');
        useCommandPaletteStore.getState().setSelectedIndex(5);
        useCommandPaletteStore.getState().setResults([mockSearchResult]);
        useCommandPaletteStore.getState().setIsSearching(true);
        useCommandPaletteStore.getState().setSearchError('error');
        useCommandPaletteStore.getState().addRecentItem(mockRecentItem);
        useCommandPaletteStore.getState().reset();
      });

      const state = useCommandPaletteStore.getState();
      expect(state.isOpen).toBe(false);
      expect(state.query).toBe('');
      expect(state.activeCategory).toBe('all');
      expect(state.selectedIndex).toBe(0);
      expect(state.results).toEqual([]);
      expect(state.isSearching).toBe(false);
      expect(state.searchError).toBeNull();
      // Recent items should be preserved
      expect(state.recentItems).toHaveLength(1);
    });
  });

  describe('Selectors', () => {
    beforeEach(() => {
      act(() => {
        useCommandPaletteStore.getState().open();
        useCommandPaletteStore.getState().setQuery('test query');
        useCommandPaletteStore.getState().setActiveCategory('documents');
        useCommandPaletteStore.getState().setResults([mockSearchResult]);
        // Set selectedIndex after setResults since setResults resets it to 0
        useCommandPaletteStore.getState().setSelectedIndex(2);
        useCommandPaletteStore.getState().setIsSearching(true);
        useCommandPaletteStore.getState().setSearchError('test error');
        useCommandPaletteStore.getState().addRecentItem(mockRecentItem);
      });
    });

    it('selectIsOpen returns isOpen', () => {
      expect(selectIsOpen(useCommandPaletteStore.getState())).toBe(true);
    });

    it('selectQuery returns query', () => {
      expect(selectQuery(useCommandPaletteStore.getState())).toBe('test query');
    });

    it('selectActiveCategory returns activeCategory', () => {
      expect(selectActiveCategory(useCommandPaletteStore.getState())).toBe('documents');
    });

    it('selectSelectedIndex returns selectedIndex', () => {
      expect(selectSelectedIndex(useCommandPaletteStore.getState())).toBe(2);
    });

    it('selectResults returns results', () => {
      expect(selectResults(useCommandPaletteStore.getState())).toEqual([mockSearchResult]);
    });

    it('selectIsSearching returns isSearching', () => {
      expect(selectIsSearching(useCommandPaletteStore.getState())).toBe(true);
    });

    it('selectSearchError returns searchError', () => {
      expect(selectSearchError(useCommandPaletteStore.getState())).toBe('test error');
    });

    it('selectRecentItems returns recentItems', () => {
      const recentItems = selectRecentItems(useCommandPaletteStore.getState());
      expect(recentItems).toHaveLength(1);
      expect(recentItems[0].id).toBe(mockRecentItem.id);
    });
  });
});
