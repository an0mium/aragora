/**
 * AsyncPaginator Tests
 *
 * Comprehensive tests for the pagination helpers, covering:
 * - Auto-pagination across multiple pages
 * - Total count tracking
 * - Empty results handling
 * - Error handling during pagination
 * - Different response formats (items, data, raw lists)
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AsyncPaginator, paginate, paginateWith, type FetchPage } from '../pagination';
import type { AragoraClient } from '../client';

// Mock interface for test items
interface TestItem {
  id: number;
  name: string;
}

// Create a mock client
function createMockClient(requestFn: ReturnType<typeof vi.fn>): AragoraClient {
  return {
    request: requestFn,
  } as unknown as AragoraClient;
}

describe('AsyncPaginator', () => {
  let mockRequest: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockRequest = vi.fn();
  });

  describe('basic iteration', () => {
    it('should iterate through a single page of results', async () => {
      const items = [
        { id: 1, name: 'Item 1' },
        { id: 2, name: 'Item 2' },
      ];

      mockRequest.mockResolvedValueOnce({ items, total: 2 });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items', { pageSize: 10 });

      const results: TestItem[] = [];
      for await (const item of paginator) {
        results.push(item);
      }

      expect(results).toHaveLength(2);
      expect(results[0].id).toBe(1);
      expect(results[1].id).toBe(2);
      expect(mockRequest).toHaveBeenCalledTimes(1);
    });

    it('should auto-paginate across multiple pages', async () => {
      // Page 1
      mockRequest.mockResolvedValueOnce({
        items: [{ id: 1, name: 'Item 1' }, { id: 2, name: 'Item 2' }],
        total: 5,
      });

      // Page 2
      mockRequest.mockResolvedValueOnce({
        items: [{ id: 3, name: 'Item 3' }, { id: 4, name: 'Item 4' }],
        total: 5,
      });

      // Page 3 (partial)
      mockRequest.mockResolvedValueOnce({
        items: [{ id: 5, name: 'Item 5' }],
        total: 5,
      });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items', { pageSize: 2 });

      const results = await paginator.toArray();

      expect(results).toHaveLength(5);
      expect(results.map((r) => r.id)).toEqual([1, 2, 3, 4, 5]);
      expect(mockRequest).toHaveBeenCalledTimes(3);
    });

    it('should pass offset and limit parameters correctly', async () => {
      mockRequest.mockResolvedValueOnce({ items: [{ id: 1, name: 'Item 1' }] });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items', {
        pageSize: 25,
        params: { status: 'active' },
      });

      await paginator.toArray();

      expect(mockRequest).toHaveBeenCalledWith('GET', '/api/items', {
        params: {
          status: 'active',
          limit: 25,
          offset: 0,
        },
      });
    });

    it('should increment offset for subsequent pages', async () => {
      mockRequest
        .mockResolvedValueOnce({ items: [{ id: 1, name: 'Item 1' }, { id: 2, name: 'Item 2' }], total: 4 })
        .mockResolvedValueOnce({ items: [{ id: 3, name: 'Item 3' }, { id: 4, name: 'Item 4' }] });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items', { pageSize: 2 });

      await paginator.toArray();

      expect(mockRequest).toHaveBeenNthCalledWith(1, 'GET', '/api/items', {
        params: { limit: 2, offset: 0 },
      });
      expect(mockRequest).toHaveBeenNthCalledWith(2, 'GET', '/api/items', {
        params: { limit: 2, offset: 2 },
      });
    });
  });

  describe('total count tracking', () => {
    it('should track total count from items response', async () => {
      mockRequest.mockResolvedValueOnce({ items: [{ id: 1, name: 'Item 1' }], total: 100 });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      // Before fetching, total should be null
      expect(paginator.total).toBeNull();

      // Iterate to trigger fetch
      const iterator = paginator[Symbol.asyncIterator]();
      await iterator.next();

      // After fetching, total should be populated
      expect(paginator.total).toBe(100);
    });

    it('should track total count from data response', async () => {
      mockRequest.mockResolvedValueOnce({ data: [{ id: 1, name: 'Item 1' }], total: 50 });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const iterator = paginator[Symbol.asyncIterator]();
      await iterator.next();

      expect(paginator.total).toBe(50);
    });

    it('should handle response without total count', async () => {
      mockRequest.mockResolvedValueOnce({ items: [{ id: 1, name: 'Item 1' }] });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      await paginator.toArray();

      expect(paginator.total).toBeNull();
    });
  });

  describe('empty results handling', () => {
    it('should handle empty first page', async () => {
      mockRequest.mockResolvedValueOnce({ items: [], total: 0 });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results = await paginator.toArray();

      expect(results).toHaveLength(0);
      expect(mockRequest).toHaveBeenCalledTimes(1);
    });

    it('should handle empty items array in response', async () => {
      mockRequest.mockResolvedValueOnce({ items: [] });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results = await paginator.toArray();

      expect(results).toHaveLength(0);
    });

    it('should handle empty data array in response', async () => {
      mockRequest.mockResolvedValueOnce({ data: [] });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results = await paginator.toArray();

      expect(results).toHaveLength(0);
    });

    it('should handle raw empty array response', async () => {
      mockRequest.mockResolvedValueOnce([]);

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results = await paginator.toArray();

      expect(results).toHaveLength(0);
    });
  });

  describe('different response formats', () => {
    it('should handle items response format', async () => {
      mockRequest.mockResolvedValueOnce({
        items: [{ id: 1, name: 'Item 1' }],
        total: 1,
      });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results = await paginator.toArray();

      expect(results).toHaveLength(1);
      expect(results[0].id).toBe(1);
    });

    it('should handle data response format', async () => {
      mockRequest.mockResolvedValueOnce({
        data: [{ id: 1, name: 'Item 1' }],
        total: 1,
      });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results = await paginator.toArray();

      expect(results).toHaveLength(1);
      expect(results[0].id).toBe(1);
    });

    it('should handle raw array response format', async () => {
      mockRequest.mockResolvedValueOnce([{ id: 1, name: 'Item 1' }]);

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results = await paginator.toArray();

      expect(results).toHaveLength(1);
      expect(results[0].id).toBe(1);
    });

    it('should prefer items over data when both present', async () => {
      mockRequest.mockResolvedValueOnce({
        items: [{ id: 1, name: 'From Items' }],
        data: [{ id: 2, name: 'From Data' }],
      });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results = await paginator.toArray();

      expect(results).toHaveLength(1);
      expect(results[0].name).toBe('From Items');
    });
  });

  describe('cursor-based pagination', () => {
    it('should use cursor when provided in response', async () => {
      mockRequest
        .mockResolvedValueOnce({
          items: [{ id: 1, name: 'Item 1' }, { id: 2, name: 'Item 2' }],
          next_cursor: 'cursor_abc',
          total: 4,
        })
        .mockResolvedValueOnce({
          items: [{ id: 3, name: 'Item 3' }, { id: 4, name: 'Item 4' }],
        });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items', { pageSize: 2 });

      await paginator.toArray();

      // When cursor is available, it should be used instead of offset
      expect(mockRequest).toHaveBeenNthCalledWith(2, 'GET', '/api/items', {
        params: { limit: 2, cursor: 'cursor_abc' },
      });
    });
  });

  describe('error handling', () => {
    it('should propagate errors during pagination', async () => {
      mockRequest.mockRejectedValueOnce(new Error('Network error'));

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      await expect(paginator.toArray()).rejects.toThrow('Network error');
    });

    it('should propagate errors on subsequent pages', async () => {
      mockRequest
        .mockResolvedValueOnce({ items: [{ id: 1, name: 'Item 1' }], total: 5 })
        .mockRejectedValueOnce(new Error('Server error'));

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items', { pageSize: 1 });

      const results: TestItem[] = [];
      await expect(async () => {
        for await (const item of paginator) {
          results.push(item);
        }
      }).rejects.toThrow('Server error');

      // First item should have been collected before error
      expect(results).toHaveLength(1);
    });

    it('should throw when no client or fetch function is provided', async () => {
      const paginator = new AsyncPaginator<TestItem>(null as unknown as AragoraClient, '/api/items');

      await expect(paginator.toArray()).rejects.toThrow('No client or fetch function provided');
    });
  });

  describe('utility methods', () => {
    it('should take a limited number of items', async () => {
      mockRequest.mockResolvedValueOnce({
        items: [
          { id: 1, name: 'Item 1' },
          { id: 2, name: 'Item 2' },
          { id: 3, name: 'Item 3' },
        ],
      });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results = await paginator.take(2);

      expect(results).toHaveLength(2);
      expect(results[0].id).toBe(1);
      expect(results[1].id).toBe(2);
    });

    it('should find first matching item', async () => {
      mockRequest.mockResolvedValueOnce({
        items: [
          { id: 1, name: 'Item 1' },
          { id: 2, name: 'Target' },
          { id: 3, name: 'Item 3' },
        ],
      });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const result = await paginator.find((item) => item.name === 'Target');

      expect(result).toBeDefined();
      expect(result!.id).toBe(2);
    });

    it('should return undefined when find has no match', async () => {
      mockRequest.mockResolvedValueOnce({
        items: [{ id: 1, name: 'Item 1' }],
      });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const result = await paginator.find((item) => item.name === 'Not Found');

      expect(result).toBeUndefined();
    });

    it('should map items through transform function', async () => {
      mockRequest.mockResolvedValueOnce({
        items: [{ id: 1, name: 'Item 1' }, { id: 2, name: 'Item 2' }],
      });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results: string[] = [];
      for await (const name of paginator.map((item) => item.name)) {
        results.push(name);
      }

      expect(results).toEqual(['Item 1', 'Item 2']);
    });

    it('should filter items based on predicate', async () => {
      mockRequest.mockResolvedValueOnce({
        items: [
          { id: 1, name: 'Item 1' },
          { id: 2, name: 'Item 2' },
          { id: 3, name: 'Item 3' },
        ],
      });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results: TestItem[] = [];
      for await (const item of paginator.filter((item) => item.id % 2 === 1)) {
        results.push(item);
      }

      expect(results).toHaveLength(2);
      expect(results[0].id).toBe(1);
      expect(results[1].id).toBe(3);
    });
  });

  describe('fromFetch factory', () => {
    it('should create paginator from custom fetch function', async () => {
      const mockFetch: FetchPage<TestItem> = vi.fn().mockResolvedValueOnce({
        items: [{ id: 1, name: 'Item 1' }],
      });

      const paginator = AsyncPaginator.fromFetch(mockFetch, { pageSize: 10 });

      const results = await paginator.toArray();

      expect(results).toHaveLength(1);
      expect(mockFetch).toHaveBeenCalledWith({ limit: 10, offset: 0 });
    });
  });

  describe('paginate helper function', () => {
    it('should create paginator via helper function', async () => {
      mockRequest.mockResolvedValueOnce({ items: [{ id: 1, name: 'Item 1' }] });

      const client = createMockClient(mockRequest);
      const paginator = paginate<TestItem>(client, '/api/items', { pageSize: 5 });

      const results = await paginator.toArray();

      expect(results).toHaveLength(1);
      expect(mockRequest).toHaveBeenCalledWith('GET', '/api/items', {
        params: { limit: 5, offset: 0 },
      });
    });
  });

  describe('paginateWith helper function', () => {
    it('should create paginator from fetch function via helper', async () => {
      const mockFetch: FetchPage<TestItem> = vi.fn().mockResolvedValueOnce({
        data: [{ id: 1, name: 'Item 1' }],
        total: 1,
      });

      const paginator = paginateWith(mockFetch, { pageSize: 20 });

      const results = await paginator.toArray();

      expect(results).toHaveLength(1);
      expect(paginator.total).toBe(1);
    });
  });

  describe('edge cases', () => {
    it('should handle exactly one page of results', async () => {
      mockRequest.mockResolvedValueOnce({
        items: [{ id: 1, name: 'Item 1' }, { id: 2, name: 'Item 2' }],
        total: 2,
      });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items', { pageSize: 2 });

      const results = await paginator.toArray();

      expect(results).toHaveLength(2);
      expect(mockRequest).toHaveBeenCalledTimes(1);
    });

    it('should stop when receiving fewer items than page size', async () => {
      mockRequest.mockResolvedValueOnce({
        items: [{ id: 1, name: 'Item 1' }],
        // No total provided
      });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items', { pageSize: 10 });

      const results = await paginator.toArray();

      expect(results).toHaveLength(1);
      expect(mockRequest).toHaveBeenCalledTimes(1);
    });

    it('should stop when offset reaches total', async () => {
      mockRequest
        .mockResolvedValueOnce({ items: [{ id: 1, name: 'Item 1' }, { id: 2, name: 'Item 2' }], total: 2 })
        // This should never be called
        .mockResolvedValueOnce({ items: [{ id: 3, name: 'Item 3' }] });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items', { pageSize: 2 });

      const results = await paginator.toArray();

      expect(results).toHaveLength(2);
      expect(mockRequest).toHaveBeenCalledTimes(1);
    });

    it('should handle response with undefined items/data', async () => {
      mockRequest.mockResolvedValueOnce({ something_else: true });

      const client = createMockClient(mockRequest);
      const paginator = new AsyncPaginator<TestItem>(client, '/api/items');

      const results = await paginator.toArray();

      expect(results).toHaveLength(0);
    });
  });
});
