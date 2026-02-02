/**
 * Aragora Pagination Helpers
 *
 * Provides auto-paginating async iterators for list endpoints, allowing users to
 * iterate through all results without manually handling pagination.
 */

import type { AragoraClient } from './client';

/**
 * Options for configuring the paginator behavior.
 */
export interface PaginatorOptions {
  /** Number of items to fetch per page (default: 20) */
  pageSize?: number;
  /** Additional query parameters to include in requests */
  params?: Record<string, unknown>;
}

/**
 * Response format that includes items in a 'data' field.
 */
interface DataResponse<T> {
  data: T[];
  total?: number;
}

/**
 * Response format that includes items in an 'items' field.
 */
interface ItemsResponse<T> {
  items: T[];
  total?: number;
}

/**
 * Union type for all supported response formats.
 */
type PaginatedApiResponse<T> = DataResponse<T> | ItemsResponse<T> | T[];

/**
 * Fetch function type for making API requests.
 * This allows the paginator to work with different request implementations.
 */
export type FetchPage<T> = (params: Record<string, unknown>) => Promise<PaginatedApiResponse<T>>;

/**
 * AsyncPaginator - Auto-paginating asynchronous iterator for list endpoints.
 *
 * Automatically fetches additional pages as needed while iterating.
 * Supports both cursor-based and offset/limit pagination patterns.
 *
 * @example Using with AragoraClient
 * ```typescript
 * import { AsyncPaginator } from '@aragora/sdk';
 *
 * const paginator = new AsyncPaginator<Debate>(
 *   client,
 *   '/api/v1/debates',
 *   { params: { status: 'active' }, pageSize: 10 }
 * );
 *
 * for await (const debate of paginator) {
 *   console.log(debate.id);
 * }
 *
 * // Or collect all items
 * const allDebates = await paginator.toArray();
 * ```
 *
 * @example Using with custom fetch function
 * ```typescript
 * const paginator = AsyncPaginator.fromFetch<Debate>(
 *   async (params) => client.request('GET', '/api/v1/debates', { params }),
 *   { pageSize: 50 }
 * );
 *
 * for await (const debate of paginator) {
 *   console.log(debate.id);
 * }
 * ```
 */
export class AsyncPaginator<T> implements AsyncIterable<T> {
  private client: AragoraClient | null;
  private path: string;
  private params: Record<string, unknown>;
  private pageSize: number;
  private offset: number = 0;
  private cursor: string | null = null;
  private buffer: T[] = [];
  private exhausted: boolean = false;
  private _total: number | null = null;
  private customFetch: FetchPage<T> | null = null;

  /**
   * Create a new AsyncPaginator using an AragoraClient.
   *
   * @param client - The AragoraClient instance to use for requests.
   * @param path - The API endpoint path.
   * @param options - Configuration options.
   */
  constructor(
    client: AragoraClient,
    path: string,
    options?: PaginatorOptions
  );

  /**
   * Create a new AsyncPaginator using a custom fetch function.
   * Internal constructor overload.
   */
  constructor(
    client: AragoraClient | null,
    path: string,
    options: PaginatorOptions,
    customFetch: FetchPage<T>
  );

  constructor(
    client: AragoraClient | null,
    path: string,
    options?: PaginatorOptions,
    customFetch?: FetchPage<T>
  ) {
    this.client = client;
    this.path = path;
    this.params = options?.params ?? {};
    this.pageSize = options?.pageSize ?? 20;
    this.customFetch = customFetch ?? null;
  }

  /**
   * Create an AsyncPaginator from a custom fetch function.
   *
   * @param fetchFn - Function that fetches a page of results.
   * @param options - Configuration options.
   * @returns A new AsyncPaginator instance.
   */
  static fromFetch<U>(
    fetchFn: FetchPage<U>,
    options: PaginatorOptions = {}
  ): AsyncPaginator<U> {
    return new AsyncPaginator<U>(null, '', options, fetchFn);
  }

  /**
   * Returns the async iterator for this paginator.
   */
  [Symbol.asyncIterator](): AsyncIterator<T> {
    return {
      next: async (): Promise<IteratorResult<T>> => {
        if (this.buffer.length === 0) {
          if (this.exhausted) {
            return { done: true, value: undefined };
          }
          await this.fetchPage();
        }

        if (this.buffer.length === 0) {
          return { done: true, value: undefined };
        }

        const value = this.buffer.shift()!;
        return { done: false, value };
      },
    };
  }

  /**
   * Fetch the next page of results.
   */
  private async fetchPage(): Promise<void> {
    const params: Record<string, unknown> = {
      ...this.params,
      limit: this.pageSize,
    };

    // Support both cursor-based and offset-based pagination
    // When using cursor, don't include offset
    if (this.cursor !== null) {
      params.cursor = this.cursor;
    } else {
      params.offset = this.offset;
    }

    let response: PaginatedApiResponse<T>;

    if (this.customFetch) {
      response = await this.customFetch(params);
    } else if (this.client) {
      response = await this.client.request<PaginatedApiResponse<T>>('GET', this.path, { params });
    } else {
      throw new Error('No client or fetch function provided');
    }

    // Handle different response formats
    let items: T[];
    if (Array.isArray(response)) {
      // Response is a raw array
      items = response;
    } else if ('items' in response && Array.isArray(response.items)) {
      // Response has 'items' field
      items = response.items;
      if (response.total !== undefined) {
        this._total = response.total;
      }
    } else if ('data' in response && Array.isArray(response.data)) {
      // Response has 'data' field
      items = response.data;
      if (response.total !== undefined) {
        this._total = response.total;
      }
    } else {
      // Empty or unexpected response format
      items = [];
    }

    // Check for cursor-based pagination in response
    const responseWithCursor = response as unknown as Record<string, unknown>;
    if (responseWithCursor.next_cursor && typeof responseWithCursor.next_cursor === 'string') {
      this.cursor = responseWithCursor.next_cursor;
    }

    if (items.length > 0) {
      this.buffer.push(...items);
      this.offset += items.length;

      // Check if we've exhausted all results
      if (items.length < this.pageSize) {
        this.exhausted = true;
      } else if (this._total !== null && this.offset >= this._total) {
        this.exhausted = true;
      } else if (this.cursor === null && responseWithCursor.next_cursor === undefined) {
        // No cursor provided in response, check if we might have more
        // based on the number of items returned
      }
    } else {
      this.exhausted = true;
    }
  }

  /**
   * Returns the total number of items, if known from the API response.
   * This value is populated after at least one page has been fetched.
   */
  get total(): number | null {
    return this._total;
  }

  /**
   * Collect all items from the paginator into an array.
   * This will fetch all remaining pages.
   *
   * @returns Promise resolving to array of all items.
   */
  async toArray(): Promise<T[]> {
    const items: T[] = [];
    for await (const item of this) {
      items.push(item);
    }
    return items;
  }

  /**
   * Take at most `limit` items from the paginator.
   *
   * @param limit - Maximum number of items to take.
   * @returns Promise resolving to array of items.
   */
  async take(limit: number): Promise<T[]> {
    const items: T[] = [];
    for await (const item of this) {
      items.push(item);
      if (items.length >= limit) {
        break;
      }
    }
    return items;
  }

  /**
   * Find the first item matching a predicate.
   *
   * @param predicate - Function to test each item.
   * @returns Promise resolving to the first matching item, or undefined.
   */
  async find(predicate: (item: T) => boolean): Promise<T | undefined> {
    for await (const item of this) {
      if (predicate(item)) {
        return item;
      }
    }
    return undefined;
  }

  /**
   * Map items through a transform function.
   *
   * @param mapper - Function to transform each item.
   * @returns A new async iterator yielding transformed items.
   */
  async *map<U>(mapper: (item: T) => U): AsyncGenerator<U> {
    for await (const item of this) {
      yield mapper(item);
    }
  }

  /**
   * Filter items based on a predicate.
   *
   * @param predicate - Function to test each item.
   * @returns A new async iterator yielding only items that pass the test.
   */
  async *filter(predicate: (item: T) => boolean): AsyncGenerator<T> {
    for await (const item of this) {
      if (predicate(item)) {
        yield item;
      }
    }
  }
}

/**
 * Convenience function to create an AsyncPaginator.
 *
 * @param client - The AragoraClient instance.
 * @param path - The API endpoint path.
 * @param options - Paginator options.
 * @returns A new AsyncPaginator instance.
 */
export function paginate<T>(
  client: AragoraClient,
  path: string,
  options?: PaginatorOptions
): AsyncPaginator<T> {
  return new AsyncPaginator<T>(client, path, options);
}

/**
 * Create an AsyncPaginator from a custom fetch function.
 *
 * @param fetchFn - Function that fetches a page of results.
 * @param options - Paginator options.
 * @returns A new AsyncPaginator instance.
 */
export function paginateWith<T>(
  fetchFn: FetchPage<T>,
  options?: PaginatorOptions
): AsyncPaginator<T> {
  return AsyncPaginator.fromFetch(fetchFn, options);
}
