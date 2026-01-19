/**
 * Type declarations for Jest mocks.
 * These augment modules with mock-specific exports used in tests.
 */

import type { Mock } from 'jest';

declare module 'next/navigation' {
  export const mockRouter: {
    push: Mock;
    replace: Mock;
    prefetch: Mock;
    back: Mock;
    forward: Mock;
    refresh: Mock;
  };
}
