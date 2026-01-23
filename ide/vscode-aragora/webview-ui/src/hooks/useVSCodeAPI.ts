import { useMemo } from 'react';
import type { WebviewMessage } from '../types';

interface VSCodeAPI {
  postMessage(message: WebviewMessage): void;
  getState<T>(): T | undefined;
  setState<T>(state: T): void;
}

declare global {
  interface Window {
    vscodeApi?: VSCodeAPI;
    acquireVsCodeApi?(): VSCodeAPI;
  }
}

/**
 * Hook to access the VSCode API for webview communication
 */
export function useVSCodeAPI(): VSCodeAPI {
  return useMemo(() => {
    // Check if already initialized
    if (window.vscodeApi) {
      return window.vscodeApi;
    }

    // Try to acquire the API
    if (window.acquireVsCodeApi) {
      const api = window.acquireVsCodeApi();
      window.vscodeApi = api;
      return api;
    }

    // Fallback for development outside VSCode
    console.warn('VSCode API not available - using mock');
    return {
      postMessage: (message: WebviewMessage) => {
        console.log('VSCode postMessage (mock):', message);
      },
      getState: <T>() => undefined as T | undefined,
      setState: <T>(_state: T) => {
        // no-op
      },
    };
  }, []);
}
