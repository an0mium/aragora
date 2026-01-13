import '@testing-library/jest-dom';

process.env.NEXT_PUBLIC_API_URL = 'http://localhost:8080';
process.env.NEXT_PUBLIC_WS_URL = 'ws://localhost:8080';

jest.mock('next/navigation');

// Mock fetch globally
global.fetch = jest.fn();

// Mock window.matchMedia for components that use it (e.g., ThemeToggle)
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock ResizeObserver for components that use it
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock IntersectionObserver for lazy loading components
global.IntersectionObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Mock scrollTo for components that use window scrolling
window.scrollTo = jest.fn();

// Reset mocks between tests
beforeEach(() => {
  jest.clearAllMocks();
});
