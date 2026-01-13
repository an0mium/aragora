const mockRouter = {
  push: jest.fn(),
  replace: jest.fn(),
  prefetch: jest.fn(),
  back: jest.fn(),
  forward: jest.fn(),
  refresh: jest.fn(),
};

const mockSearchParams = new URLSearchParams();

const useRouter = jest.fn(() => mockRouter);
const useSearchParams = jest.fn(() => mockSearchParams);
const usePathname = jest.fn(() => '/');
const useParams = jest.fn(() => ({}));

export { mockRouter, useRouter, useSearchParams, usePathname, useParams };
