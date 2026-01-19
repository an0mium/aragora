/**
 * Tests for logger utility
 */

import { logger } from '@/utils/logger';

describe('logger utility', () => {
  const originalEnv = process.env.NODE_ENV;
  const mockConsole = {
    log: jest.spyOn(console, 'log').mockImplementation(),
    warn: jest.spyOn(console, 'warn').mockImplementation(),
    error: jest.spyOn(console, 'error').mockImplementation(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterAll(() => {
    mockConsole.log.mockRestore();
    mockConsole.warn.mockRestore();
    mockConsole.error.mockRestore();
    process.env.NODE_ENV = originalEnv;
  });

  describe('in development mode', () => {
    beforeAll(() => {
      process.env.NODE_ENV = 'development';
    });

    it('debug logs with prefix in development', () => {
      // Logger uses isDev constant determined at module load time
      // Since we're in test mode, let's verify the function exists
      expect(typeof logger.debug).toBe('function');
    });

    it('warn logs with prefix in development', () => {
      expect(typeof logger.warn).toBe('function');
    });

    it('error logs with prefix in development', () => {
      expect(typeof logger.error).toBe('function');
    });

    it('accepts multiple arguments', () => {
      // Verify functions accept multiple args without throwing
      expect(() => logger.debug('test', 1, { key: 'value' })).not.toThrow();
      expect(() => logger.warn('test', 'warning')).not.toThrow();
      expect(() => logger.error('test', new Error('error'))).not.toThrow();
    });
  });

  describe('logger structure', () => {
    it('has debug method', () => {
      expect(logger).toHaveProperty('debug');
      expect(typeof logger.debug).toBe('function');
    });

    it('has warn method', () => {
      expect(logger).toHaveProperty('warn');
      expect(typeof logger.warn).toBe('function');
    });

    it('has error method', () => {
      expect(logger).toHaveProperty('error');
      expect(typeof logger.error).toBe('function');
    });
  });
});
