/**
 * Tests for formatter utilities
 */

import {
  formatEloChange,
  formatAge,
  formatTimeAgo,
  formatPercent,
  formatNumber,
  formatDate,
  formatTime,
  formatDateTime,
  truncate,
} from '@/utils/formatters';

describe('formatter utilities', () => {
  describe('formatEloChange', () => {
    it('adds + prefix to positive values', () => {
      expect(formatEloChange(15)).toBe('+15');
    });

    it('shows negative values with - prefix', () => {
      expect(formatEloChange(-10)).toBe('-10');
    });

    it('shows zero without prefix', () => {
      expect(formatEloChange(0)).toBe('0');
    });

    it('handles large positive values', () => {
      expect(formatEloChange(100)).toBe('+100');
    });

    it('handles large negative values', () => {
      expect(formatEloChange(-100)).toBe('-100');
    });
  });

  describe('formatAge', () => {
    it('formats seconds', () => {
      expect(formatAge(30)).toBe('30s');
    });

    it('rounds seconds', () => {
      expect(formatAge(30.5)).toBe('31s');
    });

    it('formats minutes', () => {
      expect(formatAge(120)).toBe('2m');
    });

    it('formats at 60 seconds as 1 minute', () => {
      expect(formatAge(60)).toBe('1m');
    });

    it('formats hours', () => {
      expect(formatAge(7200)).toBe('2h');
    });

    it('formats at 3600 seconds as 1 hour', () => {
      expect(formatAge(3600)).toBe('1h');
    });

    it('formats days', () => {
      expect(formatAge(172800)).toBe('2d');
    });

    it('formats at 86400 seconds as 1 day', () => {
      expect(formatAge(86400)).toBe('1d');
    });

    it('handles zero', () => {
      expect(formatAge(0)).toBe('0s');
    });
  });

  describe('formatTimeAgo', () => {
    beforeEach(() => {
      jest.useFakeTimers();
      jest.setSystemTime(new Date('2024-01-15T12:00:00Z'));
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('returns "just now" for recent timestamps', () => {
      const now = Date.now();
      expect(formatTimeAgo(now - 30000)).toBe('just now');
    });

    it('formats minutes ago', () => {
      const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
      expect(formatTimeAgo(fiveMinutesAgo)).toBe('5 minutes ago');
    });

    it('formats single minute', () => {
      const oneMinuteAgo = Date.now() - 60 * 1000;
      expect(formatTimeAgo(oneMinuteAgo)).toBe('1 minute ago');
    });

    it('formats hours ago', () => {
      const threeHoursAgo = Date.now() - 3 * 60 * 60 * 1000;
      expect(formatTimeAgo(threeHoursAgo)).toBe('3 hours ago');
    });

    it('formats single hour', () => {
      const oneHourAgo = Date.now() - 60 * 60 * 1000;
      expect(formatTimeAgo(oneHourAgo)).toBe('1 hour ago');
    });

    it('formats days ago', () => {
      const twoDaysAgo = Date.now() - 2 * 24 * 60 * 60 * 1000;
      expect(formatTimeAgo(twoDaysAgo)).toBe('2 days ago');
    });

    it('formats single day', () => {
      const oneDayAgo = Date.now() - 24 * 60 * 60 * 1000;
      expect(formatTimeAgo(oneDayAgo)).toBe('1 day ago');
    });

    it('accepts Date object', () => {
      const date = new Date(Date.now() - 2 * 60 * 1000);
      expect(formatTimeAgo(date)).toBe('2 minutes ago');
    });

    it('accepts ISO string', () => {
      const isoString = new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString();
      expect(formatTimeAgo(isoString)).toBe('3 hours ago');
    });

    it('accepts timestamp number', () => {
      const timestamp = Date.now() - 30 * 60 * 1000;
      expect(formatTimeAgo(timestamp)).toBe('30 minutes ago');
    });
  });

  describe('formatPercent', () => {
    it('formats decimal as percentage', () => {
      expect(formatPercent(0.75)).toBe('75%');
    });

    it('formats percentage value as-is', () => {
      expect(formatPercent(75)).toBe('75%');
    });

    it('handles zero', () => {
      expect(formatPercent(0)).toBe('0%');
    });

    it('handles 100%', () => {
      expect(formatPercent(1)).toBe('100%');
    });

    it('supports decimal places', () => {
      expect(formatPercent(0.756, 1)).toBe('75.6%');
    });

    it('supports multiple decimal places', () => {
      expect(formatPercent(0.7567, 2)).toBe('75.67%');
    });

    it('handles values slightly over 1 as percentages', () => {
      // Values > 1 are treated as already being percentages
      expect(formatPercent(1.5)).toBe('2%'); // rounds to nearest
    });

    it('handles large percentage values', () => {
      expect(formatPercent(150)).toBe('150%');
    });
  });

  describe('formatNumber', () => {
    it('formats small numbers', () => {
      expect(formatNumber(42)).toBe('42');
    });

    it('formats thousands with separator', () => {
      expect(formatNumber(1000)).toMatch(/1[,.]?000/);
    });

    it('formats millions', () => {
      expect(formatNumber(1000000)).toMatch(/1[,.]?000[,.]?000/);
    });

    it('handles zero', () => {
      expect(formatNumber(0)).toBe('0');
    });

    it('handles negative numbers', () => {
      const result = formatNumber(-1000);
      expect(result).toMatch(/-1[,.]?000/);
    });
  });

  describe('formatDate', () => {
    it('formats date string', () => {
      const result = formatDate('2024-01-15');
      expect(result).toBeTruthy();
      expect(typeof result).toBe('string');
    });

    it('formats Date object', () => {
      const result = formatDate(new Date('2024-01-15'));
      expect(result).toBeTruthy();
    });
  });

  describe('formatTime', () => {
    it('formats time string', () => {
      const result = formatTime('2024-01-15T14:30:00');
      expect(result).toBeTruthy();
      expect(typeof result).toBe('string');
    });

    it('formats Date object', () => {
      const result = formatTime(new Date('2024-01-15T14:30:00'));
      expect(result).toBeTruthy();
    });
  });

  describe('formatDateTime', () => {
    it('formats date and time string', () => {
      const result = formatDateTime('2024-01-15T14:30:00');
      expect(result).toBeTruthy();
      expect(typeof result).toBe('string');
    });

    it('includes both date and time', () => {
      const result = formatDateTime(new Date('2024-01-15T14:30:00'));
      expect(result.length).toBeGreaterThan(10);
    });
  });

  describe('truncate', () => {
    it('returns unchanged if under max length', () => {
      expect(truncate('hello', 10)).toBe('hello');
    });

    it('returns unchanged if exactly max length', () => {
      expect(truncate('hello', 5)).toBe('hello');
    });

    it('truncates and adds ellipsis', () => {
      expect(truncate('hello world', 8)).toBe('hello...');
    });

    it('handles empty string', () => {
      expect(truncate('', 10)).toBe('');
    });

    it('handles max length of 3 (minimum for ellipsis)', () => {
      expect(truncate('hello', 3)).toBe('...');
    });

    it('handles longer strings', () => {
      const long = 'This is a very long string that needs truncation';
      expect(truncate(long, 20)).toBe('This is a very lo...');
    });
  });
});
