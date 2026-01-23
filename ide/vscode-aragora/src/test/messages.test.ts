/**
 * Tests for types/messages.ts - type guards and utility functions.
 */

import {
  isExtensionMessage,
  isWebviewMessage,
  getAgentColor,
  compareSeverity,
  SEVERITY_PRIORITY,
  AGENT_COLORS,
  type Severity,
  type ExtensionMessage,
  type WebviewMessage,
} from '../types/messages';

describe('Type Guards', () => {
  describe('isExtensionMessage', () => {
    it('should return true for valid extension messages', () => {
      const validMessages: ExtensionMessage[] = [
        { type: 'debate_started', debate: {} as any },
        { type: 'error', message: 'test error' },
        { type: 'info', message: 'test info' },
        { type: 'security_finding', finding: {} as any },
      ];

      for (const msg of validMessages) {
        expect(isExtensionMessage(msg)).toBe(true);
      }
    });

    it('should return false for invalid messages', () => {
      expect(isExtensionMessage(null)).toBe(false);
      expect(isExtensionMessage(undefined)).toBe(false);
      expect(isExtensionMessage('string')).toBe(false);
      expect(isExtensionMessage(123)).toBe(false);
      expect(isExtensionMessage({})).toBe(false);
      expect(isExtensionMessage({ notType: 'test' })).toBe(false);
      expect(isExtensionMessage({ type: 123 })).toBe(false);
    });
  });

  describe('isWebviewMessage', () => {
    it('should return true for valid webview messages', () => {
      const validMessages: WebviewMessage[] = [
        { type: 'ready' },
        { type: 'get_state' },
        { type: 'start_debate', question: 'test?' },
        { type: 'fix_finding', findingId: '123' },
      ];

      for (const msg of validMessages) {
        expect(isWebviewMessage(msg)).toBe(true);
      }
    });

    it('should return false for invalid messages', () => {
      expect(isWebviewMessage(null)).toBe(false);
      expect(isWebviewMessage(undefined)).toBe(false);
      expect(isWebviewMessage('string')).toBe(false);
      expect(isWebviewMessage(123)).toBe(false);
      expect(isWebviewMessage({})).toBe(false);
    });
  });
});

describe('getAgentColor', () => {
  it('should return correct color for known agents', () => {
    expect(getAgentColor('claude')).toBe(AGENT_COLORS.claude);
    expect(getAgentColor('Claude')).toBe(AGENT_COLORS.claude);
    expect(getAgentColor('gpt-4')).toBe(AGENT_COLORS['gpt-4']);
    expect(getAgentColor('GPT-4o')).toBe(AGENT_COLORS['gpt-4o']);
    expect(getAgentColor('gemini-pro')).toBe(AGENT_COLORS.gemini);
    expect(getAgentColor('mistral-large')).toBe(AGENT_COLORS.mistral);
    expect(getAgentColor('grok-2')).toBe(AGENT_COLORS.grok);
    expect(getAgentColor('llama-3')).toBe(AGENT_COLORS.llama);
    expect(getAgentColor('deepseek-coder')).toBe(AGENT_COLORS.deepseek);
    expect(getAgentColor('qwen-plus')).toBe(AGENT_COLORS.qwen);
  });

  it('should return default color for unknown agents', () => {
    expect(getAgentColor('unknown-agent')).toBe(AGENT_COLORS.default);
    expect(getAgentColor('custom-llm')).toBe(AGENT_COLORS.default);
    expect(getAgentColor('')).toBe(AGENT_COLORS.default);
  });

  it('should be case-insensitive', () => {
    expect(getAgentColor('CLAUDE')).toBe(AGENT_COLORS.claude);
    expect(getAgentColor('Claude-3')).toBe(AGENT_COLORS.claude);
    expect(getAgentColor('GEMINI')).toBe(AGENT_COLORS.gemini);
  });
});

describe('compareSeverity', () => {
  it('should correctly order severities', () => {
    const severities: Severity[] = ['info', 'low', 'medium', 'high', 'critical'];
    const sorted = [...severities].sort(compareSeverity);

    expect(sorted).toEqual(['critical', 'high', 'medium', 'low', 'info']);
  });

  it('should return 0 for equal severities', () => {
    expect(compareSeverity('critical', 'critical')).toBe(0);
    expect(compareSeverity('high', 'high')).toBe(0);
    expect(compareSeverity('medium', 'medium')).toBe(0);
    expect(compareSeverity('low', 'low')).toBe(0);
    expect(compareSeverity('info', 'info')).toBe(0);
  });

  it('should return negative for higher severity first', () => {
    expect(compareSeverity('critical', 'high')).toBeLessThan(0);
    expect(compareSeverity('high', 'medium')).toBeLessThan(0);
    expect(compareSeverity('medium', 'low')).toBeLessThan(0);
    expect(compareSeverity('low', 'info')).toBeLessThan(0);
  });

  it('should return positive for lower severity first', () => {
    expect(compareSeverity('info', 'low')).toBeGreaterThan(0);
    expect(compareSeverity('low', 'medium')).toBeGreaterThan(0);
    expect(compareSeverity('medium', 'high')).toBeGreaterThan(0);
    expect(compareSeverity('high', 'critical')).toBeGreaterThan(0);
  });
});

describe('SEVERITY_PRIORITY', () => {
  it('should have correct priority order', () => {
    expect(SEVERITY_PRIORITY.critical).toBe(0);
    expect(SEVERITY_PRIORITY.high).toBe(1);
    expect(SEVERITY_PRIORITY.medium).toBe(2);
    expect(SEVERITY_PRIORITY.low).toBe(3);
    expect(SEVERITY_PRIORITY.info).toBe(4);
  });

  it('should have lower numbers for higher severity', () => {
    expect(SEVERITY_PRIORITY.critical).toBeLessThan(SEVERITY_PRIORITY.high);
    expect(SEVERITY_PRIORITY.high).toBeLessThan(SEVERITY_PRIORITY.medium);
    expect(SEVERITY_PRIORITY.medium).toBeLessThan(SEVERITY_PRIORITY.low);
    expect(SEVERITY_PRIORITY.low).toBeLessThan(SEVERITY_PRIORITY.info);
  });
});

describe('AGENT_COLORS', () => {
  it('should have valid hex color values', () => {
    const hexColorRegex = /^#[0-9A-Fa-f]{6}$/;

    for (const [agent, color] of Object.entries(AGENT_COLORS)) {
      expect(color).toMatch(hexColorRegex);
    }
  });

  it('should have a default color', () => {
    expect(AGENT_COLORS.default).toBeDefined();
  });

  it('should have colors for major LLM providers', () => {
    expect(AGENT_COLORS.claude).toBeDefined();
    expect(AGENT_COLORS['gpt-4']).toBeDefined();
    expect(AGENT_COLORS.gemini).toBeDefined();
    expect(AGENT_COLORS.mistral).toBeDefined();
  });
});
