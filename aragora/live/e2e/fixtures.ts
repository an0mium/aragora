import { test as base, expect } from '@playwright/test';

/**
 * Extended test fixtures for Aragora E2E tests
 */

// Test data factories
export const testData = {
  debate: {
    simple: {
      topic: 'Should AI systems be open source?',
      agents: ['claude', 'gpt'],
      rounds: 3,
    },
    complex: {
      topic: 'What is the best approach to implementing rate limiting in distributed systems?',
      agents: ['claude', 'gpt', 'gemini'],
      rounds: 5,
      options: {
        consensus: 'majority',
        enableEvidence: true,
      },
    },
  },
  user: {
    test: {
      email: 'test@aragora.ai',
      password: 'test-password-123',
    },
  },
};

// Custom page object helpers
export class AragoraPage {
  constructor(private page: import('@playwright/test').Page) {}

  async waitForAppReady() {
    // Wait for Next.js hydration
    await this.page.waitForLoadState('networkidle');
    // Wait for any loading spinners to disappear
    await this.page.waitForSelector('[data-testid="loading"]', { state: 'hidden' }).catch(() => {});
  }

  async getToast() {
    return this.page.locator('[data-testid="toast"]').first();
  }

  async waitForToast(text: string) {
    await expect(this.page.locator('[data-testid="toast"]').filter({ hasText: text })).toBeVisible();
  }

  async dismissToast() {
    const toast = await this.getToast();
    if (await toast.isVisible()) {
      await toast.locator('button[aria-label="Close"]').click().catch(() => {});
    }
  }
}

// Extended test with Aragora fixtures
export const test = base.extend<{
  aragoraPage: AragoraPage;
}>({
  aragoraPage: async ({ page }, use) => {
    const aragoraPage = new AragoraPage(page);
    await use(aragoraPage);
  },
});

export { expect };

// Common selectors
export const selectors = {
  // Navigation
  nav: {
    home: '[data-testid="nav-home"]',
    debates: '[data-testid="nav-debates"]',
    agents: '[data-testid="nav-agents"]',
    plugins: '[data-testid="nav-plugins"]',
    settings: '[data-testid="nav-settings"]',
  },
  // Debate creation
  debateInput: {
    topic: '[data-testid="debate-topic-input"]',
    agentSelect: '[data-testid="agent-select"]',
    startButton: '[data-testid="start-debate-button"]',
  },
  // Debate viewer
  debateViewer: {
    container: '[data-testid="debate-viewer"]',
    messages: '[data-testid="debate-messages"]',
    agentPanel: '[data-testid="agent-panel"]',
    exportButton: '[data-testid="export-button"]',
  },
  // Auth
  auth: {
    signInButton: '[data-testid="sign-in-button"]',
    signOutButton: '[data-testid="sign-out-button"]',
    userMenu: '[data-testid="user-menu"]',
  },
  // Landing page
  landing: {
    hero: '[data-testid="hero-section"]',
    cta: '[data-testid="cta-button"]',
    features: '[data-testid="features-section"]',
  },
};

// API mocking helpers
export async function mockApiResponse(
  page: import('@playwright/test').Page,
  url: string | RegExp,
  response: object,
  status = 200
) {
  await page.route(url, async (route) => {
    await route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify(response),
    });
  });
}

// Mock debate data for testing
export const mockDebate = {
  id: 'test-debate-123',
  topic: 'Test debate topic',
  status: 'completed',
  created_at: new Date().toISOString(),
  agents: ['claude', 'gpt'],
  messages: [
    {
      id: 'msg-1',
      agent: 'claude',
      role: 'proposer',
      content: 'I propose that...',
      round: 1,
    },
    {
      id: 'msg-2',
      agent: 'gpt',
      role: 'critic',
      content: 'I critique this because...',
      round: 1,
    },
  ],
  consensus: {
    reached: true,
    type: 'majority',
    summary: 'Agents agreed on the main points.',
  },
};

// Mock agents data
export const mockAgents = [
  { name: 'claude', provider: 'anthropic', status: 'available', elo: 1500 },
  { name: 'gpt', provider: 'openai', status: 'available', elo: 1480 },
  { name: 'gemini', provider: 'google', status: 'available', elo: 1460 },
];
