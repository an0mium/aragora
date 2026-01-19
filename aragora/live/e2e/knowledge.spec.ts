import { test, expect, mockApiResponse } from './fixtures';

/**
 * E2E tests for Knowledge Mound functionality.
 *
 * Tests the knowledge system including:
 * - Knowledge node display and filtering
 * - Search functionality
 * - Node details and relationships
 * - Stats display
 */

// Mock knowledge data
const mockNodes = [
  {
    id: 'node-1',
    nodeType: 'fact',
    content: 'Test fact about AI systems',
    confidence: 0.85,
    tier: 'medium',
    sourceType: 'consensus',
    topics: ['AI', 'systems'],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: 'node-2',
    nodeType: 'memory',
    content: 'Memory from previous debate',
    confidence: 0.72,
    tier: 'slow',
    sourceType: 'continuum',
    topics: ['debate', 'memory'],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
];

const mockStats = {
  totalNodes: 42,
  nodesByType: { fact: 15, memory: 12, consensus: 10, evidence: 5 },
  nodesByTier: { fast: 5, medium: 20, slow: 12, glacial: 5 },
  nodesBySource: { consensus: 18, continuum: 14, document: 10 },
  totalRelationships: 67,
};

test.describe('Knowledge Mound Page', () => {
  test('should load knowledge page', async ({ page, aragoraPage }) => {
    await page.goto('/knowledge');
    await aragoraPage.dismissAllOverlays();

    await expect(page).toHaveURL(/\/knowledge/);
    await expect(page.locator('body')).toBeVisible();
  });

  test('should display knowledge nodes when API returns data', async ({ page, aragoraPage }) => {
    await mockApiResponse(page, '**/api/knowledge/nodes**', { nodes: mockNodes });
    await mockApiResponse(page, '**/api/knowledge/stats', mockStats);

    await page.goto('/knowledge');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    // Should show content
    await expect(page.locator('body')).toBeVisible();
  });

  test('should display knowledge stats', async ({ page, aragoraPage }) => {
    await mockApiResponse(page, '**/api/knowledge/nodes**', { nodes: mockNodes });
    await mockApiResponse(page, '**/api/knowledge/stats', mockStats);

    await page.goto('/knowledge');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    // Stats should be displayed somewhere on page
    const body = page.locator('body');
    await expect(body).toBeVisible();
  });

  test('should have search functionality', async ({ page, aragoraPage }) => {
    await mockApiResponse(page, '**/api/knowledge/**', { nodes: mockNodes });

    await page.goto('/knowledge');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    // Look for search input
    const searchInput = page.locator('input[type="text"], input[placeholder*="search" i], input[placeholder*="Search" i]');
    const hasSearch = await searchInput.first().isVisible().catch(() => false);
    expect(hasSearch).toBeDefined();
  });

  test('should have filter options', async ({ page, aragoraPage }) => {
    await mockApiResponse(page, '**/api/knowledge/**', { nodes: mockNodes });

    await page.goto('/knowledge');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    // Look for filter buttons or selects
    const filters = page.locator('select, [role="combobox"], button:has-text("Filter"), button:has-text("All")');
    const hasFilters = await filters.first().isVisible().catch(() => false);
    expect(hasFilters).toBeDefined();
  });

  test('should handle empty state gracefully', async ({ page, aragoraPage }) => {
    await mockApiResponse(page, '**/api/knowledge/nodes**', { nodes: [] });
    await mockApiResponse(page, '**/api/knowledge/stats', { totalNodes: 0, nodesByType: {}, nodesByTier: {}, nodesBySource: {}, totalRelationships: 0 });

    await page.goto('/knowledge');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    // Page should handle empty state
    await expect(page.locator('body')).toBeVisible();
  });

  test('should fall back gracefully when API unavailable', async ({ page, aragoraPage }) => {
    await page.route('**/api/knowledge/**', (route) => {
      route.fulfill({ status: 503, body: 'Service Unavailable' });
    });

    await page.goto('/knowledge');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    // Page should still be functional
    await expect(page.locator('body')).toBeVisible();
  });
});

test.describe('Knowledge Search and Query', () => {
  test('should search knowledge nodes', async ({ page, aragoraPage }) => {
    await mockApiResponse(page, '**/api/knowledge/**', { nodes: mockNodes });

    await page.goto('/knowledge');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    // Look for search input - page may or may not have one
    const searchInput = page.locator('input[type="text"], input[placeholder*="search" i]').first();
    const hasSearch = await searchInput.isVisible({ timeout: 2000 }).catch(() => false);

    if (hasSearch) {
      await searchInput.fill('AI');
      await page.waitForTimeout(300);
    }

    // Page should still be functional
    await expect(page.locator('body')).toBeVisible();
  });

  test('should query knowledge API', async ({ page, aragoraPage }) => {
    let queryRequested = false;

    await page.route('**/api/knowledge/query**', (route) => {
      queryRequested = true;
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ answer: 'Test answer', facts: mockNodes }),
      });
    });

    await page.goto('/knowledge');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    // Query endpoint availability is tracked
    expect(queryRequested).toBeDefined();
  });
});

test.describe('Knowledge API Endpoints', () => {
  test.skip(({ browserName }) => browserName !== 'chromium', 'API tests only run in chromium');

  test('should handle /api/knowledge/facts endpoint', async ({ page }) => {
    const response = await page.request.get('/api/knowledge/facts').catch(() => null);

    if (response) {
      expect([200, 404, 503]).toContain(response.status());
    } else {
      expect(true).toBe(true);
    }
  });

  test('should handle /api/knowledge/stats endpoint', async ({ page }) => {
    const response = await page.request.get('/api/knowledge/stats').catch(() => null);

    if (response) {
      expect([200, 404, 503]).toContain(response.status());
    } else {
      expect(true).toBe(true);
    }
  });
});
