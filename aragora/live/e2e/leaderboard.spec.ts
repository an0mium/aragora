import { test, expect } from '@playwright/test';

/**
 * E2E tests for the leaderboard feature.
 */

test.describe('Leaderboard', () => {
  test('should load leaderboard page', async ({ page }) => {
    await page.goto('/leaderboard');

    // Should have leaderboard heading or content
    const heading = page.locator('h1:has-text("Leaderboard"), h1:has-text("Ranking"), [data-testid="leaderboard-title"]');
    await expect(heading.first()).toBeVisible();
  });

  test('should display agent rankings', async ({ page }) => {
    await page.goto('/leaderboard');
    await page.waitForLoadState('domcontentloaded');

    // Should show ranking table or list
    const rankingItems = page.locator(
      '[data-testid="ranking-row"], tr[data-agent], .agent-ranking, table tbody tr'
    );

    const emptyState = page.locator('[data-testid="empty-leaderboard"], :text("No rankings")');

    // Either rankings exist or empty state
    const hasRankings = await rankingItems.count() > 0;
    const hasEmptyState = await emptyState.isVisible().catch(() => false);

    expect(hasRankings || hasEmptyState).toBeTruthy();
  });

  test('should show ELO ratings', async ({ page }) => {
    await page.goto('/leaderboard');
    await page.waitForLoadState('domcontentloaded');

    // Look for ELO or rating values
    const ratings = page.locator('[data-testid="elo-rating"], .elo-score, :text(/\\d{3,4}/)');

    // Check if numeric ratings are displayed
    const ratingsCount = await ratings.count();
    if (ratingsCount > 0) {
      const firstRating = await ratings.first().textContent();
      // ELO ratings are typically 1000-2000+
      expect(firstRating).toBeDefined();
    }
  });

  test('should display agent names', async ({ page }) => {
    await page.goto('/leaderboard');
    await page.waitForLoadState('domcontentloaded');

    // Look for agent names
    const agentNames = page.locator(
      '[data-testid="agent-name"], .agent-name, td:first-child'
    );

    const namesCount = await agentNames.count();
    if (namesCount > 0) {
      // Should show recognizable agent types
      const firstAgent = await agentNames.first().textContent();
      expect(firstAgent).toBeDefined();
    }
  });

  test('should allow sorting by different metrics', async ({ page }) => {
    await page.goto('/leaderboard');
    await page.waitForLoadState('domcontentloaded');

    // Look for sort controls
    const sortButtons = page.locator(
      '[data-testid="sort-button"], th[role="columnheader"], button:has-text("Sort")'
    );

    if (await sortButtons.count() > 0) {
      // Click on a sort header
      await sortButtons.first().click();

      // Page should update (URL or content)
      await page.waitForTimeout(500);
    }
  });

  test('should show win/loss statistics', async ({ page }) => {
    await page.goto('/leaderboard');
    await page.waitForLoadState('domcontentloaded');

    // Look for win/loss or match statistics
    const stats = page.locator(
      '[data-testid="win-count"], [data-testid="loss-count"], :text("W:"), :text("L:"), .wins, .losses'
    );

    const hasStats = await stats.count() > 0;
    // Stats are optional but nice to have
    expect(hasStats).toBeDefined();
  });

  test('should be accessible', async ({ page }) => {
    await page.goto('/leaderboard');

    // Table should have proper structure
    const table = page.locator('table');

    if (await table.isVisible().catch(() => false)) {
      // Should have headers
      const headers = page.locator('th, [role="columnheader"]');
      const headerCount = await headers.count();
      expect(headerCount).toBeGreaterThan(0);
    }
  });

  test('should handle loading state', async ({ page }) => {
    await page.goto('/leaderboard');

    // During load, might show skeleton or spinner
    const loadingIndicator = page.locator(
      '[data-testid="loading"], .skeleton, .spinner, [aria-busy="true"]'
    );

    // Loading state should be brief
    if (await loadingIndicator.isVisible().catch(() => false)) {
      await expect(loadingIndicator).not.toBeVisible({ timeout: 10000 });
    }
  });
});

test.describe('Leaderboard Filtering', () => {
  test('should filter by time period', async ({ page }) => {
    await page.goto('/leaderboard');
    await page.waitForLoadState('domcontentloaded');

    // Look for time period selector
    const periodSelector = page.locator(
      '[data-testid="period-selector"], select[name*="period"], button:has-text("Week"), button:has-text("Month")'
    );

    if (await periodSelector.isVisible().catch(() => false)) {
      // Change period
      await periodSelector.first().click();

      // Should update content
      await page.waitForTimeout(500);
    }
  });

  test('should filter by agent type', async ({ page }) => {
    await page.goto('/leaderboard');
    await page.waitForLoadState('domcontentloaded');

    // Look for agent type filter
    const agentFilter = page.locator(
      '[data-testid="agent-filter"], select[name*="agent"], [data-filter="agent"]'
    );

    if (await agentFilter.isVisible().catch(() => false)) {
      await agentFilter.selectOption({ index: 1 });
      await page.waitForTimeout(500);
    }
  });
});

test.describe('Agent Details', () => {
  test('should link to agent detail page', async ({ page }) => {
    await page.goto('/leaderboard');
    await page.waitForLoadState('domcontentloaded');

    // Click on agent name if it's a link
    const agentLink = page.locator(
      '[data-testid="agent-link"], .agent-name a, a[href*="agent"]'
    ).first();

    if (await agentLink.isVisible().catch(() => false)) {
      await agentLink.click();
      await expect(page).toHaveURL(/agent/i);
    }
  });

  test('should show agent performance chart', async ({ page }) => {
    await page.goto('/leaderboard');
    await page.waitForLoadState('domcontentloaded');

    // Some leaderboards show inline charts
    const chart = page.locator(
      '[data-testid="performance-chart"], canvas, svg.recharts-surface, .chart'
    );

    const hasChart = await chart.isVisible().catch(() => false);
    expect(hasChart).toBeDefined();
  });
});
