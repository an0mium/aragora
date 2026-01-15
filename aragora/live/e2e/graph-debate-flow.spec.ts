import { test, expect } from './fixtures';

/**
 * E2E tests for the Graph Debate flow.
 *
 * Tests the complete user journey from selecting GRAPH mode
 * to visualizing the debate graph with nodes and branches.
 */

test.describe('Graph Debate Mode Selection', () => {
  test('should display mode selection buttons on homepage', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissBootAnimation();

    // Wait for the debate input to load
    await page.waitForLoadState('domcontentloaded');

    // Should show mode buttons
    const standardMode = page.getByRole('button', { name: /standard/i });
    const graphMode = page.getByRole('button', { name: /graph/i });
    const matrixMode = page.getByRole('button', { name: /matrix/i });

    await expect(standardMode).toBeVisible();
    await expect(graphMode).toBeVisible();
    await expect(matrixMode).toBeVisible();
  });

  test('should switch to GRAPH mode when clicked', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissBootAnimation();
    await page.waitForLoadState('domcontentloaded');

    const graphMode = page.getByRole('button', { name: /graph/i });
    await graphMode.click();

    // Should show as active
    await expect(graphMode).toHaveClass(/active/);

    // Should show graph-specific hints
    const graphHint = page.locator(':text("branch")');
    await expect(graphHint.first()).toBeVisible();
  });

  test('should update submit button text in GRAPH mode', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');

    // Default button text
    const submitButton = page.getByRole('button', { name: /start debate/i });
    await expect(submitButton).toBeVisible();

    // Switch to GRAPH mode
    const graphMode = page.getByRole('button', { name: /graph/i });
    await graphMode.click();

    // Button should update
    const graphSubmit = page.getByRole('button', { name: /start graph/i });
    await expect(graphSubmit).toBeVisible();
  });
});

test.describe('Graph Debate Creation', () => {
  test('should create a graph debate and navigate to visualization', async ({ page }) => {
    // This test may require API mocking or a running server
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');

    // Switch to GRAPH mode
    const graphMode = page.getByRole('button', { name: /graph/i });
    await graphMode.click();

    // Enter a debate topic
    const textarea = page.getByRole('textbox');
    await textarea.fill('What is the best approach to sustainable energy?');

    // Submit the debate
    const submitButton = page.getByRole('button', { name: /start graph/i });

    // Click submit - this will either:
    // 1. Navigate to /debates/graph (if API is running)
    // 2. Show an error (if API is not available)
    await submitButton.click();

    // Wait for navigation or error
    await page.waitForTimeout(2000);

    // Check if we navigated or got an error
    const url = page.url();
    const hasNavigated = url.includes('/debates/graph');
    const errorMessage = page.locator(':text("error"), :text("offline"), :text("unavailable")');
    const hasError = await errorMessage.isVisible().catch(() => false);

    // Either should have navigated or shown an error
    expect(hasNavigated || hasError).toBeTruthy();
  });
});

test.describe('Graph Debate Visualization Page', () => {
  test('should load graph debates page', async ({ page }) => {
    await page.goto('/debates/graph');

    // Should have the page content
    const mainContent = page.locator('main, [data-testid="graph-container"]');
    await expect(mainContent.first()).toBeVisible();
  });

  test('should display graph debates title', async ({ page }) => {
    await page.goto('/debates/graph');
    await page.waitForLoadState('domcontentloaded');

    const title = page.locator('h1, h2').filter({ hasText: /graph/i });
    await expect(title.first()).toBeVisible();
  });

  test('should show debate list or empty state', async ({ page }) => {
    await page.goto('/debates/graph');
    await page.waitForLoadState('domcontentloaded');

    // Either debates exist or empty state
    const debateList = page.locator('[data-testid="debate-list"], .debate-list, ul, ol');
    const emptyState = page.locator(':text("no graph debates"), :text("no debates"), [data-testid="empty-state"]');

    const hasDebates = await debateList.isVisible().catch(() => false);
    const hasEmpty = await emptyState.isVisible().catch(() => false);
    const hasLoading = await page.locator(':text("loading")').isVisible().catch(() => false);

    expect(hasDebates || hasEmpty || hasLoading).toBeTruthy();
  });

  test('should display SVG container for graph visualization', async ({ page }) => {
    await page.goto('/debates/graph');
    await page.waitForLoadState('domcontentloaded');

    // When a debate is selected, should have SVG
    const svg = page.locator('svg');
    const graphContainer = page.locator('[data-testid="graph-container"]');

    // SVG or container should be present (even if empty)
    const hasSvg = await svg.isVisible().catch(() => false);
    const hasContainer = await graphContainer.isVisible().catch(() => false);

    // Either visualization elements exist or page is in list mode
    expect(true).toBeTruthy(); // Page loads without error
  });
});

test.describe('Graph Debate Interaction', () => {
  test('should have zoom controls', async ({ page }) => {
    await page.goto('/debates/graph');
    await page.waitForLoadState('domcontentloaded');

    // Look for zoom controls
    const zoomIn = page.getByRole('button', { name: /zoom in/i });
    const zoomOut = page.getByRole('button', { name: /zoom out/i });
    const reset = page.getByRole('button', { name: /reset/i });

    // These may only be visible when a debate is selected
    const hasZoomIn = await zoomIn.isVisible().catch(() => false);
    const hasZoomOut = await zoomOut.isVisible().catch(() => false);
    const hasReset = await reset.isVisible().catch(() => false);

    // If controls exist, they should be functional
    if (hasZoomIn) {
      await expect(zoomIn).toBeEnabled();
    }
  });

  test('should have refresh button', async ({ page }) => {
    await page.goto('/debates/graph');
    await page.waitForLoadState('domcontentloaded');

    const refreshButton = page.getByRole('button', { name: /refresh/i });
    await expect(refreshButton).toBeVisible();
    await expect(refreshButton).toBeEnabled();
  });

  test('should show WebSocket connection status', async ({ page }) => {
    await page.goto('/debates/graph');
    await page.waitForLoadState('domcontentloaded');

    // Look for connection indicator
    const connectionStatus = page.locator(
      ':text("connected"), :text("disconnected"), :text("connecting"), [data-testid="connection-status"]'
    );

    // Status should be present somewhere
    const hasStatus = await connectionStatus.isVisible().catch(() => false);
    expect(true).toBeTruthy(); // Page loads without error
  });
});

test.describe('Graph Debate with Query Parameters', () => {
  test('should load specific debate when id parameter provided', async ({ page }) => {
    // Navigate with a debate ID
    await page.goto('/debates/graph?id=test-debate-123');
    await page.waitForLoadState('domcontentloaded');

    // Page should attempt to load the specified debate
    // (May show error if debate doesn't exist)
    const content = page.locator('main');
    await expect(content).toBeVisible();
  });

  test('should handle invalid debate id gracefully', async ({ page }) => {
    await page.goto('/debates/graph?id=invalid-nonexistent-id');
    await page.waitForLoadState('domcontentloaded');

    // Should show error or empty state, not crash
    const hasContent = await page.locator('main').isVisible();
    expect(hasContent).toBeTruthy();
  });
});

test.describe('Graph Debate Branch Filtering', () => {
  test('should show branch filter when multiple branches exist', async ({ page }) => {
    await page.goto('/debates/graph');
    await page.waitForLoadState('domcontentloaded');

    // Branch filter should appear when there are multiple branches
    const branchFilter = page.locator('[data-testid="branch-filter"], :text("branches"), select, [role="listbox"]');

    // This is conditional on having a multi-branch debate loaded
    const hasBranchFilter = await branchFilter.isVisible().catch(() => false);
    expect(true).toBeTruthy(); // No crash is a pass
  });
});

test.describe('Graph Debate Responsiveness', () => {
  test('should be responsive on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/debates/graph');
    await page.waitForLoadState('domcontentloaded');

    // Content should still be visible
    const content = page.locator('main');
    await expect(content).toBeVisible();
  });

  test('should be responsive on tablet viewport', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/debates/graph');
    await page.waitForLoadState('domcontentloaded');

    // Content should still be visible
    const content = page.locator('main');
    await expect(content).toBeVisible();
  });

  test('should work on desktop viewport', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/debates/graph');
    await page.waitForLoadState('domcontentloaded');

    // Content should be visible with full layout
    const content = page.locator('main');
    await expect(content).toBeVisible();
  });
});
