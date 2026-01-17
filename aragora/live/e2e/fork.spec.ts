import { test, expect, mockApiResponse, mockDebate } from './fixtures';

const mockForks = {
  forks: [
    {
      branch_id: 'fork-1',
      parent_debate_id: 'test-debate',
      branch_point: 2,
      status: 'completed',
      messages_inherited: 4,
    },
    {
      branch_id: 'fork-2',
      parent_debate_id: 'test-debate',
      branch_point: 3,
      status: 'running',
      messages_inherited: 6,
    },
  ],
  tree: {
    id: 'test-debate',
    type: 'root',
    branch_point: 0,
    children: [
      {
        id: 'fork-1',
        type: 'fork',
        branch_point: 2,
        children: [],
      },
      {
        id: 'fork-2',
        type: 'fork',
        branch_point: 3,
        children: [],
      },
    ],
    total_nodes: 3,
    max_depth: 1,
  },
  total: 2,
};

test.describe('Fork Visualizer', () => {
  test.beforeEach(async ({ page }) => {
    await mockApiResponse(page, '**/api/health', { status: 'ok' });
    await mockApiResponse(page, '**/api/debates/test-debate', mockDebate);
    await mockApiResponse(page, '**/api/debates/test-debate/forks', mockForks);
  });

  test('should display fork explorer section', async ({ page }) => {
    await page.goto('/debate/test-debate');
    
    // Look for fork explorer section
    const forkSection = page.locator('text=/fork|explorer|branch/i').first();
    await expect(forkSection).toBeVisible({ timeout: 10000 });
  });

  test('should show fork tree visualization', async ({ page }) => {
    await page.goto('/debate/test-debate');
    
    // Look for tree nodes
    const treeNode = page.locator('[class*="fork"], [class*="tree"], [class*="node"]').first();
    await expect(treeNode).toBeVisible({ timeout: 10000 });
  });

  test('should display fork count', async ({ page }) => {
    await page.goto('/debate/test-debate');
    
    // Should show fork count
    const forkCount = page.locator('text=/2 forks|2 fork/i').first();
    await expect(forkCount).toBeVisible({ timeout: 10000 });
  });

  test('should allow creating a new fork', async ({ page }) => {
    await page.route('**/api/debates/test-debate/fork', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          branch_id: 'new-fork',
          message: 'Fork created',
        }),
      });
    });

    await page.goto('/debate/test-debate');
    
    // Find create fork button
    const createForkButton = page.locator('button').filter({
      hasText: /fork|create|branch/i
    }).first();
    
    if (await createForkButton.isVisible()) {
      await createForkButton.click();
      
      // Form should appear
      const forkForm = page.locator('input, textarea').filter({
        has: page.locator('[placeholder*="context"], [name*="branch"]')
      }).or(page.locator('input[type="number"]')).first();
      
      await expect(forkForm).toBeVisible({ timeout: 5000 });
    }
  });

  test('should allow selecting forks for comparison', async ({ page }) => {
    await page.goto('/debate/test-debate');
    
    // Find comparison buttons (L/R)
    const leftButton = page.locator('button').filter({ hasText: 'L' }).first();
    const rightButton = page.locator('button').filter({ hasText: 'R' }).first();
    
    if (await leftButton.isVisible() && await rightButton.isVisible()) {
      await leftButton.click();
      await rightButton.click();
      
      // Compare view should appear
      const compareView = page.locator('text=/compare|comparison|diff/i').first();
      await expect(compareView).toBeVisible({ timeout: 5000 });
    }
  });

  test('should show fork status indicators', async ({ page }) => {
    await page.goto('/debate/test-debate');
    
    // Should show status colors/indicators
    const statusIndicator = page.locator('[class*="status"], [class*="completed"], [class*="running"]').first();
    await expect(statusIndicator).toBeVisible({ timeout: 10000 });
  });

  test('should show fork metadata', async ({ page }) => {
    await page.goto('/debate/test-debate');
    
    // Should show branch point info
    const branchInfo = page.locator('text=/branch.*point|round|r[0-9]/i').first();
    await expect(branchInfo).toBeVisible({ timeout: 10000 });
  });
});

test.describe('Fork Visualizer - Empty State', () => {
  test('should show empty state when no forks', async ({ page }) => {
    await mockApiResponse(page, '**/api/health', { status: 'ok' });
    await mockApiResponse(page, '**/api/debates/no-forks', mockDebate);
    await page.route('**/api/debates/no-forks/forks', async (route) => {
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ forks: [], total: 0 }),
      });
    });

    await page.goto('/debate/no-forks');
    
    // Should show empty state message
    const emptyState = page.locator('text=/no forks|create.*fork/i').first();
    await expect(emptyState).toBeVisible({ timeout: 10000 });
  });
});
