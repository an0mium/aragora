import { test, expect, AragoraPage } from './fixtures';

/**
 * E2E tests for the Aragora homepage and navigation.
 */

test.describe('Homepage', () => {
  test('should load successfully', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // Should have a title
    await expect(page).toHaveTitle(/Aragora/i);

    // Should show main heading or logo
    const heading = page.locator('h1, [data-testid="logo"]').first();
    await expect(heading).toBeVisible();
  });

  test('should display navigation', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // Check for navigation elements - homepage uses header/links instead of nav landmark
    const navLinks = page.locator('a[href="/debates"], a[href="/leaderboard"], a[href="/agents"]');
    await expect(navLinks.first()).toBeVisible();
  });

  test('should be responsive on mobile', async ({ page, aragoraPage }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // Page should still be functional
    await expect(page).toHaveTitle(/Aragora/i);

    // Content should not overflow horizontally
    const body = page.locator('body');
    const bodyBox = await body.boundingBox();
    expect(bodyBox?.width).toBeLessThanOrEqual(375);
  });

  test('should have no console errors on load', async ({ page, aragoraPage }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    await page.goto('/');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    // Filter out expected errors:
    // - WebSocket: connection may fail in test environment
    // - favicon: missing favicon is not critical
    // - CORS: expected when testing cross-origin (e.g., localhost -> live.aragora.ai)
    // - ERR_FAILED: usually accompanies CORS errors
    // - 404: some resources may not exist in production
    const unexpectedErrors = consoleErrors.filter(
      (err) =>
        !err.includes('WebSocket') &&
        !err.includes('favicon') &&
        !err.includes('CORS') &&
        !err.includes('ERR_FAILED') &&
        !err.includes('404')
    );

    expect(unexpectedErrors).toHaveLength(0);
  });

  test('should have accessible page structure', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // Should have a main landmark (use first() to handle multiple mains)
    const main = page.locator('main, [role="main"]').first();
    await expect(main).toBeVisible();

    // Should have skip link or proper heading structure
    const headings = page.locator('h1, h2, h3');
    const headingCount = await headings.count();
    expect(headingCount).toBeGreaterThan(0);
  });
});

test.describe('Navigation', () => {
  test('should navigate to debates page', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // Click on debates link
    const debatesLink = page.locator('a[href="/debates"]').first();
    await debatesLink.click();
    await expect(page).toHaveURL(/debate/i);
  });

  test('should navigate to leaderboard', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // Click on leaderboard link
    const leaderboardLink = page.locator('a[href="/leaderboard"]').first();
    await leaderboardLink.click();
    await expect(page).toHaveURL(/leaderboard/i);
  });

  test('should navigate back to homepage from any page', async ({ page, aragoraPage }) => {
    await page.goto('/debates');
    await aragoraPage.dismissAllOverlays();

    // Click on logo or home link
    const homeLink = page.locator('a[href="/"], [data-testid="logo"]').first();
    await homeLink.click();
    await expect(page).toHaveURL(/.*\/$/);
  });
});
