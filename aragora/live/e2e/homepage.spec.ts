import { test, expect } from './fixtures';

/**
 * E2E tests for the Aragora homepage and navigation.
 */

test.describe('Homepage', () => {
  test('should load successfully', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();
    await expect(page).toHaveURL(/\/landing\/?$/);

    // Should have a title
    await expect(page).toHaveTitle(/Aragora/i);

    // Should show main heading or logo
    const heading = page.locator('h1, [data-testid="logo"]').first();
    await expect(heading).toBeVisible();
  });

  test('should display navigation', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // The public landing surface exposes public-nav links rather than app-sidebar links.
    const navLinks = page.locator('a[href="/about"], a[href="/pricing"], a[href="/docs"], a[href="/signup"]');
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
        !err.includes('404') &&
        !err.includes('500 (Internal Server Error)')
    );

    expect(unexpectedErrors).toHaveLength(0);
  });

  test('should have accessible page structure', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // The standalone landing route may render without an explicit main landmark.
    const primarySurface = page.locator('main, [role="main"], body').first();
    await expect(primarySurface).toBeVisible();

    // Should have skip link or proper heading structure
    const headings = page.locator('h1, h2, h3');
    const headingCount = await headings.count();
    expect(headingCount).toBeGreaterThan(0);
  });
});

test.describe('Navigation', () => {
  test('should navigate to about page from the public landing surface', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    const aboutLink = page.locator('a[href="/about"]').first();
    await aboutLink.click();
    await expect(page).toHaveURL(/\/about\/?$/);
  });

  test('should navigate to pricing from the public landing surface', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    const pricingLink = page.locator('a[href="/pricing"]').last();
    await pricingLink.scrollIntoViewIfNeeded();
    await pricingLink.click();
    await expect(page).toHaveURL(/\/pricing\/?$/);
  });

  test('should navigate back to the landing page from a public page', async ({ page, aragoraPage }) => {
    await page.goto('/quickstart');
    await aragoraPage.dismissAllOverlays();

    const homeLink = page.locator('a[href="/landing"], a[href="/"], [data-testid="logo"]').first();
    await homeLink.click();
    await expect(page).toHaveURL(/\/landing\/?$/);
  });
});
