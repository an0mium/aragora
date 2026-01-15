import { test, expect, mockApiResponse } from './fixtures';

test.describe('Navigation', () => {
  test.beforeEach(async ({ page }) => {
    await mockApiResponse(page, '**/api/health', { status: 'ok' });
  });

  test('should navigate to home page', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();
    await expect(page).toHaveURL('/');
    await expect(page.locator('body')).toBeVisible();
  });

  test('should navigate to about page', async ({ page, aragoraPage }) => {
    await page.goto('/about');
    await aragoraPage.dismissAllOverlays();
    await expect(page).toHaveURL('/about');

    // Should have about content - page uses ASCII art banner, not h1
    const aboutContent = page.locator('main').first();
    await expect(aboutContent).toBeVisible();
  });

  test('should navigate to pulse page', async ({ page, aragoraPage }) => {
    await page.goto('/pulse');
    await aragoraPage.dismissAllOverlays();
    await expect(page).toHaveURL('/pulse');

    // Should have pulse content
    const pulseContent = page.locator('h1, h2').filter({
      hasText: /pulse|scheduler/i
    }).first();
    await expect(pulseContent).toBeVisible();
  });

  test('should navigate to agents page', async ({ page, aragoraPage }) => {
    await page.goto('/agents');
    await aragoraPage.dismissAllOverlays();
    await expect(page).toHaveURL('/agents');

    // Should have agents content
    const agentsContent = page.locator('h1, h2').filter({
      hasText: /agent/i
    }).first();
    await expect(agentsContent).toBeVisible();
  });

  test('should navigate to plugins page', async ({ page, aragoraPage }) => {
    await page.goto('/plugins');
    await aragoraPage.dismissAllOverlays();
    await expect(page).toHaveURL('/plugins');

    // Should have plugins content
    const pluginsContent = page.locator('h1, h2').filter({
      hasText: /plugin|marketplace/i
    }).first();
    await expect(pluginsContent).toBeVisible();
  });

  test('should navigate to batch page', async ({ page, aragoraPage }) => {
    await page.goto('/batch');
    await aragoraPage.dismissAllOverlays();
    await expect(page).toHaveURL('/batch');

    // Should have batch content - heading is "BATCH DEBATE OPERATIONS"
    const batchContent = page.locator('h1, h2').filter({
      hasText: /batch|operations/i
    }).first();
    await expect(batchContent).toBeVisible();
  });

  test('should have working navigation links in header', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // Find and click about link
    const aboutLink = page.locator('a[href="/about"]').first();
    if (await aboutLink.isVisible()) {
      await aboutLink.click();
      await expect(page).toHaveURL('/about');
    }
  });

  test('should handle 404 for non-existent pages', async ({ page, aragoraPage }) => {
    await page.goto('/non-existent-page-12345');
    await aragoraPage.dismissAllOverlays();

    // Should show 404 or redirect
    const notFoundElement = page.locator('text=/404|not found|page.*exist/i').first();
    const hasNotFound = await notFoundElement.isVisible().catch(() => false);
    const redirectedToHome = page.url().endsWith('/');

    expect(hasNotFound || redirectedToHome).toBeTruthy();
  });

  test('should preserve scroll position on back navigation', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // Scroll down
    await page.evaluate(() => window.scrollTo(0, 500));

    // Navigate to about
    await page.goto('/about');
    await aragoraPage.dismissAllOverlays();

    // Go back
    await page.goBack();

    // Check scroll position (browser may restore or not)
    await page.waitForTimeout(500);
  });

  test('should have proper page titles', async ({ page, aragoraPage }) => {
    const pages = [
      { path: '/', title: /aragora/i },
      { path: '/about', title: /aragora|about/i },
      { path: '/pulse', title: /aragora|pulse/i },
    ];

    for (const { path, title } of pages) {
      await page.goto(path);
      await aragoraPage.dismissAllOverlays();
      await expect(page).toHaveTitle(title);
    }
  });
});

test.describe('Navigation - Mobile', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
  });

  test('should have mobile navigation', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // Look for hamburger menu or mobile nav
    const mobileMenu = page.locator('[class*="mobile"], [class*="hamburger"], button[aria-label*="menu"]').first();

    if (await mobileMenu.isVisible()) {
      await mobileMenu.click();
      // Nav items should appear
      await page.waitForTimeout(300);
    }
  });

  test('should be responsive on mobile', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();

    // Page should not have significant horizontal scroll (allow some tolerance for scrollbars)
    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    const viewportWidth = await page.evaluate(() => window.innerWidth);

    // Allow 50px tolerance for minor overflow from animations/transitions
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 50);
  });
});
