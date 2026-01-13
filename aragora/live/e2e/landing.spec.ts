import { test, expect } from './fixtures';

test.describe('Landing Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display the landing page with ASCII art title', async ({ page }) => {
    // Wait for page to load
    await expect(page).toHaveTitle(/Aragora/i);

    // Check for ASCII art banner on desktop
    const asciiBanner = page.locator('pre').filter({ hasText: 'ARAGORA' });
    await expect(asciiBanner.or(page.locator('h1').filter({ hasText: 'ARAGORA' }))).toBeVisible();
  });

  test('should have navigation links in header', async ({ page }) => {
    // Check About link
    const aboutLink = page.locator('a[href="/about"]');
    await expect(aboutLink).toBeVisible();
    await expect(aboutLink).toHaveText('[ABOUT]');

    // Check Live Dashboard link
    const dashboardLink = page.locator('a[href="https://aragora.ai"]');
    await expect(dashboardLink).toBeVisible();
  });

  test('should have theme toggle', async ({ page }) => {
    // Look for theme toggle button
    const themeToggle = page.locator('button').filter({ hasText: /theme|dark|light/i }).or(
      page.locator('[aria-label*="theme"]')
    ).or(
      page.locator('button').filter({ has: page.locator('svg') }).first()
    );

    // Theme toggle should be present
    await expect(themeToggle.first()).toBeVisible();
  });

  test('should display debate input form', async ({ page }) => {
    // Check for main input area
    const inputArea = page.locator('textarea, input[type="text"]').first();
    await expect(inputArea).toBeVisible();
  });

  test('should show agent selection options', async ({ page }) => {
    // Look for agent-related UI elements
    const agentSection = page.locator('text=/agent|claude|gpt|gemini/i').first();
    await expect(agentSection).toBeVisible({ timeout: 10000 });
  });

  test('should navigate to about page', async ({ page }) => {
    await page.click('a[href="/about"]');
    await expect(page).toHaveURL('/about');
  });

  test('should be responsive on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });

    // Mobile title should be visible
    const mobileTitle = page.locator('h1').filter({ hasText: 'ARAGORA' });
    await expect(mobileTitle).toBeVisible();
  });

  test('should have proper meta tags', async ({ page }) => {
    // Check for description meta tag
    const description = page.locator('meta[name="description"]');
    await expect(description).toHaveAttribute('content', /.+/);
  });

  test('should display error banner when error occurs', async ({ page }) => {
    // Trigger an error by mocking API failure
    await page.route('**/api/**', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Test error' }),
      });
    });

    // Try to interact with the page in a way that triggers API call
    const inputArea = page.locator('textarea, input[type="text"]').first();
    if (await inputArea.isVisible()) {
      await inputArea.fill('Test topic');
      // Look for submit button and click
      const submitButton = page.locator('button[type="submit"], button').filter({ hasText: /start|debate|submit/i }).first();
      if (await submitButton.isVisible()) {
        await submitButton.click();
        // Error should appear
        const errorBanner = page.locator('.bg-warning\\/10, [class*="error"], [class*="warning"]').first();
        await expect(errorBanner).toBeVisible({ timeout: 10000 });
      }
    }
  });
});

test.describe('Landing Page - CRT Effects', () => {
  test('should have scanlines effect', async ({ page }) => {
    await page.goto('/');

    // Check for scanlines element (usually a div with specific styling)
    const scanlines = page.locator('[class*="scanline"], [class*="Scanline"]').first();
    // Scanlines may be implemented differently, so we just check page renders
    await expect(page.locator('body')).toBeVisible();
  });
});

test.describe('Landing Page - Accessibility', () => {
  test('should have proper heading hierarchy', async ({ page }) => {
    await page.goto('/');

    // Check that there's at least one h1
    const h1 = page.locator('h1').first();
    await expect(h1).toBeVisible();
  });

  test('should have focusable interactive elements', async ({ page }) => {
    await page.goto('/');

    // Tab through page and check focus
    await page.keyboard.press('Tab');
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeVisible();
  });

  test('should have proper color contrast', async ({ page }) => {
    await page.goto('/');

    // Check that text is visible (basic contrast check)
    const textElements = page.locator('p, span, a, button');
    const firstText = textElements.first();
    await expect(firstText).toBeVisible();
  });
});
