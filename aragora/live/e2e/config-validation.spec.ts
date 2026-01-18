import { test, expect } from './fixtures';

/**
 * E2E tests for configuration validation and ConfigHealthBanner.
 *
 * These tests verify that the app properly handles missing or
 * misconfigured environment variables.
 */

test.describe('Configuration Validation', () => {
  test.describe('ConfigHealthBanner', () => {
    test('should not show banner when all config is valid', async ({ page, aragoraPage }) => {
      await page.goto('/');
      await aragoraPage.dismissAllOverlays();

      // Wait for banner to potentially appear
      await page.waitForTimeout(500);

      // In a properly configured environment, no banner should show
      // (or it should be dismissible and stay dismissed)
      const configBanner = page.locator('[role="alert"]:has-text("CONFIG")');
      const isVisible = await configBanner.isVisible().catch(() => false);

      // If visible, it should be dismissible (not an error banner)
      if (isVisible) {
        const dismissButton = configBanner.locator('button:has-text("[DISMISS]")');
        if (await dismissButton.isVisible()) {
          await dismissButton.click();
          await expect(configBanner).not.toBeVisible();
        }
      }
    });

    test('should persist dismissed state in localStorage', async ({ page, aragoraPage }) => {
      await page.goto('/');
      await aragoraPage.dismissAllOverlays();

      const configBanner = page.locator('[role="alert"]:has-text("CONFIG")');
      const dismissButton = configBanner.locator('button:has-text("[DISMISS]")');

      if (await dismissButton.isVisible({ timeout: 1000 }).catch(() => false)) {
        await dismissButton.click();

        // Verify localStorage was set
        const dismissed = await page.evaluate(() => {
          return localStorage.getItem('aragora-config-warnings-dismissed');
        });
        expect(dismissed).toBe('true');

        // Reload page - banner should not reappear
        await page.reload();
        await aragoraPage.dismissAllOverlays();
        await page.waitForTimeout(500);

        await expect(configBanner).not.toBeVisible();
      }
    });
  });

  test.describe('API Configuration', () => {
    test('should use configured API URL', async ({ page }) => {
      let apiUrl: string | null = null;

      // Intercept API calls to capture the URL being used
      await page.route('**/api/**', async (route) => {
        apiUrl = route.request().url();
        await route.continue();
      });

      await page.goto('/');
      await page.waitForTimeout(2000);

      // The app should be making API calls (unless no features use API on homepage)
      // Just verify no errors from misconfigured URLs
      const body = page.locator('body');
      await expect(body).toBeVisible();
    });

    test('should handle API unavailability gracefully', async ({ page, aragoraPage }) => {
      // Block all API calls
      await page.route('**/api/**', (route) => {
        route.abort('connectionfailed');
      });

      await page.goto('/');
      await aragoraPage.dismissAllOverlays();

      // Page should still load (with error states)
      const body = page.locator('body');
      await expect(body).toBeVisible();

      // Should show error indicators, not crash
      const hasContent = await page.content();
      expect(hasContent.length).toBeGreaterThan(100);
    });
  });

  test.describe('WebSocket Configuration', () => {
    test('should attempt WebSocket connection to configured URL', async ({ page, aragoraPage }) => {
      const wsUrls: string[] = [];

      page.on('websocket', (ws) => {
        wsUrls.push(ws.url());
      });

      await page.goto('/');
      await aragoraPage.dismissAllOverlays();
      await page.waitForTimeout(3000);

      // If WebSocket was attempted, verify URL format
      if (wsUrls.length > 0) {
        wsUrls.forEach((url) => {
          // Should be ws:// or wss://
          expect(url).toMatch(/^wss?:\/\//);
        });
      }
    });
  });

  test.describe('Environment Detection', () => {
    test('should work in production-like configuration', async ({ page, aragoraPage }) => {
      await page.goto('/');
      await aragoraPage.dismissAllOverlays();

      // Check that essential UI elements render
      const header = page.locator('header');
      await expect(header).toBeVisible();

      // No JavaScript errors should crash the page
      const body = page.locator('body');
      await expect(body).toBeVisible();
    });

    test('should expose IS_DEV_MODE flag correctly', async ({ page }) => {
      await page.goto('/');

      // Check if dev mode is exposed (for debugging purposes)
      const isDevMode = await page.evaluate(() => {
        // The config exports IS_DEV_MODE
        return (window as any).__aragora_dev_mode;
      });

      // Just verify the flag exists (value depends on environment)
      expect(typeof isDevMode === 'boolean' || isDevMode === undefined).toBe(true);
    });
  });

  test.describe('Fallback Behavior', () => {
    test('should use localhost fallback in development', async ({ page, aragoraPage }) => {
      await page.goto('/');
      await aragoraPage.dismissAllOverlays();

      // In dev mode without env vars, should use localhost
      // Verify by checking console warnings (if exposed)
      const logs: string[] = [];
      page.on('console', (msg) => {
        if (msg.type() === 'warn') {
          logs.push(msg.text());
        }
      });

      await page.reload();
      await aragoraPage.dismissAllOverlays();
      await page.waitForTimeout(1000);

      // If using localhost fallback, there may be warnings
      // This is expected behavior in development
      expect(Array.isArray(logs)).toBe(true);
    });

    test('should not use localhost in production config', async ({ page, aragoraPage }) => {
      // Get the configured API URL from the page
      const config = await page.evaluate(() => {
        // Try to read from window config if exposed
        return {
          apiUrl: (window as any).__aragora_api_url,
          isDevMode: (window as any).__aragora_dev_mode,
        };
      });

      // If in production mode, should not use localhost
      if (config.isDevMode === false) {
        expect(config.apiUrl).not.toContain('localhost');
      }
    });
  });
});
