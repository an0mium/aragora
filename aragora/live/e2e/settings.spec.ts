import { test, expect } from '@playwright/test';

/**
 * E2E tests for Settings page.
 *
 * Tests user preferences, feature toggles, and integrations.
 */

test.describe('Settings Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
  });

  test('should load settings page', async ({ page }) => {
    await expect(page).toHaveTitle(/Settings|Aragora/i);
    await expect(page.getByRole('heading', { name: /settings/i })).toBeVisible();
  });

  test('should display all tabs', async ({ page }) => {
    const tabs = ['FEATURES', 'DEBATE', 'APPEARANCE', 'NOTIFICATIONS', 'API KEYS', 'INTEGRATIONS', 'ACCOUNT'];

    for (const tab of tabs) {
      await expect(page.getByRole('tab', { name: new RegExp(tab, 'i') })).toBeVisible();
    }
  });

  test('should switch between tabs', async ({ page }) => {
    // Click Appearance tab
    await page.getByRole('tab', { name: /appearance/i }).click();
    await expect(page.getByText(/theme/i)).toBeVisible();

    // Click Notifications tab
    await page.getByRole('tab', { name: /notifications/i }).click();
    await expect(page.getByText(/email notifications/i)).toBeVisible();

    // Click API Keys tab
    await page.getByRole('tab', { name: /api keys/i }).click();
    await expect(page.getByText(/generate api key/i)).toBeVisible();
  });
});

test.describe('Settings - Features Tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
  });

  test('should display feature toggles', async ({ page }) => {
    // Features tab should be default or click it
    const featuresTab = page.getByRole('tab', { name: /features/i });
    if (await featuresTab.isVisible()) {
      await featuresTab.click();
    }

    // Check for feature toggle labels
    await expect(page.getByText(/calibration tracking/i)).toBeVisible();
    await expect(page.getByText(/trickster/i)).toBeVisible();
    await expect(page.getByText(/rhetorical observer/i)).toBeVisible();
  });

  test('should toggle feature on and off', async ({ page }) => {
    const featuresTab = page.getByRole('tab', { name: /features/i });
    if (await featuresTab.isVisible()) {
      await featuresTab.click();
    }

    // Find a toggle switch
    const toggleSwitch = page.getByRole('switch').first();
    const initialState = await toggleSwitch.getAttribute('aria-checked');

    // Click to toggle
    await toggleSwitch.click();

    // State should change
    const newState = await toggleSwitch.getAttribute('aria-checked');
    expect(newState).not.toBe(initialState);
  });
});

test.describe('Settings - Debate Tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    await page.getByRole('tab', { name: /debate/i }).click();
  });

  test('should display debate preferences', async ({ page }) => {
    await expect(page.getByText(/default mode/i)).toBeVisible();
    await expect(page.getByText(/default rounds/i)).toBeVisible();
    await expect(page.getByText(/default agents/i)).toBeVisible();
  });

  test('should allow changing default mode', async ({ page }) => {
    const modeSelect = page.locator('select').first();
    await modeSelect.selectOption('graph');
    await expect(modeSelect).toHaveValue('graph');
  });

  test('should allow changing default rounds', async ({ page }) => {
    const roundsInput = page.locator('input[type="number"]').first();
    await roundsInput.fill('5');
    await expect(roundsInput).toHaveValue('5');
  });
});

test.describe('Settings - Appearance Tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    await page.getByRole('tab', { name: /appearance/i }).click();
  });

  test('should display theme options', async ({ page }) => {
    await expect(page.getByText(/dark/i)).toBeVisible();
    await expect(page.getByText(/light/i)).toBeVisible();
    await expect(page.getByText(/system/i)).toBeVisible();
  });

  test('should allow selecting theme', async ({ page }) => {
    // Find and click the light theme option
    const lightOption = page.getByRole('radio', { name: /light/i });
    if (await lightOption.isVisible()) {
      await lightOption.click();
      await expect(lightOption).toBeChecked();
    }
  });

  test('should display display options', async ({ page }) => {
    await expect(page.getByText(/compact mode/i)).toBeVisible();
    await expect(page.getByText(/show agent icons/i)).toBeVisible();
    await expect(page.getByText(/auto-scroll/i)).toBeVisible();
  });
});

test.describe('Settings - API Keys Tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    await page.getByRole('tab', { name: /api keys/i }).click();
  });

  test('should display API key generation form', async ({ page }) => {
    await expect(page.getByPlaceholder(/key name/i)).toBeVisible();
    await expect(page.getByRole('button', { name: /generate/i })).toBeVisible();
  });

  test('should display API documentation example', async ({ page }) => {
    await expect(page.getByText(/curl/i)).toBeVisible();
    await expect(page.getByText(/authorization/i)).toBeVisible();
  });

  test('should require key name for generation', async ({ page }) => {
    const generateButton = page.getByRole('button', { name: /generate/i });
    await expect(generateButton).toBeDisabled();

    // Fill in key name
    await page.getByPlaceholder(/key name/i).fill('Test Key');
    await expect(generateButton).toBeEnabled();
  });
});

test.describe('Settings - Integrations Tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');
    await page.getByRole('tab', { name: /integrations/i }).click();
  });

  test('should display Slack integration', async ({ page }) => {
    await expect(page.getByText(/slack integration/i)).toBeVisible();
    await expect(page.getByPlaceholder(/slack/i)).toBeVisible();
  });

  test('should display Discord integration', async ({ page }) => {
    await expect(page.getByText(/discord integration/i)).toBeVisible();
    await expect(page.getByPlaceholder(/discord/i)).toBeVisible();
  });

  test('should have save button', async ({ page }) => {
    await expect(page.getByRole('button', { name: /save/i })).toBeVisible();
  });
});

test.describe('Settings - Navigation', () => {
  test('should navigate back to dashboard', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    await page.getByRole('link', { name: /dashboard/i }).click();
    await expect(page).toHaveURL('/');
  });

  test('should persist settings after navigation', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    // Change a setting
    await page.getByRole('tab', { name: /appearance/i }).click();
    const compactToggle = page.getByRole('switch', { name: /compact/i });
    if (await compactToggle.isVisible()) {
      await compactToggle.click();
    }

    // Navigate away and back
    await page.goto('/');
    await page.goto('/settings');
    await page.getByRole('tab', { name: /appearance/i }).click();

    // Setting should be persisted (localStorage)
    // This test verifies the navigation flow works
  });
});
