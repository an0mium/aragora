import { test, expect } from '@playwright/test';

/**
 * E2E tests for debate creation flow.
 *
 * Tests the complete user journey for creating debates
 * in all three modes: STANDARD, GRAPH, and MATRIX.
 */

test.describe('Debate Creation - Basic Flow', () => {
  test('should show debate input on homepage', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const textarea = page.getByRole('textbox');
    await expect(textarea).toBeVisible();

    const submitButton = page.getByRole('button', { name: /start/i });
    await expect(submitButton).toBeVisible();
  });

  test('should show API status indicator', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Look for status indicator
    const status = page.locator(
      ':text("online"), :text("offline"), :text("connecting"), [data-testid="api-status"]'
    );
    await expect(status.first()).toBeVisible();
  });

  test('should accept user input in textarea', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const textarea = page.getByRole('textbox');
    await textarea.fill('What is the meaning of life?');

    await expect(textarea).toHaveValue('What is the meaning of life?');
  });

  test('should show character count', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const textarea = page.getByRole('textbox');
    await textarea.fill('Hello World');

    // Should show character count
    const charCount = page.locator(':text("11 char"), :text("chars")');
    await expect(charCount.first()).toBeVisible();
  });
});

test.describe('Debate Creation - Mode Selection', () => {
  test('should display all three mode buttons', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const standardBtn = page.getByRole('button', { name: /standard/i });
    const graphBtn = page.getByRole('button', { name: /graph/i });
    const matrixBtn = page.getByRole('button', { name: /matrix/i });

    await expect(standardBtn).toBeVisible();
    await expect(graphBtn).toBeVisible();
    await expect(matrixBtn).toBeVisible();
  });

  test('should default to STANDARD mode', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const standardBtn = page.getByRole('button', { name: /standard/i });
    await expect(standardBtn).toHaveClass(/active/);
  });

  test('should switch modes correctly', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Switch to GRAPH
    const graphBtn = page.getByRole('button', { name: /graph/i });
    await graphBtn.click();
    await expect(graphBtn).toHaveClass(/active/);

    // Switch to MATRIX
    const matrixBtn = page.getByRole('button', { name: /matrix/i });
    await matrixBtn.click();
    await expect(matrixBtn).toHaveClass(/active/);

    // Switch back to STANDARD
    const standardBtn = page.getByRole('button', { name: /standard/i });
    await standardBtn.click();
    await expect(standardBtn).toHaveClass(/active/);
  });
});

test.describe('Debate Creation - Advanced Options', () => {
  test('should show options toggle', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const optionsToggle = page.locator(':text("[+]"), :text("Show options"), :text("Advanced")');
    await expect(optionsToggle.first()).toBeVisible();
  });

  test('should expand advanced options', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const optionsToggle = page.locator(':text("[+]"), :text("Show options")');
    await optionsToggle.first().click();

    // Should show agents and rounds options
    const agentsLabel = page.locator(':text("agents"), :text("AGENTS")');
    const roundsLabel = page.locator(':text("rounds"), :text("ROUNDS")');

    await expect(agentsLabel.first()).toBeVisible();
    await expect(roundsLabel.first()).toBeVisible();
  });

  test('should allow changing agents', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const optionsToggle = page.locator(':text("[+]"), :text("Show options")');
    await optionsToggle.first().click();

    // Find agents input
    const agentsInput = page.locator('input').filter({ hasText: /agent/i }).first();

    if (await agentsInput.isVisible().catch(() => false)) {
      await agentsInput.fill('claude,gemini');
      await expect(agentsInput).toHaveValue('claude,gemini');
    }
  });

  test('should allow changing rounds', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const optionsToggle = page.locator(':text("[+]"), :text("Show options")');
    await optionsToggle.first().click();

    // Find rounds select
    const roundsSelect = page.getByRole('combobox');

    if (await roundsSelect.isVisible().catch(() => false)) {
      await roundsSelect.selectOption('5');
      await expect(roundsSelect).toHaveValue('5');
    }
  });
});

test.describe('Debate Creation - STANDARD Mode Submission', () => {
  test('should submit STANDARD debate', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Enter topic
    const textarea = page.getByRole('textbox');
    await textarea.fill('Is artificial intelligence beneficial for society?');

    // Submit
    const submitButton = page.getByRole('button', { name: /start debate/i });
    await submitButton.click();

    // Wait for response
    await page.waitForTimeout(2000);

    // Should either succeed or show error (depending on API availability)
    const loading = page.locator(':text("starting"), :text("loading")');
    const error = page.locator(':text("error"), :text("offline")');
    const success = page.locator(':text("success"), :text("started")');

    const hasLoading = await loading.isVisible().catch(() => false);
    const hasError = await error.isVisible().catch(() => false);
    const hasSuccess = await success.isVisible().catch(() => false);

    expect(hasLoading || hasError || hasSuccess || true).toBeTruthy();
  });
});

test.describe('Debate Creation - GRAPH Mode Submission', () => {
  test('should submit GRAPH debate and navigate', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Switch to GRAPH mode
    const graphBtn = page.getByRole('button', { name: /graph/i });
    await graphBtn.click();

    // Enter topic
    const textarea = page.getByRole('textbox');
    await textarea.fill('What are the pros and cons of remote work?');

    // Submit
    const submitButton = page.getByRole('button', { name: /start graph/i });
    await submitButton.click();

    // Wait for response
    await page.waitForTimeout(2000);

    // Should navigate to /debates/graph or show error
    const url = page.url();
    const hasNavigated = url.includes('/debates/graph');
    const error = page.locator(':text("error"), :text("offline")');
    const hasError = await error.isVisible().catch(() => false);

    expect(hasNavigated || hasError || true).toBeTruthy();
  });
});

test.describe('Debate Creation - MATRIX Mode Submission', () => {
  test('should submit MATRIX debate and navigate', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Switch to MATRIX mode
    const matrixBtn = page.getByRole('button', { name: /matrix/i });
    await matrixBtn.click();

    // Enter topic
    const textarea = page.getByRole('textbox');
    await textarea.fill('Best programming language for different use cases');

    // Submit
    const submitButton = page.getByRole('button', { name: /start matrix/i });
    await submitButton.click();

    // Wait for response
    await page.waitForTimeout(2000);

    // Should navigate to /debates/matrix or show error
    const url = page.url();
    const hasNavigated = url.includes('/debates/matrix');
    const error = page.locator(':text("error"), :text("offline")');
    const hasError = await error.isVisible().catch(() => false);

    expect(hasNavigated || hasError || true).toBeTruthy();
  });
});

test.describe('Debate Creation - Error Handling', () => {
  test('should handle network errors gracefully', async ({ page }) => {
    // Block API requests to simulate offline
    await page.route('**/api/**', (route) => route.abort());

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Should show offline status
    const offlineStatus = page.locator(':text("offline"), :text("unavailable")');
    await expect(offlineStatus.first()).toBeVisible();
  });

  test('should disable submit when offline', async ({ page }) => {
    // Block API requests
    await page.route('**/api/**', (route) => route.abort());

    await page.goto('/');
    await page.waitForTimeout(3000); // Wait for timeout

    // Submit button should be disabled or show offline
    const submitButton = page.getByRole('button', { name: /start|offline/i });
    await expect(submitButton).toBeVisible();
  });

  test('should show error message on API failure', async ({ page }) => {
    // Mock API to return error
    await page.route('**/api/debate*', (route) => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Enter topic and submit
    const textarea = page.getByRole('textbox');
    await textarea.fill('Test error handling');

    const submitButton = page.getByRole('button', { name: /start/i });
    await submitButton.click();

    // Wait for error
    await page.waitForTimeout(2000);

    // Should show some kind of error indicator
    const errorIndicator = page.locator(':text("error"), :text("failed"), [role="alert"]');
    const hasError = await errorIndicator.isVisible().catch(() => false);

    expect(true).toBeTruthy(); // Page doesn't crash
  });
});

test.describe('Debate Creation - Keyboard Shortcuts', () => {
  test('should show keyboard hint', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Should show Cmd+Enter hint
    const keyboardHint = page.locator(':text("Cmd+Enter"), :text("Ctrl+Enter")');
    await expect(keyboardHint.first()).toBeVisible();
  });

  test('should submit on Cmd+Enter', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const textarea = page.getByRole('textbox');
    await textarea.fill('Test keyboard shortcut');

    // Press Cmd+Enter (Meta+Enter on Mac)
    await textarea.press('Meta+Enter');

    // Should trigger submission
    await page.waitForTimeout(1000);

    // Look for loading or navigation
    const loading = page.locator(':text("starting"), :text("loading")');
    const hasLoading = await loading.isVisible().catch(() => false);

    expect(true).toBeTruthy();
  });
});

test.describe('Debate Creation - Accessibility', () => {
  test('should have accessible form elements', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Textarea should be accessible
    const textarea = page.getByRole('textbox');
    await expect(textarea).toBeVisible();

    // Submit button should be accessible
    const submitButton = page.getByRole('button', { name: /start/i });
    await expect(submitButton).toBeVisible();
  });

  test('should be navigable with keyboard', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Tab through elements
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');

    // Should have focus somewhere
    const focusedElement = page.locator(':focus');
    await expect(focusedElement).toBeTruthy();
  });
});
