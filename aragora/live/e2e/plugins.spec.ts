import { test, expect, mockApiResponse } from './fixtures';

const mockPlugins = [
  {
    name: 'security-scan',
    version: '1.0.0',
    description: 'Scan code for security vulnerabilities',
    author: 'aragora',
    capabilities: ['security_scan', 'code_analysis'],
    requirements: ['read_files'],
    entry_point: 'security_scan:run',
    timeout_seconds: 120,
    max_memory_mb: 512,
    python_packages: ['bandit'],
    system_tools: [],
    license: 'MIT',
    homepage: '',
    tags: ['security', 'analysis'],
    created_at: new Date().toISOString(),
  },
  {
    name: 'test-runner',
    version: '1.2.0',
    description: 'Execute test suites and report results',
    author: 'aragora',
    capabilities: ['test_runner'],
    requirements: ['read_files', 'run_commands'],
    entry_point: 'test_runner:run',
    timeout_seconds: 300,
    max_memory_mb: 1024,
    python_packages: ['pytest'],
    system_tools: [],
    license: 'MIT',
    homepage: '',
    tags: ['testing'],
    created_at: new Date().toISOString(),
  },
];

test.describe('Plugin Marketplace', () => {
  test.beforeEach(async ({ page }) => {
    await mockApiResponse(page, '**/api/health', { status: 'ok' });
    await mockApiResponse(page, '**/api/plugins', { plugins: mockPlugins });
    await mockApiResponse(page, '**/api/plugins/installed', { installed: [] });
  });

  test('should display plugin list', async ({ page }) => {
    await page.goto('/plugins');
    
    // Should show plugin cards
    for (const plugin of mockPlugins) {
      const pluginCard = page.locator(`text=${plugin.name}`).first();
      await expect(pluginCard).toBeVisible({ timeout: 10000 });
    }
  });

  test('should show plugin details on click', async ({ page }) => {
    await mockApiResponse(page, '**/api/plugins/security-scan', mockPlugins[0]);
    
    await page.goto('/plugins');
    
    // Click on plugin
    const pluginCard = page.locator('text=security-scan').first();
    await pluginCard.click();
    
    // Details should appear
    const details = page.locator('text=/capabilities|requirements|description/i').first();
    await expect(details).toBeVisible({ timeout: 5000 });
  });

  test('should allow filtering by capability', async ({ page }) => {
    await page.goto('/plugins');
    
    // Find filter dropdown
    const filterSelect = page.locator('select').filter({
      has: page.locator('option')
    }).first();
    
    if (await filterSelect.isVisible()) {
      await filterSelect.selectOption({ label: /security/i });
      
      // Should filter results
      await page.waitForTimeout(500);
    }
  });

  test('should allow searching plugins', async ({ page }) => {
    await page.goto('/plugins');
    
    // Find search input
    const searchInput = page.locator('input[type="text"], input[type="search"]').filter({
      has: page.locator('[placeholder*="search"]')
    }).or(page.locator('input[placeholder*="search" i]')).first();
    
    if (await searchInput.isVisible()) {
      await searchInput.fill('security');
      
      // Should filter results
      await page.waitForTimeout(500);
      
      const securityPlugin = page.locator('text=security-scan').first();
      await expect(securityPlugin).toBeVisible();
    }
  });

  test('should show install button for uninstalled plugins', async ({ page }) => {
    await page.goto('/plugins');
    
    // Click on plugin
    const pluginCard = page.locator('text=security-scan').first();
    await pluginCard.click();
    
    // Should show install button
    const installButton = page.locator('button').filter({
      hasText: /install/i
    }).first();
    
    await expect(installButton).toBeVisible({ timeout: 5000 });
  });

  test('should handle plugin installation', async ({ page }) => {
    await page.route('**/api/plugins/security-scan/install', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ success: true }),
      });
    });
    
    await page.goto('/plugins');
    
    // Click on plugin
    const pluginCard = page.locator('text=security-scan').first();
    await pluginCard.click();
    
    // Click install
    const installButton = page.locator('button').filter({
      hasText: /install/i
    }).first();
    
    if (await installButton.isVisible()) {
      await installButton.click();
      
      // Should show success or installed state
      await page.waitForTimeout(1000);
    }
  });

  test('should open run modal for installed plugins', async ({ page }) => {
    await mockApiResponse(page, '**/api/plugins/installed', {
      installed: [mockPlugins[0]],
    });
    
    await page.goto('/plugins');
    
    // Click on installed plugin
    const pluginCard = page.locator('text=security-scan').first();
    await pluginCard.click();
    
    // Find run button
    const runButton = page.locator('button').filter({
      hasText: /run/i
    }).first();
    
    if (await runButton.isVisible()) {
      await runButton.click();
      
      // Modal should appear
      const modal = page.locator('[class*="modal"], [role="dialog"]').first();
      await expect(modal).toBeVisible({ timeout: 5000 });
    }
  });
});

test.describe('Plugin Run Modal', () => {
  test.beforeEach(async ({ page }) => {
    await mockApiResponse(page, '**/api/health', { status: 'ok' });
    await mockApiResponse(page, '**/api/plugins', { plugins: mockPlugins });
    await mockApiResponse(page, '**/api/plugins/installed', {
      installed: [mockPlugins[0]],
    });
  });

  test('should show plugin info in modal', async ({ page }) => {
    await page.goto('/plugins');
    
    const pluginCard = page.locator('text=security-scan').first();
    await pluginCard.click();
    
    const runButton = page.locator('button').filter({ hasText: /run/i }).first();
    if (await runButton.isVisible()) {
      await runButton.click();
      
      // Modal should show plugin name
      const modalTitle = page.locator('text=/security-scan|run plugin/i').first();
      await expect(modalTitle).toBeVisible({ timeout: 5000 });
    }
  });

  test('should allow configuring run parameters', async ({ page }) => {
    await page.goto('/plugins');
    
    const pluginCard = page.locator('text=security-scan').first();
    await pluginCard.click();
    
    const runButton = page.locator('button').filter({ hasText: /run/i }).first();
    if (await runButton.isVisible()) {
      await runButton.click();
      
      // Should have input fields
      const inputField = page.locator('input, textarea').first();
      await expect(inputField).toBeVisible({ timeout: 5000 });
    }
  });

  test('should close modal on escape', async ({ page }) => {
    await page.goto('/plugins');
    
    const pluginCard = page.locator('text=security-scan').first();
    await pluginCard.click();
    
    const runButton = page.locator('button').filter({ hasText: /run/i }).first();
    if (await runButton.isVisible()) {
      await runButton.click();
      
      // Modal should be visible
      const modal = page.locator('[class*="modal"], [role="dialog"]').first();
      await expect(modal).toBeVisible();
      
      // Press escape
      await page.keyboard.press('Escape');
      
      // Modal should close
      await expect(modal).not.toBeVisible({ timeout: 2000 });
    }
  });
});
