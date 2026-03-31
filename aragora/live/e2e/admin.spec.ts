import { test, expect, mockApiResponse } from './fixtures';
import type { Page } from '@playwright/test';

const mockHealthData = {
  status: 'healthy',
  uptime_seconds: 3600,
  version: '1.0.0',
  components: {
    database: { status: 'ok', latency_ms: 5 },
    agents: { status: 'ok', available: 6, total: 6 },
    memory: { status: 'ok', usage_mb: 256 },
    websocket: { status: 'ok', connections: 10 },
  },
};

async function openAdmin(
  page: Page,
  aragoraPage: { dismissAllOverlays: () => Promise<void> },
) {
  await mockApiResponse(page, '**/api/health*', mockHealthData);
  await page.goto('/admin');
  await aragoraPage.dismissAllOverlays();
  await page.waitForLoadState('domcontentloaded');
}

test.describe('Admin Overview', () => {
  test('loads the overview shell', async ({ page, aragoraPage }) => {
    await openAdmin(page, aragoraPage);

    await expect(page).toHaveTitle(/Admin|Aragora/i);
    await expect(page.getByRole('heading', { name: /admin overview/i })).toBeVisible();
    await expect(
      page.getByText(/system health, usage metrics, and recent activity at a glance/i),
    ).toBeVisible();
  });

  test('shows quick actions and refresh control', async ({ page, aragoraPage }) => {
    await openAdmin(page, aragoraPage);

    await expect(page.getByRole('button', { name: /refresh/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /invite user/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /create organization/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /view audit logs/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /check billing/i })).toBeVisible();
  });

  test('renders the system health card without crashing', async ({ page, aragoraPage }) => {
    await openAdmin(page, aragoraPage);

    await expect(page.getByRole('heading', { name: /system health/i })).toBeVisible();
    await expect(page.locator('body')).not.toContainText(/cannot read properties|runtime error/i);
  });

  test('tolerates minimal public health payloads', async ({ page, aragoraPage }) => {
    await mockApiResponse(page, '**/api/health*', {
      status: 'healthy',
      timestamp: new Date().toISOString(),
    });
    await page.goto('/admin');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    await expect(page.getByRole('heading', { name: /admin overview/i })).toBeVisible();
    await expect(page.getByRole('heading', { name: /system health/i })).toBeVisible();
    await expect(page.getByText(/agents/i).first()).toBeVisible();
    await expect(page.locator('body')).not.toContainText(/cannot read properties|runtime error/i);
  });

  test('refreshes without crashing', async ({ page, aragoraPage }) => {
    await openAdmin(page, aragoraPage);

    const refreshButton = page.getByRole('button', { name: /refresh/i });
    await refreshButton.click();
    await expect(refreshButton).toContainText(/refreshing/i);
    await expect(refreshButton).toContainText(/^refresh$/i, { timeout: 10000 });
  });

  test('shows recent activity section', async ({ page, aragoraPage }) => {
    await openAdmin(page, aragoraPage);

    await expect(page.getByRole('heading', { name: /recent activity/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /view all activity/i })).toBeVisible();
  });
});

test.describe('Admin Overview Layout', () => {
  test('renders on mobile', async ({ page, aragoraPage }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await openAdmin(page, aragoraPage);

    await expect(page.getByRole('heading', { name: /admin overview/i })).toBeVisible();
    await expect(page.locator('body')).toBeVisible();
  });

  test('renders grid content on desktop', async ({ page, aragoraPage }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await openAdmin(page, aragoraPage);

    await expect(page.locator('[class*="grid"]').first()).toBeVisible();
    await expect(page.getByRole('heading', { name: /system health/i })).toBeVisible();
  });
});
