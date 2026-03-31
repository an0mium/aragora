import { test, expect, mockApiResponse } from './fixtures';

const adminOverviewFixtures = {
  health: {
    status: 'healthy',
    uptime_seconds: 3600,
    version: '1.0.0',
    components: {
      database: { status: 'healthy', latency_ms: 5 },
      agents: { status: 'healthy', available: 6, total: 6 },
      memory: { status: 'healthy', usage_mb: 256 },
      websocket: { status: 'healthy', connections: 10 },
    },
  },
  activity: {
    activities: [
      {
        id: 'activity-1',
        type: 'action_completed',
        title: 'Deployment completed',
        timestamp: '2026-03-31T10:00:00Z',
      },
      {
        id: 'activity-2',
        type: 'meeting_scheduled',
        title: 'Audit log exported',
        timestamp: '2026-03-31T11:00:00Z',
      },
    ],
  },
  debateTrends: {
    data_points: [
      { period: '2026-03-30T00:00:00Z', total: 12 },
      { period: '2026-03-31T00:00:00Z', total: 15 },
    ],
  },
  usageTrends: {
    data_points: [
      { period: '2026-03-30T00:00:00Z', requests: 42, tokens: 4200 },
      { period: '2026-03-31T00:00:00Z', requests: 55, tokens: 5500 },
    ],
  },
};

async function mockAdminOverviewData(page: import('@playwright/test').Page) {
  await mockApiResponse(page, /\/api\/health\/?$/, adminOverviewFixtures.health);
  await mockApiResponse(page, '**/api/v1/dashboard/activity?limit=10', adminOverviewFixtures.activity);
  await mockApiResponse(page, '**/api/analytics/debates/trends?time_range=30d', adminOverviewFixtures.debateTrends);
  await mockApiResponse(page, '**/api/analytics/usage/tokens?time_range=30d', adminOverviewFixtures.usageTrends);
}

async function openAdminOverview(
  page: import('@playwright/test').Page,
  aragoraPage: { dismissAllOverlays: () => Promise<void> },
  viewport?: { width: number; height: number }
) {
  await mockAdminOverviewData(page);
  if (viewport) {
    await page.setViewportSize(viewport);
  }
  await page.goto('/admin');
  await aragoraPage.dismissAllOverlays();
  await page.waitForLoadState('domcontentloaded');
  await expect(page.getByRole('heading', { name: 'Admin Overview' })).toBeVisible({ timeout: 10000 });
}

test.describe('Admin Overview', () => {
  test.beforeEach(async ({ page, aragoraPage }) => {
    await openAdminOverview(page, aragoraPage);
  });

  test('loads the admin overview shell and current layout', async ({ page }) => {
    await expect(page).toHaveTitle(/Admin|Aragora/i);
    await expect(
      page.getByText(/System health, usage metrics, and recent activity at a glance\./i)
    ).toBeVisible();
    await expect(page.getByRole('banner')).toBeVisible();
    await expect(page.getByRole('complementary', { name: 'Main navigation' })).toBeVisible();
    await expect(page.locator('aside').filter({ hasText: 'ADMIN PANEL' })).toBeVisible();
  });

  test('shows the non-admin access warning', async ({ page }) => {
    await expect(
      page.getByText(/Admin access required\. Some features may be restricted\./i)
    ).toBeVisible();
  });

  test('renders quick actions and summary cards for the overview dashboard', async ({ page }) => {
    await expect(page.getByRole('link', { name: /Invite User/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /Create Organization/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /View Audit Logs/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /Check Billing/i })).toBeVisible();

    await expect(page.getByText('Total Users')).toBeVisible();
    await expect(page.getByText('Organizations').last()).toBeVisible();
    await expect(page.getByText('Active (24h)')).toBeVisible();
    await expect(page.getByText('Debates (Month)')).toBeVisible();
    await expect(page.getByText('API Calls (Today)')).toBeVisible();
  });

  test('renders system health details from the health endpoint', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'System Health' })).toBeVisible();
    await expect(page.getByText('Uptime')).toBeVisible();
    await expect(page.getByText('1h 0m')).toBeVisible();
    await expect(page.getByText('Version')).toBeVisible();
    await expect(page.getByText('1.0.0')).toBeVisible();
    await expect(page.getByText('6/6')).toBeVisible();
    await expect(page.getByText('10 conn')).toBeVisible();
    await expect(page.getByRole('link', { name: /View Full System Status/i })).toBeVisible();
  });

  test('renders charts and recent activity', async ({ page }) => {
    await expect(page.getByText('DEBATES PER DAY')).toBeVisible();
    await expect(page.getByText('API CALLS PER DAY')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Recent Activity' })).toBeVisible();
    await expect(page.getByText('Deployment completed')).toBeVisible();
    await expect(page.getByText('Audit log exported')).toBeVisible();
    await expect(page.getByRole('link', { name: /View All Activity/i })).toBeVisible();
  });

  test('keeps the refresh action available', async ({ page }) => {
    await expect(page.getByRole('button', { name: /refresh/i })).toBeVisible();
    await expect(page.getByRole('button', { name: /refresh/i })).toBeEnabled();
  });

  test('renders sidebar navigation for admin destinations', async ({ page }) => {
    const sidebar = page.locator('aside').filter({ hasText: 'ADMIN PANEL' });

    await expect(sidebar.getByRole('link', { name: /Overview/i })).toBeVisible();
    await expect(sidebar.getByRole('link', { name: /Users/i })).toBeVisible();
    await expect(sidebar.getByRole('link', { name: /Organizations/i })).toBeVisible();
    await expect(sidebar.getByRole('link', { name: /Audit Logs/i })).toBeVisible();
    await expect(sidebar.getByRole('button', { name: /collapse sidebar/i })).toBeVisible();
  });

  test('renders stable header controls', async ({ page }) => {
    await expect(page.getByRole('link', { name: /Aragora \[ARAGORA\]/i })).toBeVisible();
    await expect(page.getByRole('link', { name: /\[ADMIN\]/i })).toBeVisible();
    await expect(page.getByRole('button', { name: 'PROD', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'DEV', exact: true })).toBeVisible();
  });
});

test.describe('Admin Overview responsive shell', () => {
  test('renders on mobile', async ({ page, aragoraPage }) => {
    await openAdminOverview(page, aragoraPage, { width: 375, height: 667 });
    await expect(page.locator('body')).toBeVisible();
  });

  test('renders on tablet', async ({ page, aragoraPage }) => {
    await openAdminOverview(page, aragoraPage, { width: 768, height: 1024 });
    await expect(page.locator('body')).toBeVisible();
  });

  test('keeps the overview content visible on desktop', async ({ page, aragoraPage }) => {
    await openAdminOverview(page, aragoraPage, { width: 1920, height: 1080 });
    await expect(page.getByRole('main', { name: 'Main content' })).toBeVisible({ timeout: 10000 });
    await expect(page.getByRole('heading', { name: 'System Health' })).toBeVisible();
  });
});
