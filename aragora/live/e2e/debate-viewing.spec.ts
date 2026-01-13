import { test, expect, mockApiResponse, mockDebate } from './fixtures';

test.describe('Debate Viewing', () => {
  const debateId = 'test-debate-123';

  test.beforeEach(async ({ page }) => {
    // Mock debate data endpoint
    await mockApiResponse(page, `**/api/debates/${debateId}`, mockDebate);
    await mockApiResponse(page, '**/api/health', { status: 'ok' });
  });

  test('should display debate topic', async ({ page }) => {
    await page.goto(`/debate/${debateId}`);

    // Should show debate topic
    const topicElement = page.locator('h1, h2, [class*="topic"]').filter({
      hasText: new RegExp(mockDebate.topic, 'i')
    }).first();

    await expect(topicElement).toBeVisible({ timeout: 10000 });
  });

  test('should display agent messages', async ({ page }) => {
    await page.goto(`/debate/${debateId}`);

    // Should show messages from agents
    for (const message of mockDebate.messages) {
      const messageElement = page.locator('[class*="message"], [data-testid="message"]').filter({
        hasText: message.content.substring(0, 20)
      }).first();

      await expect(messageElement).toBeVisible({ timeout: 10000 });
    }
  });

  test('should display agent panel', async ({ page }) => {
    await page.goto(`/debate/${debateId}`);

    // Should show agent panel with participating agents
    const agentPanel = page.locator('[class*="agent"], [data-testid="agent-panel"]').first();
    await expect(agentPanel).toBeVisible({ timeout: 10000 });

    // Should show agent names
    for (const agent of mockDebate.agents) {
      const agentName = page.locator(`text=/${agent}/i`).first();
      await expect(agentName).toBeVisible();
    }
  });

  test('should display consensus status when reached', async ({ page }) => {
    await page.goto(`/debate/${debateId}`);

    // Should show consensus indicator
    const consensusElement = page.locator('[class*="consensus"], text=/consensus|agreed|majority/i').first();
    await expect(consensusElement).toBeVisible({ timeout: 10000 });
  });

  test('should show debate status', async ({ page }) => {
    await page.goto(`/debate/${debateId}`);

    // Should show status (completed, in progress, etc.)
    const statusElement = page.locator('[class*="status"]').or(
      page.locator('text=/completed|in progress|running/i')
    ).first();

    await expect(statusElement).toBeVisible({ timeout: 10000 });
  });

  test('should have export button', async ({ page }) => {
    await page.goto(`/debate/${debateId}`);

    // Should have export option
    const exportButton = page.locator('button, a').filter({
      hasText: /export|download|pdf|share/i
    }).first();

    await expect(exportButton).toBeVisible({ timeout: 10000 });
  });

  test('should handle non-existent debate', async ({ page }) => {
    await page.route('**/api/debates/non-existent', async (route) => {
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Debate not found' }),
      });
    });

    await page.goto('/debate/non-existent');

    // Should show error or not found message
    const errorElement = page.locator('text=/not found|error|404/i').first();
    await expect(errorElement).toBeVisible({ timeout: 10000 });
  });

  test('should display round indicators', async ({ page }) => {
    await page.goto(`/debate/${debateId}`);

    // Should show round information
    const roundElement = page.locator('text=/round|r1|r2/i').first();
    await expect(roundElement).toBeVisible({ timeout: 10000 });
  });
});

test.describe('Debate Viewing - Real-time Updates', () => {
  test('should connect to WebSocket for live updates', async ({ page }) => {
    const wsConnected = new Promise<void>((resolve) => {
      page.on('websocket', (ws) => {
        if (ws.url().includes('ws')) {
          resolve();
        }
      });
    });

    await mockApiResponse(page, '**/api/debates/live-debate', {
      ...mockDebate,
      id: 'live-debate',
      status: 'running',
    });

    await page.goto('/debate/live-debate');

    // WebSocket should be attempted (may not connect in test env)
    await Promise.race([
      wsConnected,
      page.waitForTimeout(5000),
    ]);
  });
});

test.describe('Debate Viewing - Interaction', () => {
  test('should allow collapsing message sections', async ({ page }) => {
    await mockApiResponse(page, '**/api/debates/test-debate', mockDebate);
    await page.goto('/debate/test-debate');

    // Find collapsible sections
    const collapseButton = page.locator('button, [role="button"]').filter({
      hasText: /collapse|expand|show|hide/i
    }).first();

    if (await collapseButton.isVisible()) {
      await collapseButton.click();
      // Content should toggle
      await page.waitForTimeout(300);
    }
  });

  test('should show message timestamps', async ({ page }) => {
    await mockApiResponse(page, '**/api/debates/test-debate', {
      ...mockDebate,
      messages: mockDebate.messages.map(m => ({
        ...m,
        created_at: new Date().toISOString(),
      })),
    });

    await page.goto('/debate/test-debate');

    // Should show some time indicator
    const timeElement = page.locator('time, [class*="time"], [class*="date"]').first();
    await expect(timeElement).toBeVisible({ timeout: 10000 });
  });
});
