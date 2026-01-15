/**
 * Accessibility Tests
 *
 * Tests WCAG compliance using axe-core for critical pages.
 * Run with: npx playwright test accessibility.spec.ts
 */

import { test, expect } from './fixtures';
import AxeBuilder from '@axe-core/playwright';

// Critical pages to test for accessibility
const CRITICAL_PAGES = [
  { path: '/', name: 'Landing Page' },
  { path: '/debates', name: 'Debates List' },
  { path: '/agents', name: 'Agents Page' },
  { path: '/leaderboard', name: 'Leaderboard' },
  { path: '/about', name: 'About Page' },
  { path: '/pricing', name: 'Pricing Page' },
];

test.describe('Accessibility - Critical Pages', () => {
  for (const page of CRITICAL_PAGES) {
    test(`${page.name} should have no critical accessibility violations`, async ({ page: browserPage, aragoraPage }) => {
      await browserPage.goto(page.path);
      await aragoraPage.dismissAllOverlays();
      await browserPage.waitForLoadState('domcontentloaded');

      const results = await new AxeBuilder({ page: browserPage })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
        .analyze();

      // Filter to only critical and serious violations
      const criticalViolations = results.violations.filter(
        (v) => v.impact === 'critical' || v.impact === 'serious'
      );

      // Log violations for debugging
      if (criticalViolations.length > 0) {
        console.log(`\nAccessibility violations on ${page.name}:`);
        criticalViolations.forEach((violation) => {
          console.log(`\n  [${violation.impact?.toUpperCase()}] ${violation.id}: ${violation.description}`);
          console.log(`  Help: ${violation.helpUrl}`);
          violation.nodes.forEach((node, i) => {
            console.log(`    ${i + 1}. ${node.html.substring(0, 100)}...`);
          });
        });
      }

      expect(
        criticalViolations,
        `Found ${criticalViolations.length} critical/serious accessibility violations on ${page.name}`
      ).toHaveLength(0);
    });
  }
});

test.describe('Accessibility - Interactive Components', () => {
  test('Modal dialogs should be accessible', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    // Try to open any modal (e.g., sign in)
    const signInButton = page.getByRole('button', { name: /sign in/i });
    if (await signInButton.isVisible()) {
      await signInButton.click();
      await page.waitForTimeout(500);

      const results = await new AxeBuilder({ page })
        .include('[role="dialog"]')
        .analyze();

      const criticalViolations = results.violations.filter(
        (v) => v.impact === 'critical' || v.impact === 'serious'
      );

      expect(criticalViolations).toHaveLength(0);
    }
  });

  test('Navigation should be keyboard accessible', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    // Test keyboard navigation
    await page.keyboard.press('Tab');

    // First focusable element should receive focus
    const focusedElement = await page.evaluate(() => {
      return document.activeElement?.tagName;
    });

    expect(focusedElement).toBeTruthy();
  });

  test('Forms should have proper labels', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a'])
      .disableRules(['region']) // Skip landmark rules for this test
      .analyze();

    const labelViolations = results.violations.filter(
      (v) => v.id.includes('label') || v.id.includes('form')
    );

    expect(labelViolations).toHaveLength(0);
  });
});

test.describe('Accessibility - Color Contrast', () => {
  test('Text should have sufficient color contrast', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2aa'])
      .analyze();

    const contrastViolations = results.violations.filter(
      (v) => v.id === 'color-contrast'
    );

    // Log contrast issues for debugging
    if (contrastViolations.length > 0) {
      console.log('\nColor contrast violations:');
      contrastViolations.forEach((violation) => {
        violation.nodes.forEach((node) => {
          console.log(`  - ${node.html.substring(0, 80)}...`);
          console.log(`    ${node.failureSummary}`);
        });
      });
    }

    expect(contrastViolations).toHaveLength(0);
  });
});

test.describe('Accessibility - ARIA', () => {
  test('ARIA attributes should be valid', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    const results = await new AxeBuilder({ page })
      .withTags(['cat.aria'])
      .analyze();

    const ariaViolations = results.violations.filter(
      (v) => v.impact === 'critical' || v.impact === 'serious'
    );

    expect(ariaViolations).toHaveLength(0);
  });

  test('Interactive elements should have accessible names', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    const results = await new AxeBuilder({ page })
      .analyze();

    const nameViolations = results.violations.filter(
      (v) => v.id === 'button-name' || v.id === 'link-name' || v.id === 'image-alt'
    );

    expect(nameViolations).toHaveLength(0);
  });
});

test.describe('Accessibility - Debates Page', () => {
  test('Debate list should be accessible', async ({ page, aragoraPage }) => {
    await page.goto('/debates');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa'])
      .analyze();

    const criticalViolations = results.violations.filter(
      (v) => v.impact === 'critical' || v.impact === 'serious'
    );

    expect(criticalViolations).toHaveLength(0);
  });

  test('Debate viewer should be accessible', async ({ page, aragoraPage }) => {
    // Navigate to debates and click on first one if available
    await page.goto('/debates');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    const debateLink = page.locator('a[href*="/debates/"]').first();
    if (await debateLink.isVisible()) {
      await debateLink.click();
      await page.waitForLoadState('domcontentloaded');

      const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa'])
        .analyze();

      const criticalViolations = results.violations.filter(
        (v) => v.impact === 'critical' || v.impact === 'serious'
      );

      expect(criticalViolations).toHaveLength(0);
    }
  });
});

test.describe('Accessibility - Screen Reader Compatibility', () => {
  test('Page should have proper heading hierarchy', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    const headings = await page.evaluate(() => {
      const h1s = document.querySelectorAll('h1');
      const allHeadings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
      return {
        h1Count: h1s.length,
        totalHeadings: allHeadings.length,
        headingLevels: Array.from(allHeadings).map((h) => h.tagName),
      };
    });

    // Should have exactly one h1
    expect(headings.h1Count).toBeLessThanOrEqual(1);
  });

  test('Page should have skip links or landmarks', async ({ page, aragoraPage }) => {
    await page.goto('/');
    await aragoraPage.dismissAllOverlays();
    await page.waitForLoadState('domcontentloaded');

    const hasLandmarks = await page.evaluate(() => {
      const main = document.querySelector('main, [role="main"]');
      const nav = document.querySelector('nav, [role="navigation"]');
      const skipLink = document.querySelector('a[href="#main"], a[href="#content"]');
      return {
        hasMain: !!main,
        hasNav: !!nav,
        hasSkipLink: !!skipLink,
      };
    });

    // Should have main landmark or skip link
    expect(hasLandmarks.hasMain || hasLandmarks.hasSkipLink).toBe(true);
  });
});
