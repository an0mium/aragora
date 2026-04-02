jest.mock('@playwright/test', () => ({
  defineConfig: <T>(config: T) => config,
  devices: {
    'Desktop Chrome': {},
    'Desktop Firefox': {},
    'Desktop Safari': {},
    'Pixel 5': {},
    'iPhone 12': {},
  },
}));

const {
  default: config,
  SPECIALTY_PROJECT_TEST_IGNORE,
} = require('../playwright.config') as typeof import('../playwright.config');

type PlaywrightProject = {
  name?: string;
  testDir?: string;
  testIgnore?: unknown;
  testMatch?: unknown;
};

const projects = (config.projects ?? []) as PlaywrightProject[];

function getProject(name: string): PlaywrightProject {
  const project = projects.find(candidate => candidate.name === name);
  expect(project).toBeDefined();
  return project as PlaywrightProject;
}

describe('playwright config', () => {
  it('keeps specialty suites out of the default browser matrix', () => {
    for (const projectName of ['chromium', 'firefox', 'webkit', 'Mobile Chrome', 'Mobile Safari']) {
      expect(getProject(projectName).testIgnore).toEqual(SPECIALTY_PROJECT_TEST_IGNORE);
    }
  });

  it('keeps the specialty projects scoped to their dedicated specs', () => {
    expect(getProject('accessibility').testMatch).toEqual(/accessibility\.spec\.ts/);
    expect(getProject('visual-regression').testMatch).toEqual(/visual-regression\.spec\.ts/);
    expect(getProject('mobile-audit').testDir).toBe('./e2e/mobile');
    expect(getProject('mobile-audit').testMatch).toEqual(/mobile-audit\.spec\.ts/);
  });
});
