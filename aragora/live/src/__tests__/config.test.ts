describe('config local dev API fallback', () => {
  const originalApiUrl = process.env.NEXT_PUBLIC_API_URL;
  const originalWsUrl = process.env.NEXT_PUBLIC_WS_URL;

  beforeEach(() => {
    jest.resetModules();
    delete process.env.NEXT_PUBLIC_API_URL;
    delete process.env.NEXT_PUBLIC_WS_URL;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  afterAll(() => {
    if (originalApiUrl === undefined) {
      delete process.env.NEXT_PUBLIC_API_URL;
    } else {
      process.env.NEXT_PUBLIC_API_URL = originalApiUrl;
    }
    if (originalWsUrl === undefined) {
      delete process.env.NEXT_PUBLIC_WS_URL;
    } else {
      process.env.NEXT_PUBLIC_WS_URL = originalWsUrl;
    }
  });

  it('uses the same-origin API proxy on localhost when NEXT_PUBLIC_API_URL is unset', async () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});

    const config = await import('../config');

    expect(config.API_BASE_URL).toBe('');
    expect(config.getEnvWarnings()).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: 'NEXT_PUBLIC_API_URL',
          message: 'API URL not set, using same-origin /api proxy',
        }),
      ]),
    );
    expect(warnSpy).toHaveBeenCalledWith(
      '[Aragora] NEXT_PUBLIC_API_URL not set, using same-origin /api proxy (local dev mode).'
    );
  });

  it('treats only local dev hosts as same-origin proxy candidates', async () => {
    const config = await import('../config');

    expect(config.isLocalDevHostname('localhost')).toBe(true);
    expect(config.isLocalDevHostname('127.0.0.1')).toBe(true);
    expect(config.isLocalDevHostname('192.168.1.24')).toBe(true);
    expect(config.isLocalDevHostname('preview.example.test')).toBe(false);
  });
});
