import { beforeEach, describe, expect, it, vi } from 'vitest';

import { MarketplaceAPI } from '../marketplace';

function createMockClient() {
  return {
    request: vi.fn(),
    browseMarketplace: vi.fn(),
    getMarketplaceTemplate: vi.fn(),
    publishTemplate: vi.fn(),
    importTemplate: vi.fn(),
    rateTemplate: vi.fn(),
    reviewTemplate: vi.fn(),
    getFeaturedTemplates: vi.fn(),
    getTrendingTemplates: vi.fn(),
    getMarketplaceCategories: vi.fn(),
    getMarketplaceIndustries: vi.fn(),
    getNewMarketplaceReleases: vi.fn(),
    getMarketplaceReviews: vi.fn(),
    purchaseTemplate: vi.fn(),
    downloadTemplate: vi.fn(),
    getMyMarketplacePurchases: vi.fn(),
    updateMarketplaceTemplate: vi.fn(),
    unpublishTemplate: vi.fn(),
    exportMarketplaceTemplate: vi.fn(),
  };
}

describe('MarketplaceAPI', () => {
  let mockClient: ReturnType<typeof createMockClient>;
  let api: MarketplaceAPI;

  beforeEach(() => {
    vi.clearAllMocks();
    mockClient = createMockClient();
    api = new MarketplaceAPI(mockClient);
  });

  describe('rate', () => {
    it('returns the average rating from the v2 ratings endpoint', async () => {
      mockClient.request.mockResolvedValueOnce({ average_rating: 4.6 });

      const result = await api.rate('tpl/1', 5);

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v2/marketplace/templates/tpl%2F1/ratings',
        { body: { score: 5 } }
      );
      expect(result).toEqual({ new_rating: 4.6 });
    });

    it('preserves legacy new_rating responses', async () => {
      mockClient.request.mockResolvedValueOnce({ new_rating: 4.2 });

      const result = await api.rate('tpl/1', 4);

      expect(result).toEqual({ new_rating: 4.2 });
    });

    it('throws when the response omits the rating value', async () => {
      mockClient.request.mockResolvedValueOnce({ success: true });

      await expect(api.rate('tpl/1', 4)).rejects.toThrow(
        'Marketplace rating response missing average_rating'
      );
    });
  });
});
