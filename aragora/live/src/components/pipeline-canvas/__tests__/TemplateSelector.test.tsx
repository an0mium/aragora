import { render, waitFor } from '@testing-library/react';

import { TemplateSelector } from '../TemplateSelector';

const mockFetch = jest.fn();
global.fetch = mockFetch as typeof fetch;

describe('TemplateSelector', () => {
  beforeEach(() => {
    mockFetch.mockReset();
    localStorage.clear();
    localStorage.setItem('aragora-backend', 'production');
  });

  it('loads templates from the selected backend', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        templates: [
          {
            name: 'product_launch',
            display_name: 'Product Launch',
            description: 'Launch a product.',
            category: 'product',
            idea_count: 3,
            tags: [],
            vertical_profile: null,
          },
        ],
      }),
    });

    render(<TemplateSelector onSelectTemplate={jest.fn()} onStartBlank={jest.fn()} />);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        'https://api.aragora.ai/api/v1/canvas/pipeline/templates',
      );
    });
  });
});
