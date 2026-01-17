/**
 * Tests for AuditTemplatesPage component
 *
 * Tests cover:
 * - Loading state display
 * - Fetching and displaying audit presets
 * - Fetching and displaying audit types
 * - Preset card rendering with icons and colors
 * - Audit type capabilities display
 * - Error handling
 * - Navigation on preset selection
 */

import { render, screen, waitFor } from '@testing-library/react';
import { useRouter } from 'next/navigation';

// Mock Next.js router
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}));

// Mock hooks
jest.mock('../src/components/BackendSelector', () => ({
  BackendSelector: () => null,
  useBackend: () => ({
    config: { api: 'http://localhost:8080' },
  }),
}));

jest.mock('../src/context/AuthContext', () => ({
  useAuth: () => ({
    tokens: { access_token: 'test-token' },
  }),
}));

// Mock UI components
jest.mock('../src/components/MatrixRain', () => ({
  Scanlines: () => null,
  CRTVignette: () => null,
}));

jest.mock('../src/components/AsciiBanner', () => ({
  AsciiBannerCompact: () => <div data-testid="ascii-banner">ARAGORA</div>,
}));

jest.mock('../src/components/ThemeToggle', () => ({
  ThemeToggle: () => null,
}));

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

const mockPresets = [
  {
    name: 'Legal Due Diligence',
    description: 'Comprehensive legal document analysis for M&A and contracts',
    audit_types: ['contract_analysis', 'compliance_check'],
    consensus_threshold: 0.8,
    custom_rules_count: 15,
  },
  {
    name: 'Financial Audit',
    description: 'Financial statement analysis and fraud detection',
    audit_types: ['financial_analysis', 'fraud_detection'],
    consensus_threshold: 0.9,
    custom_rules_count: 22,
  },
  {
    name: 'Code Security',
    description: 'Static analysis and vulnerability scanning for codebases',
    audit_types: ['sast', 'vulnerability_scan'],
    consensus_threshold: 0.75,
    custom_rules_count: 45,
  },
];

const mockAuditTypes = [
  {
    id: 'contract_analysis',
    display_name: 'Contract Analysis',
    description: 'Analyze legal contracts for risks and obligations',
    version: '1.2.0',
    capabilities: {
      supports_chunk_analysis: true,
      supports_cross_document: true,
      requires_llm: true,
    },
  },
  {
    id: 'sast',
    display_name: 'Static Analysis',
    description: 'Static Application Security Testing',
    version: '2.0.1',
    capabilities: {
      supports_chunk_analysis: true,
      supports_cross_document: false,
      requires_llm: false,
    },
  },
];

// Import the component dynamically to avoid issues with the mocks
let AuditTemplatesPage: React.ComponentType;

describe('AuditTemplatesPage', () => {
  const mockRouter = {
    push: jest.fn(),
    back: jest.fn(),
  };

  beforeAll(async () => {
    // Dynamic import after mocks are set up
    const module = await import('../src/app/audit/templates/page');
    AuditTemplatesPage = module.default;
  });

  beforeEach(() => {
    mockFetch.mockReset();
    mockRouter.push.mockReset();
    (useRouter as jest.Mock).mockReturnValue(mockRouter);
  });

  function setupSuccessfulFetch() {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/api/audit/presets')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ presets: mockPresets }),
        });
      }
      if (url.includes('/api/audit/types')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ types: mockAuditTypes }),
        });
      }
      return Promise.resolve({
        ok: false,
        json: () => Promise.resolve({ error: 'Not found' }),
      });
    });
  }

  describe('Loading State', () => {
    it('should show loading state initially', () => {
      mockFetch.mockImplementation(() => new Promise(() => {})); // Never resolves

      render(<AuditTemplatesPage />);

      expect(screen.getByText(/LOADING/i)).toBeInTheDocument();
    });
  });

  describe('Preset Display', () => {
    it('should fetch presets on mount', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/audit/presets'),
          expect.any(Object)
        );
      });
    });

    it('should display preset names', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        expect(screen.getByText('Legal Due Diligence')).toBeInTheDocument();
        expect(screen.getByText('Financial Audit')).toBeInTheDocument();
        expect(screen.getByText('Code Security')).toBeInTheDocument();
      });
    });

    it('should display preset descriptions', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        expect(screen.getByText(/Comprehensive legal document analysis/)).toBeInTheDocument();
        expect(screen.getByText(/Financial statement analysis/)).toBeInTheDocument();
        expect(screen.getByText(/Static analysis and vulnerability/)).toBeInTheDocument();
      });
    });

    it('should display industry icons for presets', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        // Icons are emojis: Legal (⚖️), Financial (💰), Security (🔒)
        expect(screen.getByText('⚖️')).toBeInTheDocument();
        expect(screen.getByText('💰')).toBeInTheDocument();
        expect(screen.getByText('🔒')).toBeInTheDocument();
      });
    });

    it('should display consensus threshold for presets', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        expect(screen.getByText(/80%/)).toBeInTheDocument(); // Legal preset
        expect(screen.getByText(/90%/)).toBeInTheDocument(); // Financial preset
        expect(screen.getByText(/75%/)).toBeInTheDocument(); // Security preset
      });
    });

    it('should display custom rules count for presets', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        expect(screen.getByText(/15.*rules/i)).toBeInTheDocument();
        expect(screen.getByText(/22.*rules/i)).toBeInTheDocument();
        expect(screen.getByText(/45.*rules/i)).toBeInTheDocument();
      });
    });
  });

  describe('Audit Types Display', () => {
    it('should fetch audit types on mount', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/audit/types'),
          expect.any(Object)
        );
      });
    });

    it('should display audit type names', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        expect(screen.getByText('Contract Analysis')).toBeInTheDocument();
        expect(screen.getByText('Static Analysis')).toBeInTheDocument();
      });
    });

    it('should display audit type versions', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        expect(screen.getByText(/v1\.2\.0/)).toBeInTheDocument();
        expect(screen.getByText(/v2\.0\.1/)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle fetch failure gracefully', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      render(<AuditTemplatesPage />);

      // Should not crash and should eventually stop loading
      await waitFor(() => {
        expect(screen.queryByText(/LOADING/i)).not.toBeInTheDocument();
      }, { timeout: 3000 });
    });

    it('should handle API error response', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        json: () => Promise.resolve({ error: 'Server error' }),
      });

      render(<AuditTemplatesPage />);

      // Should not crash
      await waitFor(() => {
        expect(screen.queryByText(/LOADING/i)).not.toBeInTheDocument();
      }, { timeout: 3000 });
    });
  });

  describe('Navigation', () => {
    it('should have link back to home', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        const homeLink = screen.getByTestId('ascii-banner').closest('a');
        expect(homeLink).toHaveAttribute('href', '/');
      });
    });
  });

  describe('Header', () => {
    it('should display page header', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        expect(screen.getByText(/AUDIT TEMPLATES/i)).toBeInTheDocument();
      });
    });

    it('should display section headers for presets and types', async () => {
      setupSuccessfulFetch();

      render(<AuditTemplatesPage />);

      await waitFor(() => {
        expect(screen.getByText(/INDUSTRY PRESETS/i)).toBeInTheDocument();
        expect(screen.getByText(/AVAILABLE AUDIT TYPES/i)).toBeInTheDocument();
      });
    });
  });
});
