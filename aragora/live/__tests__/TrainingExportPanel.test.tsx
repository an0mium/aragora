/**
 * Tests for TrainingExportPanel component
 *
 * Tests cover:
 * - Loading states
 * - Panel rendering
 * - Tab navigation
 * - Export type display
 */

import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { TrainingExportPanel } from '../src/components/TrainingExportPanel';

// Mock the useAragoraClient hook
jest.mock('../src/hooks/useAragoraClient', () => ({
  useAragoraClient: () => mockClient,
}));

const mockClient = {
  training: {
    stats: jest.fn(),
    formats: jest.fn(),
    exportSFT: jest.fn(),
    exportDPO: jest.fn(),
    exportGauntlet: jest.fn(),
  },
};

const mockStats = {
  available_exporters: ['sft', 'dpo', 'gauntlet'],
  export_directory: '/exports',
  exported_files: [
    {
      name: 'sft_export_2026-01-13.json',
      size_bytes: 1024,
      created_at: '2026-01-13T10:00:00Z',
      modified_at: '2026-01-13T10:00:00Z',
    },
  ],
  sft_available: true,
};

const mockFormats = {
  formats: {
    sft: {
      description: 'Supervised Fine-Tuning format',
      schema: { instruction: 'string', response: 'string' },
      use_case: 'Fine-tuning language models',
    },
    dpo: {
      description: 'Direct Preference Optimization format',
      schema: { chosen: 'string', rejected: 'string' },
      use_case: 'Preference learning',
    },
    gauntlet: {
      description: 'Adversarial testing format',
      schema: { scenario: 'string', result: 'string' },
      use_case: 'Red team testing',
    },
  },
  output_formats: ['json', 'jsonl'],
  endpoints: {},
};

function setupSuccessfulMocks() {
  mockClient.training.stats.mockResolvedValue(mockStats);
  mockClient.training.formats.mockResolvedValue(mockFormats);
}

describe('TrainingExportPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    setupSuccessfulMocks();
  });

  describe('Loading States', () => {
    it('shows loading spinner initially', async () => {
      mockClient.training.stats.mockImplementation(() => new Promise(() => {}));
      mockClient.training.formats.mockImplementation(() => new Promise(() => {}));

      await act(async () => {
        render(<TrainingExportPanel />);
      });

      expect(document.querySelector('.animate-spin')).toBeInTheDocument();
    });
  });

  describe('Panel Rendering', () => {
    it('shows panel title after loading', async () => {
      await act(async () => {
        render(<TrainingExportPanel />);
      });

      await waitFor(() => {
        expect(screen.getByText('Training Data Export')).toBeInTheDocument();
      });
    });

    it('shows description text', async () => {
      await act(async () => {
        render(<TrainingExportPanel />);
      });

      await waitFor(() => {
        expect(screen.getByText(/export debate data/i)).toBeInTheDocument();
      });
    });
  });

  describe('Tab Navigation', () => {
    it('renders all tabs', async () => {
      await act(async () => {
        render(<TrainingExportPanel />);
      });

      await waitFor(() => {
        expect(screen.getByText('Export')).toBeInTheDocument();
        expect(screen.getByText('Formats')).toBeInTheDocument();
        expect(screen.getByText('History')).toBeInTheDocument();
      });
    });

    it('switches to formats tab', async () => {
      await act(async () => {
        render(<TrainingExportPanel />);
      });

      await waitFor(() => {
        expect(screen.getByText('Formats')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText('Formats'));

      await waitFor(() => {
        expect(screen.getByText(/Supervised Fine-Tuning format/i)).toBeInTheDocument();
      });
    });

    it('switches to history tab', async () => {
      await act(async () => {
        render(<TrainingExportPanel />);
      });

      await waitFor(() => {
        expect(screen.getByText('History')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText('History'));

      await waitFor(() => {
        expect(screen.getByText(/sft_export/i)).toBeInTheDocument();
      });
    });
  });

  describe('Export Type Selection', () => {
    it('shows export type buttons', async () => {
      await act(async () => {
        render(<TrainingExportPanel />);
      });

      await waitFor(() => {
        expect(screen.getByText('SFT')).toBeInTheDocument();
        expect(screen.getByText('DPO')).toBeInTheDocument();
        expect(screen.getByText('GAUNTLET')).toBeInTheDocument();
      });
    });

    it('shows export type descriptions', async () => {
      await act(async () => {
        render(<TrainingExportPanel />);
      });

      await waitFor(() => {
        expect(screen.getByText('Supervised Fine-Tuning')).toBeInTheDocument();
        expect(screen.getByText('Direct Preference Optimization')).toBeInTheDocument();
      });
    });
  });
});
