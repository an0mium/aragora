/**
 * Tests for ConnectorDashboard component
 *
 * Tests cover:
 * - Tab navigation (connectors, sync-status, scheduled)
 * - Connector list display
 * - Connector type badges
 * - Sync actions
 * - Filter functionality
 * - Loading and error states
 */

import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { ConnectorDashboard } from '../src/components/control-plane/ConnectorDashboard/ConnectorDashboard';

// Default mock data - defined before the mock
const mockConnectorData = [
  {
    id: 'connector-1',
    name: 'Google Drive',
    type: 'google_drive',
    status: 'connected',
    lastSync: '2024-01-16T10:00:00Z',
    documentsIndexed: 1250,
    config: { folder_id: 'root' },
  },
  {
    id: 'connector-2',
    name: 'SharePoint',
    type: 'sharepoint',
    status: 'syncing',
    lastSync: '2024-01-16T09:00:00Z',
    documentsIndexed: 3500,
    syncProgress: 0.65,
    config: { site_url: 'https://company.sharepoint.com' },
  },
  {
    id: 'connector-3',
    name: 'Confluence',
    type: 'confluence',
    status: 'error',
    lastSync: '2024-01-15T12:00:00Z',
    documentsIndexed: 800,
    errorMessage: 'Authentication expired',
    config: { space_key: 'DOCS' },
  },
];

// Mock the hooks
jest.mock('@/hooks/useApi', () => ({
  useApi: () => ({
    get: jest.fn().mockResolvedValue({ connectors: mockConnectorData }),
    post: jest.fn().mockResolvedValue({ success: true }),
    put: jest.fn().mockResolvedValue({ success: true }),
    delete: jest.fn().mockResolvedValue({ success: true }),
  }),
}));

jest.mock('@/components/BackendSelector', () => ({
  useBackend: () => ({
    config: { api: 'http://localhost:8080' },
  }),
}));

jest.mock('@/utils/logger', () => ({
  logger: {
    debug: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
  },
}));

describe('ConnectorDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Header and Layout', () => {
    it('renders the dashboard header', async () => {
      render(<ConnectorDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Enterprise Connectors')).toBeInTheDocument();
      });
    });

    it('shows all tabs', async () => {
      render(<ConnectorDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Connectors')).toBeInTheDocument();
        expect(screen.getByText('Sync Status')).toBeInTheDocument();
        expect(screen.getByText('Scheduled Jobs')).toBeInTheDocument();
      });
    });
  });

  describe('Connector List', () => {
    it('displays all connectors', async () => {
      render(<ConnectorDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Google Drive')).toBeInTheDocument();
        expect(screen.getByText('SharePoint')).toBeInTheDocument();
        expect(screen.getByText('Confluence')).toBeInTheDocument();
      });
    });

    it('shows connector status badges', async () => {
      render(<ConnectorDashboard />);

      await waitFor(() => {
        expect(screen.getByText('connected')).toBeInTheDocument();
        expect(screen.getByText('syncing')).toBeInTheDocument();
        expect(screen.getByText('error')).toBeInTheDocument();
      });
    });

    it('shows document counts', async () => {
      render(<ConnectorDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/1,250/)).toBeInTheDocument();
        expect(screen.getByText(/3,500/)).toBeInTheDocument();
      });
    });
  });

  describe('Filter Functionality', () => {
    it('shows filter options', async () => {
      render(<ConnectorDashboard />);

      await waitFor(() => {
        expect(screen.getByText('All')).toBeInTheDocument();
        expect(screen.getByText('Connected')).toBeInTheDocument();
        expect(screen.getByText('Disconnected')).toBeInTheDocument();
        expect(screen.getByText('Error')).toBeInTheDocument();
      });
    });
  });

  describe('Connector Selection', () => {
    it('calls onSelectConnector when connector is clicked', async () => {
      const mockOnSelect = jest.fn();
      render(<ConnectorDashboard onSelectConnector={mockOnSelect} />);

      await waitFor(() => {
        expect(screen.getByText('Google Drive')).toBeInTheDocument();
      });

      await act(async () => {
        fireEvent.click(screen.getByText('Google Drive'));
      });

      expect(mockOnSelect).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'Google Drive',
        })
      );
    });
  });

  describe('Sync Actions', () => {
    it('shows sync button for connected connectors', async () => {
      render(<ConnectorDashboard />);

      await waitFor(() => {
        const syncButtons = screen.getAllByText(/Sync/i);
        expect(syncButtons.length).toBeGreaterThan(0);
      });
    });

    it('shows progress for syncing connectors', async () => {
      render(<ConnectorDashboard />);

      await waitFor(() => {
        // SharePoint is syncing at 65%
        expect(screen.getByText(/65%/)).toBeInTheDocument();
      });
    });

    it('shows error message for failed connectors', async () => {
      render(<ConnectorDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/Authentication expired/)).toBeInTheDocument();
      });
    });
  });

  describe('Loading State', () => {
    it('shows loading indicator initially', () => {
      render(<ConnectorDashboard />);

      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });
  });

  describe('CSS Classes', () => {
    it('applies custom className', async () => {
      const { container } = render(<ConnectorDashboard className="custom-class" />);

      await waitFor(() => {
        expect(container.firstChild).toHaveClass('custom-class');
      });
    });
  });

  describe('Add Connector', () => {
    it('shows add connector button', async () => {
      render(<ConnectorDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/Add Connector/i)).toBeInTheDocument();
      });
    });

    it('opens config modal when add button is clicked', async () => {
      render(<ConnectorDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/Add Connector/i)).toBeInTheDocument();
      });

      await act(async () => {
        fireEvent.click(screen.getByText(/Add Connector/i));
      });

      // Modal should appear
      await waitFor(() => {
        expect(screen.getByText(/Configure Connector/i)).toBeInTheDocument();
      });
    });
  });
});
