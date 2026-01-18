'use client';

import { useState, useMemo } from 'react';
import { ModelSelector, type AvailableModel } from './ModelSelector';
import { TrainingConfig, type TrainingParameters } from './TrainingConfig';
import { JobMonitor } from './JobMonitor';

export interface FineTuningJob {
  id: string;
  name: string;
  vertical: string;
  baseModel: string;
  status: 'queued' | 'preparing' | 'training' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  currentEpoch?: number;
  totalEpochs?: number;
  currentStep?: number;
  totalSteps?: number;
  loss?: number;
  trainingExamples: number;
  startedAt?: string;
  completedAt?: string;
  outputPath?: string;
  error?: string;
}

export interface FineTuningDashboardProps {
  jobs?: FineTuningJob[];
  onStartJob?: (config: TrainingParameters & { model: AvailableModel }) => void;
  onCancelJob?: (jobId: string) => void;
  className?: string;
}

type TabId = 'new' | 'jobs' | 'models';

// Mock jobs for demonstration
const MOCK_JOBS: FineTuningJob[] = [
  {
    id: 'ft_001',
    name: 'Legal Specialist v2',
    vertical: 'legal',
    baseModel: 'nlpaueb/legal-bert-base-uncased',
    status: 'training',
    progress: 0.45,
    currentEpoch: 2,
    totalEpochs: 3,
    currentStep: 450,
    totalSteps: 1000,
    loss: 0.324,
    trainingExamples: 2500,
    startedAt: '2024-01-16T10:00:00Z',
  },
  {
    id: 'ft_002',
    name: 'Healthcare Compliance Model',
    vertical: 'healthcare',
    baseModel: 'medicalai/ClinicalBERT',
    status: 'completed',
    progress: 1.0,
    currentEpoch: 3,
    totalEpochs: 3,
    trainingExamples: 1800,
    startedAt: '2024-01-15T14:00:00Z',
    completedAt: '2024-01-15T18:30:00Z',
    outputPath: '/models/healthcare_compliance_v1',
  },
  {
    id: 'ft_003',
    name: 'Code Review Assistant',
    vertical: 'software',
    baseModel: 'codellama/CodeLlama-7b-Instruct-hf',
    status: 'queued',
    progress: 0,
    trainingExamples: 5000,
  },
];

export function FineTuningDashboard({
  jobs = MOCK_JOBS,
  onStartJob,
  onCancelJob,
  className = '',
}: FineTuningDashboardProps) {
  const [activeTab, setActiveTab] = useState<TabId>('jobs');
  const [selectedModel, setSelectedModel] = useState<AvailableModel | null>(null);

  const stats = useMemo(() => ({
    running: jobs.filter(j => j.status === 'training' || j.status === 'preparing').length,
    queued: jobs.filter(j => j.status === 'queued').length,
    completed: jobs.filter(j => j.status === 'completed').length,
    failed: jobs.filter(j => j.status === 'failed').length,
  }), [jobs]);

  const handleStartTraining = (params: TrainingParameters) => {
    if (selectedModel && onStartJob) {
      onStartJob({ ...params, model: selectedModel });
      setActiveTab('jobs');
    }
  };

  return (
    <div className={`bg-surface border border-border rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-bg">
        <h3 className="text-sm font-mono font-bold text-acid-green">
          FINE-TUNING PIPELINE
        </h3>
        <p className="text-xs text-text-muted mt-1">
          Train domain-specific models with LoRA adapters
        </p>
      </div>

      {/* Stats Bar */}
      <div className="grid grid-cols-4 gap-3 p-4 border-b border-border">
        <div className="text-center">
          <div className="text-xl font-bold text-cyan-400">{stats.running}</div>
          <div className="text-xs text-text-muted">Training</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-bold text-yellow-400">{stats.queued}</div>
          <div className="text-xs text-text-muted">Queued</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-bold text-acid-green">{stats.completed}</div>
          <div className="text-xs text-text-muted">Completed</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-bold text-red-400">{stats.failed}</div>
          <div className="text-xs text-text-muted">Failed</div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border">
        {(['jobs', 'new', 'models'] as TabId[]).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`
              px-4 py-2 text-xs font-mono uppercase transition-colors
              ${activeTab === tab
                ? 'text-acid-green border-b-2 border-acid-green bg-bg'
                : 'text-text-muted hover:text-text'
              }
            `}
          >
            {tab === 'new' ? 'New Job' : tab === 'jobs' ? 'Active Jobs' : 'Available Models'}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="p-4">
        {activeTab === 'jobs' && (
          <JobMonitor
            jobs={jobs}
            onCancelJob={onCancelJob}
          />
        )}

        {activeTab === 'new' && (
          <div className="space-y-6">
            <ModelSelector
              selectedModel={selectedModel}
              onSelectModel={setSelectedModel}
            />
            {selectedModel && (
              <TrainingConfig
                model={selectedModel}
                onStartTraining={handleStartTraining}
              />
            )}
          </div>
        )}

        {activeTab === 'models' && (
          <ModelSelector
            showAllModels
            onSelectModel={(model) => {
              setSelectedModel(model);
              setActiveTab('new');
            }}
          />
        )}
      </div>
    </div>
  );
}
