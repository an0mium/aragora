# Fine-Tuning Guide

This guide covers Aragora's fine-tuning pipeline for training domain-specific models using debate outcomes as training data. The system supports LoRA-based fine-tuning with multiple export formats.

## Overview

The fine-tuning pipeline allows you to:
1. Export debate outcomes as training data (SFT, DPO formats)
2. Configure and run LoRA fine-tuning jobs
3. Monitor training progress in real-time
4. Deploy trained adapters for specialized tasks

## Dashboard

Access the fine-tuning dashboard at `/training` or through the admin panel at `/admin/training`.

### Dashboard Sections

**Stats Bar** - Shows job counts by status:
- Training (running jobs)
- Queued (pending jobs)
- Completed (successful jobs)
- Failed (error jobs)

**Tabs:**
- **Active Jobs** - Monitor running and recent jobs
- **New Job** - Configure and start new training
- **Available Models** - Browse base models for fine-tuning

## Creating a Training Job

### 1. Select a Base Model

Choose from available base models. Models are categorized by:
- **Provider** - Anthropic, OpenAI, HuggingFace, etc.
- **Size** - Parameter count (7B, 13B, 70B)
- **Vertical** - Domain specialization (legal, medical, code)
- **Capabilities** - What the model can do (debate, critique, summarize)

Recommended models are marked for common use cases.

### 2. Configure Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `jobName` | Required | Unique name for this training job |
| `loraR` | 8 | LoRA rank (higher = more parameters) |
| `loraAlpha` | 16 | LoRA scaling factor |
| `loraDropout` | 0.05 | Dropout for regularization |
| `numEpochs` | 3 | Training epochs |
| `batchSize` | 4 | Training batch size |
| `learningRate` | 2e-4 | Learning rate |
| `maxSeqLength` | 2048 | Maximum sequence length |
| `quantization` | `4bit` | Quantization level (`4bit`, `8bit`, `none`) |
| `gradientCheckpointing` | true | Enable gradient checkpointing |
| `datasetPath` | Optional | Custom dataset path |

### 3. Start Training

After configuration, click "Start Training" to:
1. Create the job in `queued` status
2. Automatically start the training process
3. Redirect to the Active Jobs tab for monitoring

## Training Job Lifecycle

Jobs progress through these statuses:

```
queued → preparing → training → completed
                           ↓
                        failed
                           ↓
                      cancelled
```

| Status | Description |
|--------|-------------|
| `queued` | Job created, waiting to start |
| `preparing` | Loading model and preparing data |
| `training` | Actively training |
| `completed` | Training finished successfully |
| `failed` | Error during training |
| `cancelled` | Manually cancelled by user |

## Monitoring Jobs

The Job Monitor shows:
- **Progress bar** - Visual progress indicator
- **Current epoch/step** - Training position
- **Loss** - Current training loss
- **Training examples** - Number of examples processed
- **Start/completion time** - Timestamps

### Job Actions

- **Cancel** - Stop a running job (cannot be resumed)
- **View Metrics** - See detailed training metrics
- **View Artifacts** - Access model checkpoints and logs

## Training Data Export

Export debate outcomes for external fine-tuning or analysis.

### SFT Format (Supervised Fine-Tuning)

Standard instruction-following format:

```json
{
  "instruction": "Analyze the following debate topic...",
  "input": "Should AI systems have explicit uncertainty estimates?",
  "output": "Yes, AI systems should provide explicit uncertainty...",
  "metadata": {
    "debate_id": "debate-123",
    "consensus_confidence": 0.87,
    "rounds": 3
  }
}
```

**API Endpoint:**
```
GET /api/training/export/sft?min_confidence=0.7&limit=1000
```

**Parameters:**
- `min_confidence` - Minimum consensus confidence (0.0-1.0)
- `limit` - Maximum records to export

### DPO Format (Direct Preference Optimization)

Preference pairs from debate outcomes:

```json
{
  "prompt": "What is the best approach to...",
  "chosen": "The consensus answer from the winning side...",
  "rejected": "The alternative answer that lost...",
  "metadata": {
    "debate_id": "debate-123",
    "confidence_diff": 0.35
  }
}
```

**API Endpoint:**
```
GET /api/training/export/dpo?min_confidence_diff=0.3&limit=1000
```

**Parameters:**
- `min_confidence_diff` - Minimum confidence difference between sides
- `limit` - Maximum records to export

### Gauntlet Format

Exports from Gauntlet adversarial testing:

```json
{
  "prompt": "Attack scenario prompt...",
  "response": "Model response...",
  "attack_type": "prompt_injection",
  "success": false,
  "findings": ["Resisted manipulation", "Maintained reasoning"]
}
```

## API Reference

### List Jobs

```
GET /api/training/jobs
GET /api/training/jobs?vertical=legal&status=completed
```

### Get Job Details

```
GET /api/training/jobs/:id
```

### Create Job

```
POST /api/training/jobs
Content-Type: application/json

{
  "name": "legal_specialist_v1",
  "vertical": "legal",
  "base_model": "nlpaueb/legal-bert-base-uncased",
  "training_config": {
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 0.0002,
    "max_seq_length": 2048,
    "quantization": "4bit",
    "gradient_checkpointing": true
  }
}
```

### Start Job

```
POST /api/training/jobs/:id/start
```

### Cancel Job

```
DELETE /api/training/jobs/:id
```

### Get Job Metrics

```
GET /api/training/jobs/:id/metrics
```

Returns:
```json
{
  "job_id": "job-123",
  "status": "completed",
  "training_data_examples": 1500,
  "training_data_debates": 300,
  "final_loss": 0.245,
  "elo_rating": 1450,
  "win_rate": 0.67,
  "vertical_accuracy": {
    "legal": 0.82,
    "technical": 0.75
  }
}
```

### Get Job Artifacts

```
GET /api/training/jobs/:id/artifacts
```

Returns:
```json
{
  "job_id": "job-123",
  "checkpoint_path": "/models/job-123/checkpoint-final",
  "data_directory": "/data/training/job-123",
  "files": [
    { "name": "adapter_config.json", "size_bytes": 1024, "type": "config" },
    { "name": "adapter_model.bin", "size_bytes": 52428800, "type": "weights" }
  ]
}
```

### Get Available Formats

```
GET /api/training/formats
```

Returns documentation of all export formats and their schemas.

### Get Training Stats

```
GET /api/training/stats
```

Returns:
```json
{
  "available_exporters": ["sft", "dpo", "gauntlet"],
  "export_directory": "/data/exports",
  "exported_files": [
    { "name": "sft_export_2024-01.jsonl", "size_bytes": 1048576, "created_at": "..." }
  ],
  "sft_available": true
}
```

## React Hook

Use the `useFineTuning` hook for frontend integration:

```tsx
import { useFineTuning } from '@/hooks/useFineTuning';

function TrainingDashboard() {
  const {
    jobs,
    stats,
    loading,
    error,
    createJob,
    startJob,
    cancelJob,
    exportSFT,
  } = useFineTuning({ autoLoad: true, pollInterval: 30000 });

  // Create and start a job
  const handleCreate = async () => {
    const job = await createJob({
      name: 'my_model_v1',
      vertical: 'general',
      base_model: 'meta-llama/Llama-2-7b',
      training_config: {
        lora_r: 8,
        lora_alpha: 16,
        lora_dropout: 0.05,
        num_epochs: 3,
        batch_size: 4,
        learning_rate: 2e-4,
        max_seq_length: 2048,
        quantization: '4bit',
        gradient_checkpointing: true,
      },
    });

    if (job) {
      await startJob(job.id);
    }
  };

  return (
    <div>
      <p>Running: {stats.running}</p>
      <p>Completed: {stats.completed}</p>
      {jobs.map(job => (
        <JobCard key={job.id} job={job} onCancel={() => cancelJob(job.id)} />
      ))}
    </div>
  );
}
```

## Best Practices

### Data Quality

- Use debates with high consensus confidence (>0.7)
- Include diverse topics for generalization
- Balance verticals to avoid specialization bias

### Training Configuration

- Start with smaller LoRA rank (8) and increase if needed
- Use 4-bit quantization for memory efficiency
- Enable gradient checkpointing for larger models

### Monitoring

- Watch for loss plateaus (may indicate overfitting)
- Compare validation metrics across verticals
- Test on held-out debates before deployment

### Deployment

- Export adapters in standard format
- Test with multiple prompt styles
- Compare against base model performance

## Troubleshooting

### "Training pipeline not available"

The training service is not running. Check:
1. Backend server is running
2. Training dependencies installed (`pip install transformers peft`)
3. GPU/CUDA available (if using quantization)

### Job Stuck in "Preparing"

- Check available disk space for model download
- Verify network access to model hub
- Check backend logs for download errors

### High Loss / No Convergence

- Reduce learning rate
- Increase LoRA rank
- Check data quality (min confidence threshold)
- Try more epochs

### Out of Memory

- Reduce batch size
- Enable gradient checkpointing
- Use 4-bit quantization
- Use a smaller base model

## Related Documentation

- [Training Data Export](./FEATURES.md#traininghandler) - More on export formats
- [Gauntlet Guide](./GAUNTLET.md) - Adversarial testing
- [API Usage](./API_USAGE.md) - General API documentation
- [Observability](./OBSERVABILITY.md) - Monitoring and metrics
