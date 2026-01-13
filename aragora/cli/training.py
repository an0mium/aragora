"""
CLI commands for Tinker training operations.

Provides commands for:
- Exporting training data
- Scheduling training jobs
- Managing trained models
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="training",
    help="Tinker training operations for fine-tuning models on Aragora debate data",
)


@app.command("export-sft")
def export_sft(
    output: str = typer.Option(
        "sft_training_data.jsonl", "-o", "--output", help="Output file path"
    ),
    min_confidence: float = typer.Option(0.7, "--min-confidence", help="Minimum debate confidence"),
    min_success_rate: float = typer.Option(
        0.6, "--min-success-rate", help="Minimum pattern success rate"
    ),
    limit: int = typer.Option(1000, "--limit", help="Maximum records to export"),
    db_path: str = typer.Option("agora_memory.db", "--db-path", help="Database path"),
):
    """Export SFT (Supervised Fine-Tuning) training data from debates."""
    from aragora.training.exporters import SFTExporter

    exporter = SFTExporter(db_path=db_path)
    metadata = exporter.export_to_file(
        output,
        min_confidence=min_confidence,
        min_success_rate=min_success_rate,
        limit=limit,
    )

    typer.echo(f"Exported {metadata.total_records} SFT records to {output}")
    typer.echo(f"Filters: confidence >= {min_confidence}, success_rate >= {min_success_rate}")


@app.command("export-dpo")
def export_dpo(
    output: str = typer.Option(
        "dpo_training_data.jsonl", "-o", "--output", help="Output file path"
    ),
    min_elo_difference: float = typer.Option(50.0, "--min-elo-diff", help="Minimum ELO difference"),
    min_debates: int = typer.Option(3, "--min-debates", help="Minimum debates between agents"),
    limit: int = typer.Option(500, "--limit", help="Maximum records to export"),
):
    """Export DPO (Direct Preference Optimization) training data from ELO rankings."""
    from aragora.training.exporters import DPOExporter

    exporter = DPOExporter()
    metadata = exporter.export_to_file(
        output,
        min_elo_difference=min_elo_difference,
        min_debates=min_debates,
        limit=limit,
    )

    typer.echo(f"Exported {metadata.total_records} DPO records to {output}")
    typer.echo(f"Filters: elo_diff >= {min_elo_difference}, debates >= {min_debates}")


@app.command("export-gauntlet")
def export_gauntlet(
    output: str = typer.Option(
        "gauntlet_training_data.jsonl", "-o", "--output", help="Output file path"
    ),
    min_robustness: float = typer.Option(0.3, "--min-robustness", help="Minimum robustness score"),
    limit: int = typer.Option(200, "--limit", help="Maximum records to export"),
):
    """Export Gauntlet adversarial training data."""
    from aragora.training.exporters import GauntletExporter

    exporter = GauntletExporter()
    metadata = exporter.export_to_file(
        output,
        min_robustness_score=min_robustness,
        limit=limit,
    )

    typer.echo(f"Exported {metadata.total_records} Gauntlet records to {output}")


@app.command("export-all")
def export_all(
    output_dir: str = typer.Option("training_data", "-d", "--output-dir", help="Output directory"),
    limit: int = typer.Option(1000, "--limit", help="Maximum records per export type"),
):
    """Export all training data types."""
    from aragora.training.exporters import SFTExporter, DPOExporter, GauntletExporter

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # SFT
    sft_exporter = SFTExporter()
    sft_metadata = sft_exporter.export_to_file(
        output_path / "sft_data.jsonl",
        limit=limit,
    )

    # DPO
    dpo_exporter = DPOExporter()
    dpo_metadata = dpo_exporter.export_to_file(
        output_path / "dpo_data.jsonl",
        limit=limit // 2,
    )

    # Gauntlet
    gauntlet_exporter = GauntletExporter()
    gauntlet_metadata = gauntlet_exporter.export_to_file(
        output_path / "gauntlet_data.jsonl",
        limit=limit // 5,
    )

    typer.echo(f"\nExported training data to {output_dir}/")
    typer.echo(f"  - SFT: {sft_metadata.total_records} records")
    typer.echo(f"  - DPO: {dpo_metadata.total_records} records")
    typer.echo(f"  - Gauntlet: {gauntlet_metadata.total_records} records")


@app.command("test-connection")
def test_connection():
    """Test connection to Tinker API."""
    import os
    from aragora.training.tinker_client import TinkerClient, TinkerAPIError

    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        typer.echo("Error: TINKER_API_KEY environment variable not set", err=True)
        raise typer.Exit(1)

    async def _test():
        client = TinkerClient()
        try:
            await client.test_connection()
            typer.echo("Connection successful!")
        except TinkerAPIError as e:
            typer.echo(f"Connection failed: {e}", err=True)
            raise typer.Exit(1)
        finally:
            await client.close()

    asyncio.run(_test())


@app.command("train-sft")
def train_sft(
    model: str = typer.Option("llama-3.3-70b", "--model", help="Base model to fine-tune"),
    adapter_name: Optional[str] = typer.Option(None, "--adapter-name", help="Name for the adapter"),
    min_confidence: float = typer.Option(0.7, "--min-confidence", help="Minimum debate confidence"),
    limit: int = typer.Option(1000, "--limit", help="Maximum training examples"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for completion"),
):
    """Schedule an SFT training job."""
    from aragora.training import TrainingScheduler

    async def _train():
        scheduler = TrainingScheduler()
        try:
            job = await scheduler.schedule_sft(
                model=model,
                adapter_name=adapter_name,
                min_confidence=min_confidence,
                limit=limit,
            )

            typer.echo(f"Scheduled SFT job: {job.job_id}")
            typer.echo(f"  Model: {job.model}")
            typer.echo(f"  Status: {job.status.value}")

            if wait:
                typer.echo("\nWaiting for completion...")
                job = await scheduler.wait_for_job(job.job_id)

                if job.status.value == "completed":
                    typer.echo(f"\nTraining completed!")
                    typer.echo(f"  Model ID: {job.model_id}")
                    if job.result:
                        typer.echo(f"  Final loss: {job.result.final_loss}")
                        typer.echo(f"  Training time: {job.result.training_time_seconds:.0f}s")
                else:
                    typer.echo(f"\nTraining failed: {job.error}", err=True)
                    raise typer.Exit(1)

        finally:
            await scheduler.close()

    asyncio.run(_train())


@app.command("train-dpo")
def train_dpo(
    model: str = typer.Option("llama-3.3-70b", "--model", help="Base model to fine-tune"),
    adapter_name: Optional[str] = typer.Option(None, "--adapter-name", help="Name for the adapter"),
    min_elo_diff: float = typer.Option(50.0, "--min-elo-diff", help="Minimum ELO difference"),
    limit: int = typer.Option(500, "--limit", help="Maximum training examples"),
    beta: float = typer.Option(0.1, "--beta", help="DPO temperature parameter"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for completion"),
):
    """Schedule a DPO training job."""
    from aragora.training import TrainingScheduler

    async def _train():
        scheduler = TrainingScheduler()
        try:
            job = await scheduler.schedule_dpo(
                model=model,
                adapter_name=adapter_name,
                min_elo_difference=min_elo_diff,
                limit=limit,
                beta=beta,
            )

            typer.echo(f"Scheduled DPO job: {job.job_id}")
            typer.echo(f"  Model: {job.model}")
            typer.echo(f"  Beta: {beta}")

            if wait:
                typer.echo("\nWaiting for completion...")
                job = await scheduler.wait_for_job(job.job_id)

                if job.status.value == "completed":
                    typer.echo(f"\nTraining completed!")
                    typer.echo(f"  Model ID: {job.model_id}")
                else:
                    typer.echo(f"\nTraining failed: {job.error}", err=True)
                    raise typer.Exit(1)

        finally:
            await scheduler.close()

    asyncio.run(_train())


@app.command("train-combined")
def train_combined(
    model: str = typer.Option("llama-3.3-70b", "--model", help="Base model to fine-tune"),
    adapter_name: Optional[str] = typer.Option(None, "--adapter-name", help="Name for the adapter"),
    sft_limit: int = typer.Option(1000, "--sft-limit", help="SFT training examples"),
    dpo_limit: int = typer.Option(500, "--dpo-limit", help="DPO training examples"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for completion"),
):
    """Schedule a combined SFT + DPO training pipeline."""
    from aragora.training import TrainingScheduler

    async def _train():
        scheduler = TrainingScheduler()
        try:
            job = await scheduler.schedule_combined(
                model=model,
                adapter_name=adapter_name,
                sft_limit=sft_limit,
                dpo_limit=dpo_limit,
            )

            typer.echo(f"Scheduled combined job: {job.job_id}")
            typer.echo(f"  Model: {job.model}")
            typer.echo(f"  Phase 1: SFT ({sft_limit} examples)")
            typer.echo(f"  Phase 2: DPO ({dpo_limit} examples)")

            if wait:
                typer.echo("\nWaiting for completion...")
                job = await scheduler.wait_for_job(job.job_id, timeout=7200)  # 2 hour timeout

                if job.status.value == "completed":
                    typer.echo(f"\nTraining completed!")
                    typer.echo(f"  Model ID: {job.model_id}")
                else:
                    typer.echo(f"\nTraining failed: {job.error}", err=True)
                    raise typer.Exit(1)

        finally:
            await scheduler.close()

    asyncio.run(_train())


@app.command("list-models")
def list_models():
    """List available fine-tuned models."""
    from aragora.training.tinker_client import TinkerClient

    async def _list():
        client = TinkerClient()
        try:
            models = await client.list_models()

            if not models:
                typer.echo("No fine-tuned models found")
                return

            typer.echo(f"Found {len(models)} models:\n")
            for model in models:
                typer.echo(f"  {model.get('model_id', 'unknown')}")
                typer.echo(f"    Base: {model.get('base_model', 'unknown')}")
                typer.echo(f"    Created: {model.get('created_at', 'unknown')}")
                typer.echo()

        finally:
            await client.close()

    asyncio.run(_list())


@app.command("sample")
def sample(
    prompt: str = typer.Argument(..., help="Prompt to generate from"),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Fine-tuned model ID"),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", help="Sampling temperature"),
):
    """Generate text from a fine-tuned model."""
    from aragora.training.tinker_client import TinkerClient

    async def _sample():
        client = TinkerClient()
        try:
            response = await client.sample(
                prompt=prompt,
                model_id=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            typer.echo(response)

        finally:
            await client.close()

    asyncio.run(_sample())


@app.command("stats")
def show_stats():
    """Show training data statistics."""
    from aragora.memory.store import CritiqueStore
    from aragora.ranking.elo import EloSystem

    # Get CritiqueStore stats
    store = CritiqueStore()
    critique_stats = store.get_stats()

    # Get ELO stats
    elo = EloSystem()
    elo_stats = elo.get_stats()

    typer.echo("Training Data Statistics")
    typer.echo("=" * 40)
    typer.echo("\nDebate Data (SFT source):")
    typer.echo(f"  Total debates: {critique_stats.get('total_debates', 0)}")
    typer.echo(f"  Consensus debates: {critique_stats.get('consensus_debates', 0)}")
    typer.echo(f"  Total critiques: {critique_stats.get('total_critiques', 0)}")
    typer.echo(f"  Total patterns: {critique_stats.get('total_patterns', 0)}")
    typer.echo(f"  Avg confidence: {critique_stats.get('avg_consensus_confidence', 0):.2f}")

    typer.echo("\nELO Data (DPO source):")
    typer.echo(f"  Total agents: {elo_stats.get('total_agents', 0)}")
    typer.echo(f"  Total matches: {elo_stats.get('total_matches', 0)}")
    typer.echo(f"  Average ELO: {elo_stats.get('average_elo', 1000):.0f}")

    patterns_by_type = critique_stats.get("patterns_by_type", {})
    if patterns_by_type:
        typer.echo("\nPatterns by Type:")
        for ptype, count in sorted(patterns_by_type.items(), key=lambda x: -x[1]):
            typer.echo(f"  {ptype}: {count}")


if __name__ == "__main__":
    app()
