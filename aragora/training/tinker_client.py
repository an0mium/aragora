"""
Tinker API client for fine-tuning open-source LLMs.

Tinker (thinkingmachines.ai) provides a training API for LoRA fine-tuning
of models like Llama, Qwen, and DeepSeek.

API Operations:
- forward_backward: Gradient accumulation
- optim_step: Weight updates
- sample: Generate from fine-tuned model
- save_state: Checkpoint persistence
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)


class TinkerModel(str, Enum):
    """Supported base models for Tinker fine-tuning."""

    LLAMA_3_3_70B = "llama-3.3-70b"
    LLAMA_3_1_8B = "llama-3.1-8b"
    QWEN_2_5_72B = "qwen-2.5-72b"
    QWEN_3_32B = "qwen-3-32b"
    DEEPSEEK_V3 = "deepseek-v3"
    DEEPSEEK_R1 = "deepseek-r1"


class TrainingState(str, Enum):
    """Training job state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TinkerConfig:
    """Configuration for Tinker API client."""

    api_key: str = field(default_factory=lambda: os.getenv("TINKER_API_KEY", ""))
    base_url: str = "https://api.thinkingmachines.ai/v1"
    base_model: str = field(
        default_factory=lambda: os.getenv("TINKER_BASE_MODEL", TinkerModel.LLAMA_3_3_70B.value)
    )
    lora_rank: int = 16
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    warmup_steps: int = 100
    save_steps: int = 100
    timeout: float = 300.0


@dataclass
class TrainingProgress:
    """Progress of a training job."""

    step: int
    total_steps: int
    loss: float
    learning_rate: float
    elapsed_seconds: float
    eta_seconds: float

    @property
    def progress_pct(self) -> float:
        """Progress as percentage."""
        return (self.step / self.total_steps) * 100 if self.total_steps > 0 else 0


@dataclass
class TrainingResult:
    """Result of a training job."""

    job_id: str
    state: TrainingState
    model_id: str | None
    final_loss: float | None
    total_steps: int
    training_time_seconds: float
    checkpoint_path: str | None
    error_message: str | None = None


class TinkerClient:
    """
    Client for the Tinker training API.

    Provides methods for fine-tuning open-source LLMs using LoRA.

    Example:
        client = TinkerClient()

        # Train SFT model
        result = await client.train_sft(
            training_data=data,
            model=TinkerModel.LLAMA_3_3_70B,
        )

        # Generate from fine-tuned model
        response = await client.sample(
            prompt="Critique this proposal: ...",
            model_id=result.model_id,
        )
    """

    def __init__(self, config: TinkerConfig | None = None):
        self.config = config or TinkerConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "TinkerClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def test_connection(self) -> bool:
        """Test API connection and authentication.

        Returns:
            True if connection is successful.

        Raises:
            TinkerAPIError: If connection fails.
        """
        if not self.config.api_key:
            raise TinkerAPIError("TINKER_API_KEY not set")

        client = await self._get_client()
        try:
            response = await client.get("/health")
            response.raise_for_status()
            logger.info("Tinker API connection successful")
            return True
        except httpx.HTTPError as e:
            raise TinkerAPIError(f"Connection failed: {e}") from e

    async def train_sft(
        self,
        training_data: list[dict[str, Any]],
        model: str | TinkerModel = TinkerModel.LLAMA_3_3_70B,
        adapter_name: str | None = None,
        **kwargs,
    ) -> TrainingResult:
        """
        Train a model using Supervised Fine-Tuning (SFT).

        Args:
            training_data: List of {"instruction": ..., "response": ...} dicts
            model: Base model to fine-tune
            adapter_name: Name for the LoRA adapter (auto-generated if None)
            **kwargs: Override default config values

        Returns:
            TrainingResult with job info and model ID
        """
        if isinstance(model, TinkerModel):
            model = model.value

        adapter_name = adapter_name or f"aragora-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Prepare training request
        request = {
            "type": "sft",
            "base_model": model,
            "adapter_name": adapter_name,
            "training_data": training_data,
            "config": {
                "lora_rank": kwargs.get("lora_rank", self.config.lora_rank),
                "learning_rate": kwargs.get("learning_rate", self.config.learning_rate),
                "batch_size": kwargs.get("batch_size", self.config.batch_size),
                "gradient_accumulation_steps": kwargs.get(
                    "gradient_accumulation_steps",
                    self.config.gradient_accumulation_steps,
                ),
                "max_steps": kwargs.get("max_steps", self.config.max_steps),
                "warmup_steps": kwargs.get("warmup_steps", self.config.warmup_steps),
                "save_steps": kwargs.get("save_steps", self.config.save_steps),
            },
        }

        return await self._submit_training_job(request)

    async def train_dpo(
        self,
        preference_data: list[dict[str, Any]],
        model: str | TinkerModel = TinkerModel.LLAMA_3_3_70B,
        adapter_name: str | None = None,
        beta: float = 0.1,
        **kwargs,
    ) -> TrainingResult:
        """
        Train a model using Direct Preference Optimization (DPO).

        Args:
            preference_data: List of {"prompt": ..., "chosen": ..., "rejected": ...} dicts
            model: Base model to fine-tune
            adapter_name: Name for the LoRA adapter
            beta: DPO temperature parameter (lower = stronger preference)
            **kwargs: Override default config values

        Returns:
            TrainingResult with job info and model ID
        """
        if isinstance(model, TinkerModel):
            model = model.value

        adapter_name = adapter_name or f"aragora-dpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        request = {
            "type": "dpo",
            "base_model": model,
            "adapter_name": adapter_name,
            "preference_data": preference_data,
            "config": {
                "lora_rank": kwargs.get("lora_rank", self.config.lora_rank),
                "learning_rate": kwargs.get("learning_rate", self.config.learning_rate),
                "batch_size": kwargs.get("batch_size", self.config.batch_size),
                "beta": beta,
                "max_steps": kwargs.get("max_steps", self.config.max_steps),
            },
        }

        return await self._submit_training_job(request)

    async def _submit_training_job(self, request: dict[str, Any]) -> TrainingResult:
        """Submit a training job and wait for completion."""
        client = await self._get_client()

        try:
            # Submit job
            response = await client.post("/training/jobs", json=request)
            response.raise_for_status()
            job_data = response.json()
            job_id = job_data["job_id"]

            logger.info("Submitted training job: %s", job_id)

            # Poll for completion
            return await self._wait_for_job(job_id)

        except httpx.HTTPError as e:
            raise TinkerAPIError(f"Training submission failed: {e}") from e

    async def _wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 10.0,
    ) -> TrainingResult:
        """Wait for a training job to complete."""
        client = await self._get_client()

        while True:
            try:
                response = await client.get(f"/training/jobs/{job_id}")
                response.raise_for_status()
                job_data = response.json()

                state = TrainingState(job_data["state"])

                if state == TrainingState.COMPLETED:
                    return TrainingResult(
                        job_id=job_id,
                        state=state,
                        model_id=job_data.get("model_id"),
                        final_loss=job_data.get("final_loss"),
                        total_steps=job_data.get("total_steps", 0),
                        training_time_seconds=job_data.get("training_time_seconds", 0),
                        checkpoint_path=job_data.get("checkpoint_path"),
                    )

                if state == TrainingState.FAILED:
                    return TrainingResult(
                        job_id=job_id,
                        state=state,
                        model_id=None,
                        final_loss=None,
                        total_steps=job_data.get("total_steps", 0),
                        training_time_seconds=job_data.get("training_time_seconds", 0),
                        checkpoint_path=None,
                        error_message=job_data.get("error"),
                    )

                if state == TrainingState.CANCELLED:
                    return TrainingResult(
                        job_id=job_id,
                        state=state,
                        model_id=None,
                        final_loss=None,
                        total_steps=0,
                        training_time_seconds=0,
                        checkpoint_path=None,
                        error_message="Job cancelled",
                    )

                # Still running - log progress and continue
                if "progress" in job_data:
                    progress = TrainingProgress(**job_data["progress"])
                    logger.info(
                        "Training progress: %d/%d steps (%.1f%%), loss=%.4f",
                        progress.step,
                        progress.total_steps,
                        progress.progress_pct,
                        progress.loss,
                    )

                await asyncio.sleep(poll_interval)

            except httpx.HTTPError as e:
                logger.warning("Failed to poll job status: %s", e)
                await asyncio.sleep(poll_interval)

    async def sample(
        self,
        prompt: str,
        model_id: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate text from a fine-tuned model.

        Args:
            prompt: Input prompt
            model_id: ID of fine-tuned model (uses base model if None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        client = await self._get_client()

        request = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        if model_id:
            request["model_id"] = model_id
        else:
            request["base_model"] = self.config.base_model

        try:
            response = await client.post("/sample", json=request)
            response.raise_for_status()
            return response.json()["text"]

        except httpx.HTTPError as e:
            raise TinkerAPIError(f"Sampling failed: {e}") from e

    async def sample_stream(
        self,
        prompt: str,
        model_id: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream text generation from a fine-tuned model.

        Args:
            prompt: Input prompt
            model_id: ID of fine-tuned model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks
        """
        client = await self._get_client()

        request = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        if model_id:
            request["model_id"] = model_id
        else:
            request["base_model"] = self.config.base_model

        try:
            async with client.stream("POST", "/sample", json=request) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "text" in data:
                            yield data["text"]

        except httpx.HTTPError as e:
            raise TinkerAPIError(f"Streaming failed: {e}") from e

    async def save_checkpoint(self, model_id: str, path: str) -> str:
        """
        Save a model checkpoint.

        Args:
            model_id: ID of the model to save
            path: Path to save checkpoint

        Returns:
            Path to saved checkpoint
        """
        client = await self._get_client()

        try:
            response = await client.post(
                "/checkpoints",
                json={"model_id": model_id, "path": path},
            )
            response.raise_for_status()
            return response.json()["checkpoint_path"]

        except httpx.HTTPError as e:
            raise TinkerAPIError(f"Checkpoint save failed: {e}") from e

    async def load_checkpoint(self, checkpoint_path: str) -> str:
        """
        Load a model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Model ID of loaded model
        """
        client = await self._get_client()

        try:
            response = await client.post(
                "/checkpoints/load",
                json={"path": checkpoint_path},
            )
            response.raise_for_status()
            return response.json()["model_id"]

        except httpx.HTTPError as e:
            raise TinkerAPIError(f"Checkpoint load failed: {e}") from e

    async def list_models(self) -> list[dict[str, Any]]:
        """List available fine-tuned models."""
        client = await self._get_client()

        try:
            response = await client.get("/models")
            response.raise_for_status()
            return response.json()["models"]

        except httpx.HTTPError as e:
            raise TinkerAPIError(f"Model list failed: {e}") from e

    async def delete_model(self, model_id: str) -> bool:
        """Delete a fine-tuned model."""
        client = await self._get_client()

        try:
            response = await client.delete(f"/models/{model_id}")
            response.raise_for_status()
            return True

        except httpx.HTTPError as e:
            raise TinkerAPIError(f"Model deletion failed: {e}") from e


class TinkerAPIError(Exception):
    """Error from Tinker API."""

    pass


# CLI support for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tinker API client")
    parser.add_argument("--test-connection", action="store_true", help="Test API connection")
    args = parser.parse_args()

    if args.test_connection:

        async def test():
            client = TinkerClient()
            try:
                await client.test_connection()
                print("Connection successful!")
            except TinkerAPIError as e:
                print(f"Connection failed: {e}")
            finally:
                await client.close()

        asyncio.run(test())
