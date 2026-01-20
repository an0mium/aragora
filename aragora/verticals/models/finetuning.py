"""
Vertical Specialist Fine-tuning Pipeline.

Provides LoRA fine-tuning for domain-specific models using
debate transcripts and domain data.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FinetuningConfig:
    """Configuration for fine-tuning a specialist model."""

    # Base model
    base_model_id: str
    vertical_id: str

    # Output
    output_dir: str
    adapter_name: str = "lora_adapter"

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training configuration
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048

    # Optimization
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_model_id": self.base_model_id,
            "vertical_id": self.vertical_id,
            "output_dir": self.output_dir,
            "adapter_name": self.adapter_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_seq_length": self.max_seq_length,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
        }


@dataclass
class TrainingExample:
    """A single training example for fine-tuning."""

    instruction: str
    input: str
    output: str
    vertical: str
    source: str = "custom"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt(self, template: str = "alpaca") -> str:
        """
        Convert to prompt format.

        Args:
            template: Prompt template ("alpaca", "chatml", "llama")

        Returns:
            Formatted prompt string
        """
        if template == "alpaca":
            if self.input:
                return (
                    f"### Instruction:\n{self.instruction}\n\n"
                    f"### Input:\n{self.input}\n\n"
                    f"### Response:\n{self.output}"
                )
            else:
                return f"### Instruction:\n{self.instruction}\n\n" f"### Response:\n{self.output}"

        elif template == "chatml":
            messages = [
                {"role": "system", "content": f"You are a {self.vertical} specialist."},
                {
                    "role": "user",
                    "content": (
                        f"{self.instruction}\n\n{self.input}" if self.input else self.instruction
                    ),
                },
                {"role": "assistant", "content": self.output},
            ]
            return "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages)

        elif template == "llama":
            return (
                f"<s>[INST] <<SYS>>\nYou are a {self.vertical} specialist.\n<</SYS>>\n\n"
                f"{self.instruction}"
                + (f"\n\n{self.input}" if self.input else "")
                + f" [/INST] {self.output} </s>"
            )

        else:
            return f"{self.instruction}\n{self.input}\n{self.output}"


class VerticalFineTuningPipeline:
    """
    Fine-tuning pipeline for vertical specialist models.

    Supports:
    - LoRA/QLoRA fine-tuning
    - Dataset preparation from debate transcripts
    - Training progress tracking
    - Model evaluation
    """

    def __init__(
        self,
        config: FinetuningConfig,
        data_dir: Optional[str] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            config: Fine-tuning configuration
            data_dir: Directory containing training data
        """
        self.config = config
        self.data_dir = data_dir
        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._training_examples: List[TrainingExample] = []

    def load_base_model(self, quantization: Optional[str] = "4bit") -> None:
        """
        Load the base model for fine-tuning.

        Args:
            quantization: "4bit" or "8bit" for QLoRA
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch

            logger.info(f"Loading base model: {self.config.base_model_id}")

            # Configure quantization for QLoRA
            bnb_config = None
            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif quantization == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            # Load model
            model_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
            }
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config

            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                **model_kwargs,
            )

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_id,
                trust_remote_code=True,
            )

            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Enable gradient checkpointing
            if self.config.gradient_checkpointing:
                self._model.gradient_checkpointing_enable()

            logger.info("Base model loaded successfully")

        except ImportError as e:
            logger.error(f"Missing dependencies for fine-tuning: {e}")
            raise

    def prepare_lora_model(self) -> None:
        """Configure LoRA adapters on the base model."""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if self._model is None:
                raise ValueError("Base model not loaded")

            logger.info("Configuring LoRA adapters")

            # Prepare model for k-bit training
            self._model = prepare_model_for_kbit_training(self._model)

            # Configure LoRA
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Apply LoRA
            self._model = get_peft_model(self._model, lora_config)

            # Print trainable parameters
            trainable, total = self._model.get_nb_trainable_parameters()
            logger.info(
                f"Trainable parameters: {trainable:,} / {total:,} "
                f"({100 * trainable / total:.2f}%)"
            )

        except ImportError as e:
            logger.error(f"peft library required for LoRA: {e}")
            raise

    def add_training_example(self, example: TrainingExample) -> None:
        """Add a training example."""
        self._training_examples.append(example)

    def add_training_examples_from_file(
        self,
        file_path: str,
        format: str = "jsonl",
    ) -> int:
        """
        Load training examples from a file.

        Args:
            file_path: Path to training data
            format: "jsonl" or "json"

        Returns:
            Number of examples loaded
        """
        path = Path(file_path)
        count = 0

        if format == "jsonl":
            with open(path) as f:
                for line in f:
                    data = json.loads(line.strip())
                    example = TrainingExample(
                        instruction=data.get("instruction", ""),
                        input=data.get("input", ""),
                        output=data.get("output", ""),
                        vertical=data.get("vertical", self.config.vertical_id),
                        source=data.get("source", "file"),
                        metadata=data.get("metadata", {}),
                    )
                    self._training_examples.append(example)
                    count += 1

        elif format == "json":
            with open(path) as f:
                data = json.load(f)
                for item in data:
                    example = TrainingExample(
                        instruction=item.get("instruction", ""),
                        input=item.get("input", ""),
                        output=item.get("output", ""),
                        vertical=item.get("vertical", self.config.vertical_id),
                        source=item.get("source", "file"),
                        metadata=item.get("metadata", {}),
                    )
                    self._training_examples.append(example)
                    count += 1

        logger.info(f"Loaded {count} training examples from {file_path}")
        return count

    def add_debate_transcript(
        self,
        debate_id: str,
        topic: str,
        messages: List[Dict[str, str]],
        final_answer: str,
    ) -> None:
        """
        Convert a debate transcript to training examples.

        Args:
            debate_id: Unique debate identifier
            topic: Debate topic/question
            messages: List of debate messages
            final_answer: Consensus or final answer
        """
        # Create instruction from debate topic
        instruction = (
            f"As a {self.config.vertical_id} specialist, analyze and provide guidance on: {topic}"
        )

        # Build context from key messages
        key_insights = []
        for msg in messages:
            if msg.get("role") in ("critic", "expert"):
                key_insights.append(f"- {msg.get('content', '')[:200]}")

        input_text = "\n".join(key_insights[:5]) if key_insights else ""

        example = TrainingExample(
            instruction=instruction,
            input=input_text,
            output=final_answer,
            vertical=self.config.vertical_id,
            source="debate",
            metadata={"debate_id": debate_id},
        )

        self._training_examples.append(example)
        logger.debug(f"Added debate transcript: {debate_id}")

    def prepare_dataset(
        self,
        template: str = "alpaca",
        train_split: float = 0.9,
    ) -> Any:
        """
        Prepare the dataset for training.

        Args:
            template: Prompt template
            train_split: Fraction for training (rest is eval)

        Returns:
            DatasetDict with train and eval splits
        """
        try:
            from datasets import Dataset, DatasetDict

            if not self._training_examples:
                raise ValueError("No training examples added")

            # Convert to prompts
            texts = [ex.to_prompt(template) for ex in self._training_examples]

            # Create dataset
            dataset = Dataset.from_dict({"text": texts})

            # Split
            split = dataset.train_test_split(test_size=1 - train_split, seed=42)

            logger.info(
                f"Dataset prepared: {len(split['train'])} train, "
                f"{len(split['test'])} eval examples"
            )

            return split

        except ImportError as e:
            logger.error(f"datasets library required: {e}")
            raise

    def train(
        self,
        dataset: Any,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the fine-tuning training.

        Args:
            dataset: Prepared dataset
            resume_from_checkpoint: Checkpoint to resume from

        Returns:
            Training metrics
        """
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            from trl import SFTTrainer

            if self._model is None or self._tokenizer is None:
                raise ValueError("Model not loaded")

            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                logging_steps=self.config.logging_steps,
                eval_strategy="steps",
                eval_steps=self.config.eval_steps,
                save_strategy="steps",
                save_steps=self.config.save_steps,
                save_total_limit=3,
                load_best_model_at_end=True,
                report_to=["tensorboard"],
                remove_unused_columns=False,
            )

            # Create trainer
            self._trainer = SFTTrainer(
                model=self._model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                tokenizer=self._tokenizer,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
            )

            logger.info("Starting training...")

            # Train
            train_result = self._trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            # Save final model
            self._trainer.save_model(os.path.join(self.config.output_dir, self.config.adapter_name))

            # Save training metrics
            metrics = train_result.metrics
            self._trainer.log_metrics("train", metrics)
            self._trainer.save_metrics("train", metrics)

            logger.info("Training completed")
            return metrics

        except ImportError as e:
            logger.error(f"Missing dependencies for training: {e}")
            raise

    def save_adapter(self, path: Optional[str] = None) -> str:
        """
        Save the trained LoRA adapter.

        Args:
            path: Path to save adapter (default: output_dir/adapter_name)

        Returns:
            Path where adapter was saved
        """
        if self._model is None:
            raise ValueError("Model not loaded")

        save_path = path or os.path.join(
            self.config.output_dir,
            self.config.adapter_name,
        )

        self._model.save_pretrained(save_path)

        # Also save config
        config_path = os.path.join(save_path, "finetuning_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Adapter saved to: {save_path}")
        return save_path

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "num_examples": len(self._training_examples),
            "verticals": list(set(ex.vertical for ex in self._training_examples)),
            "sources": list(set(ex.source for ex in self._training_examples)),
            "config": self.config.to_dict(),
        }
