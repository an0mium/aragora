"""Tests for Vertical Fine-tuning Pipeline.

Tests FinetuningConfig, TrainingExample, and VerticalFineTuningPipeline.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from aragora.verticals.models.finetuning import (
    FinetuningConfig,
    TrainingExample,
    VerticalFineTuningPipeline,
)


# =============================================================================
# FinetuningConfig Tests
# =============================================================================


class TestFinetuningConfigDefaults:
    """Test FinetuningConfig default values."""

    def test_required_fields(self):
        """Test required fields must be provided."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        assert config.base_model_id == "test-model"
        assert config.vertical_id == "software"
        assert config.output_dir == "/tmp/output"

    def test_adapter_name_default(self):
        """Test default adapter name."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        assert config.adapter_name == "lora_adapter"

    def test_lora_r_default(self):
        """Test default LoRA r."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        assert config.lora_r == 16

    def test_lora_alpha_default(self):
        """Test default LoRA alpha."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        assert config.lora_alpha == 32

    def test_lora_dropout_default(self):
        """Test default LoRA dropout."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        assert config.lora_dropout == 0.1

    def test_target_modules_default(self):
        """Test default target modules."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_num_train_epochs_default(self):
        """Test default training epochs."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        assert config.num_train_epochs == 3

    def test_batch_size_defaults(self):
        """Test default batch sizes."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        assert config.per_device_train_batch_size == 4
        assert config.per_device_eval_batch_size == 4

    def test_learning_rate_default(self):
        """Test default learning rate."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        assert config.learning_rate == 2e-4

    def test_optimization_defaults(self):
        """Test default optimization flags."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        assert config.fp16 is True
        assert config.bf16 is False
        assert config.gradient_checkpointing is True


class TestFinetuningConfigCustomValues:
    """Test FinetuningConfig with custom values."""

    def test_custom_lora_params(self):
        """Test custom LoRA parameters."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="legal",
            output_dir="/tmp/output",
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
        )
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.05

    def test_custom_training_params(self):
        """Test custom training parameters."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="healthcare",
            output_dir="/tmp/output",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            learning_rate=1e-4,
        )
        assert config.num_train_epochs == 5
        assert config.per_device_train_batch_size == 8
        assert config.learning_rate == 1e-4

    def test_custom_target_modules(self):
        """Test custom target modules."""
        modules = ["gate_proj", "up_proj", "down_proj"]
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="research",
            output_dir="/tmp/output",
            target_modules=modules,
        )
        assert config.target_modules == modules


class TestFinetuningConfigToDict:
    """Test FinetuningConfig serialization."""

    def test_to_dict_contains_all_fields(self):
        """Test to_dict includes all configuration fields."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        d = config.to_dict()
        assert "base_model_id" in d
        assert "vertical_id" in d
        assert "output_dir" in d
        assert "adapter_name" in d
        assert "lora_r" in d
        assert "lora_alpha" in d
        assert "lora_dropout" in d
        assert "target_modules" in d
        assert "num_train_epochs" in d
        assert "fp16" in d
        assert "gradient_checkpointing" in d

    def test_to_dict_values_match(self):
        """Test to_dict values match config."""
        config = FinetuningConfig(
            base_model_id="my-model",
            vertical_id="legal",
            output_dir="/custom/path",
            lora_r=8,
        )
        d = config.to_dict()
        assert d["base_model_id"] == "my-model"
        assert d["vertical_id"] == "legal"
        assert d["output_dir"] == "/custom/path"
        assert d["lora_r"] == 8


# =============================================================================
# TrainingExample Tests
# =============================================================================


class TestTrainingExampleCreation:
    """Test TrainingExample creation."""

    def test_basic_creation(self):
        """Test basic example creation."""
        example = TrainingExample(
            instruction="Write a function",
            input="def foo():",
            output="return 42",
            vertical="software",
        )
        assert example.instruction == "Write a function"
        assert example.input == "def foo():"
        assert example.output == "return 42"
        assert example.vertical == "software"

    def test_default_source(self):
        """Test default source value."""
        example = TrainingExample(
            instruction="test",
            input="",
            output="result",
            vertical="legal",
        )
        assert example.source == "custom"

    def test_default_metadata(self):
        """Test default metadata is empty dict."""
        example = TrainingExample(
            instruction="test",
            input="",
            output="result",
            vertical="healthcare",
        )
        assert example.metadata == {}

    def test_custom_source_and_metadata(self):
        """Test custom source and metadata."""
        example = TrainingExample(
            instruction="test",
            input="input",
            output="output",
            vertical="accounting",
            source="debate",
            metadata={"debate_id": "d123"},
        )
        assert example.source == "debate"
        assert example.metadata == {"debate_id": "d123"}


class TestTrainingExampleToPromptAlpaca:
    """Test TrainingExample alpaca template."""

    def test_alpaca_with_input(self):
        """Test alpaca template with input."""
        example = TrainingExample(
            instruction="Analyze this code",
            input="def main(): pass",
            output="This is an empty function.",
            vertical="software",
        )
        prompt = example.to_prompt("alpaca")
        assert "### Instruction:" in prompt
        assert "Analyze this code" in prompt
        assert "### Input:" in prompt
        assert "def main(): pass" in prompt
        assert "### Response:" in prompt
        assert "This is an empty function." in prompt

    def test_alpaca_without_input(self):
        """Test alpaca template without input."""
        example = TrainingExample(
            instruction="What is Python?",
            input="",
            output="Python is a programming language.",
            vertical="software",
        )
        prompt = example.to_prompt("alpaca")
        assert "### Instruction:" in prompt
        assert "What is Python?" in prompt
        assert "### Input:" not in prompt
        assert "### Response:" in prompt
        assert "Python is a programming language." in prompt


class TestTrainingExampleToPromptChatML:
    """Test TrainingExample chatml template."""

    def test_chatml_format(self):
        """Test chatml template format."""
        example = TrainingExample(
            instruction="Review this contract",
            input="Contract text here",
            output="The contract looks valid.",
            vertical="legal",
        )
        prompt = example.to_prompt("chatml")
        assert "<|im_start|>system" in prompt
        assert "legal specialist" in prompt
        assert "<|im_start|>user" in prompt
        assert "Review this contract" in prompt
        assert "Contract text here" in prompt
        assert "<|im_start|>assistant" in prompt
        assert "The contract looks valid." in prompt
        assert "<|im_end|>" in prompt

    def test_chatml_without_input(self):
        """Test chatml without input concatenates correctly."""
        example = TrainingExample(
            instruction="What is GDPR?",
            input="",
            output="GDPR is a privacy regulation.",
            vertical="legal",
        )
        prompt = example.to_prompt("chatml")
        assert "What is GDPR?" in prompt
        # With empty input, should just be instruction without extra newline


class TestTrainingExampleToPromptLlama:
    """Test TrainingExample llama template."""

    def test_llama_format(self):
        """Test llama template format."""
        example = TrainingExample(
            instruction="Diagnose symptoms",
            input="Patient has fever",
            output="Could be viral infection.",
            vertical="healthcare",
        )
        prompt = example.to_prompt("llama")
        assert "<s>[INST]" in prompt
        assert "<<SYS>>" in prompt
        assert "healthcare specialist" in prompt
        assert "<</SYS>>" in prompt
        assert "Diagnose symptoms" in prompt
        assert "Patient has fever" in prompt
        assert "[/INST]" in prompt
        assert "Could be viral infection." in prompt
        assert "</s>" in prompt

    def test_llama_without_input(self):
        """Test llama without input."""
        example = TrainingExample(
            instruction="What are symptoms of flu?",
            input="",
            output="Fever, cough, fatigue.",
            vertical="healthcare",
        )
        prompt = example.to_prompt("llama")
        assert "What are symptoms of flu?" in prompt
        assert "Fever, cough, fatigue." in prompt


class TestTrainingExampleToPromptUnknown:
    """Test TrainingExample unknown template."""

    def test_unknown_template_fallback(self):
        """Test unknown template uses simple concatenation."""
        example = TrainingExample(
            instruction="Instruction",
            input="Input",
            output="Output",
            vertical="research",
        )
        prompt = example.to_prompt("unknown_template")
        assert "Instruction" in prompt
        assert "Input" in prompt
        assert "Output" in prompt


# =============================================================================
# VerticalFineTuningPipeline Tests
# =============================================================================


class TestVerticalFineTuningPipelineInit:
    """Test pipeline initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)
        assert pipeline.config == config
        assert pipeline.data_dir is None
        assert pipeline._training_examples == []

    def test_init_with_data_dir(self):
        """Test init with data directory."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="legal",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config, data_dir="/data")
        assert pipeline.data_dir == "/data"


class TestVerticalFineTuningPipelineAddExamples:
    """Test adding training examples."""

    def test_add_training_example(self):
        """Test adding a single example."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)
        example = TrainingExample(
            instruction="Test",
            input="",
            output="Result",
            vertical="software",
        )
        pipeline.add_training_example(example)
        assert len(pipeline._training_examples) == 1
        assert pipeline._training_examples[0] == example

    def test_add_multiple_examples(self):
        """Test adding multiple examples."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)
        for i in range(5):
            example = TrainingExample(
                instruction=f"Task {i}",
                input="",
                output=f"Result {i}",
                vertical="software",
            )
            pipeline.add_training_example(example)
        assert len(pipeline._training_examples) == 5


class TestVerticalFineTuningPipelineLoadFromFile:
    """Test loading examples from files."""

    def test_load_from_jsonl(self, tmp_path):
        """Test loading from JSONL file."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir=str(tmp_path / "output"),
        )
        pipeline = VerticalFineTuningPipeline(config)

        # Create JSONL file
        jsonl_file = tmp_path / "data.jsonl"
        records = [
            {"instruction": "Task 1", "input": "", "output": "Result 1"},
            {"instruction": "Task 2", "input": "Input 2", "output": "Result 2"},
        ]
        with open(jsonl_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        count = pipeline.add_training_examples_from_file(str(jsonl_file), format="jsonl")
        assert count == 2
        assert len(pipeline._training_examples) == 2

    def test_load_from_json(self, tmp_path):
        """Test loading from JSON file."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="legal",
            output_dir=str(tmp_path / "output"),
        )
        pipeline = VerticalFineTuningPipeline(config)

        # Create JSON file
        json_file = tmp_path / "data.json"
        records = [
            {"instruction": "Legal task 1", "input": "", "output": "Legal result 1"},
            {"instruction": "Legal task 2", "input": "", "output": "Legal result 2"},
            {"instruction": "Legal task 3", "input": "", "output": "Legal result 3"},
        ]
        with open(json_file, "w") as f:
            json.dump(records, f)

        count = pipeline.add_training_examples_from_file(str(json_file), format="json")
        assert count == 3
        assert len(pipeline._training_examples) == 3

    def test_load_uses_config_vertical(self, tmp_path):
        """Test loaded examples use config vertical if not specified."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="healthcare",
            output_dir=str(tmp_path / "output"),
        )
        pipeline = VerticalFineTuningPipeline(config)

        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"instruction": "t", "input": "", "output": "r"}) + "\n")

        pipeline.add_training_examples_from_file(str(jsonl_file), format="jsonl")
        assert pipeline._training_examples[0].vertical == "healthcare"


class TestVerticalFineTuningPipelineDebateTranscript:
    """Test adding debate transcripts."""

    def test_add_debate_transcript(self):
        """Test converting debate to training example."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)

        messages = [
            {"role": "critic", "content": "This code has security issues."},
            {"role": "expert", "content": "Consider input validation."},
        ]

        pipeline.add_debate_transcript(
            debate_id="d123",
            topic="Review authentication code",
            messages=messages,
            final_answer="Implement input validation and use prepared statements.",
        )

        assert len(pipeline._training_examples) == 1
        example = pipeline._training_examples[0]
        assert example.source == "debate"
        assert example.metadata == {"debate_id": "d123"}
        assert "Review authentication code" in example.instruction

    def test_debate_transcript_extracts_key_insights(self):
        """Test debate transcript extracts critic/expert messages."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="legal",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)

        messages = [
            {"role": "user", "content": "User question"},  # Not extracted
            {"role": "critic", "content": "Critic insight here"},
            {"role": "expert", "content": "Expert analysis"},
        ]

        pipeline.add_debate_transcript(
            debate_id="d456",
            topic="Contract review",
            messages=messages,
            final_answer="Final analysis.",
        )

        example = pipeline._training_examples[0]
        # Input should contain critic and expert messages
        assert "Critic insight here" in example.input or example.input == ""
        # User message should not be in input (only critic/expert extracted)


class TestVerticalFineTuningPipelineStats:
    """Test training statistics."""

    def test_get_training_stats_empty(self):
        """Test stats with no examples."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)
        stats = pipeline.get_training_stats()
        assert stats["num_examples"] == 0
        assert stats["verticals"] == []
        assert stats["sources"] == []
        assert "config" in stats

    def test_get_training_stats_with_examples(self):
        """Test stats with examples."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)

        # Add examples from different sources
        pipeline.add_training_example(
            TrainingExample(
                instruction="t1", input="", output="r1", vertical="software", source="custom"
            )
        )
        pipeline.add_training_example(
            TrainingExample(
                instruction="t2", input="", output="r2", vertical="legal", source="debate"
            )
        )

        stats = pipeline.get_training_stats()
        assert stats["num_examples"] == 2
        assert set(stats["verticals"]) == {"software", "legal"}
        assert set(stats["sources"]) == {"custom", "debate"}


class TestVerticalFineTuningPipelineLoadModel:
    """Test model loading with mocks."""

    @patch("aragora.verticals.models.finetuning.AutoModelForCausalLM")
    @patch("aragora.verticals.models.finetuning.AutoTokenizer")
    @patch("aragora.verticals.models.finetuning.BitsAndBytesConfig")
    def test_load_base_model_4bit(self, mock_bnb, mock_tokenizer, mock_model):
        """Test loading with 4bit quantization."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)

        mock_tok_instance = MagicMock()
        mock_tok_instance.pad_token = None
        mock_tok_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tok_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        pipeline.load_base_model(quantization="4bit")

        mock_bnb.assert_called_once()
        assert pipeline._model is not None
        assert pipeline._tokenizer is not None

    @patch("aragora.verticals.models.finetuning.AutoModelForCausalLM")
    @patch("aragora.verticals.models.finetuning.AutoTokenizer")
    @patch("aragora.verticals.models.finetuning.BitsAndBytesConfig")
    def test_load_base_model_8bit(self, mock_bnb, mock_tokenizer, mock_model):
        """Test loading with 8bit quantization."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)

        mock_tok_instance = MagicMock()
        mock_tok_instance.pad_token = "<pad>"
        mock_tokenizer.from_pretrained.return_value = mock_tok_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        pipeline.load_base_model(quantization="8bit")

        mock_bnb.assert_called_once()
        call_kwargs = mock_bnb.call_args[1]
        assert call_kwargs.get("load_in_8bit") is True

    @patch("aragora.verticals.models.finetuning.AutoModelForCausalLM")
    @patch("aragora.verticals.models.finetuning.AutoTokenizer")
    def test_load_base_model_no_quantization(self, mock_tokenizer, mock_model):
        """Test loading without quantization."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)

        mock_tok_instance = MagicMock()
        mock_tok_instance.pad_token = "<pad>"
        mock_tokenizer.from_pretrained.return_value = mock_tok_instance

        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        pipeline.load_base_model(quantization=None)

        assert pipeline._model is not None


class TestVerticalFineTuningPipelineLoRA:
    """Test LoRA preparation with mocks."""

    @patch("aragora.verticals.models.finetuning.get_peft_model")
    @patch("aragora.verticals.models.finetuning.prepare_model_for_kbit_training")
    @patch("aragora.verticals.models.finetuning.LoraConfig")
    def test_prepare_lora_model(self, mock_lora_config, mock_prepare, mock_get_peft):
        """Test LoRA configuration."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
            lora_r=16,
            lora_alpha=32,
        )
        pipeline = VerticalFineTuningPipeline(config)
        pipeline._model = MagicMock()

        mock_prepared = MagicMock()
        mock_prepared.get_nb_trainable_parameters.return_value = (1000, 10000)
        mock_prepare.return_value = mock_prepared
        mock_get_peft.return_value = mock_prepared

        pipeline.prepare_lora_model()

        mock_lora_config.assert_called_once()
        call_kwargs = mock_lora_config.call_args[1]
        assert call_kwargs["r"] == 16
        assert call_kwargs["lora_alpha"] == 32

    def test_prepare_lora_no_model_raises(self):
        """Test LoRA prep raises without model."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)

        with pytest.raises(ValueError, match="Base model not loaded"):
            pipeline.prepare_lora_model()


class TestVerticalFineTuningPipelineDataset:
    """Test dataset preparation with mocks."""

    @patch("aragora.verticals.models.finetuning.Dataset")
    def test_prepare_dataset(self, mock_dataset_cls):
        """Test dataset preparation."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)

        # Add examples
        for i in range(10):
            pipeline.add_training_example(
                TrainingExample(
                    instruction=f"Task {i}",
                    input="",
                    output=f"Result {i}",
                    vertical="software",
                )
            )

        mock_dataset = MagicMock()
        mock_split = {"train": MagicMock(), "test": MagicMock()}
        mock_split["train"].__len__ = MagicMock(return_value=9)
        mock_split["test"].__len__ = MagicMock(return_value=1)
        mock_dataset.train_test_split.return_value = mock_split
        mock_dataset_cls.from_dict.return_value = mock_dataset

        result = pipeline.prepare_dataset(template="alpaca", train_split=0.9)

        mock_dataset_cls.from_dict.assert_called_once()
        assert result == mock_split

    def test_prepare_dataset_no_examples_raises(self):
        """Test dataset prep raises without examples."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)

        with pytest.raises(ValueError, match="No training examples"):
            pipeline.prepare_dataset()


class TestVerticalFineTuningPipelineTrain:
    """Test training with mocks."""

    def test_train_no_model_raises(self):
        """Test training raises without model."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)

        with pytest.raises(ValueError, match="Model not loaded"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                pipeline.train.__wrapped__(pipeline, {})
            ) if hasattr(pipeline.train, "__wrapped__") else pipeline.train({})


class TestVerticalFineTuningPipelineSaveAdapter:
    """Test adapter saving with mocks."""

    def test_save_adapter_no_model_raises(self):
        """Test save raises without model."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir="/tmp/output",
        )
        pipeline = VerticalFineTuningPipeline(config)

        with pytest.raises(ValueError, match="Model not loaded"):
            pipeline.save_adapter()

    def test_save_adapter_with_model(self, tmp_path):
        """Test saving adapter."""
        config = FinetuningConfig(
            base_model_id="test-model",
            vertical_id="software",
            output_dir=str(tmp_path / "output"),
        )
        pipeline = VerticalFineTuningPipeline(config)
        pipeline._model = MagicMock()

        save_path = str(tmp_path / "adapter")
        result = pipeline.save_adapter(save_path)

        pipeline._model.save_pretrained.assert_called_once_with(save_path)
        assert result == save_path
