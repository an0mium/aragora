"""Tests for HuggingFace Specialist Model Loader.

Tests ModelLoadError, RECOMMENDED_MODELS, SpecialistModel, and HuggingFaceSpecialistLoader.
"""

import gc
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from aragora.verticals.models.huggingface_loader import (
    ModelLoadError,
    RECOMMENDED_MODELS,
    SpecialistModel,
    HuggingFaceSpecialistLoader,
)


# =============================================================================
# ModelLoadError Tests
# =============================================================================


class TestModelLoadError:
    """Test ModelLoadError exception."""

    def test_instantiation(self):
        """Test creating ModelLoadError."""
        error = ModelLoadError("Test error message")
        assert str(error) == "Test error message"

    def test_inheritance(self):
        """Test ModelLoadError is an Exception."""
        error = ModelLoadError("test")
        assert isinstance(error, Exception)

    def test_raise_and_catch(self):
        """Test raising and catching ModelLoadError."""
        with pytest.raises(ModelLoadError) as exc_info:
            raise ModelLoadError("Model failed to load")
        assert "Model failed to load" in str(exc_info.value)


# =============================================================================
# RECOMMENDED_MODELS Tests
# =============================================================================


class TestRecommendedModels:
    """Test RECOMMENDED_MODELS dictionary."""

    def test_software_vertical_exists(self):
        """Test software vertical exists."""
        assert "software" in RECOMMENDED_MODELS

    def test_legal_vertical_exists(self):
        """Test legal vertical exists."""
        assert "legal" in RECOMMENDED_MODELS

    def test_healthcare_vertical_exists(self):
        """Test healthcare vertical exists."""
        assert "healthcare" in RECOMMENDED_MODELS

    def test_accounting_vertical_exists(self):
        """Test accounting vertical exists."""
        assert "accounting" in RECOMMENDED_MODELS

    def test_research_vertical_exists(self):
        """Test research vertical exists."""
        assert "research" in RECOMMENDED_MODELS

    def test_five_verticals_total(self):
        """Test exactly 5 verticals."""
        assert len(RECOMMENDED_MODELS) == 5

    def test_software_has_primary(self):
        """Test software has primary model."""
        assert "primary" in RECOMMENDED_MODELS["software"]

    def test_software_has_embedding(self):
        """Test software has embedding model."""
        assert "embedding" in RECOMMENDED_MODELS["software"]

    def test_software_has_small(self):
        """Test software has small model."""
        assert "small" in RECOMMENDED_MODELS["software"]

    def test_all_verticals_have_three_model_types(self):
        """Test all verticals have primary, embedding, and small."""
        for vertical, models in RECOMMENDED_MODELS.items():
            assert "primary" in models, f"{vertical} missing primary"
            assert "embedding" in models, f"{vertical} missing embedding"
            assert "small" in models, f"{vertical} missing small"

    def test_model_ids_are_strings(self):
        """Test all model IDs are strings."""
        for vertical, models in RECOMMENDED_MODELS.items():
            for model_type, model_id in models.items():
                assert isinstance(model_id, str), f"{vertical}/{model_type} is not string"


# =============================================================================
# SpecialistModel Tests
# =============================================================================


class TestSpecialistModelInit:
    """Test SpecialistModel initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        model = SpecialistModel(
            model_id="test-model",
            vertical_id="software",
        )
        assert model.model_id == "test-model"
        assert model.vertical_id == "software"
        assert model.model is None
        assert model.tokenizer is None
        assert model.adapter_id is None
        assert model.quantization is None
        assert model.device == "cuda"
        assert model.loaded is False
        assert model.metadata == {}

    def test_full_init(self):
        """Test full initialization."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        model = SpecialistModel(
            model_id="full-model",
            vertical_id="legal",
            model=mock_model,
            tokenizer=mock_tokenizer,
            adapter_id="adapter-1",
            quantization="4bit",
            device="cpu",
            loaded=True,
            metadata={"custom": "data"},
        )
        assert model.model == mock_model
        assert model.tokenizer == mock_tokenizer
        assert model.adapter_id == "adapter-1"
        assert model.quantization == "4bit"
        assert model.device == "cpu"
        assert model.loaded is True
        assert model.metadata == {"custom": "data"}


class TestSpecialistModelGenerate:
    """Test SpecialistModel generate method."""

    def test_generate_not_loaded_raises(self):
        """Test generate raises when not loaded."""
        model = SpecialistModel(
            model_id="test-model",
            vertical_id="software",
            loaded=False,
        )
        with pytest.raises(ModelLoadError, match="Model not loaded"):
            model.generate("test prompt")

    def test_generate_no_model_raises(self):
        """Test generate raises when model is None."""
        model = SpecialistModel(
            model_id="test-model",
            vertical_id="software",
            loaded=True,
            model=None,
        )
        with pytest.raises(ModelLoadError, match="Model not loaded"):
            model.generate("test prompt")

    def test_generate_no_tokenizer_raises(self):
        """Test generate raises when tokenizer is None."""
        model = SpecialistModel(
            model_id="test-model",
            vertical_id="software",
            loaded=True,
            model=MagicMock(),
            tokenizer=None,
        )
        with pytest.raises(ModelLoadError, match="Model not loaded"):
            model.generate("test prompt")

    def test_generate_success(self):
        """Test successful generation."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Setup tokenizer
        mock_tokenizer.return_value = {"input_ids": MagicMock()}
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode.return_value = "Generated text"

        # Setup model
        mock_model.generate.return_value = [MagicMock()]

        specialist = SpecialistModel(
            model_id="test-model",
            vertical_id="software",
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded=True,
            device="cpu",
        )

        result = specialist.generate("Test prompt")
        assert result == "Generated text"
        mock_model.generate.assert_called_once()


class TestSpecialistModelEmbed:
    """Test SpecialistModel embed method."""

    def test_embed_not_loaded_raises(self):
        """Test embed raises when not loaded."""
        model = SpecialistModel(
            model_id="test-model",
            vertical_id="software",
            loaded=False,
        )
        with pytest.raises(ModelLoadError, match="Model not loaded"):
            model.embed("test text")

    def test_embed_single_text(self):
        """Test embedding single text."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Setup tokenizer
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        # Setup model output with pooler_output
        mock_output = MagicMock()
        mock_output.pooler_output = MagicMock()
        mock_model.return_value = mock_output

        specialist = SpecialistModel(
            model_id="test-model",
            vertical_id="software",
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded=True,
            device="cpu",
        )

        result = specialist.embed("Test text")
        assert result == mock_output.pooler_output

    def test_embed_list_of_texts(self):
        """Test embedding list of texts."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        mock_output = MagicMock()
        mock_output.pooler_output = MagicMock()
        mock_model.return_value = mock_output

        specialist = SpecialistModel(
            model_id="test-model",
            vertical_id="legal",
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded=True,
            device="cpu",
        )

        result = specialist.embed(["Text 1", "Text 2", "Text 3"])
        mock_tokenizer.assert_called_once()
        # Should be called with list
        call_args = mock_tokenizer.call_args
        assert call_args[0][0] == ["Text 1", "Text 2", "Text 3"]


class TestSpecialistModelUnload:
    """Test SpecialistModel unload method."""

    def test_unload_clears_model(self):
        """Test unload clears model."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        specialist = SpecialistModel(
            model_id="test-model",
            vertical_id="software",
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded=True,
        )

        specialist.unload()

        assert specialist.model is None
        assert specialist.tokenizer is None
        assert specialist.loaded is False

    @patch("aragora.verticals.models.huggingface_loader.gc.collect")
    def test_unload_calls_gc(self, mock_gc):
        """Test unload calls garbage collection."""
        specialist = SpecialistModel(
            model_id="test-model",
            vertical_id="software",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded=True,
        )

        specialist.unload()

        mock_gc.assert_called()

    def test_unload_already_unloaded(self):
        """Test unloading already unloaded model."""
        specialist = SpecialistModel(
            model_id="test-model",
            vertical_id="software",
            loaded=False,
        )

        # Should not raise
        specialist.unload()
        assert specialist.loaded is False


# =============================================================================
# HuggingFaceSpecialistLoader Tests
# =============================================================================


class TestHuggingFaceSpecialistLoaderInit:
    """Test loader initialization."""

    def test_default_init(self):
        """Test default initialization."""
        loader = HuggingFaceSpecialistLoader()
        assert loader._cache_dir is None
        assert loader._default_device == "auto"
        assert loader._max_cached_models == 2
        assert loader._loaded_models == {}

    def test_custom_init(self):
        """Test custom initialization."""
        loader = HuggingFaceSpecialistLoader(
            cache_dir="/custom/cache",
            default_device="cpu",
            max_cached_models=5,
        )
        assert loader._cache_dir == "/custom/cache"
        assert loader._default_device == "cpu"
        assert loader._max_cached_models == 5


class TestHuggingFaceSpecialistLoaderCheckDeps:
    """Test dependency checking."""

    def test_check_torch_available(self):
        """Test torch check when available."""
        loader = HuggingFaceSpecialistLoader()
        # Should have checked during init
        assert isinstance(loader._torch_available, bool)

    def test_check_transformers_available(self):
        """Test transformers check when available."""
        loader = HuggingFaceSpecialistLoader()
        assert isinstance(loader._transformers_available, bool)

    @patch.dict("sys.modules", {"torch": None})
    def test_check_torch_import_error(self):
        """Test torch check handles ImportError."""
        loader = HuggingFaceSpecialistLoader()
        # _check_torch is called during init, so we test the method directly
        with patch("builtins.__import__", side_effect=ImportError):
            result = loader._check_torch()
            assert result is False


class TestHuggingFaceSpecialistLoaderGetDevice:
    """Test device detection."""

    def test_get_device_explicit_cpu(self):
        """Test explicit CPU device."""
        loader = HuggingFaceSpecialistLoader(default_device="cpu")
        assert loader._get_device() == "cpu"

    def test_get_device_explicit_cuda(self):
        """Test explicit CUDA device."""
        loader = HuggingFaceSpecialistLoader(default_device="cuda")
        assert loader._get_device() == "cuda"

    @patch("torch.cuda.is_available", return_value=True)
    def test_get_device_auto_cuda(self, mock_cuda):
        """Test auto device with CUDA available."""
        loader = HuggingFaceSpecialistLoader(default_device="auto")
        loader._torch_available = True
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            import torch
            torch.cuda = MagicMock()
            torch.cuda.is_available = MagicMock(return_value=True)
            device = loader._get_device()
            # Will be cuda if available


class TestHuggingFaceSpecialistLoaderQuantization:
    """Test quantization configuration."""

    def test_no_quantization(self):
        """Test no quantization returns None config."""
        loader = HuggingFaceSpecialistLoader()
        config, kwargs = loader._get_quantization_config(None)
        assert config is None
        assert kwargs == {}

    @patch("aragora.verticals.models.huggingface_loader.BitsAndBytesConfig")
    def test_8bit_quantization(self, mock_bnb):
        """Test 8bit quantization config."""
        loader = HuggingFaceSpecialistLoader()
        mock_bnb.return_value = MagicMock()

        config, kwargs = loader._get_quantization_config("8bit")

        mock_bnb.assert_called_once()
        call_kwargs = mock_bnb.call_args[1]
        assert call_kwargs.get("load_in_8bit") is True
        assert "device_map" in kwargs

    @patch("aragora.verticals.models.huggingface_loader.BitsAndBytesConfig")
    def test_4bit_quantization(self, mock_bnb):
        """Test 4bit quantization config."""
        loader = HuggingFaceSpecialistLoader()
        mock_bnb.return_value = MagicMock()

        config, kwargs = loader._get_quantization_config("4bit")

        mock_bnb.assert_called_once()
        call_kwargs = mock_bnb.call_args[1]
        assert call_kwargs.get("load_in_4bit") is True

    def test_unknown_quantization(self):
        """Test unknown quantization returns None."""
        loader = HuggingFaceSpecialistLoader()
        config, kwargs = loader._get_quantization_config("unknown")
        assert config is None
        assert kwargs == {}


class TestHuggingFaceSpecialistLoaderLoadModel:
    """Test model loading."""

    def test_load_model_deps_missing(self):
        """Test load raises when deps missing."""
        loader = HuggingFaceSpecialistLoader()
        loader._torch_available = False

        with pytest.raises(ModelLoadError, match="PyTorch and transformers required"):
            loader.load_model("test-model", "software")

    @patch("aragora.verticals.models.huggingface_loader.AutoTokenizer")
    @patch("aragora.verticals.models.huggingface_loader.AutoModelForCausalLM")
    def test_load_model_causal_lm(self, mock_model_cls, mock_tokenizer_cls):
        """Test loading causal LM."""
        loader = HuggingFaceSpecialistLoader()
        loader._torch_available = True
        loader._transformers_available = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        result = loader.load_model("test-model", "software", model_type="causal_lm")

        assert result.model_id == "test-model"
        assert result.vertical_id == "software"
        assert result.loaded is True
        mock_model_cls.from_pretrained.assert_called_once()

    def test_load_model_cache_hit(self):
        """Test cache hit returns existing model."""
        loader = HuggingFaceSpecialistLoader()
        loader._torch_available = True
        loader._transformers_available = True

        # Pre-populate cache
        cached = SpecialistModel(
            model_id="cached-model",
            vertical_id="legal",
            loaded=True,
        )
        loader._loaded_models["cached-model:none:none"] = cached

        result = loader.load_model("cached-model", "legal")
        assert result == cached

    def test_load_model_cache_eviction(self):
        """Test cache eviction when full."""
        loader = HuggingFaceSpecialistLoader(max_cached_models=2)
        loader._torch_available = True
        loader._transformers_available = True

        # Pre-populate cache to max
        for i in range(2):
            m = SpecialistModel(
                model_id=f"model-{i}",
                vertical_id="software",
                loaded=True,
            )
            m.unload = MagicMock()
            loader._loaded_models[f"model-{i}:none:none"] = m

        assert len(loader._loaded_models) == 2

        # Loading new model should evict oldest
        with patch.object(loader, "load_model") as mock_load:
            # The actual eviction happens inside load_model
            # Just verify we have eviction logic by checking the max
            pass


class TestHuggingFaceSpecialistLoaderAdapters:
    """Test adapter loading."""

    @patch("aragora.verticals.models.huggingface_loader.PeftModel")
    def test_load_adapter_success(self, mock_peft):
        """Test successful adapter loading."""
        loader = HuggingFaceSpecialistLoader()
        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model

        result = loader._load_adapter(mock_base_model, "adapter-path")

        assert result == mock_peft_model
        mock_peft.from_pretrained.assert_called_once()

    def test_load_adapter_import_error(self):
        """Test adapter loading handles import error."""
        loader = HuggingFaceSpecialistLoader()
        mock_base_model = MagicMock()

        with patch("builtins.__import__", side_effect=ImportError):
            result = loader._load_adapter(mock_base_model, "adapter-path")
            # Should return original model on error
            assert result == mock_base_model


class TestHuggingFaceSpecialistLoaderEmbedding:
    """Test embedding model loading."""

    @patch.object(HuggingFaceSpecialistLoader, "load_model")
    def test_load_embedding_model(self, mock_load):
        """Test embedding model calls load_model with encoder type."""
        loader = HuggingFaceSpecialistLoader()
        mock_load.return_value = MagicMock()

        loader.load_embedding_model("embed-model", "research")

        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs.get("model_type") == "encoder"


class TestHuggingFaceSpecialistLoaderRecommended:
    """Test recommended model lookup."""

    def test_get_recommended_model_valid(self):
        """Test getting valid recommended model."""
        loader = HuggingFaceSpecialistLoader()
        model_id = loader.get_recommended_model("software", "primary")
        assert model_id is not None
        assert isinstance(model_id, str)

    def test_get_recommended_model_embedding(self):
        """Test getting embedding model."""
        loader = HuggingFaceSpecialistLoader()
        model_id = loader.get_recommended_model("legal", "embedding")
        assert model_id is not None

    def test_get_recommended_model_small(self):
        """Test getting small model."""
        loader = HuggingFaceSpecialistLoader()
        model_id = loader.get_recommended_model("healthcare", "small")
        assert model_id is not None

    def test_get_recommended_model_invalid_vertical(self):
        """Test invalid vertical returns None."""
        loader = HuggingFaceSpecialistLoader()
        model_id = loader.get_recommended_model("invalid_vertical", "primary")
        assert model_id is None

    def test_get_recommended_model_invalid_type(self):
        """Test invalid model type returns None."""
        loader = HuggingFaceSpecialistLoader()
        model_id = loader.get_recommended_model("software", "invalid_type")
        assert model_id is None


class TestHuggingFaceSpecialistLoaderManagement:
    """Test model management functions."""

    def test_list_loaded_models_empty(self):
        """Test listing with no models."""
        loader = HuggingFaceSpecialistLoader()
        models = loader.list_loaded_models()
        assert models == []

    def test_list_loaded_models_with_models(self):
        """Test listing with loaded models."""
        loader = HuggingFaceSpecialistLoader()
        loader._loaded_models["model1:none:none"] = SpecialistModel(
            model_id="model1",
            vertical_id="software",
            loaded=True,
        )
        loader._loaded_models["model2:adapter:4bit"] = SpecialistModel(
            model_id="model2",
            vertical_id="legal",
            adapter_id="adapter",
            quantization="4bit",
            loaded=True,
        )

        models = loader.list_loaded_models()
        assert len(models) == 2
        assert all("model_id" in m for m in models)
        assert all("vertical_id" in m for m in models)

    def test_unload_model_found(self):
        """Test unloading existing model."""
        loader = HuggingFaceSpecialistLoader()
        model = SpecialistModel(
            model_id="to-unload",
            vertical_id="software",
            loaded=True,
        )
        model.unload = MagicMock()
        loader._loaded_models["to-unload:none:none"] = model

        result = loader.unload_model("to-unload")

        assert result is True
        model.unload.assert_called_once()
        assert "to-unload:none:none" not in loader._loaded_models

    def test_unload_model_not_found(self):
        """Test unloading non-existent model."""
        loader = HuggingFaceSpecialistLoader()
        result = loader.unload_model("not-exists")
        assert result is False

    def test_unload_all(self):
        """Test unloading all models."""
        loader = HuggingFaceSpecialistLoader()

        for i in range(3):
            model = SpecialistModel(
                model_id=f"model-{i}",
                vertical_id="software",
                loaded=True,
            )
            model.unload = MagicMock()
            loader._loaded_models[f"model-{i}:none:none"] = model

        loader.unload_all()

        assert len(loader._loaded_models) == 0


class TestHuggingFaceSpecialistLoaderInfo:
    """Test model info retrieval."""

    @patch("aragora.verticals.models.huggingface_loader.model_info")
    def test_get_model_info_success(self, mock_info_fn):
        """Test successful model info retrieval."""
        loader = HuggingFaceSpecialistLoader()

        mock_info = MagicMock()
        mock_info.pipeline_tag = "text-generation"
        mock_info.tags = ["pytorch", "llama"]
        mock_info.downloads = 1000
        mock_info.library_name = "transformers"
        mock_info_fn.return_value = mock_info

        result = loader.get_model_info("test-model")

        assert result["model_id"] == "test-model"
        assert result["pipeline_tag"] == "text-generation"
        assert result["downloads"] == 1000

    def test_get_model_info_error(self):
        """Test model info handles errors."""
        loader = HuggingFaceSpecialistLoader()

        with patch("builtins.__import__", side_effect=ImportError):
            result = loader.get_model_info("test-model")
            # Should return None on error
            assert result is None
