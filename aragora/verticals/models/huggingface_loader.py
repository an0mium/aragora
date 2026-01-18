"""
HuggingFace Specialist Model Loader.

Loads and manages domain-specific models from HuggingFace,
with support for quantization and LoRA fine-tuned adapters.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Error loading a model."""

    pass


# Recommended models per vertical based on domain expertise
RECOMMENDED_MODELS: Dict[str, Dict[str, str]] = {
    "software": {
        "primary": "codellama/CodeLlama-34b-Instruct-hf",
        "embedding": "microsoft/codebert-base",
        "small": "codellama/CodeLlama-7b-Instruct-hf",
    },
    "legal": {
        "primary": "nlpaueb/legal-bert-base-uncased",
        "embedding": "lexlms/legal-roberta-base",
        "small": "nlpaueb/legal-bert-small-uncased",
    },
    "healthcare": {
        "primary": "medicalai/ClinicalBERT",
        "embedding": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "small": "dmis-lab/biobert-base-cased-v1.1",
    },
    "accounting": {
        "primary": "ProsusAI/finbert",
        "embedding": "yiyanghkust/finbert-tone",
        "small": "ProsusAI/finbert",
    },
    "research": {
        "primary": "allenai/scibert_scivocab_uncased",
        "embedding": "allenai/specter",
        "small": "sentence-transformers/all-MiniLM-L6-v2",
    },
}


@dataclass
class SpecialistModel:
    """
    Container for a loaded specialist model.

    Manages the model, tokenizer, and optional adapter.
    """

    model_id: str
    vertical_id: str
    model: Any = None  # The actual model
    tokenizer: Any = None
    adapter_id: Optional[str] = None
    quantization: Optional[str] = None
    device: str = "cuda"
    loaded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> str:
        """
        Generate text from the model.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self.loaded or self.model is None or self.tokenizer is None:
            raise ModelLoadError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def embed(self, texts: Union[str, List[str]]) -> Any:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            Embeddings tensor
        """
        if not self.loaded or self.model is None or self.tokenizer is None:
            raise ModelLoadError("Model not loaded")

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Use CLS token embedding or mean pooling
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            # Mean pooling over token embeddings
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            return (
                (token_embeddings * input_mask_expanded).sum(1)
                / input_mask_expanded.sum(1).clamp(min=1e-9)
            )

    def unload(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.loaded = False

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info(f"Unloaded model: {self.model_id}")


class HuggingFaceSpecialistLoader:
    """
    Loader for HuggingFace specialist models.

    Supports:
    - Loading models from HuggingFace Hub
    - Quantization (8bit/4bit via bitsandbytes)
    - LoRA adapter loading (via PEFT)
    - Memory management and model caching
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        default_device: str = "auto",
        max_cached_models: int = 2,
    ):
        """
        Initialize the loader.

        Args:
            cache_dir: Directory for caching models
            default_device: Default device ("cuda", "cpu", "auto")
            max_cached_models: Maximum models to keep in cache
        """
        self._cache_dir = cache_dir
        self._default_device = default_device
        self._max_cached_models = max_cached_models
        self._loaded_models: Dict[str, SpecialistModel] = {}
        self._torch_available = self._check_torch()
        self._transformers_available = self._check_transformers()

    def _check_torch(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch

            return True
        except ImportError:
            logger.warning("PyTorch not available - model loading disabled")
            return False

    def _check_transformers(self) -> bool:
        """Check if transformers is available."""
        try:
            import transformers

            return True
        except ImportError:
            logger.warning("transformers not available - model loading disabled")
            return False

    def _get_device(self) -> str:
        """Determine the best device to use."""
        if self._default_device != "auto":
            return self._default_device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def _get_quantization_config(
        self,
        quantization: Optional[str],
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Get quantization configuration.

        Args:
            quantization: "8bit" or "4bit" or None

        Returns:
            Tuple of (BitsAndBytesConfig, load_kwargs)
        """
        if not quantization:
            return None, {}

        try:
            from transformers import BitsAndBytesConfig
            import torch

            if quantization == "8bit":
                return (
                    BitsAndBytesConfig(
                        load_in_8bit=True,
                    ),
                    {"device_map": "auto"},
                )
            elif quantization == "4bit":
                return (
                    BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    ),
                    {"device_map": "auto"},
                )
            else:
                logger.warning(f"Unknown quantization: {quantization}")
                return None, {}

        except ImportError:
            logger.warning("bitsandbytes not available for quantization")
            return None, {}

    def load_model(
        self,
        model_id: str,
        vertical_id: str,
        model_type: str = "causal_lm",
        quantization: Optional[str] = None,
        adapter_id: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> SpecialistModel:
        """
        Load a specialist model from HuggingFace.

        Args:
            model_id: HuggingFace model ID
            vertical_id: Vertical this model is for
            model_type: "causal_lm", "encoder", "seq2seq"
            quantization: "8bit" or "4bit" for quantization
            adapter_id: Optional LoRA adapter to load
            trust_remote_code: Trust remote code in model
            **kwargs: Additional loading arguments

        Returns:
            Loaded SpecialistModel

        Raises:
            ModelLoadError: If loading fails
        """
        if not self._torch_available or not self._transformers_available:
            raise ModelLoadError("PyTorch and transformers required for model loading")

        # Check cache
        cache_key = f"{model_id}:{adapter_id or 'none'}:{quantization or 'none'}"
        if cache_key in self._loaded_models:
            logger.info(f"Using cached model: {model_id}")
            return self._loaded_models[cache_key]

        # Evict old models if needed
        while len(self._loaded_models) >= self._max_cached_models:
            oldest_key = next(iter(self._loaded_models))
            self._loaded_models[oldest_key].unload()
            del self._loaded_models[oldest_key]
            logger.info(f"Evicted model from cache: {oldest_key}")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

            device = self._get_device()
            quant_config, load_kwargs = self._get_quantization_config(quantization)

            logger.info(f"Loading model: {model_id} on {device}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=self._cache_dir,
                trust_remote_code=trust_remote_code,
            )

            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model based on type
            model_kwargs = {
                "cache_dir": self._cache_dir,
                "trust_remote_code": trust_remote_code,
                **load_kwargs,
                **kwargs,
            }

            if quant_config:
                model_kwargs["quantization_config"] = quant_config

            if model_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            elif model_type == "encoder":
                model = AutoModel.from_pretrained(model_id, **model_kwargs)
            else:
                # Default to auto model
                model = AutoModel.from_pretrained(model_id, **model_kwargs)

            # Load LoRA adapter if specified
            if adapter_id:
                model = self._load_adapter(model, adapter_id)

            # Move to device if not using device_map
            if "device_map" not in load_kwargs and device != "cpu":
                model = model.to(device)

            # Set evaluation mode
            model.eval()

            specialist_model = SpecialistModel(
                model_id=model_id,
                vertical_id=vertical_id,
                model=model,
                tokenizer=tokenizer,
                adapter_id=adapter_id,
                quantization=quantization,
                device=device,
                loaded=True,
                metadata={
                    "model_type": model_type,
                    "trust_remote_code": trust_remote_code,
                },
            )

            self._loaded_models[cache_key] = specialist_model
            logger.info(f"Successfully loaded model: {model_id}")

            return specialist_model

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise ModelLoadError(f"Failed to load model {model_id}: {e}") from e

    def _load_adapter(self, model: Any, adapter_id: str) -> Any:
        """
        Load a LoRA adapter onto a model.

        Args:
            model: Base model
            adapter_id: Adapter path or HuggingFace ID

        Returns:
            Model with adapter loaded
        """
        try:
            from peft import PeftModel

            logger.info(f"Loading adapter: {adapter_id}")
            model = PeftModel.from_pretrained(
                model,
                adapter_id,
                cache_dir=self._cache_dir,
            )
            logger.info(f"Loaded adapter: {adapter_id}")
            return model

        except ImportError:
            logger.warning("peft not available for adapter loading")
            return model
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_id}: {e}")
            return model

    def load_embedding_model(
        self,
        model_id: str,
        vertical_id: str,
        **kwargs: Any,
    ) -> SpecialistModel:
        """
        Load a model for embeddings.

        Args:
            model_id: HuggingFace model ID
            vertical_id: Vertical this model is for
            **kwargs: Additional loading arguments

        Returns:
            Loaded SpecialistModel for embeddings
        """
        return self.load_model(
            model_id=model_id,
            vertical_id=vertical_id,
            model_type="encoder",
            **kwargs,
        )

    def get_recommended_model(
        self,
        vertical_id: str,
        model_type: str = "primary",
    ) -> Optional[str]:
        """
        Get the recommended model ID for a vertical.

        Args:
            vertical_id: Vertical identifier
            model_type: "primary", "embedding", or "small"

        Returns:
            Model ID or None if not found
        """
        vertical_models = RECOMMENDED_MODELS.get(vertical_id, {})
        return vertical_models.get(model_type)

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """
        List currently loaded models.

        Returns:
            List of model information dictionaries
        """
        return [
            {
                "model_id": m.model_id,
                "vertical_id": m.vertical_id,
                "adapter_id": m.adapter_id,
                "quantization": m.quantization,
                "device": m.device,
                "loaded": m.loaded,
            }
            for m in self._loaded_models.values()
        ]

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a specific model.

        Args:
            model_id: Model ID to unload

        Returns:
            True if model was unloaded
        """
        for key, model in list(self._loaded_models.items()):
            if model.model_id == model_id:
                model.unload()
                del self._loaded_models[key]
                return True
        return False

    def unload_all(self) -> None:
        """Unload all models."""
        for model in self._loaded_models.values():
            model.unload()
        self._loaded_models.clear()
        logger.info("Unloaded all models")

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a HuggingFace model.

        Args:
            model_id: Model ID to query

        Returns:
            Model information or None
        """
        try:
            from huggingface_hub import model_info

            info = model_info(model_id)
            return {
                "model_id": model_id,
                "pipeline_tag": info.pipeline_tag,
                "tags": info.tags,
                "downloads": info.downloads,
                "library_name": info.library_name,
            }
        except Exception as e:
            logger.warning(f"Could not get model info for {model_id}: {e}")
            return None
