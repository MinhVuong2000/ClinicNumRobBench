"""
Hugging Face Transformers Integration for Large Language Models

This module provides a comprehensive interface for loading and using open-source
large language models from Hugging Face, including QwQ-32B, DeepSeek-R1,
and other state-of-the-art models.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import platform
import sys
import subprocess

# Check if running on macOS before importing mlx-lm
if platform.system() == "Darwin":
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load as mlx_load, generate as mlx_generate, batch_generate as mlx_batch_generate
        from mlx_lm.sample_utils import make_sampler
        MLX_AVAILABLE = True
    except ImportError as e:
        print(f"MLX not available on macOS: {e}")
        print("Installing mlx-lm via poetry...")
        try:
            subprocess.check_call(["poetry", "add", "mlx-lm"])
            # Try importing again after installation
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm import load as mlx_load, generate as mlx_generate, batch_generate as mlx_batch_generate
            from mlx_lm.sample_utils import make_sampler
            MLX_AVAILABLE = True
            print("MLX-LM successfully installed via poetry and imported.")
        except (subprocess.CalledProcessError, ImportError) as install_error:
            print(f"Failed to install or import MLX-LM via poetry: {install_error}")
            MLX_AVAILABLE = False
            # Create dummy imports to prevent import errors
            mx = None
            nn = None
            mlx_load = None
            mlx_generate = None
            mlx_batch_generate = None
            make_sampler = None
else:
    print("MLX is only supported on macOS. Skipping MLX imports.")
    MLX_AVAILABLE = False
    # Create dummy imports to prevent import errors
    mx = None
    nn = None
    mlx_load = None
    mlx_generate = None
    mlx_batch_generate = None
    make_sampler = None

from llms.base_language_model import BaseLanguageModel


logger = logging.getLogger(__name__)


class MLXCausalLLM(BaseLanguageModel):
    """
    A comprehensive interface for loading and using Hugging Face causal language models from mlx-lm.

    Supports various model architectures including:
    - Qwen
    - DeepSeek
    - Mistral models
    - Phi models
    - And other mlx-based causal language models

    Features:
    - Automatic device management (CPU/GPU)
    - Quantization support (4-bit, 8-bit)
    - Memory-efficient loading
    - Batch processing
    - Custom generation parameters

    This requires macOS 15.0 or higher to work
    check https://github.com/ml-explore/mlx-lm/tree/main#large-models for more details
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "fp16",
        trust_remote_code: bool = True,
    ):
        """
        Initialize the Hugging Face causal model using mlx-lm.

        Args:
            model_name: Hugging Face model identifier (e.g., "Qwen/QwQ-32B")
            dtype: Data type for the model
            trust_remote_code: Whether to trust remote code in model files
        """
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            dtype=dtype,
        )

        # Model components
        self.tokenizer_config = {"dtype": self.dtype, "trust_remote_code": True if trust_remote_code else None}
        self.tokenizer = None
        self.model = None

        # Load model and tokenizer
        self.load_model()

    def load_model(self, **kwargs):
        """Load the model and tokenizer from Hugging Face."""
        try:
            logger.info("Loading model: %s with mlx (Optimized for macOS 15.0 or higher)", self.model_name)

            # Load tokenizer
            self.model, self.tokenizer = mlx_load(self.model_name, tokenizer_config=self.tokenizer_config)

            logger.info("Successfully loaded model: %s", self.model_name)

        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_name, str(e))
            raise

    def _tokenize(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        Tokenize input text or list of texts.

        Args:
            texts: Input text or list of texts

        Returns:
            List of tokens or list of list of tokens
        """

        if isinstance(texts, str):
            return self.tokenizer.encode(texts, add_special_tokens=self.tokenizer.bos_token is None or not texts.startswith(self.tokenizer.bos_token))
        else:
            return [
                self.tokenizer.encode(t, add_special_tokens=self.tokenizer.bos_token is None or not t.startswith(self.tokenizer.bos_token))
                for t in texts
            ]

    def prepare_model_prompt(self, messages: List[Dict[str, str]], return_tokens: bool = False, **kwargs) ->Tuple[str|List[int], int]:
        """
        Prepare model prompt for Hugging Face models.

        Args:
            messages: List of messages
            return_tokens: Whether to return tokens
            **kwargs: Additional arguments

        Returns:
            Tuple with prompt, number of input tokens
        """

        if self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Tokenize input
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self._tokenize(prompt) # list[int]

        # Count input tokens
        num_input_tokens = len(model_inputs)

        if return_tokens:
            return model_inputs, num_input_tokens
        else:
            return prompt, num_input_tokens

    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, draft_model: Optional[nn.Module] = None, return_num_tokens: bool = True, verbose: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Generate response from the model using the base class interface.

        Args:
            messages: Input messages, a list of dictionaries with role and content of system and user
            max_tokens: Maximum number of tokens to generate. Default is 512.
            temperature: Sampling temperature (higher = more random). Default is 0.7.
            top_p: Nucleus sampling parameter. Default is 0.9.
            top_k: Top-k sampling parameter. Default is 50.
            draft_model: Draft model for speculative decoding. Default is None.
            return_num_tokens: Whether to return token counts. Default is True.
            verbose: Whether to print verbose output. Default is False.
            **kwargs: Additional arguments, following the arguments of mlx_generate

        Returns:
            Dictionary with generated text, input tokens, output tokens, and generation time
        """

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        prompt, num_input_tokens = self.prepare_model_prompt(messages)

        # Set generation parameters
        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)

        # Generate
        generate_start_time = time.time()
        generated_text = mlx_generate(self.model, self.tokenizer, prompt=prompt, sampler=sampler, max_tokens=max_tokens, draft_model=draft_model, verbose=verbose, **kwargs)

        generation_time = time.time() - generate_start_time

        # Count output tokens
        if return_num_tokens:
            num_output_tokens = len(self.tokenizer.encode(generated_text, add_special_tokens = self.tokenizer.eos_token is None or not generated_text.endswith(self.tokenizer.eos_token)))
        else:
            num_output_tokens = None

        # Clear cache for efficient memory usage
        # self.clear_cache()

        return {
            "generated_text": generated_text,
            "num_input_tokens": num_input_tokens if return_num_tokens else None,
            "num_output_tokens": num_output_tokens if return_num_tokens else None,
            "generation_time": generation_time,
        }

    def batch_generate_response(self, messages: List[List[Dict[str, str]]], max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50, return_num_tokens: bool = True, verbose: bool = False,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 2048,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of messages.

        Args:
            messages: List of messages
            max_tokens: Maximum number of tokens to generate. Default is 512.
            temperature: Sampling temperature (higher = more random). Default is 0.7.
            top_p: Nucleus sampling parameter. Default is 0.9.
            top_k: Top-k sampling parameter. Default is 50.
            return_num_tokens: Whether to return token counts. Default is True.
            verbose: Whether to print verbose output. Default is False.
            completion_batch_size: Batch size for completion. Default is 32.
            prefill_batch_size: Batch size for prefill. Default is 8.
            prefill_step_size: Step size for prefill. Default is 2048.
        Returns:
            List of dictionaries with generated text (list of str), list of num_input_tokens (list of int), list of num_output_tokens (list of int), and generation time(int)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Build chat prompts per sample
        prompts: List[List[int]] = []
        num_input_tokens: List[int] = []
        for chat in messages:
            prompt, n_in = self.prepare_model_prompt(chat, return_tokens=True)
            prompts.append(prompt)
            if return_num_tokens:
                num_input_tokens.append(n_in)

        sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)

        generate_start_time = time.time()
        generated_responses = mlx_batch_generate(
            self.model,
            self.tokenizer,
            prompts=prompts,
            sampler=sampler,
            max_tokens=max_tokens,
            verbose=verbose,
            completion_batch_size=completion_batch_size,
            prefill_batch_size=prefill_batch_size,
            prefill_step_size=prefill_step_size,
        )

        generated_texts = generated_responses.texts
        num_output_tokens = [len(self._tokenize(text)) for text in generated_texts] if return_num_tokens else None

        generation_time = time.time() - generate_start_time

        # Clear cache for efficient memory usage
        # self.clear_cache()

        return {
            "generated_text": generated_texts,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "generation_time": [generation_time]*len(messages),
        }

    @classmethod
    def supported_models(self) -> List[str]:
        """
        Get the list of supported models.
        """
        return ["lmstudio-community/DeepSeek-R1-0528-Qwen3-8B-MLX-8bit"]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "Model not loaded"}

        info = {
            "model_name": self.model_name,
            "num_parameters": sum(p.numel() for p in self.model.parameters()) if self.model is not None else None,
            "vocab_size": self.tokenizer.vocab_size,
        }

        return info

    def clear_cache(self) -> None:
        """Clear GPU cache to free memory."""
        mx.clear_cache()
