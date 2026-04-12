"""
Hugging Face Transformers Integration for Large Language Models

This module provides a comprehensive interface for loading and using open-source
large language models from Hugging Face, including QwQ-32B, DeepSeek-R1,
and other state-of-the-art models.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
import gc

from const import HF_TOKEN
from llms.base_language_model import BaseLanguageModel


logger = logging.getLogger(__name__)


class EosListStoppingCriteria(StoppingCriteria):
  def __init__(self, eos_sequence = [32007]):
      self.eos_sequence = eos_sequence

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
      last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
      return self.eos_sequence in last_ids

  def __len__(self):
      return len(self.eos_sequence)


class HFCasualLLM(BaseLanguageModel):
    """
    A comprehensive interface for loading and using Hugging Face causal language models.

    Supports various model architectures including:
    - Qwen
    - DeepSeek
    - Mistral models
    - Phi models
    - And other transformer-based causal language models

    Features:
    - Automatic device management (CPU/GPU)
    - Quantization support (4-bit, 8-bit)
    - Memory-efficient loading
    - Batch processing
    - Custom generation parameters
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "fp16",
        max_retries: int = 3,
        timeout: int = 30,
        device_map: Optional[Union[str, Dict[str, Any]]] = "auto",
        quantization_config: Optional[BitsAndBytesConfig] = None,
        trust_remote_code: bool = True,
        use_cache: bool = False,
        low_cpu_mem_usage: bool = True,
        **kwargs
    ):
        """
        Initialize the Hugging Face causal model.

        Args:
            model_name: Hugging Face model identifier (e.g., "Qwen/QwQ-32B")
            dtype: Data type for the model
            max_retries: Maximum retries for API calls
            timeout: Timeout for API calls
            device_map: Device mapping strategy ("auto", "cpu", "cuda", or custom dict)
            quantization_config: Quantization configuration for memory efficiency
            trust_remote_code: Whether to trust remote code in model files
            use_cache: Whether to use KV cache for faster generation
            low_cpu_mem_usage: Whether to use low CPU memory usage during loading
            **kwargs: Additional arguments passed to model loading
        """
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            dtype=dtype,
        )

        # Hugging Face specific attributes
        self.device_map = device_map
        self.quantization_config = quantization_config
        self.trust_remote_code = trust_remote_code if "phi-3" not in model_name.lower() else False
        self.use_cache = use_cache
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.max_retries = max_retries
        self.timeout = timeout

        # Model components
        self.tokenizer = None
        self.model = None
        self.device = None

        # Load model and tokenizer
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        """Load the model and tokenizer from Hugging Face."""
        try:
            logger.info("Loading model: %s", self.model_name)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left", token=HF_TOKEN if HF_TOKEN else None)

            # Set pad token iif not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine device
            if torch.cuda.is_available() and self.device_map != "cpu":
                self.device = "cuda"
                logger.info("Using CUDA for model inference")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and self.device_map != "cpu":
                self.device = "mps"
                logger.info("Using MPS for model inference")
            else:
                self.device = "cpu"
                logger.info("Using CPU for model inference")

            # Load model
            model_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "dtype": self.dtype,
                "device_map": self.device_map,
                # "use_cache": self.use_cache,
                "low_cpu_mem_usage": self.low_cpu_mem_usage,
            }

            if self.quantization_config:
                model_kwargs["quantization_config"] = self.quantization_config

            # if 'mediphi' in self.model_name.lower():
            #     model_kwargs["attn_implementation"] = "flash_attention_2"

            # Add any additional kwargs
            model_kwargs.update(kwargs)

            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=HF_TOKEN if HF_TOKEN else None, **model_kwargs)
            if 'mediphi' in self.model_name.lower():
                self.model.forward = torch.compile(self.model.forward)
            # if "huatuo" not in self.model_name.lower() and not self.model.generation_config.pad_token_id:
                # self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

            logger.info("Successfully loaded model: %s", self.model_name)

        except Exception as e:
            logger.error("Failed to load model %s: %s", self.model_name, str(e))
            raise

    def _tokenize(self, texts: Union[str, List[str]], return_tensors="pt", padding: bool = True, truncation: bool = True) -> dict:
        """
        Tokenize input text or list of texts.
        """
        tokenized = self.tokenizer(texts, return_tensors=return_tensors, padding=padding, truncation=truncation)

        # Move tokenized inputs to the same device as the model
        if hasattr(self.model, 'device'):
            model_device = next(self.model.parameters()).device
        else:
            model_device = torch.device(self.device)
        if "mediphi" in self.model_name.lower():
            return {k: v.to(model_device).contiguous() for k, v in tokenized.items()}
        return {k: v.to(model_device) for k, v in tokenized.items()}

    def __prepare_template_prompt(self, messages: List[Dict[str, str]]|List[List[Dict[str, str]]], reasoning_effort: str = "none") -> str:
        """
        Prepare template prompt for Hugging Face models.
        Args:
            messages: List of messages or list of list of messages
            reasoning_effort: Reasoning effort

        Returns:
            Template prompt
        """
        if "deepseek" in self.model_name.lower():
            # merge system prompt to user prompt
            messages = [
                {"role": "user", "content": "\n".join({messages[0]["content"],messages[1]["content"]})}
            ] if isinstance(messages[0],dict) else [
                [{"role": "user", "content": "\n".join({msg[0]["content"],msg[1]["content"]})}] for msg in messages
            ]

        if "qwen3" in self.model_name.lower():
            print(f"{self.model_name} Reasoning effort: {reasoning_effort}")
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False if reasoning_effort == "none" else True
            )
        elif "meditron" in self.model_name.lower():
            prompt = f"{messages[0]['content']}\n\n### User: {messages[1]['content']}\n### Assistant:" if isinstance(messages[0],dict) else [
                f"{msg[0]['content']}\n\n### User: {msg[1]['content']}\n### Assistant:" for msg in messages
            ]
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        return prompt


    def prepare_model_prompt(self, messages: List[Dict[str, str]] | List[List[Dict[str, str]]], return_tokens: bool = False, **kwargs) ->Tuple[dict, int|List[int]]:
        """
        Prepare model prompt for Hugging Face models.

        Args:
            messages: List of messages or list of list of messages
            return_tokens: Whether to return tokens
            **kwargs: Additional arguments

        Returns:
            model inputs (dict) and number of input tokens (int or list of int)
        """
        if self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        prompt = self.__prepare_template_prompt(messages, kwargs.get("reasoning_effort", "none"))

        model_inputs = self._tokenize(prompt, return_tensors="pt", padding=True, truncation=True)
        if return_tokens:
            num_input_tokens = (model_inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1).tolist()
            num_input_tokens = num_input_tokens[0] if len(num_input_tokens) == 1 else num_input_tokens # only one message
        else:
            num_input_tokens = None
        return model_inputs, num_input_tokens


    @torch.inference_mode()
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        return_num_tokens: bool = True,
        verbose: bool = False,
        return_logprobs: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the model.

        Args:
            messages: Input messages
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            do_sample: Whether to use sampling
            return_num_tokens: Whether to return token counts, default is True
            verbose: Whether to print verbose output
            return_logprobs: Whether to return log probabilities
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with generated text, input tokens, output tokens, and generation time
        """

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        model_inputs, num_input_tokens = self.prepare_model_prompt(messages, return_tokens=True)

        # Set generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            # "return_dict_in_generate": True,
            "output_scores": return_logprobs,
            "return_dict_in_generate": return_logprobs,
        }

        # Add any additional kwargs
        generation_kwargs.update(kwargs)

        # Generate
        generate_start_time = time.time()
        outputs = self.model.generate(**model_inputs, **generation_kwargs)

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[num_input_tokens:], # start from the first token of the generated text
            skip_special_tokens=True
        ) if not return_logprobs else self.tokenizer.decode(outputs.sequences[0][num_input_tokens:], skip_special_tokens=True)

        generated_logprobs = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)[0] if return_logprobs else None
        generation_time = time.time() - generate_start_time

        # Count output tokens
        num_output_tokens = outputs[0].shape[1] - num_input_tokens if return_num_tokens else None

        return {
            "generated_text": generated_text,
            "num_input_tokens": num_input_tokens if return_num_tokens else None,
            "num_output_tokens": num_output_tokens,
            "generation_time": generation_time,
            "logprobs": generated_logprobs if return_logprobs else None, # tensor
        }

    @torch.inference_mode()
    def batch_generate_response(self, messages: List[List[Dict[str, str]]], max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 40, return_num_tokens: bool = True, verbose: bool = False, return_logprobs: bool = False, reasoning_effort: str = "none", **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of messages.
        """

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        model_inputs, num_input_tokens = self.prepare_model_prompt(messages, return_tokens=True, reasoning_effort=reasoning_effort)

        # Set generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature if "mediphi" not in self.model_name.lower() else 0,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": True if "mediphi" not in self.model_name.lower() else False,
            # "return_dict_in_generate": True,
            "output_scores": return_logprobs,
            "return_dict_in_generate": return_logprobs,
        }
        # if "mediphi" in self.model_name.lower():
        #     generation_kwargs["stopping_criteria"] = EosListStoppingCriteria()

        # Add any additional kwargs
        generation_kwargs.update(kwargs)

        # Generate
        generate_start_time = time.time()
        outputs = self.model.generate(**model_inputs, **generation_kwargs)

        # Decode output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], outputs)
        ] if not return_logprobs else [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], outputs.sequences)
        ]
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_logprobs = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True) if return_logprobs else None
        generation_time = time.time() - generate_start_time

        # Count output tokens
        num_output_tokens = [generated_id.shape[0] for generated_id in generated_ids] if return_num_tokens else None

        return {
            "generated_text": generated_text,
            "num_input_tokens": num_input_tokens if return_num_tokens else None,
            "num_output_tokens": num_output_tokens,
            "generation_time": [generation_time]*len(messages),
            "logprobs": generated_logprobs.tolist() if return_logprobs else None, # list of tensors
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "Model not loaded"}

        info = {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": getattr(self.model.config, "max_position_embeddings", None),
            "hidden_size": getattr(self.model.config, "hidden_size", None),
            "num_layers": getattr(self.model.config, "num_hidden_layers", None),
            "num_attention_heads": getattr(self.model.config, "num_attention_heads", None),
        }

        return info

    def clear_cache(self) -> None:
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    @classmethod
    def create_quantized_model(
        cls,
        model_name: str,
        quantization_type: str = "4bit",
        device_map: str = "auto",
        **kwargs
    ) -> "HFCasualLLM":
        """
        Create a quantized model for memory efficiency.

        Args:
            model_name: Hugging Face model identifier
            quantization_type: Type of quantization ("4bit", "8bit")
            device_map: Device mapping strategy
            **kwargs: Additional arguments

        Returns:
            Quantized HFCasualLLM instance
        """
        if quantization_type == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization_type == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")

        return cls(
            model_name=model_name,
            device_map=device_map,
            quantization_config=quantization_config,
            **kwargs
        )

    # @classmethod
    # def get_supported_models(cls) -> List[str]:
    #     """Get a list of supported model names."""
    #     return [
    #         # "Qwen/QwQ-32B", # not enough memory
    #         "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    #         "Qwen/Qwen3-VL-30B-A3B-Instruct",
    #         "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    #         "mistralai/Magistral-Small-2509",
    #         "microsoft/Phi-4-reasoning",
    #     ]
