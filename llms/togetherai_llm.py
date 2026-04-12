"""
TogetherAI API Integration for Large Language Models

This module provides a comprehensive interface for using TogetherAI's API
to query language models with support for logprobs and batch processing.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer
from huggingface_hub import InferenceClient

from const import TOGETHERAI_API_KEY, HF_TOKEN
from llms.base_language_model import BaseLanguageModel


logger = logging.getLogger(__name__)


class TogetherAILLM(BaseLanguageModel):
    """
    A comprehensive interface for using TogetherAI's API for language models.

    Supports various features including:
    - Logprobs extraction
    - Batch processing
    - Custom generation parameters
    - Token counting
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: str = "https://api.together.xyz/v1",
        max_retries: int = 3,
        timeout: int = 120,
        num_workers: int = 4,
        **kwargs
    ):
        """
        Initialize the TogetherAI model.

        Args:
            model_name: TogetherAI model identifier (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
            api_key: TogetherAI API key (defaults to TOGETHERAI_API_KEY from const)
            api_base: Base URL for TogetherAI API
            max_retries: Maximum retries for API calls
            timeout: Timeout for API calls in seconds
            **kwargs: Additional arguments
        """
        # Initialize parent class
        super().__init__(
            model_name=model_name,
            dtype="fp16",  # Not applicable for API calls, but required by base class
            api_key=api_key or TOGETHERAI_API_KEY,
            api_base=api_base,
        )

        # TogetherAI specific attributes
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_base = api_base
        self.num_workers = num_workers
        if not self.api_key:
            raise ValueError("TogetherAI API key is required. Set TOGETHERAI_API_KEY environment variable or pass api_key parameter.")

        # Load model (validate API key)
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        """Initialize TogetherAI client and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left", token=HF_TOKEN if HF_TOKEN else None, trust_remote_code=True)

        try:
            logger.info("Initializing TogetherAI client for model: %s", self.model_name)
            # Initialize TogetherAI client
            self.client = InferenceClient(
                provider="together",
                api_key=self.api_key,
            )
            logger.info("Successfully initialized TogetherAI client")
        except Exception as e:
            logger.error("Failed to initialize TogetherAI client: %s", str(e))
            raise


    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        return_num_tokens: bool = True,
        verbose: bool = False,
        return_logprobs: bool = False,
        reasoning_effort: str = "low",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the TogetherAI API.

        Args:
            messages: Input messages
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            return_num_tokens: Whether to return token counts
            verbose: Whether to print verbose output
            return_logprobs: Whether to return log probabilities
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with generated text, input tokens, output tokens, generation time, and logprobs
        """

        # Set logprobs parameter
        top_logprobs = 1 if return_logprobs else 0

        # Generate
        if verbose:
            logger.info("Generating response with TogetherAI model: %s", self.model_name)
        generate_start_time = time.time()

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            # repetition_penalty=repetition_penalty,
            logprobs=return_logprobs,
            top_logprobs=top_logprobs,
            extra_body={
                "reasoning_effort": reasoning_effort if reasoning_effort in ["medium", "high"] else "low",
                "repetition_penalty": repetition_penalty,
            }
        )

        generated_text = completion.choices[0].message.content
        num_input_tokens = completion.usage.prompt_tokens
        num_output_tokens = completion.usage.completion_tokens
        generation_time = time.time() - generate_start_time
        if return_logprobs:
            if completion.choices[0].logprobs.token_logprobs:
                logprobs = completion.choices[0].logprobs.token_logprobs
            else:
                if 'gpt-oss' in self.model_name:
                    tokens = []
                    logprobs = []
                    for token in completion.choices[0].logprobs.content[3:-1]: # ignore the first 3 tokens: <channel>, analysis, <message>, ..., <|return|>
                        if token.token.startswith('<|') and token.token.endswith('|>'):
                            continue
                        tokens.append(token.token)
                        logprobs.append(token.logprob)
                    generated_text = ''.join(tokens)
                else:
                    raise ValueError(f"Logprobs are not supported for model {self.model_name}")
        else:
            logprobs = None

        return {
            "generated_text": generated_text,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "generation_time": generation_time,
            "logprobs": logprobs,
        }

    def batch_generate_response(
        self,
        messages: List[List[Dict[str, str]]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        return_num_tokens: bool = True,
        verbose: bool = False,
        return_logprobs: bool = False,
        reasoning_effort: str = "low",
        labels: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate responses for a batch of messages.

        Args:
            messages: List of message lists
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            return_num_tokens: Whether to return token counts
            verbose: Whether to print verbose output
            return_logprobs: Whether to return log probabilities
            labels: List of labels
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with generated texts, input tokens, output tokens, generation times, and logprobs
        """
        generated_texts = []
        num_input_tokens_list = []
        num_output_tokens_list = []
        generation_times = []
        logprobs_list = []

        # Process each prompt
        if verbose:
            logger.info("Generating batch responses with TogetherAI model: %s", self.model_name)

        # Use ThreadPoolExecutor with default NUM_WORKERS workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.generate_response, message, max_tokens, temperature, top_p, top_k, repetition_penalty, return_num_tokens, verbose, return_logprobs, reasoning_effort, **kwargs) for message in messages]

            for future in futures:
                response = future.result()
                generated_texts.append(response["generated_text"])
                num_input_tokens_list.append(response["num_input_tokens"])
                num_output_tokens_list.append(response["num_output_tokens"])
                generation_times.append(response["generation_time"])
                logprobs_list.append(response["logprobs"])

        return {
            "generated_text": generated_texts,
            "num_input_tokens": num_input_tokens_list if return_num_tokens else None,
            "num_output_tokens": num_output_tokens_list if return_num_tokens else None,
            "generation_time": generation_times,
            "logprobs": logprobs_list if return_logprobs else None,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "api_base": self.api_base,
            "provider": "TogetherAI",
        }

    def clear_cache(self) -> None:
        """Clear cache (no-op for API-based models)."""
        # API-based models don't maintain local cache
