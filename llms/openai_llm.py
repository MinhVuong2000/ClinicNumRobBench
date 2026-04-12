import math
import time
from typing import Union, Iterator, Optional, List, Dict, Any
import asyncio
import openai

from const import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
from llms.base_language_model import BaseLanguageModel


class OpenAILLM(BaseLanguageModel):
    def __init__(self, model_name: str, api_key: str = AZURE_OPENAI_API_KEY, api_base: str = AZURE_OPENAI_ENDPOINT, api_version: str = AZURE_OPENAI_API_VERSION):
        super().__init__(model_name, api_key=api_key, api_base=api_base, api_version=api_version)
        self.load_model()

    def load_model(self, **kwargs):
        if "azure" in self.api_base:
            self.client = openai.AzureOpenAI(api_key=self.api_key, azure_endpoint=self.api_base, api_version=self.api_version)
            self.async_client = openai.AsyncAzureOpenAI(api_key=self.api_key, azure_endpoint=self.api_base, api_version=self.api_version)
        else:
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
            self.async_client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)

    def _build_params(
        self,
        messages: list[dict],
        model_name: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        reasoning_effort: str,
        prompt_cache_key: Optional[str] = None,
    ) -> dict:
        if "gpt" in model_name:
            max_token_param_name = "max_completion_tokens"
        else:
            max_token_param_name = "max_tokens"

        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            max_token_param_name: max_tokens,
        }
        if prompt_cache_key:
            params["prompt_cache_key"] = prompt_cache_key

        if "gpt-4.1" in model_name:
            params.pop("top_k")
        elif "gpt-5" in model_name:
            params["temperature"] = 1
            params.pop("top_p")
            params.pop("top_k")
            params["reasoning_effort"] = reasoning_effort
        return params


    def generate_response(
        self,
        messages: list[dict],
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        max_tokens: int = None,
        reasoning_effort: str = None,
        return_logprobs: bool = False,
    ) -> Union[str, Iterator[str]]:
        """
        Sync call to OpenAI.
        - stream=False: trả về str (hoặc parsed JSON khi dùng beta.parse).
        - stream=True: trả về Iterator[str], mỗi lần next() một chunk.
        """

        try:
            params = self._build_params(
                messages=messages,
                model_name=self.model_name,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            if return_logprobs:
                params["logprobs"] = True
                params["top_logprobs"] = 1

            generate_start_time = time.time()
            completion_or_stream = self.client.chat.completions.create(**params)
            num_input_tokens = completion_or_stream.usage.prompt_tokens
            num_output_tokens = completion_or_stream.usage.completion_tokens
            generation_time = time.time() - generate_start_time
            generated_text = completion_or_stream.choices[0].message.content
            logprobs = None
            if return_logprobs:
                logprobs = [token.logprob for token in completion_or_stream.choices[0].logprobs.content]
            return {
                "generated_text": generated_text,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "generation_time": generation_time,
                "logprobs": logprobs,
            }
        except Exception as e:
            print(f"OpenAI request failed: {e}")
            raise Exception(f"OpenAI request failed: {e}")

    async def generate_response_async(self, messages: list[dict], temperature: float = None, top_p: float = None, top_k: int = None, max_tokens: int = None, reasoning_effort: str = None, return_logprobs: bool = False) -> Union[str, Iterator[str]]:
        try:
            params = self._build_params(
                messages=messages,
                model_name=self.model_name,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            if return_logprobs:
                if 'gpt-5' in self.model_name:
                    raise NotImplementedError("Logprobs are not supported for gpt-5 models")
                else:
                    params["logprobs"] = True
                    params["top_logprobs"] = 1

            generate_start_time = time.time()
            completion_or_stream = await self.async_client.chat.completions.create(**params)
            num_input_tokens = completion_or_stream.usage.prompt_tokens
            num_output_tokens = completion_or_stream.usage.completion_tokens
            generation_time = time.time() - generate_start_time
            generated_text = completion_or_stream.choices[0].message.content
            logprobs = None
            if return_logprobs:
                logprobs = [token.logprob for token in completion_or_stream.choices[0].logprobs.content]

            return {
                "generated_text": generated_text,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "generation_time": generation_time,
                "logprobs": logprobs,
            }
        except Exception as e:
            print(f"OpenAI request failed: {e}")
            raise Exception(f"OpenAI request failed: {e}")

    async def batch_generate_response_async(self, messages_batch: List[List[dict]], temperature: float = None, top_p: float = None, top_k: int = None, max_tokens: int = None, reasoning_effort: str = None, labels: List[dict] = None, return_logprobs: bool = False) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of messages asynchronously
        """

        async def generate_single(messages, label):
            if label and label["num_output_tokens"] and not math.isnan(label["num_output_tokens"]) and (label["num_output_tokens"]==max_tokens or (isinstance(label["generated_text"], str) and len(label["generated_text"]) > 0)):
                return {
                    "generated_text": label["generated_text"],
                    "num_input_tokens": label["num_input_tokens"],
                    "num_output_tokens": label["num_output_tokens"],
                    "generation_time": label["generation_time"],
                    "logprobs": label["logprobs"] if "logprobs" in label else None,
                }
            return await self.generate_response_async(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                return_logprobs=return_logprobs,
            )

        # Create tasks for all messages
        labels = [labels] * len(messages_batch) if labels is None else labels
        tasks = [generate_single(messages, label) for messages, label in zip(messages_batch, labels)]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Extract individual components
        generated_texts = [result.get("generated_text") for result in results]
        num_input_tokens = [result.get("num_input_tokens") for result in results]
        num_output_tokens = [result.get("num_output_tokens") for result in results]
        generation_times = [result.get("generation_time") for result in results]
        logprobs = [result.get("logprobs") for result in results]

        return {
            "generated_text": generated_texts,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "generation_time": generation_times,
            "logprobs": logprobs,
        }

    def batch_generate_response(self, messages_batch: List[List[dict]], temperature: float = None, top_p: float = None, top_k: int = None, max_tokens: int = None, reasoning_effort: str = None, labels: List[dict] = None, return_logprobs: bool = False) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of messages synchronously
        """
        return asyncio.run(self.batch_generate_response_async(messages_batch, temperature, top_p, top_k, max_tokens, reasoning_effort, labels, return_logprobs))
