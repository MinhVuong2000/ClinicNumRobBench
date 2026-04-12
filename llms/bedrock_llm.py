import math
import time
from typing import Optional, List, Dict, Any
import asyncio
import requests
import httpx

from const import BEDROCK_ENDPOINT, BEDROCK_API_KEY
from llms.base_language_model import BaseLanguageModel


class BedrockLLM(BaseLanguageModel):
    def __init__(self, model_name: str, api_key: str = BEDROCK_API_KEY, api_base: str = BEDROCK_ENDPOINT):
        super().__init__(model_name, api_key=api_key, api_base=api_base)
        print(BEDROCK_API_KEY[-5:])

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


    def generate_response(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 2048,
        reasoning_effort: str = None,
        return_logprobs: bool = False,
    ) -> Dict[str, Any]:
        """
        Sync call to Bedrock.
        """
        try:
            generate_start_time = time.time()
            result = self._call_bedrock_model(
                model_name=self.model_name,
                messages=messages,
                temperature=temperature or 0.7,
                top_p=top_p or 0.9,
                max_tokens=max_tokens or 2048,
            )
            generation_time = time.time() - generate_start_time

            return {
                "generated_text": result["generated_text"],
                "num_input_tokens": result["num_input_tokens"],
                "num_output_tokens": result["num_output_tokens"],
                "generation_time": generation_time,
                "logprobs": None,
            }
        except Exception as e:
            print(f"Bedrock request failed: {e}")
            raise Exception(f"Bedrock request failed: {e}")

    async def generate_response_async(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 2048,
        reasoning_effort: str = None,
        return_logprobs: bool = False
    ) -> Dict[str, Any]:
        """
        Async call to Bedrock.
        """
        try:
            generate_start_time = time.time()
            result = await self._call_bedrock_model_async(
                model_name=self.model_name,
                messages=messages,
                temperature=temperature or 0.7,
                top_p=top_p or 0.9,
                max_tokens=max_tokens or 2048,
            )
            generation_time = time.time() - generate_start_time

            return {
                "generated_text": result["generated_text"],
                "num_input_tokens": result["num_input_tokens"],
                "num_output_tokens": result["num_output_tokens"],
                "generation_time": generation_time,
                "logprobs": None,
            }
        except Exception as e:
            print(f"Bedrock request failed: {e}")
            raise Exception(f"Bedrock request failed: {e}")

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
                    "logprobs": None,
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


    def _call_bedrock_model(self, model_name: str, messages: List[Dict[str, Any]], max_tokens: int = 2048, temperature: float = 0.7, top_p: float = 0.9, return_logprobs: bool = False) -> Dict[str, Any]:
        """
        Call a model via AWS Bedrock.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens in the response
            temperature: Controls randomness

        Returns:
            Dict[str, Any]: {generated_text, num_input_tokens, num_output_tokens}
        """

        if return_logprobs:
            logger.warning("Logprobs are not supported for BedrockLLM")

        if not self.api_key:
            raise ValueError("AWS_BEARER_TOKEN_BEDROCK environment variable not set")

        url = f"{self.api_base}/model/{model_name}/converse"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": msg["content"]} for msg in messages if msg["role"] == "user"]
                }
            ],
            "system": [{
                "text": msg["content"]
            } for msg in messages if msg["role"] == "system"],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            }
        }

        response = requests.post(url, json=body, headers=headers)

        if not response.ok:
            print(f"Error {response.status_code}: {response.text}")
            response.raise_for_status()

        response_body = response.json()

        generated_text = ""
        # Converse API response format
        if "output" in response_body and "message" in response_body["output"]:
            generated_text = response_body["output"]["message"]["content"][0]["text"].strip()
        elif "generation" in response_body:
            generated_text = response_body["generation"]

        usage = response_body.get("usage", {})


        return {
            "generated_text": generated_text,
            "num_input_tokens": usage.get("inputTokens", 0),
            "num_output_tokens": usage.get("outputTokens", 0),
            "logprobs": None,
        }

    async def _call_bedrock_model_async(self, model_name: str, messages: List[Dict[str, Any]], max_tokens: int = 2048, temperature: float = 0.7, top_p: float = 0.9, return_logprobs: bool = False) -> Dict[str, Any]:
        """
        Call a model via AWS Bedrock asynchronously.
        """
        if return_logprobs:
            logger.warning("Logprobs are not supported for BedrockLLM")

        if not self.api_key:
            raise ValueError("AWS_BEARER_TOKEN_BEDROCK environment variable not set")

        url = f"{self.api_base}/model/{model_name}/converse"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": msg["content"]} for msg in messages if msg["role"] == "user"]
                }
            ],
            "system": [{
                "text": msg["content"]
            } for msg in messages if msg["role"] == "system"],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            }
        }

        async with httpx.AsyncClient(timeout=100.0) as client:
            response = await client.post(url, json=body, headers=headers)

            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                response.raise_for_status()

            response_body = response.json()

            generated_text = ""
            if "output" in response_body and "message" in response_body["output"]:
                generated_text = response_body["output"]["message"]["content"][0]["text"].strip()
            elif "generation" in response_body:
                generated_text = response_body["generation"]

            usage = response_body.get("usage", {})

            return {
                "generated_text": generated_text,
                "num_input_tokens": usage.get("inputTokens", 0),
                "num_output_tokens": usage.get("outputTokens", 0),
                "logprobs": None,
            }
