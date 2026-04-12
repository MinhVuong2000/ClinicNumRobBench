from typing import List, Dict, Any, Tuple, Union

import torch

class BaseLanguageModel:
    """
    Base lanuage model. Define how to generate response by using a LLM
    Args:
        model_name: name of the model
        api_key: api key for the model
        api_base: api base for the model
        api_version: api version for the model
        api_type: api type for the model
        kwargs: keyword arguments for the model
    """
    DTYPE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

    def __init__(self, model_name: str, dtype: str="fp16", api_key: str=None, api_base: str=None, api_version: str=None):
        """
        Initialize the model
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.dtype = self.DTYPE.get(dtype, None)

    def add_args(self, **kwargs):
        """
        Add arguments to the model

        Args:
            **kwargs: additional arguments for the model
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def load_model(self, **kwargs):
        raise NotImplementedError

    def _tokenize(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        Tokenize input text or list of texts.
        """
        raise NotImplementedError


    def prepare_model_prompt(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, int]:
        """
        Add model-specific prompt to the input

        Args:
            messages (List[Dict[str, str]]): messages
            **kwargs: additional arguments for the model
        Returns:
            Tuple[str, int]: prompt, number of input tokens
        """
        raise NotImplementedError

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate response from the model

        Args:
            messages (List[Dict[str, str]]): messages
            **kwargs: additional arguments for the model
        Returns:
            Dict[str, Any]: response
        """
        raise NotImplementedError

    def batch_generate_response(self, messages: List[List[Dict[str, str]]], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of messages

        Args:
            messages (List[List[Dict[str, str]]]): messages
            **kwargs: additional arguments for the model
        Returns:
            List[Dict[str, Any]]: responses
        """
        raise NotImplementedError

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model

        Returns:
            Dict[str, Any]: model information
        """
        raise NotImplementedError

    def clear_cache(self) -> None:
        """
        Clear the cache of the model
        """
        raise NotImplementedError

    def batch_generate(self, messages: List[List[Dict[str, str]]], **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of messages

        Args:
            messages (List[List[Dict[str, str]]]): messages
            **kwargs: additional arguments for the model
        Returns:
            List[Dict[str, Any]]: responses
        """
        raise NotImplementedError
