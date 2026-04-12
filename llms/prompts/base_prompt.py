from typing import List, Dict

class BasePrompt:
    @classmethod
    def prepare_prompt(cls, *args, prompt_type: str = "zero_shot", **kwargs) -> List[Dict[str, str]]:
        raise NotImplementedError
