import os
import backoff
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from numeric_reasoning.configs.settings import settings


# begin new client
endpoint = os.getenv("ENDPOINT_URL", "https://zhuang-api.openai.azure.com/")
deployment_name = os.getenv("DEPLOYMENT_NAME", "gpt-4.1-mini")

# Initialize Azure OpenAI client with Entra ID authentication
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

provider_client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version="2025-01-01-preview",
)
# end new client

# Initialize Azure OpenAI client
old_client = AzureOpenAI(
    api_key=settings.llm.azure_openai.api_key,
    api_version="2024-07-01-preview",
    azure_endpoint=settings.llm.azure_openai.endpoint
)

@backoff.on_exception(backoff.expo, Exception, max_time=3)
def call_openai_chat_api(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    deployment_name: str = "gpt-4o-mini",
    temperature: float = 1,
    max_tokens: int = 1024
) -> str:
    """
    Makes an Azure OpenAI ChatCompletion request using:
        A system message (system_prompt) defining role & responsibilities.
        A user message (user_prompt) detailing the specific task & data.
    Returns the assistant's reply."""

    messages = [{"role": "system","content": system_prompt.strip()},{"role": "user","content": user_prompt.strip()}]

    if "4.1" in deployment_name or "5" in deployment_name:
        client = provider_client
    else:
        client = old_client

    try:
        response = client.chat.completions.create( # type: ignore
            model=deployment_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Azure OpenAI API error: {e}")
        return ""
