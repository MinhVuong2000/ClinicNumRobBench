import os
import dotenv

dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
BEDROCK_API_KEY = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
BEDROCK_ENDPOINT = os.getenv("BEDROCK_ENDPOINT","https://bedrock-runtime.us-east-2.amazonaws.com")
