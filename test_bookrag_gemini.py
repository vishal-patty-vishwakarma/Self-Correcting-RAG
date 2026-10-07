import os

from Core.configs.llm_config import LLMConfig
from Core.provider.llm import LLM


config = LLMConfig(
    model_name="gemini-3.7-flash",
    api_key=os.environ["GEMINI_API_KEY"],
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    backend="openai",
    max_tokens=200,
    temperature=0.1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    max_workers=2,
)

llm = LLM(config)

response = llm.get_completion(
    "Reply with exactly: BookRAG Gemini connection successful"
)

print(response)