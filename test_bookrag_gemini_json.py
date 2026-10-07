import os
from pydantic import BaseModel

from Core.configs.llm_config import LLMConfig
from Core.provider.llm import LLM


class TestSchema(BaseModel):
    topic: str
    answer: str


config = LLMConfig(
    model_name="gemini-3.7-flash",
    api_key=os.environ["GEMINI_API_KEY"],
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    backend="openai",
    max_tokens=500,
    temperature=0.1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    max_workers=2,
)

llm = LLM(config)

response = llm.get_json_completion(
    "The topic is photosynthesis. Give a short answer explaining what it is.",
    TestSchema,
)

print(response)
print("Topic:", response.topic)
print("Answer:", response.answer)