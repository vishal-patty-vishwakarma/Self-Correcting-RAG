import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-3.7-flash",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Gemini connection successful"
        }
    ]
)

print(response.choices[0].message.content)