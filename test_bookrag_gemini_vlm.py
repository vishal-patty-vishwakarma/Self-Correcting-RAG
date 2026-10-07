import os
from PIL import Image, ImageDraw

from Core.configs.vlm_config import VLMConfig
from Core.provider.vlm import VLM


# Create a tiny test image
image_path = "gemini_vlm_test.jpg"

img = Image.new("RGB", (400, 200), "white")
draw = ImageDraw.Draw(img)
draw.rectangle((50, 50, 150, 150), fill="red")
draw.text((190, 90), "RED BOX", fill="black")
img.save(image_path)


config = VLMConfig(
    model_name="gemini-3.7-flash",
    api_key=os.environ["GEMINI_API_KEY"],
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    backend="gpt",
    temperature=0.1,
    max_tokens=500,
)

vlm = VLM(config)

response = vlm.generate(
    "Describe what you see in this image in one short sentence.",
    images=[image_path],
)

print(response)