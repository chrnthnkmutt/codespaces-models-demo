import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
import openai
from pydantic import BaseModel
from typing import List, Literal, Optional # Make sure List, Literal, Optional are imported

# Ensure you have your OPENAI_API_KEY set as an environment variable
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI(base_url="https://models.github.ai/inference", api_key=os.environ["GITHUB_TOKEN"])

class Object(BaseModel):
  name: str
  confidence: float
  attributes: str 

class ImageDescription(BaseModel):
  summary: str
  objects: List[Object]
  scene: str
  colors: List[str]
  time_of_day: Literal['Morning', 'Afternoon', 'Evening', 'Night']
  setting: Literal['Indoor', 'Outdoor', 'Unknown']
  text_content: Optional[str] = None

# Function to encode the image
def encode_image_to_base64(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

path = './sample.png' # Replace with your actual image path
# Ensure the image path is correct and the file exists.
# For demonstration, let's assume you have a placeholder or an actual image.
# If the path is invalid, the encode_image_to_base64 function will fail.

try:
    base64_image = encode_image_to_base64(path)

    prompt_text = f"""Analyze this image and describe what you see, including any objects, the scene, colors and any text you can detect.
Please provide the output in a JSON format that strictly adheres to the following Pydantic schema:
{ImageDescription.model_json_schema()}
"""

    response = client.chat.completions.create(
      model="openai/gpt-4o", # Or "gpt-4-vision-preview"
      response_format={"type": "json_object"},
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": prompt_text},
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      temperature=0,
    )

    # Extract the JSON string from the response
    json_response_content = response.choices[0].message.content
    image_description = ImageDescription.model_validate_json(json_response_content)
    print(image_description)

except FileNotFoundError:
    print(f"Error: Image file not found at {path}. Please update the path.")
except Exception as e:
    print(f"An error occurred: {e}")