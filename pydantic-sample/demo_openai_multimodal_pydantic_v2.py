import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
# import openai # openai import is redundant if OpenAI is imported directly
from pydantic import BaseModel
from typing import List, Literal, Optional

# Load environment variables
load_dotenv()

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

class ImageAnalysisAgent:
    def __init__(self):
        # Ensure you have your OPENAI_API_KEY or GITHUB_TOKEN set as an environment variable
        # self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable not set.")
        self.client = OpenAI(base_url="https://models.github.ai/inference", api_key=github_token)

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encodes an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_image(self, image_path: str) -> ImageDescription:
        """
        Analyzes an image and returns a structured description.
        """
        try:
            base64_image = self._encode_image_to_base64(image_path)

            prompt_text = f"""Analyze this image and describe what you see, including any objects, the scene, colors and any text you can detect.
Please provide the output in a JSON format that strictly adheres to the following Pydantic schema:
{ImageDescription.model_json_schema()}
"""

            response = self.client.chat.completions.create(
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

            json_response_content = response.choices[0].message.content
            if json_response_content is None:
                raise ValueError("Received an empty response from the API.")
            image_description = ImageDescription.model_validate_json(json_response_content)
            return image_description

        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}. Please update the path.")
            raise
        except Exception as e:
            print(f"An error occurred during image analysis: {e}")
            raise

if __name__ == "__main__":
    image_path = './sample.png' # Replace with your actual image path
    
    # Ensure the image path is correct and the file exists.
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}. Please create a sample.png or update the path.")
    else:
        try:
            agent = ImageAnalysisAgent()
            description = agent.analyze_image(image_path)
            print(description.model_dump_json(indent=2))
        except ValueError as ve: # Catch specific errors like missing token
            print(f"Configuration error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")