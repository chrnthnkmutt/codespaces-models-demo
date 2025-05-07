import os
import base64
import asyncio
from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field
from azure.ai.foundry import FoundryClient
from azure.core.credentials import AzureKeyCredential


# Define Pydantic models for data validation
class ModelInfo(BaseModel):
    """Information about LLM capabilities"""
    vision: bool = True
    json_output: bool = False
    function_calling: bool = False
    family: str = "unknown"
    structured_output: bool = False


class AzureConfig(BaseModel):
    """Configuration for Azure AI Foundry"""
    endpoint: str
    api_key: str
    model_info: ModelInfo


class TextContent(BaseModel):
    """Text content for messages"""
    type: Literal["text"] = "text"
    data: str


class ImageContent(BaseModel):
    """Image content for messages"""
    type: Literal["image"] = "image"
    data: bytes
    format: str = "jpeg"


class MultiModalMessage(BaseModel):
    """Message containing text and/or images"""
    content: List[Union[TextContent, ImageContent]]
    source: str


class AssistantResponse(BaseModel):
    """Response from the assistant"""
    role: str = "assistant"
    content: str


class VisionAgent:
    """Agent that processes images using Azure AI Foundry"""
    
    def __init__(self, config: AzureConfig):
        """Initialize the agent with configuration"""
        self.config = config
        self.name = "vision_agent"
        self.client = FoundryClient(
            endpoint=self.config.endpoint,
            credential=AzureKeyCredential(self.config.api_key)
        )
    
    async def process_message(self, message: MultiModalMessage) -> AssistantResponse:
        """Process a multimodal message and return a response"""
        # Extract text and images from the message
        prompt = " ".join([item.data for item in message.content if item.type == "text"])
        image_items = [item for item in message.content if item.type == "image"]
        
        print(f"Processing request with {len(image_items)} image(s)")
        print(f"Prompt: {prompt}")
        
        try:
            # Prepare for API call
            print("\nSending request to Azure AI Foundry...")
            headers = {"Content-Type": "application/json"}
            
            # Prepare base64 encoded images
            image_contents = []
            for item in image_items:
                encoded = base64.b64encode(item.data).decode('utf-8')
                image_contents.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}
                })
            
            # Combine text and images into API payload
            payload = {
                "documents": [
                    {
                        "id": "1",
                        "language": "en",
                        "text": prompt
                    }
                ]
            }
            
            # Make the API call
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.endpoint}/foundry/v1.0/analyze", 
                    headers=headers, 
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API error (status {response.status}): {error_text}")
                    
                    result = await response.json()
                    return AssistantResponse(content=result["documents"][0]["entities"])
                    
        except Exception as e:
            error_message = f"Error processing image with Azure AI Foundry: {str(e)}"
            print(error_message)
            return AssistantResponse(content=f"Error: {str(e)}")


async def main():
    """Main entry point for the application"""
    print("=== Pydantic Vision Agent for Azure AI Foundry ===")
    print("Using structured data validation with Pydantic")
    
    # Check Azure AI Foundry requirements
    print("Prerequisites:")
    print("1. Make sure you have an Azure account and API key")
    print("2. Make sure you have the endpoint URL for Azure AI Foundry")
    
    # Initialize configuration
    config = AzureConfig(
        endpoint="https://<your-endpoint>.cognitiveservices.azure.com/",
        api_key="<your-api-key>",
        model_info=ModelInfo(vision=True)
    )
    
    # Initialize the agent
    agent = VisionAgent(config)
    
    # Prepare the image
    image_path = "GettyImages.jpg"
    
    # Verify the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        print(f"Current working directory: {os.getcwd()}")
        print("Available files:")
        print(os.listdir("."))
        return
    
    # Load the image data
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Create a structured multimodal message
    message = MultiModalMessage(
        content=[
            TextContent(data="Can you describe the content of this image?"),
            ImageContent(data=image_data)
        ],
        source="user"
    )
    
    # Process the message and get response
    response = await agent.process_message(message)
    print("\nResponse from Azure AI Foundry:")
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())