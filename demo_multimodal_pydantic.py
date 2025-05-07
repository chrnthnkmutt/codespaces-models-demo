import os
import asyncio
import sys
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from io import BytesIO
import requests
from PIL import Image as PILImage
from azure.core.credentials import AzureKeyCredential
from autogen_core.models import UserMessage, ModelFamily
from autogen_core import Image as AGImage
from autogen_ext.models.azure import AzureAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import MultiModalMessage

# Pydantic models for structured data handling
class ModelInfo(BaseModel):
    json_output: bool = False
    function_calling: bool = False
    vision: bool = True
    family: str = "unknown"
    structured_output: bool = False

class ClientConfig(BaseModel):
    model: str = "meta/Llama-4-Scout-17B-16E-Instruct"
    endpoint: str = "https://models.github.ai/inference"
    credential: Optional[AzureKeyCredential] = None
    model_info: ModelInfo = Field(default_factory=ModelInfo)

class ImageSource(BaseModel):
    url: str
    timeout: int = 10

class MessageContent(BaseModel):
    text: str
    image: Optional[AGImage] = None

class VisionAgentConfig(BaseModel):
    name: str = "vision_agent"
    client_config: ClientConfig

async def fetch_image(image_source: ImageSource) -> AGImage:
    """Fetch an image from a URL and convert it to AGImage format."""
    try:
        response = requests.get(image_source.url, timeout=image_source.timeout)
        response.raise_for_status()
        pil_image = PILImage.open(BytesIO(response.content))
        return AGImage(pil_image)
    except requests.RequestException as e:
        raise ValueError(f"Error fetching image: {e}")

async def create_client(config: ClientConfig) -> AzureAIChatCompletionClient:
    """Create an Azure AI Chat Completion Client with the provided configuration."""
    return AzureAIChatCompletionClient(
        model=config.model,
        endpoint=config.endpoint,
        credential=config.credential,
        model_info=config.model_info.dict(),
    )

async def create_agent(config: VisionAgentConfig, client: AzureAIChatCompletionClient) -> AssistantAgent:
    """Create an Assistant Agent with the provided configuration and client."""
    return AssistantAgent(
        name=config.name,
        model_client=client
    )

async def process_image(agent: AssistantAgent, message: MessageContent) -> str:
    """Process an image with the agent and return the response."""
    content_list = [message.text]
    if message.image:
        content_list.append(message.image)
    
    multimodal_message = MultiModalMessage(
        content=content_list,
        source="user"
    )
    
    response = await agent.on_messages([multimodal_message], None)
    return response.chat_message

async def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get GitHub token from environment variables
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Error: GITHUB_TOKEN not found in environment variables")
        sys.exit(1)
        
    try:
        # Initialize configuration with Pydantic models
        client_config = ClientConfig(
            credential=AzureKeyCredential(github_token)
        )
        
        agent_config = VisionAgentConfig(
            client_config=client_config
        )
        
        # Create client and agent
        client = await create_client(client_config)
        agent = await create_agent(agent_config, client)
        
        try:
            # Define image source
            image_source = ImageSource(
                url="https://static.independent.co.uk/2025/04/19/19/41/GettyImages-2210861096.jpg"
            )
            
            # Fetch image
            image = await fetch_image(image_source)
            
            # Create message content
            message = MessageContent(
                text="Can you describe the content of this image?",
                image=image
            )
            
            # Process image and get response
            response = await process_image(agent, message)
            print(response)
            
        except Exception as e:
            print(f"Error processing request: {e}")
        finally:
            # Close the client
            await client.close()
    except Exception as e:
        print(f"Error initializing client: {e}")

if __name__ == "__main__":
    asyncio.run(main())
