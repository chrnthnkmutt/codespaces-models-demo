import os
import asyncio
import sys
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

async def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get GitHub token from environment variables
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Error: GITHUB_TOKEN not found in environment variables")
        sys.exit(1)
    
    try:
        # Initialize the Azure AI Chat Completion Client with vision capabilities
        client = AzureAIChatCompletionClient(
            model="meta/Llama-4-Scout-17B-16E-Instruct",
            endpoint="https://models.github.ai/inference",
            credential=AzureKeyCredential(github_token),
            model_info={
                "json_output": False,
                "function_calling": False,
                "vision": True,
                "family": "unknown",
                "structured_output": False,
            },
        )
    
        # Initialize the Assistant Agent with the client
        agent = AssistantAgent(
            name="vision_agent",
            model_client=client  # Changed from llm_config
        )
        
        try:
            # Fetch an image from the web
            image_url = "https://static.independent.co.uk/2025/04/19/19/41/GettyImages-2210861096.jpg"
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()  # Raise an exception for bad responses
            pil_image = PILImage.open(BytesIO(response.content))
            
            # Convert the PIL image to an AGImage
            ag_image = AGImage(pil_image)
            
            # Create a multimodal message with text and image
            multimodal_message = MultiModalMessage(
                content=["Can you describe the content of this image?", ag_image],
                source="user"
            )
            
            # Send the message to the agent and get the response
            response = await agent.on_messages([multimodal_message], None)  # Added None for cancellation_token
            print(response.chat_message)
            
        except requests.RequestException as e:
            print(f"Error fetching image: {e}")
        except Exception as e:
            print(f"Error processing request: {e}")
        finally:
            # Close the client
            await client.close()
    except Exception as e:
        print(f"Error initializing client: {e}")

if __name__ == "__main__":
    asyncio.run(main())
