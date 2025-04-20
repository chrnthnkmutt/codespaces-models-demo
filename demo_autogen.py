import os
import asyncio
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
    # Initialize the Azure AI Chat Completion Client with vision capabilities
    client = AzureAIChatCompletionClient(
        model="gpt-4o",  # Replace with your vision-capable model name
        endpoint="https://models.inference.ai.azure.com",
        credential=AzureKeyCredential(os.environ["GITHUB_TOKEN"]),
        model_info={
            "json_output": False,
            "function_calling": False,
            "vision": True,
            "family": ModelFamily.GPT_4O,
            "structured_output": False,
        },
    )

    # Initialize the Assistant Agent with the client
    agent = AssistantAgent(
        name="vision_agent",
        llm_config={"model_client": client}
    )

    # Fetch an image from the web
    image_url = "https://picsum.photos/300/200"
    response = requests.get(image_url)
    pil_image = PILImage.open(BytesIO(response.content))

    # Convert the PIL image to an AGImage
    ag_image = AGImage(pil_image)

    # Create a multimodal message with text and image
    multimodal_message = MultiModalMessage(
        content=["Can you describe the content of this image?", ag_image],
        source="user"
    )

    # Send the message to the agent and get the response
    response = await agent.on_messages([multimodal_message])
    print(response.chat_message)

    # Close the client
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
