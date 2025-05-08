from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.azure import AzureProvider
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class CityLocation(BaseModel):
    city: str
    country: str

def create_agent(provider_type='github', output_type=CityLocation):
    """
    Create an agent with specified provider.
    
    Args:
        provider_type: 'github', 'azure', or 'local'
        output_type: Pydantic model for structured output
    
    Returns:
        Configured Agent
    """
    if provider_type == 'github':
        api_key = os.environ.get("GITHUB_TOKEN")
        if not api_key:
            raise ValueError("GITHUB_TOKEN not set in environment")
            
        provider = OpenAIProvider(
            api_key=api_key,
            base_url='https://models.github.ai/inference'
        )
        model = OpenAIModel(
            model_name='openai/gpt-4o',
            provider=provider
        )
    
    elif provider_type == 'azure':
        azure_endpoint = os.environ.get("AZURE_ENDPOINT")
        azure_api_key = os.environ.get("AZURE_API_KEY")
        azure_api_version = os.environ.get("AZURE_API_VERSION", "2023-12-01-preview")
        
        if not azure_endpoint or not azure_api_key:
            raise ValueError("AZURE_ENDPOINT or AZURE_API_KEY not set in environment")
            
        provider = AzureProvider(
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            api_key=azure_api_key,
        )
        model = OpenAIModel(
            model_name='gpt-4o',  # Azure model deployment name
            provider=provider
        )
    
    elif provider_type == 'local':
        # For local setup using OpenAI directly
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
            
        provider = OpenAIProvider(
            api_key=openai_api_key
        )
        model = OpenAIModel(
            model_name='gpt-4o',
            provider=provider
        )
    
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")
        
    return Agent(model, output_type=output_type)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AI agent with different providers')
    parser.add_argument('--provider', type=str, default='github', 
                        choices=['github', 'azure', 'local'],
                        help='Provider to use: github, azure, or local')
    parser.add_argument('--query', type=str, default='Where were the olympics held in 2012?',
                        help='Query to send to the agent')
    
    args = parser.parse_args()
    
    try:
        # Make sure .env is loaded
        load_dotenv()
        
        # Check if running with Azure provider and environment variables are set
        if args.provider == 'azure':
            azure_endpoint = os.environ.get("AZURE_ENDPOINT")
            azure_api_key = os.environ.get("AZURE_API_KEY")
            if not azure_endpoint or not azure_api_key or azure_endpoint == "https://your-azure-endpoint.openai.azure.com/":
                print("Warning: Azure provider selected but configuration not properly set in .env file.")
                print("Please update the .env file with your actual Azure credentials.")
                # Continue anyway as the function will catch this error
        
        # Create agent with specified provider
        agent = create_agent(provider_type=args.provider)
        
        # Run the query
        print(f"Using provider: {args.provider}")
        print(f"Query: {args.query}")
        
        result = agent.run_sync(args.query)
        
        print("\nResult:")
        print(result.output)
        
        print("\nUsage:")
        print(result.usage())
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo use Azure provider, set these environment variables:")
        print("- AZURE_ENDPOINT: Your Azure OpenAI endpoint URL")
        print("- AZURE_API_KEY: Your Azure OpenAI API key")
        print("- AZURE_API_VERSION: API version (default: 2023-12-01-preview)")
        
        print("\nTo use Local provider, set:")
        print("- OPENAI_API_KEY: Your OpenAI API key")
        
        print("\nTo use Github provider, set:")
        print("- GITHUB_TOKEN: Your GitHub token")