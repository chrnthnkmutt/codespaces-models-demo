from pydantic import BaseModel, field_validator
import os
import json
from typing import Any, Dict, Type
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI

# Load environment variables
load_dotenv()

from typing import Any, Optional, List, Dict, Union

class CityLocation(BaseModel):
    city: str
    country: str

class Mountain(BaseModel):
    name: str
    height: str  # Height in meters
    location: str  # Country or region
    
class DynamicResponse(BaseModel):
    """A flexible model that can handle any type of response"""
    # Make it accept any fields without requiring a nested "response" field
    model_config = {
        "extra": "allow"
    }
    
    def __str__(self) -> str:
        """Pretty print the response"""
        return "\n".join(f"{k}: {v}" for k, v in self.model_dump().items())

class AgentResult:
    """Simple container for agent results"""
    def __init__(self, output, usage_info):
        self.output = output
        self._usage_info = usage_info
    
    def usage(self):
        return self._usage_info

class SimpleAgent:
    """A simplified agent implementation that doesn't require pydantic_ai"""
    
    def __init__(self, client, output_type: Type[BaseModel]):
        self.client = client
        self.output_type = output_type
    
    def run_sync(self, query: str) -> AgentResult:
        """Run the agent with a query and return structured output"""
        # Get the schema for the output type
        schema = self.output_type.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        
        # Create a system message based on the output type
        system_content = ""
        
        # Different instructions based on the schema type
        if self.output_type == DynamicResponse:
            system_content = """You are a helpful assistant that provides information in a structured JSON format.
Please provide your answer as a JSON object with relevant key-value pairs that best answer the user's question.
Be concise but informative. Return ONLY valid JSON without any preamble or explanation."""
        else:
            system_content = f"""You are a helpful assistant that outputs information in a specific JSON format.
The required output format follows this schema:
{schema_str}

Only return valid JSON that matches this schema, without any preamble or explanation."""
        
        # Create the messages with the appropriate system content
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]
        
        # Call the model - determine which model name format to use based on client type
        model_name = "gpt-4o"  # Default model name
        
        # Check if we're using AzureOpenAI client
        if isinstance(self.client, AzureOpenAI):
            # For Azure, we might need a deployment name instead of a model name
            # This name should match your deployment in Azure
            try:
                # Try with the model name defined in Azure
                response = self.client.chat.completions.create(
                    model=model_name,  # This should match your deployment name in Azure
                    response_format={"type": "json_object"},
                    messages=messages
                )
            except Exception as e:
                print(f"Warning: Error with model name '{model_name}'. Error: {str(e)}")
                print("Trying with Azure deployment name 'gpt-4'...")
                # Fallback to a common deployment name
                response = self.client.chat.completions.create(
                    model="gpt-4",  # Common Azure deployment name
                    response_format={"type": "json_object"},
                    messages=messages
                )
        else:
            # For GitHub and local OpenAI
            if "github.ai" in getattr(self.client, "base_url", ""):
                # GitHub AI models often need the 'openai/' prefix
                model_name = "openai/gpt-4o"
                
            response = self.client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                messages=messages
            )
        
        # Parse the response into the output type
        json_content = response.choices[0].message.content
        
        # Simple usage tracking
        usage_info = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        try:
            output = self.output_type.model_validate_json(json_content)
            return AgentResult(output, usage_info)
        except Exception as e:
            raise ValueError(f"Failed to parse model output: {e}\nOutput was: {json_content}")

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
            
        try:
            # Try standard initialization
            client = OpenAI(
                api_key=api_key,
                base_url='https://models.github.ai/inference'
            )
        except TypeError as e:
            if 'proxies' in str(e):
                # Handle the specific case of 'proxies' parameter
                import httpx
                client = OpenAI(
                    api_key=api_key,
                    base_url='https://models.github.ai/inference',
                    http_client=httpx.Client()
                )
            else:
                # Other TypeError, re-raise
                raise
    
    elif provider_type == 'azure':
        azure_endpoint = os.environ.get("AZURE_ENDPOINT")
        azure_api_key = os.environ.get("AZURE_API_KEY")
        azure_api_version = os.environ.get("AZURE_API_VERSION", "2023-12-01-preview")
        
        if not azure_endpoint or not azure_api_key:
            raise ValueError("AZURE_ENDPOINT or AZURE_API_KEY not set in environment")
            
        # Check OpenAI version and use the appropriate parameters
        try:
            # Different versions of openai library have different parameters
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version,
                api_key=azure_api_key
            )
        except TypeError as e:
            if 'proxies' in str(e):
                # Handle the specific case of 'proxies' parameter
                # This workaround handles the issue with some OpenAI library versions
                import httpx
                client = AzureOpenAI(
                    api_key=azure_api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=azure_api_version,
                    http_client=httpx.Client()
                )
            else:
                # Try with older parameter format
                client = AzureOpenAI(
                    api_key=azure_api_key,
                    azure_endpoint=azure_endpoint,
                    api_version=azure_api_version
                )
    
    elif provider_type == 'local':
        # For local setup using OpenAI directly
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
            
        try:
            # Try standard initialization
            client = OpenAI(
                api_key=openai_api_key
            )
        except TypeError as e:
            if 'proxies' in str(e):
                # Handle the specific case of 'proxies' parameter
                import httpx
                client = OpenAI(
                    api_key=openai_api_key,
                    http_client=httpx.Client()
                )
            else:
                # Other TypeError, re-raise
                raise
    
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")
        
    return SimpleAgent(client, output_type)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AI agent with different providers')
    parser.add_argument('--provider', type=str, default='github', 
                        choices=['github', 'azure', 'local'],
                        help='Provider to use: github, azure, or local')
    parser.add_argument('--query', type=str, default='Where were the olympics held in 2012?',
                        help='Query to send to the agent')
    parser.add_argument('--schema', type=str, choices=['city', 'mountain', 'dynamic'],
                        help='Schema to use for the response (defaults to dynamic)')
    
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
        
        # Use dynamic response for all queries by default
        output_type = DynamicResponse
        
        # Allow specifying a schema type via command line if needed
        if hasattr(args, 'schema') and args.schema:
            if args.schema == 'city':
                output_type = CityLocation
                print("Using CityLocation schema")
            elif args.schema == 'mountain':
                output_type = Mountain
                print("Using Mountain schema")
        
        # Create agent with specified provider and output type
        agent = create_agent(provider_type=args.provider, output_type=output_type)
        
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
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()