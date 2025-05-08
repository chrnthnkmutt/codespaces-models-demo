from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.azure import AzureProvider
import os
import logging
import json
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
def setup_logger(debug_level):
    """Set up the logger with the specified debug level."""
    logging_level = getattr(logging, debug_level.upper())
    
    # Create a logger
    logger = logging.getLogger("pydantic_agent")
    logger.setLevel(logging_level)
    
    # Clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

class CityLocation(BaseModel):
    city: str
    country: str

def create_agent(provider_type='github', output_type=CityLocation, debug_level='INFO'):
    """
    Create an agent with specified provider.
    
    Args:
        provider_type: 'github', 'azure', or 'local'
        output_type: Pydantic model for structured output
        debug_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured Agent
    """
    logger = setup_logger(debug_level)
    logger.info(f"Creating agent with provider: {provider_type}")
    
    if provider_type == 'github':
        api_key = os.environ.get("GITHUB_TOKEN")
        if not api_key:
            logger.error("GITHUB_TOKEN not set in environment")
            raise ValueError("GITHUB_TOKEN not set in environment")
        
        logger.debug("Configuring GitHub provider")    
        provider = OpenAIProvider(
            api_key=api_key,
            base_url='https://models.github.ai/inference'
        )
        model = OpenAIModel(
            model_name='openai/gpt-4o',
            provider=provider
        )
        logger.info("GitHub provider configured successfully")
    
    elif provider_type == 'azure':
        azure_endpoint = os.environ.get("AZURE_ENDPOINT")
        azure_api_key = os.environ.get("AZURE_API_KEY")
        azure_api_version = os.environ.get("AZURE_API_VERSION", "2023-12-01-preview")
        
        if not azure_endpoint or not azure_api_key:
            logger.error("AZURE_ENDPOINT or AZURE_API_KEY not set in environment")
            raise ValueError("AZURE_ENDPOINT or AZURE_API_KEY not set in environment")
        
        logger.debug(f"Configuring Azure provider with endpoint: {azure_endpoint}, version: {azure_api_version}")
        provider = AzureProvider(
            azure_endpoint=azure_endpoint,
            api_version=azure_api_version,
            api_key=azure_api_key,
        )
        model = OpenAIModel(
            model_name='gpt-4o',  # Azure model deployment name
            provider=provider
        )
        logger.info("Azure provider configured successfully")
    
    elif provider_type == 'local':
        # For local setup using OpenAI directly
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not set in environment")
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        logger.debug("Configuring local OpenAI provider")    
        provider = OpenAIProvider(
            api_key=openai_api_key
        )
        model = OpenAIModel(
            model_name='gpt-4o',
            provider=provider
        )
        logger.info("Local OpenAI provider configured successfully")
    
    else:
        logger.error(f"Unsupported provider type: {provider_type}")
        raise ValueError(f"Unsupported provider type: {provider_type}")
    
    logger.debug(f"Creating agent with output type: {output_type.__name__}")
    agent = Agent(model, output_type=output_type)
    logger.info("Agent created successfully")
    
    return agent, logger

def dump_result(result, logger, debug_level):
    """Log detailed information about the result based on debug level"""
    if debug_level.upper() == 'DEBUG':
        # Extract and log all available information from the result
        logger.debug("===== DETAILED RESULT =====")
        logger.debug(f"Output: {result.output}")
        logger.debug(f"Usage: {result.usage()}")
        
        # Try to access and log additional attributes that might be available
        try:
            # Some implementations might have raw response data
            if hasattr(result, 'raw_response'):
                logger.debug(f"Raw response: {json.dumps(result.raw_response, indent=2)}")
            
            # Log any available model info
            if hasattr(result, 'model_info'):
                logger.debug(f"Model info: {result.model_info}")
            
            # Log any available prompt details
            if hasattr(result, 'prompt'):
                logger.debug(f"Prompt: {result.prompt}")
        except Exception as e:
            logger.debug(f"Error accessing additional result attributes: {e}")
        
        logger.debug("===========================")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AI agent with different providers (Debug Version)')
    parser.add_argument('--provider', type=str, default='github', 
                        choices=['github', 'azure', 'local'],
                        help='Provider to use: github, azure, or local')
    parser.add_argument('--query', type=str, default='Where were the olympics held in 2012?',
                        help='Query to send to the agent')
    parser.add_argument('--debug', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the debug level')
    parser.add_argument('--output-json', action='store_true',
                        help='Output result as JSON')
    
    args = parser.parse_args()
    
    try:
        # Create agent with specified provider and debug level
        agent, logger = create_agent(
            provider_type=args.provider, 
            debug_level=args.debug
        )
        
        # Run the query
        logger.info(f"Sending query: {args.query}")
        result = agent.run_sync(args.query)
        
        # Log detailed result information if in debug mode
        dump_result(result, logger, args.debug)
        
        # Output results
        if args.output_json:
            # Output JSON format
            json_result = {
                "provider": args.provider,
                "query": args.query,
                "result": result.output.model_dump(),
                "usage": str(result.usage())
            }
            print(json.dumps(json_result, indent=2))
        else:
            # Output human-readable format
            print(f"Using provider: {args.provider}")
            print(f"Query: {args.query}")
            print("\nResult:")
            print(result.output)
            print("\nUsage:")
            print(result.usage())
        
    except ValueError as e:
        logger = logging.getLogger("pydantic_agent")
        logger.error(f"Configuration error: {e}")
        
        print(f"Error: {e}")
        print("\nTo use Azure provider, set these environment variables:")
        print("- AZURE_ENDPOINT: Your Azure OpenAI endpoint URL")
        print("- AZURE_API_KEY: Your Azure OpenAI API key")
        print("- AZURE_API_VERSION: API version (default: 2023-12-01-preview)")
        
        print("\nTo use Local provider, set:")
        print("- OPENAI_API_KEY: Your OpenAI API key")
        
        print("\nTo use Github provider, set:")
        print("- GITHUB_TOKEN: Your GitHub token")
        
        sys.exit(1)
    except Exception as e:
        logger = logging.getLogger("pydantic_agent")
        logger.exception(f"Unexpected error: {e}")
        
        print(f"Unexpected error: {e}")
        sys.exit(1)