from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import os # Import the os module
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

class CityLocation(BaseModel):
    city: str
    country: str

openai_api_key = os.environ.get("GITHUB_TOKEN")

openai_provider = OpenAIProvider(
    api_key=openai_api_key,
    base_url='https://models.github.ai/inference'
)

openai_model = OpenAIModel(
    model_name='openai/gpt-4o', # Changed model name for official OpenAI
    provider=openai_provider
)

agent = Agent(openai_model, output_type=CityLocation)

# Now this call will go to the official OpenAI API
result = agent.run_sync('Where were the olympics held in 2012?')

print(result.output)
# Expected output might look similar depending on the model response
# > city='London' country='United Kingdom'

print(result.usage())
# Usage details will reflect tokens used on the official OpenAI API
# """
# Usage(requests=1, request_tokens=XX, response_tokens=YY, total_tokens=ZZ, details=None)
# """