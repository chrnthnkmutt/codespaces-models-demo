 1. First, set up your .env file with the required credentials:

  # Edit the .env file
  nano .env

  2. Replace the placeholders with your actual credentials:
    - For GitHub provider: Enter your GitHub token
    - For Azure provider: Enter your Azure endpoint, API key, and version
    - For local OpenAI: Enter your OpenAI API key

  Running with GitHub provider

  # Step 1: Ensure GITHUB_TOKEN is set in .env
  # Step 2: Run the command
  python demo_agent_pydanticai.py --provider github

  Running with Azure provider

  # Step 1: Ensure these variables are set in .env:
  # - AZURE_ENDPOINT
  # - AZURE_API_KEY
  # - AZURE_API_VERSION

  # Step 2: Run the command
  python demo_agent_pydanticai.py --provider azure

  Running with local OpenAI provider

  # Step 1: Ensure OPENAI_API_KEY is set in .env
  # Step 2: Run the command
  python demo_agent_pydanticai.py --provider local

  Testing with custom queries

  # For example, with GitHub provider
  python demo_agent_pydanticai.py --provider github --query "What is the 
  capital of France?"

  # With Azure provider
  python demo_agent_pydanticai.py --provider azure --query "What is the 
  highest mountain in Japan?"

  # With local provider
  python demo_agent_pydanticai.py --provider local --query "What are the 
  largest cities in Brazil?"

  Using the debug version

  If you want more detailed information about what's happening:

  # Run with debug flag
  python demo_agent_pydanticai_simple_debug.py --provider azure --debug