import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Validate API key exists
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Import app after env is loaded
from core.api.server import create_app

app = create_app(openai_api_key, anthropic_api_key)