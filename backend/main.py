import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from core.api.server import create_app
from core.config import load_config

# Load configuration
config = load_config()

# Get API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Create FastAPI app with config
app = create_app(
    config=config,
    openai_api_key=openai_api_key,
    anthropic_api_key=anthropic_api_key
)