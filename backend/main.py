import os
from dotenv import load_dotenv
from core.api.server import create_app
from core.config import load_config
from core.observability import setup_logging, get_logger

# Load environment variables first
load_dotenv()

# Load configuration
config = load_config()

# Setup logging
logger = setup_logging(
    log_level=config.observability.log_level,
    log_format=config.observability.log_format
)

logger.info("Starting FAIForge application", extra={
    'extra_fields': {
        'environment': os.getenv('ENV', 'development'),
        'log_level': config.observability.log_level
    }
})

# Get API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY not found in environment variables")

logger.info("Creating FastAPI application")

# Create FastAPI app with config
app = create_app(
    config=config,
    openai_api_key=openai_api_key,
    anthropic_api_key=anthropic_api_key,
    gemini_api_key=gemini_api_key,
    cohere_api_key=cohere_api_key,
)

logger.info("FAIForge application created successfully")