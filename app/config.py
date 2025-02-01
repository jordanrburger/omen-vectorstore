import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)

def load_config():
    """Load configuration from environment variables."""
    config = {
        # Keboola configuration
        'KEBOOLA_API_URL': os.getenv('KEBOOLA_API_URL', 'https://connection.keboola.com'),
        'KEBOOLA_TOKEN': os.getenv('KEBOOLA_TOKEN'),

        # OpenAI configuration (optional)
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'OPENAI_MODEL': os.getenv('OPENAI_MODEL', 'text-embedding-ada-002'),

        # Sentence Transformer configuration (optional)
        'SENTENCE_TRANSFORMER_MODEL': os.getenv('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2'),

        # Qdrant configuration
        'QDRANT_HOST': os.getenv('QDRANT_HOST', 'localhost'),
        'QDRANT_PORT': int(os.getenv('QDRANT_PORT', '55000')),
        'QDRANT_COLLECTION': os.getenv('QDRANT_COLLECTION', 'keboola_metadata'),
    }

    # Validate required configuration
    if not config['KEBOOLA_TOKEN']:
        raise ValueError("KEBOOLA_TOKEN environment variable is required")

    return config
