"""
Environment variable handling for the HomeAssistant Voice LLM project.
This module provides functions to load and validate environment variables.
"""

import os
from typing import Any, Dict, Optional
from pathlib import Path

def load_env_file(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from a .env file.
    
    Args:
        env_file: Path to the .env file. If None, looks for .env in project root.
    """
    from dotenv import load_dotenv
    
    if env_file is None:
        # Look for .env in project root
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / '.env'
    
    if env_file.exists():
        load_dotenv(env_file)
    else:
        print(f"Warning: {env_file} not found. Using system environment variables.")

def get_env_var(key: str, default: Any = None) -> Any:
    """
    Get an environment variable with type conversion.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        The environment variable value, converted to appropriate type
    """
    value = os.getenv(key, default)
    
    if value is None:
        return None
        
    # Convert string values to appropriate types
    if isinstance(value, str):
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    
    return value

def validate_required_vars(required_vars: Dict[str, Any]) -> None:
    """
    Validate that required environment variables are set.
    
    Args:
        required_vars: Dictionary of variable names and their default values
        
    Raises:
        ValueError: If any required variable is not set
    """
    missing = []
    for var, default in required_vars.items():
        if get_env_var(var, default) is None:
            missing.append(var)
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Required environment variables with defaults
REQUIRED_VARS = {
    'ENV': 'development',
    'DEBUG': 'true',
    'API_V1_PREFIX': '/api/v1',
    'BACKEND_HOST': 'backend',
    'BACKEND_PORT': '8000',
    'POSTGRES_USER': 'postgres',
    'POSTGRES_PASSWORD': 'postgres',
    'POSTGRES_DB': 'home_assistant',
    'POSTGRES_HOST': 'db',
    'POSTGRES_PORT': '5432',
    'CHROMADB_HOST': 'chroma',
    'CHROMADB_PORT': '8000',
    'VOICE_HOST': 'voice',
    'VOICE_PORT': '8001',
    'FRONTEND_HOST': 'frontend',
    'FRONTEND_PORT': '3000',
    'HOME_ASSISTANT_URL': 'http://supervisor/core',
    'HOME_ASSISTANT_TOKEN': None,  # No default, must be provided
    'WAKE_WORD': 'jarvis',
    'WAKE_WORD_SENSITIVITY': '0.5',
    'STT_MODEL': 'whisper',
    'TTS_MODEL': 'coqui'
}

# Load environment variables when module is imported
load_env_file()

# Validate required variables
try:
    validate_required_vars(REQUIRED_VARS)
except ValueError as e:
    print(f"Warning: {e}") 