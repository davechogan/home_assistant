# Shared Configuration

This directory contains shared configuration for the HomeAssistant Voice LLM project.

## Configuration Structure

The configuration system uses a combination of:
- Environment variables
- Default values
- Configuration classes

## Key Components

1. `config.py`: Base configuration module that:
   - Defines common settings
   - Handles environment variables
   - Provides type-safe configuration access
   - Manages project paths

2. Environment Variables:
   - Create a `.env` file in the project root
   - Copy from `.env.template` as a starting point
   - Never commit the actual `.env` file

## Required Environment Variables

### Core Settings
- `ENV`: Environment (development/production)
- `DEBUG`: Debug mode (true/false)

### API Settings
- `API_V1_PREFIX`: API version prefix
- `BACKEND_HOST`: Backend service host
- `BACKEND_PORT`: Backend service port

### Database Settings
- `POSTGRES_USER`: PostgreSQL username
- `POSTGRES_PASSWORD`: PostgreSQL password
- `POSTGRES_DB`: Database name
- `POSTGRES_HOST`: Database host
- `POSTGRES_PORT`: Database port

### Vector Database Settings
- `CHROMADB_HOST`: ChromaDB host
- `CHROMADB_PORT`: ChromaDB port

### Voice Service Settings
- `VOICE_HOST`: Voice service host
- `VOICE_PORT`: Voice service port

### Frontend Settings
- `FRONTEND_HOST`: Frontend service host
- `FRONTEND_PORT`: Frontend service port

### Home Assistant Settings
- `HOME_ASSISTANT_URL`: Home Assistant URL
- `HOME_ASSISTANT_TOKEN`: Long-lived access token

### Voice Pipeline Settings
- `WAKE_WORD`: Wake word for voice activation
- `WAKE_WORD_SENSITIVITY`: Wake word detection sensitivity
- `STT_MODEL`: Speech-to-text model
- `TTS_MODEL`: Text-to-speech model

## Usage

Import the configuration in your Python code:

```python
from shared.config.config import config

# Access settings
debug_mode = config.DEBUG
db_url = f"postgresql://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
```

## Security Notes

1. Never commit sensitive information in configuration files
2. Use environment variables for secrets
3. Keep `.env` files out of version control
4. Use different configurations for development and production 