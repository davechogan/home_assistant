"""
Base configuration module for the HomeAssistant Voice LLM project.
This module provides a centralized configuration system using environment variables.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .env import get_env_var, REQUIRED_VARS

class Settings(BaseSettings):
    """Base settings class for the application."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    SHARED_DIR: Path = PROJECT_ROOT / 'shared'
    CONFIG_DIR: Path = SHARED_DIR / 'config'
    
    # Environment settings
    ENV: str = Field(default_factory=lambda: get_env_var('ENV', REQUIRED_VARS['ENV']))
    DEBUG: bool = Field(default_factory=lambda: get_env_var('DEBUG', REQUIRED_VARS['DEBUG']))
    
    # API settings
    API_V1_PREFIX: str = Field(default_factory=lambda: get_env_var('API_V1_PREFIX', REQUIRED_VARS['API_V1_PREFIX']))
    BACKEND_HOST: str = Field(default_factory=lambda: get_env_var('BACKEND_HOST', REQUIRED_VARS['BACKEND_HOST']))
    BACKEND_PORT: int = Field(default_factory=lambda: get_env_var('BACKEND_PORT', REQUIRED_VARS['BACKEND_PORT']))
    
    # Database settings
    POSTGRES_USER: str = Field(default_factory=lambda: get_env_var('POSTGRES_USER', REQUIRED_VARS['POSTGRES_USER']))
    POSTGRES_PASSWORD: str = Field(default_factory=lambda: get_env_var('POSTGRES_PASSWORD', REQUIRED_VARS['POSTGRES_PASSWORD']))
    POSTGRES_DB: str = Field(default_factory=lambda: get_env_var('POSTGRES_DB', REQUIRED_VARS['POSTGRES_DB']))
    POSTGRES_HOST: str = Field(default_factory=lambda: get_env_var('POSTGRES_HOST', REQUIRED_VARS['POSTGRES_HOST']))
    POSTGRES_PORT: int = Field(default_factory=lambda: get_env_var('POSTGRES_PORT', REQUIRED_VARS['POSTGRES_PORT']))
    DATABASE_URL: str = Field(default_factory=lambda: get_env_var('DATABASE_URL'))
    
    # Vector database settings
    CHROMADB_HOST: str = Field(default_factory=lambda: get_env_var('CHROMADB_HOST', REQUIRED_VARS['CHROMADB_HOST']))
    CHROMADB_PORT: int = Field(default_factory=lambda: get_env_var('CHROMADB_PORT', REQUIRED_VARS['CHROMADB_PORT']))
    CHROMADB_URL: str = Field(default_factory=lambda: get_env_var('CHROMADB_URL'))
    CHROMA_DB_PATH: str = Field(default_factory=lambda: get_env_var('CHROMA_DB_PATH'))
    
    # Voice service settings
    VOICE_HOST: str = Field(default_factory=lambda: get_env_var('VOICE_HOST', REQUIRED_VARS['VOICE_HOST']))
    VOICE_PORT: int = Field(default_factory=lambda: get_env_var('VOICE_PORT', REQUIRED_VARS['VOICE_PORT']))
    
    # Frontend settings
    FRONTEND_HOST: str = Field(default_factory=lambda: get_env_var('FRONTEND_HOST', REQUIRED_VARS['FRONTEND_HOST']))
    FRONTEND_PORT: int = Field(default_factory=lambda: get_env_var('FRONTEND_PORT', REQUIRED_VARS['FRONTEND_PORT']))
    
    # Home Assistant settings
    HOME_ASSISTANT_URL: str = Field(default_factory=lambda: get_env_var('HOME_ASSISTANT_URL', REQUIRED_VARS['HOME_ASSISTANT_URL']))
    HOME_ASSISTANT_TOKEN: Optional[str] = Field(default_factory=lambda: get_env_var('HOME_ASSISTANT_TOKEN', REQUIRED_VARS['HOME_ASSISTANT_TOKEN']))
    HA_URL: str = Field(default_factory=lambda: get_env_var('HA_URL'))
    HA_TOKEN: str = Field(default_factory=lambda: get_env_var('HA_TOKEN'))
    
    # Voice pipeline settings
    WAKE_WORD: str = Field(default_factory=lambda: get_env_var('WAKE_WORD', REQUIRED_VARS['WAKE_WORD']))
    WAKE_WORD_SENSITIVITY: float = Field(default_factory=lambda: get_env_var('WAKE_WORD_SENSITIVITY', REQUIRED_VARS['WAKE_WORD_SENSITIVITY']))
    STT_MODEL: str = Field(default_factory=lambda: get_env_var('STT_MODEL', REQUIRED_VARS['STT_MODEL']))
    TTS_MODEL: str = Field(default_factory=lambda: get_env_var('TTS_MODEL', REQUIRED_VARS['TTS_MODEL']))
    PORCUPINE_ACCESS_KEY: str = Field(default_factory=lambda: get_env_var('PORCUPINE_ACCESS_KEY'))
    
    # LLM settings
    OLLAMA_URL: str = Field(default_factory=lambda: get_env_var('OLLAMA_URL'))
    OLLAMA_MODEL: str = Field(default_factory=lambda: get_env_var('OLLAMA_MODEL'))
    
    # Audio settings
    AUDIO_INPUT_DEVICE: Optional[int] = Field(default=None)
    AUDIO_OUTPUT_DEVICE: Optional[int] = Field(default=None)
    AUDIO_SAMPLE_RATE: int = Field(default=16000)
    AUDIO_CHANNELS: int = Field(default=1)
    AUDIO_FRAME_LENGTH: int = Field(default=512)
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True
    )

# Create a singleton instance
config = Settings()

# Export commonly used settings
__all__ = ['config'] 